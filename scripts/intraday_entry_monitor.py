import os
import smtplib
from email.message import EmailMessage

import numpy as np
import pandas as pd
import yfinance as yf

from common import DATA, CONFIG, load_json, save_json, now_iso


def rsi_wilder(close, period=10):
    close = pd.to_numeric(close, errors="coerce").dropna()
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    safe_loss = avg_loss.where(avg_loss != 0, np.nan)
    rs = avg_gain / safe_loss
    rsi = (100.0 - (100.0 / (1.0 + rs))).astype("float64")
    rsi.loc[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    rsi.loc[(avg_gain == 0) & (avg_loss > 0)] = 0.0
    rsi.loc[(avg_gain == 0) & (avg_loss == 0)] = 50.0
    return rsi


def send_email(subject, body):
    username = os.getenv("SMTP_USERNAME", "").strip()
    password = os.getenv("SMTP_APP_PASSWORD", "").strip()
    if not username or not password:
        print("SMTP secrets missing; email skipped.")
        return
    msg = EmailMessage()
    msg["From"] = username
    msg["To"] = CONFIG["alert_email"]
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
        server.starttls()
        server.login(username, password)
        server.send_message(msg)


def main():
    universe = load_json(DATA / "universe.json", [])
    state = load_json(DATA / "state.json", {"active_signals": {}, "completed_trades": []})
    active = state.setdefault("active_signals", {})

    market_hist = yf.download(CONFIG["market_ticker"], period="18mo", interval="1d", auto_adjust=False, progress=False)
    if market_hist.empty:
        raise RuntimeError("Could not download ASX 200 history")
    market_close = market_hist["Close"]
    if isinstance(market_close, pd.DataFrame):
        market_close = market_close.iloc[:, 0]
    market_close = pd.to_numeric(market_close, errors="coerce").dropna()
    market_sma = market_close.rolling(CONFIG["market_sma_period"]).mean().iloc[-1]
    if pd.isna(market_sma) or float(market_close.iloc[-1]) <= float(market_sma):
        print("Market below SMA200; no new intraday entries allowed.")
        return

    new_entries = []
    batch = int(CONFIG.get("scan_batch_size", 250))
    for start in range(0, len(universe), batch):
        items = universe[start:start + batch]
        items = [x for x in items if x["symbol"] not in active]
        if not items:
            continue
        tickers = [x["ticker"] for x in items]
        try:
            daily = yf.download(tickers, period="1y", interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)
            live = yf.download(tickers, period="5d", interval="5m", group_by="ticker", auto_adjust=False, threads=True, progress=False)
        except Exception as exc:
            print("Chunk download error:", exc)
            continue
        multi = len(tickers) > 1
        for item in items:
            sym, ticker = item["symbol"], item["ticker"]
            try:
                dh = daily[ticker].copy() if multi else daily.copy()
                ih = live[ticker].copy() if multi else live.copy()
                dc = pd.to_numeric(dh["Close"], errors="coerce").dropna()
                ic = pd.to_numeric(ih["Close"], errors="coerce").dropna()
                if len(dc) < 200 or ic.empty:
                    continue
                price = float(ic.iloc[-1])
                sma200 = float(dc.rolling(200).mean().iloc[-1])
                if pd.isna(sma200) or price <= sma200:
                    continue

                closes = dc.copy()
                today = pd.Timestamp.now(tz="Australia/Sydney").date()
                if closes.index[-1].date() == today:
                    closes.iloc[-1] = price
                else:
                    closes.loc[pd.Timestamp(today)] = price
                rs = rsi_wilder(closes, CONFIG["rsi_period"]).dropna()
                if rs.empty:
                    continue
                current_rsi = float(rs.iloc[-1])
                if current_rsi >= CONFIG["entry_rsi_below"]:
                    continue

                entry = {
                    "symbol": sym,
                    "company": item["company"],
                    "ticker": ticker,
                    "entry_date": str(today),
                    "entry_price": price,
                    "entry_rsi10": current_rsi,
                    "entry_sma200": sma200,
                    "holding_trading_days": 0,
                    "latest_price": price,
                    "latest_rsi10": current_rsi,
                    "entry_source": "intraday_monitor",
                    "entry_observed_at": now_iso(),
                }
                active[sym] = entry
                new_entries.append(entry)
                print(f"ENTRY {sym}: RSI {current_rsi:.2f}, price {price:.4f}")
            except Exception as exc:
                print(f"{sym}: {exc}")

    if not new_entries:
        print("No new intraday RSI<30 entries.")
        return

    state["active_signals"] = active
    save_json(DATA / "state.json", state)

    scanner = load_json(DATA / "scanner.json", {})
    scanner["generated_at"] = now_iso()
    scanner["active_signals"] = list(active.values())
    scanner["entries_today"] = list(scanner.get("entries_today", [])) + new_entries

    # Merge newly detected intraday entries into the scanner rows immediately.
    # This lets Current Entry Conditions display them and their exact detection time
    # without waiting for the next full daily scan.
    stocks = list(scanner.get("stocks", []))
    stock_index = {row.get("symbol"): i for i, row in enumerate(stocks)}
    for entry in new_entries:
        row = {
            "symbol": entry["symbol"],
            "company": entry["company"],
            "ticker": entry["ticker"],
            "date": entry["entry_date"],
            "price": entry["latest_price"],
            "rsi10": entry["latest_rsi10"],
            "sma200": entry["entry_sma200"],
            "above_sma200": True,
            "avg_volume_20d": 0,
            "active": True,
        }
        if entry["symbol"] in stock_index:
            stocks[stock_index[entry["symbol"]]] = row
        else:
            stocks.append(row)
    scanner["stocks"] = stocks

    stats = scanner.setdefault("stats", {})
    stats["active"] = len(active)
    stats["entries_today"] = len(scanner["entries_today"])
    save_json(DATA / "scanner.json", scanner)

    lines = ["New intraday ASX RSI(10) < 30 entry signals:", ""]
    for x in new_entries:
        lines.append(f"{x['symbol']} — RSI {x['entry_rsi10']:.2f} — price ${x['entry_price']:.4f} — SMA200 ${x['entry_sma200']:.4f}")
    lines += ["", "Entry rules verified: RSI(10) < 30, stock above SMA200, ASX 200 above SMA200."]
    send_email(f"ASX RSI scanner: {len(new_entries)} new intraday entr{'y' if len(new_entries)==1 else 'ies'}", "\n".join(lines))


if __name__ == "__main__":
    main()
