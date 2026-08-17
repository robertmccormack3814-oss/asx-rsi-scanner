import os
import smtplib
import time
from datetime import datetime
from email.message import EmailMessage
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from common import DATA, CONFIG, load_json, save_json, now_iso

SYDNEY = ZoneInfo("Australia/Sydney")
MARKET_OPEN_HOUR = 10
MARKET_CLOSE_HOUR = 16
BATCH_SIZE = 60


def rsi_wilder(close, period=10):
    close = pd.to_numeric(close, errors="coerce").dropna()
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    safe_loss = avg_loss.where(avg_loss != 0, np.nan)
    rs = avg_gain / safe_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.astype("float64")
    rsi.loc[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    rsi.loc[(avg_gain == 0) & (avg_loss > 0)] = 0.0
    rsi.loc[(avg_gain == 0) & (avg_loss == 0)] = 50.0
    return rsi


def send_email(subject, body):
    username = os.getenv("SMTP_USERNAME", "").strip()
    password = os.getenv("SMTP_APP_PASSWORD", "").strip()
    if not username or not password:
        print("SMTP secrets missing; email skipped.")
        return False
    message = EmailMessage()
    message["From"] = username
    message["To"] = CONFIG["alert_email"]
    message["Subject"] = subject
    message.set_content(body)
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
        server.starttls()
        server.login(username, password)
        server.send_message(message)
    return True


def is_market_monitor_window(now_sydney):
    if now_sydney.weekday() >= 5:
        return False
    minutes = now_sydney.hour * 60 + now_sydney.minute
    return MARKET_OPEN_HOUR * 60 <= minutes <= MARKET_CLOSE_HOUR * 60 + 10


def one_history(df, ticker, multi):
    try:
        x = df[ticker].copy() if multi else df.copy()
        if "Close" not in x.columns:
            return pd.DataFrame()
        x["Close"] = pd.to_numeric(x["Close"], errors="coerce")
        return x.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()


def provisional_daily_rsi(daily_hist, latest_price, now_sydney):
    if daily_hist.empty or latest_price is None:
        return None
    close = pd.to_numeric(daily_hist["Close"], errors="coerce").dropna().copy()
    today = now_sydney.date()
    keep = []
    for idx in close.index:
        ts = pd.Timestamp(idx)
        try:
            d = ts.tz_convert(SYDNEY).date() if ts.tzinfo is not None else ts.date()
        except Exception:
            d = ts.date()
        keep.append(d < today)
    close = close[pd.Series(keep, index=close.index)]
    if len(close) < CONFIG["rsi_period"] + 1:
        return None
    provisional_index = pd.Timestamp(now_sydney.replace(tzinfo=None))
    provisional = pd.concat([close, pd.Series([float(latest_price)], index=[provisional_index])])
    rs = rsi_wilder(provisional, CONFIG["rsi_period"]).dropna()
    return float(rs.iloc[-1]) if not rs.empty else None


def signal_key(sig):
    return f"{sig.get('symbol','')}|{sig.get('entry_date','')}"


def trade_key(trade):
    return (trade.get("symbol"), trade.get("entry_date"), trade.get("exit_date"))


def holding_days(entry_date, exit_date, existing=0):
    try:
        estimated = int(np.busday_count(str(entry_date), str(exit_date)))
        return max(int(existing or 0), estimated)
    except Exception:
        return int(existing or 0)


def refresh_dashboard(dashboard, active, completed, new_exits, live_updates):
    today = datetime.now(SYDNEY).date().isoformat()
    rsi_exits = sum(
        str(t.get("exit_reason", "")).startswith("RSI(10) rose above")
        for t in completed
    )
    time_exits = sum("time exit" in str(t.get("exit_reason", "")).lower() for t in completed)
    resolved = len(completed)

    dashboard["generated_at"] = now_iso()
    dashboard["completed_trades"] = completed
    dashboard["active_signals"] = sorted(active.values(), key=lambda x: x.get("symbol", ""))

    stats = dashboard.setdefault("stats", {})
    stats["active"] = len(active)
    stats["completed_trades"] = resolved
    stats["rsi_exits"] = rsi_exits
    stats["time_exits"] = time_exits
    stats["rsi_exit_rate_pct"] = round(rsi_exits / resolved * 100.0, 1) if resolved else None
    stats["exits_today"] = sum(t.get("exit_date") == today for t in completed)

    prior_exits = dashboard.get("exits_today", [])
    by_key = {trade_key(t): t for t in prior_exits if t.get("exit_date") == today}
    for t in new_exits:
        by_key[trade_key(t)] = t
    dashboard["exits_today"] = list(by_key.values())

    for row in dashboard.get("stocks", []):
        sym = row.get("symbol")
        if sym in live_updates:
            upd = live_updates[sym]
            row["price"] = upd["price"]
            row["rsi10"] = upd["rsi10"]
            row["date"] = today
        row["active"] = sym in active

    return dashboard


def main():
    now_sydney = datetime.now(SYDNEY)
    if not is_market_monitor_window(now_sydney):
        print(f"Outside ASX monitoring window: {now_sydney.isoformat()}")
        return

    state = load_json(DATA / "state.json", {"active_signals": {}, "completed_trades": []})
    active = state.get("active_signals", {})
    completed = state.get("completed_trades", [])
    completed_keys = {trade_key(t) for t in completed}
    dashboard = load_json(DATA / "scanner.json", {})

    if not active:
        print("No active signals to monitor.")
        return

    monitor_state = load_json(DATA / "intraday_state.json", {"alerted": {}, "latest": {}, "updated_at": None})
    alerted = monitor_state.get("alerted", {})
    latest_state = monitor_state.get("latest", {})

    active_keys = {signal_key(sig) for sig in active.values()}
    alerted = {k: v for k, v in alerted.items() if k in active_keys}
    latest_state = {k: v for k, v in latest_state.items() if k in active_keys}

    signals = list(active.values())
    crossings = 0
    checked = 0
    new_exits = []
    live_updates = {}

    for start in range(0, len(signals), BATCH_SIZE):
        batch_signals = signals[start:start + BATCH_SIZE]
        tickers = [sig["ticker"] for sig in batch_signals]
        multi = len(tickers) > 1
        try:
            daily_df = yf.download(tickers, period="3mo", interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)
            intraday_df = yf.download(tickers, period="1d", interval="5m", group_by="ticker", auto_adjust=False, threads=True, progress=False)
        except Exception as exc:
            print("Batch download error:", exc)
            continue

        for sig in batch_signals:
            ticker = sig["ticker"]
            sym = sig["symbol"]
            if sym not in active:
                continue
            key = signal_key(sig)
            daily_hist = one_history(daily_df, ticker, multi)
            intraday_hist = one_history(intraday_df, ticker, multi)
            if daily_hist.empty or intraday_hist.empty:
                print(f"{sym}: missing daily or intraday data")
                continue

            latest_price = float(intraday_hist["Close"].iloc[-1])
            live_rsi = provisional_daily_rsi(daily_hist, latest_price, now_sydney)
            if live_rsi is None:
                print(f"{sym}: could not calculate provisional daily RSI")
                continue

            checked += 1
            entry_price = float(sig.get("entry_price") or 0)
            gain_pct = ((latest_price - entry_price) / entry_price) * 100.0 if entry_price else 0.0
            live_updates[sym] = {"price": latest_price, "rsi10": live_rsi}
            previous_rsi = latest_state.get(key, {}).get("provisional_rsi10")
            latest_state[key] = {
                "symbol": sym,
                "entry_date": sig.get("entry_date"),
                "checked_at": now_iso(),
                "price": latest_price,
                "provisional_rsi10": live_rsi,
                "gain_pct": gain_pct,
            }

            threshold = float(CONFIG["exit_rsi_above"])
            if live_rsi > threshold:
                exit_date = now_sydney.date().isoformat()
                held = holding_days(sig.get("entry_date"), exit_date, sig.get("holding_trading_days", 0))
                exit_trade = {
                    **sig,
                    "exit_date": exit_date,
                    "exit_price": latest_price,
                    "exit_rsi10": live_rsi,
                    "holding_trading_days": held,
                    "exit_reason": f"RSI(10) rose above {threshold}",
                    "gain_pct": gain_pct,
                    "exit_source": "intraday_monitor",
                    "exit_observed_at": now_iso(),
                }

                tkey = trade_key(exit_trade)
                if tkey not in completed_keys:
                    completed.append(exit_trade)
                    completed_keys.add(tkey)
                    new_exits.append(exit_trade)

                del active[sym]
                latest_state.pop(key, None)
                alerted.pop(key, None)

                body = (
                    f"{sig.get('company', sym)} (ASX: {sym}) has crossed above RSI(10) {threshold:.0f}.\n\n"
                    f"OFFICIAL EXIT RECORDED IMMEDIATELY\n"
                    f"Exit price: A${latest_price:.3f}\n"
                    f"RSI(10): {live_rsi:.2f}\n"
                    f"Entry price: A${entry_price:.3f}\n"
                    f"Gain from entry: {gain_pct:+.2f}%\n"
                    f"Entry date: {sig.get('entry_date','—')}\n"
                    f"Holding trading days: {held}\n"
                    f"Observed: {now_sydney.strftime('%d/%m/%Y %I:%M %p')} Sydney time\n\n"
                    f"This intraday RSI crossing is now the official scanner exit and has been written to the performance ledger."
                )
                try:
                    send_email(f"ASX RSI EXIT >40: {sym} — {gain_pct:+.2f}%", body)
                except Exception as exc:
                    print(f"{sym}: email failed after exit was recorded: {exc}")

                crossings += 1
                print(f"EXIT {sym}: provisional daily RSI {live_rsi:.2f}; {gain_pct:+.2f}%")
            else:
                print(f"{sym}: provisional daily RSI {live_rsi:.2f}")

        time.sleep(0.5)

    state["active_signals"] = active
    state["completed_trades"] = completed
    state["updated_at"] = now_iso()
    save_json(DATA / "state.json", state)

    dashboard = refresh_dashboard(dashboard, active, completed, new_exits, live_updates)
    save_json(DATA / "scanner.json", dashboard)

    remaining_keys = {signal_key(sig) for sig in active.values()}
    monitor_state = {
        "alerted": {k: v for k, v in alerted.items() if k in remaining_keys},
        "latest": {k: v for k, v in latest_state.items() if k in remaining_keys},
        "updated_at": now_iso(),
        "checked": checked,
        "new_crossing_alerts": crossings,
    }
    save_json(DATA / "intraday_state.json", monitor_state)
    print(f"Checked {checked} active signals; recorded {crossings} immediate RSI exits.")


if __name__ == "__main__":
    main()
