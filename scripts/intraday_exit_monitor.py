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

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

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
    start = MARKET_OPEN_HOUR * 60
    end = MARKET_CLOSE_HOUR * 60 + 10
    return start <= minutes <= end


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

    # Yahoo may include today's unfinished daily candle. Remove it so the
    # current intraday price is used exactly once as the provisional close.
    keep = []
    for idx in close.index:
        ts = pd.Timestamp(idx)
        try:
            if ts.tzinfo is not None:
                d = ts.tz_convert(SYDNEY).date()
            else:
                d = ts.date()
        except Exception:
            d = ts.date()
        keep.append(d < today)

    close = close[pd.Series(keep, index=close.index)]
    if len(close) < CONFIG["rsi_period"] + 1:
        return None

    provisional_index = pd.Timestamp(now_sydney.replace(tzinfo=None))
    provisional = pd.concat(
        [close, pd.Series([float(latest_price)], index=[provisional_index])]
    )
    rs = rsi_wilder(provisional, CONFIG["rsi_period"]).dropna()
    return float(rs.iloc[-1]) if not rs.empty else None


def signal_key(sig):
    return f"{sig.get('symbol','')}|{sig.get('entry_date','')}"


def main():
    now_sydney = datetime.now(SYDNEY)

    if not is_market_monitor_window(now_sydney):
        print(f"Outside ASX monitoring window: {now_sydney.isoformat()}")
        return

    state = load_json(DATA / "state.json", {"active_signals": {}})
    active = state.get("active_signals", {})

    if not active:
        print("No active signals to monitor.")
        return

    monitor_state = load_json(
        DATA / "intraday_state.json",
        {"alerted": {}, "latest": {}, "updated_at": None},
    )
    alerted = monitor_state.get("alerted", {})
    latest_state = monitor_state.get("latest", {})

    # Remove stale monitor records once an official daily scan has closed a signal.
    active_keys = {signal_key(sig) for sig in active.values()}
    alerted = {k: v for k, v in alerted.items() if k in active_keys}
    latest_state = {k: v for k, v in latest_state.items() if k in active_keys}

    signals = list(active.values())
    crossings = 0
    checked = 0

    for start in range(0, len(signals), BATCH_SIZE):
        batch_signals = signals[start:start + BATCH_SIZE]
        tickers = [sig["ticker"] for sig in batch_signals]
        multi = len(tickers) > 1

        try:
            daily_df = yf.download(
                tickers,
                period="3mo",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
            intraday_df = yf.download(
                tickers,
                period="1d",
                interval="5m",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception as exc:
            print("Batch download error:", exc)
            continue

        for sig in batch_signals:
            ticker = sig["ticker"]
            sym = sig["symbol"]
            key = signal_key(sig)

            daily_hist = one_history(daily_df, ticker, multi)
            intraday_hist = one_history(intraday_df, ticker, multi)

            if daily_hist.empty or intraday_hist.empty:
                print(f"{sym}: missing daily or intraday data")
                continue

            latest_price = float(intraday_hist["Close"].iloc[-1])
            live_rsi = provisional_daily_rsi(
                daily_hist,
                latest_price,
                now_sydney,
            )

            if live_rsi is None:
                print(f"{sym}: could not calculate provisional daily RSI")
                continue

            checked += 1
            entry_price = float(sig.get("entry_price") or 0)
            gain_pct = (
                ((latest_price - entry_price) / entry_price) * 100.0
                if entry_price
                else 0.0
            )

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
            has_crossed = live_rsi > threshold

            if has_crossed and key not in alerted:
                body = (
                    f"{sig.get('company', sym)} (ASX: {sym}) has crossed above "
                    f"RSI(10) {threshold:.0f} on the intraday monitor.\n\n"
                    f"Provisional daily RSI(10): {live_rsi:.2f}\n"
                    f"Latest price: A${latest_price:.3f}\n"
                    f"Entry price: A${entry_price:.3f}\n"
                    f"Move from entry: {gain_pct:+.2f}%\n"
                    f"Entry date: {sig.get('entry_date','—')}\n"
                    f"Observed: {now_sydney.strftime('%d/%m/%Y %I:%M %p')} Sydney time\n\n"
                    f"HEADS-UP ONLY: this uses the latest intraday price as a provisional "
                    f"today close. The official strategy exit and performance ledger remain "
                    f"based on the daily scanner confirmation."
                )

                sent = send_email(
                    f"ASX RSI INTRADAY >40: {sym} — {live_rsi:.1f}",
                    body,
                )

                if sent:
                    alerted[key] = {
                        "symbol": sym,
                        "entry_date": sig.get("entry_date"),
                        "alerted_at": now_iso(),
                        "price": latest_price,
                        "provisional_rsi10": live_rsi,
                        "gain_pct": gain_pct,
                        "previous_monitor_rsi10": previous_rsi,
                    }
                    crossings += 1
                    print(f"ALERT {sym}: provisional daily RSI {live_rsi:.2f}")
            else:
                print(f"{sym}: provisional daily RSI {live_rsi:.2f}")

        time.sleep(0.5)

    monitor_state = {
        "alerted": alerted,
        "latest": latest_state,
        "updated_at": now_iso(),
        "checked": checked,
        "new_crossing_alerts": crossings,
    }
    save_json(DATA / "intraday_state.json", monitor_state)
    print(f"Checked {checked} active signals; sent {crossings} new crossing alerts.")


if __name__ == "__main__":
    main()
