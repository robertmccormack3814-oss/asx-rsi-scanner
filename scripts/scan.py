import os
import smtplib
import time
from email.message import EmailMessage

import numpy as np
import pandas as pd
import yfinance as yf

from common import DATA, CONFIG, load_json, save_json, now_iso


def rsi_wilder(close, period=10):
    """
    Wilder RSI using completed daily closes.

    Handles zero-loss and zero-gain runs numerically so pandas NA values
    cannot crash the scanner.
    """
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
        return

    message = EmailMessage()
    message["From"] = username
    message["To"] = CONFIG["alert_email"]
    message["Subject"] = subject
    message.set_content(body)

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
        server.starttls()
        server.login(username, password)
        server.send_message(message)


def market_regime():
    hist = yf.download(
        CONFIG["market_ticker"],
        period="18mo",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if hist.empty:
        raise RuntimeError("Could not download ASX 200 history.")

    close = hist["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = pd.to_numeric(close, errors="coerce").dropna()
    sma = close.rolling(CONFIG["market_sma_period"]).mean()

    if (
        len(close) < CONFIG["market_sma_period"]
        or pd.isna(sma.iloc[-1])
    ):
        raise RuntimeError(
            "Insufficient ASX 200 data for 200-day SMA."
        )

    return {
        "ticker": CONFIG["market_ticker"],
        "name": CONFIG["market_name"],
        "date": str(close.index[-1].date()),
        "close": float(close.iloc[-1]),
        "sma200": float(sma.iloc[-1]),
        "above_sma200": bool(close.iloc[-1] > sma.iloc[-1]),
    }


def download_chunk(tickers):
    return yf.download(
        tickers,
        period="1y",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )


def one_history(df, ticker, multi):
    try:
        x = df[ticker].copy() if multi else df.copy()

        if "Close" not in x.columns:
            return pd.DataFrame()

        x["Close"] = pd.to_numeric(
            x["Close"],
            errors="coerce",
        )

        if "Volume" in x.columns:
            x["Volume"] = pd.to_numeric(
                x["Volume"],
                errors="coerce",
            )

        return x.dropna(subset=["Close"])

    except Exception:
        return pd.DataFrame()


def trading_days_since_entry(hist, entry_date):
    if hist.empty:
        return 0

    dates = pd.Index([d.date() for d in hist.index])
    entry = pd.Timestamp(entry_date).date()

    return int(sum(d > entry for d in dates))


def seeded_completed_trades():
    """
    Trades already completed by the live scanner before permanent
    completed-trade tracking was added.

    These are NOT backtested trades. They were actual scanner signals
    recorded in the repository's scan history.
    """
    return [
        {
            "symbol": "DTM",
            "company": "DART MINING NL",
            "ticker": "DTM.AX",
            "entry_date": "2026-08-12",
            "entry_price": 0.010999999940395355,
            "entry_rsi10": 20.382940355868627,
            "exit_date": "2026-08-13",
            "exit_price": 0.012000000104308128,
            "exit_rsi10": 47.71198310489537,
            "holding_trading_days": 1,
            "exit_reason": "RSI(10) rose above 40.0",
        },
        {
            "symbol": "EFE",
            "company": "EASTERN RESOURCES LIMITED",
            "ticker": "EFE.AX",
            "entry_date": "2026-08-11",
            "entry_price": 0.029999999329447746,
            "entry_rsi10": 29.92153866978198,
            "exit_date": "2026-08-13",
            "exit_price": 0.03099999949336052,
            "exit_rsi10": 49.03295772895291,
            "holding_trading_days": 2,
            "exit_reason": "RSI(10) rose above 40.0",
        },
        {
            "symbol": "EGR",
            "company": "ECOGRAF LIMITED",
            "ticker": "EGR.AX",
            "entry_date": "2026-08-12",
            "entry_price": 0.19499999284744263,
            "entry_rsi10": 23.767804508338685,
            "exit_date": "2026-08-13",
            "exit_price": 0.3100000023841858,
            "exit_rsi10": 72.29953118305919,
            "holding_trading_days": 1,
            "exit_reason": "RSI(10) rose above 40.0",
        },
        {
            "symbol": "ERD",
            "company": "EROAD LIMITED",
            "ticker": "ERD.AX",
            "entry_date": "2026-08-12",
            "entry_price": 0.7749999761581421,
            "entry_rsi10": 29.493147044868564,
            "exit_date": "2026-08-13",
            "exit_price": 0.7950000166893005,
            "exit_rsi10": 40.47373711854614,
            "holding_trading_days": 1,
            "exit_reason": "RSI(10) rose above 40.0",
        },
        {
            "symbol": "JAT",
            "company": "JATCORP LIMITED",
            "ticker": "JAT.AX",
            "entry_date": "2026-08-11",
            "entry_price": 0.10999999940395355,
            "entry_rsi10": 14.022698351640912,
            "exit_date": "2026-08-13",
            "exit_price": 0.14000000059604645,
            "exit_rsi10": 66.43305898053038,
            "holding_trading_days": 2,
            "exit_reason": "RSI(10) rose above 40.0",
        },
        {
            "symbol": "OLI",
            "company": "OLIVER'S REAL FOOD LIMITED",
            "ticker": "OLI.AX",
            "entry_date": "2026-08-12",
            "entry_price": 0.004999999888241291,
            "entry_rsi10": 15.14811209128375,
            "exit_date": "2026-08-13",
            "exit_price": 0.006000000052154064,
            "exit_rsi10": 52.055070112377976,
            "holding_trading_days": 1,
            "exit_reason": "RSI(10) rose above 40.0",
        },
        {
            "symbol": "PR1",
            "company": "PURE RESOURCES LIMITED",
            "ticker": "PR1.AX",
            "entry_date": "2026-08-11",
            "entry_price": 0.26499998569488525,
            "entry_rsi10": 27.168282877210217,
            "exit_date": "2026-08-13",
            "exit_price": 0.30000001192092896,
            "exit_rsi10": 48.90262485709933,
            "holding_trading_days": 2,
            "exit_reason": "RSI(10) rose above 40.0",
        },
        {
            "symbol": "UBI",
            "company": "UNIVERSAL BIOSENSORS INC.",
            "ticker": "UBI.AX",
            "entry_date": "2026-08-11",
            "entry_price": 0.014000000432133675,
            "entry_rsi10": 2.666227944905714,
            "exit_date": "2026-08-13",
            "exit_price": 0.014000000432133675,
            "exit_rsi10": 51.265828828483,
            "holding_trading_days": 2,
            "exit_reason": "RSI(10) rose above 40.0",
        },
    ]


def trade_key(trade):
    return (
        trade.get("symbol"),
        trade.get("entry_date"),
        trade.get("exit_date"),
    )


def ensure_completed_history(state):
    completed = state.get("completed_trades", [])

    existing_keys = {
        trade_key(trade)
        for trade in completed
    }

    for trade in seeded_completed_trades():
        key = trade_key(trade)

        if key not in existing_keys:
            completed.append(trade)
            existing_keys.add(key)

    return completed


def completed_trade_stats(completed_trades):
    completed_count = len(completed_trades)

    rsi_exits = sum(
        1
        for trade in completed_trades
        if str(trade.get("exit_reason", "")).startswith(
            "RSI(10) rose above"
        )
    )

    time_exits = sum(
        1
        for trade in completed_trades
        if "time exit" in str(
            trade.get("exit_reason", "")
        ).lower()
    )

    rsi_exit_rate = (
        round((rsi_exits / completed_count) * 100.0, 1)
        if completed_count
        else None
    )

    return {
        "completed_trades": completed_count,
        "rsi_exits": rsi_exits,
        "time_exits": time_exits,
        "rsi_exit_rate_pct": rsi_exit_rate,
    }


def main():
    universe = load_json(
        DATA / "universe.json",
        [],
    )

    state = load_json(
        DATA / "state.json",
        {
            "active_signals": {},
            "completed_trades": [],
            "last_scan_date": None,
        },
    )

    active = state.get("active_signals", {})
    completed_trades = ensure_completed_history(state)

    completed_keys = {
        trade_key(trade)
        for trade in completed_trades
    }

    market = market_regime()

    print(
        f"Market: {market['close']:.2f} vs SMA200 "
        f"{market['sma200']:.2f}; "
        f"above={market['above_sma200']}"
    )

    results = []
    entries = []
    exits = []

    errors = 0
    scanned = 0

    batch = int(
        CONFIG.get(
            "scan_batch_size",
            250,
        )
    )

    for start in range(
        0,
        len(universe),
        batch,
    ):
        items = universe[start:start + batch]

        tickers = [
            item["ticker"]
            for item in items
        ]

        try:
            df = download_chunk(tickers)

        except Exception as e:
            print(
                "Chunk error:",
                e,
            )

            errors += len(items)
            continue

        multi = len(tickers) > 1

        for item in items:
            sym = item["symbol"]
            ticker = item["ticker"]

            hist = one_history(
                df,
                ticker,
                multi,
            )

            if len(hist) < max(
                CONFIG["rsi_period"] + 2,
                25,
            ):
                errors += 1
                continue

            try:
                close = hist["Close"]

                vol = (
                    hist["Volume"]
                    if "Volume" in hist.columns
                    else pd.Series(
                        index=hist.index,
                        dtype=float,
                    )
                )

                rs = rsi_wilder(
                    close,
                    CONFIG["rsi_period"],
                )

                valid_rsi = rs.dropna()

                if valid_rsi.empty:
                    errors += 1
                    continue

                last_rsi = float(
                    valid_rsi.iloc[-1]
                )

                last_close = float(
                    close.iloc[-1]
                )

                avgvol = (
                    float(
                        vol.tail(20).mean()
                    )
                    if len(vol.dropna())
                    else 0.0
                )

                d = str(
                    hist.index[-1].date()
                )

            except Exception as e:
                print(
                    f"{sym}: indicator error: {e}"
                )

                errors += 1
                continue

            scanned += 1

            stock_sma200 = (
                float(
                    close
                    .rolling(200)
                    .mean()
                    .iloc[-1]
                )
                if len(close) >= 200
                else None
            )

            above_stock_sma200 = bool(
                stock_sma200 is not None
                and not pd.isna(stock_sma200)
                and last_close > stock_sma200
            )

            row = {
                "symbol": sym,
                "company": item["company"],
                "ticker": ticker,
                "date": d,
                "price": last_close,
                "rsi10": last_rsi,
                "sma200": stock_sma200,
                "above_sma200": above_stock_sma200,
                "avg_volume_20d": avgvol,
                "active": sym in active,
            }

            # Manage existing live scanner signals.
            if sym in active:
                sig = active[sym]

                held = trading_days_since_entry(
                    hist,
                    sig["entry_date"],
                )

                exit_reason = None

                if (
                    last_rsi
                    > CONFIG["exit_rsi_above"]
                ):
                    exit_reason = (
                        f"RSI(10) rose above "
                        f"{CONFIG['exit_rsi_above']}"
                    )

                elif (
                    held
                    >= CONFIG[
                        "max_holding_trading_days"
                    ]
                ):
                    exit_reason = (
                        f"{CONFIG['max_holding_trading_days']} "
                        "trading-day time exit"
                    )

                if exit_reason:
                    exit_row = {
                        **sig,
                        "exit_date": d,
                        "exit_price": last_close,
                        "exit_rsi10": last_rsi,
                        "holding_trading_days": held,
                        "exit_reason": exit_reason,
                    }

                    exits.append(exit_row)

                    key = trade_key(exit_row)

                    if key not in completed_keys:
                        completed_trades.append(
                            exit_row
                        )

                        completed_keys.add(key)

                    send_email(
                        (
                            f"ASX RSI EXIT: {sym} "
                            f"— {exit_reason}"
                        ),
                        (
                            f"{item['company']} "
                            f"(ASX: {sym}) exit signal.\n\n"
                            f"Exit reason: {exit_reason}\n"
                            f"Exit price: "
                            f"A${last_close:.3f}\n"
                            f"RSI(10): {last_rsi:.2f}\n"
                            f"Entry date: "
                            f"{sig['entry_date']}\n"
                            f"Entry price: "
                            f"A${sig['entry_price']:.3f}\n"
                            f"Holding trading days: "
                            f"{held}\n"
                        ),
                    )

                    del active[sym]
                    row["active"] = False

                else:
                    sig[
                        "holding_trading_days"
                    ] = held

                    sig["latest_price"] = (
                        last_close
                    )

                    sig["latest_rsi10"] = (
                        last_rsi
                    )

            # New entries are signals actually picked
            # by the live engine.
            if sym not in active:
                if (
                    above_stock_sma200
                    and (
                        last_rsi
                        < CONFIG[
                            "entry_rsi_below"
                        ]
                    )
                    and (
                        last_close
                        >= CONFIG.get(
                            "minimum_price",
                            0,
                        )
                    )
                    and (
                        avgvol
                        >= CONFIG.get(
                            "minimum_average_volume_20d",
                            0,
                        )
                    )
                ):
                    sig = {
                        "symbol": sym,
                        "company": item["company"],
                        "ticker": ticker,
                        "entry_date": d,
                        "entry_price": last_close,
                        "entry_rsi10": last_rsi,
                        "entry_sma200": stock_sma200,
                        "holding_trading_days": 0,
                        "latest_price": last_close,
                        "latest_rsi10": last_rsi,
                    }

                    active[sym] = sig
                    entries.append(sig)

                    row["active"] = True

                    send_email(
                        (
                            f"ASX RSI ENTRY: {sym} "
                            f"RSI(10) "
                            f"{last_rsi:.1f}"
                        ),
                        (
                            f"{item['company']} "
                            f"(ASX: {sym}) "
                            f"entry signal.\n\n"
                            f"ASX 200 context only: "
                            f"{market['close']:.2f} "
                            f"vs SMA(200) "
                            f"{market['sma200']:.2f}\n"
                            f"Stock price: "
                            f"A${last_close:.3f}\n"
                            f"Stock SMA(200): "
                            f"A${stock_sma200:.3f}\n"
                            f"RSI(10): "
                            f"{last_rsi:.2f}\n\n"
                            f"Entry rule: stock above "
                            f"its own SMA(200) and "
                            f"RSI(10) below "
                            f"{CONFIG['entry_rsi_below']}.\n"
                            f"Exit: RSI(10) above "
                            f"{CONFIG['exit_rsi_above']} "
                            f"or after "
                            f"{CONFIG['max_holding_trading_days']} "
                            f"trading days."
                        ),
                    )

            results.append(row)

        time.sleep(0.25)

    outcome_stats = completed_trade_stats(
        completed_trades
    )

    state = {
        "active_signals": active,
        "completed_trades": completed_trades,
        "last_scan_date": market["date"],
        "updated_at": now_iso(),
    }

    save_json(
        DATA / "state.json",
        state,
    )

    dashboard = {
        "generated_at": now_iso(),
        "market": market,
        "stats": {
            "universe": len(universe),
            "scanned": scanned,
            "entries_today": len(entries),
            "exits_today": len(exits),
            "active": len(active),
            "errors": errors,
            **outcome_stats,
        },
        "entries_today": entries,
        "exits_today": exits,
        "completed_trades": completed_trades,
        "active_signals": sorted(
            active.values(),
            key=lambda x: x["symbol"],
        ),
        "stocks": sorted(
            results,
            key=lambda x: x["rsi10"],
        ),
    }

    save_json(
        DATA / "scanner.json",
        dashboard,
    )

    print(dashboard["stats"])


if __name__ == "__main__":
    main()
