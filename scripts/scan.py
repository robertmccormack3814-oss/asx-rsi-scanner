import os, smtplib, time
from datetime import date
from email.message import EmailMessage
import pandas as pd
import yfinance as yf
from common import DATA, CONFIG, load_json, save_json, now_iso

def rsi_wilder(close, period=10):
    delta=close.diff()
    gain=delta.clip(lower=0)
    loss=-delta.clip(upper=0)
    avg_gain=gain.ewm(alpha=1/period,adjust=False,min_periods=period).mean()
    avg_loss=loss.ewm(alpha=1/period,adjust=False,min_periods=period).mean()
    rs=avg_gain/avg_loss.replace(0,pd.NA)
    rsi=100-(100/(1+rs))
    rsi=rsi.astype(float)
    # If there are gains and zero losses, RSI is 100; vice versa, 0.
    rsi[(avg_loss==0)&(avg_gain>0)]=100.0
    rsi[(avg_gain==0)&(avg_loss>0)]=0.0
    return rsi

def send_email(subject, body):
    u=os.getenv("SMTP_USERNAME","").strip()
    pw=os.getenv("SMTP_APP_PASSWORD","").strip()
    if not u or not pw:
        print("SMTP secrets missing; email skipped.")
        return
    m=EmailMessage(); m["From"]=u; m["To"]=CONFIG["alert_email"]; m["Subject"]=subject
    m.set_content(body)
    with smtplib.SMTP("smtp.gmail.com",587,timeout=30) as s:
        s.starttls(); s.login(u,pw); s.send_message(m)

def market_regime():
    hist=yf.download(CONFIG["market_ticker"],period="18mo",interval="1d",auto_adjust=False,progress=False)
    if hist.empty: raise RuntimeError("Could not download ASX 200 history.")
    close=hist["Close"]
    if isinstance(close,pd.DataFrame): close=close.iloc[:,0]
    close=close.dropna()
    sma=close.rolling(CONFIG["market_sma_period"]).mean()
    if len(close)<CONFIG["market_sma_period"] or pd.isna(sma.iloc[-1]):
        raise RuntimeError("Insufficient ASX 200 data for 200-day SMA.")
    return {
        "ticker":CONFIG["market_ticker"],
        "name":CONFIG["market_name"],
        "date":str(close.index[-1].date()),
        "close":float(close.iloc[-1]),
        "sma200":float(sma.iloc[-1]),
        "above_sma200":bool(close.iloc[-1]>sma.iloc[-1])
    }

def download_chunk(tickers):
    return yf.download(
        tickers,period="1y",interval="1d",group_by="ticker",
        auto_adjust=False,threads=True,progress=False
    )

def one_history(df,ticker,multi):
    try:
        if multi:
            x=df[ticker].copy()
        else:
            x=df.copy()
        x=x.dropna(subset=["Close"])
        return x
    except Exception:
        return pd.DataFrame()

def trading_days_since_entry(hist, entry_date):
    if hist.empty: return 0
    dates=pd.Index([d.date() for d in hist.index])
    ed=pd.Timestamp(entry_date).date()
    return int(sum(d>ed for d in dates))

def main():
    universe=load_json(DATA/"universe.json",[])
    state=load_json(DATA/"state.json",{"active_signals":{},"last_scan_date":None})
    active=state.get("active_signals",{})
    market=market_regime()
    print(f"Market: {market['close']:.2f} vs SMA200 {market['sma200']:.2f}; above={market['above_sma200']}")

    results=[]; entries=[]; exits=[]; errors=0; scanned=0
    batch=int(CONFIG.get("scan_batch_size",250))

    for start in range(0,len(universe),batch):
        items=universe[start:start+batch]
        tickers=[x["ticker"] for x in items]
        try:
            df=download_chunk(tickers)
        except Exception as e:
            print("Chunk error:",e); errors+=len(items); continue
        multi=len(tickers)>1

        for item in items:
            sym=item["symbol"]; ticker=item["ticker"]
            hist=one_history(df,ticker,multi)
            if len(hist)<max(CONFIG["rsi_period"]+2,25):
                errors+=1
                continue
            close=hist["Close"]
            vol=hist["Volume"] if "Volume" in hist.columns else pd.Series(index=hist.index,dtype=float)
            rs=rsi_wilder(close,CONFIG["rsi_period"])
            if rs.dropna().empty:
                errors+=1; continue
            last_rsi=float(rs.dropna().iloc[-1])
            last_close=float(close.iloc[-1])
            avgvol=float(vol.tail(20).mean()) if len(vol.dropna()) else 0.0
            d=str(hist.index[-1].date())
            scanned+=1

            row={"symbol":sym,"company":item["company"],"ticker":ticker,"date":d,
                 "price":last_close,"rsi10":last_rsi,"avg_volume_20d":avgvol,
                 "active":sym in active}

            # Manage existing active signals regardless of current market regime.
            if sym in active:
                sig=active[sym]
                held=trading_days_since_entry(hist,sig["entry_date"])
                exit_reason=None
                if last_rsi>CONFIG["exit_rsi_above"]:
                    exit_reason=f"RSI(10) rose above {CONFIG['exit_rsi_above']}"
                elif held>=CONFIG["max_holding_trading_days"]:
                    exit_reason=f"{CONFIG['max_holding_trading_days']} trading-day time exit"
                if exit_reason:
                    exit_row={**sig,"exit_date":d,"exit_price":last_close,
                              "exit_rsi10":last_rsi,"holding_trading_days":held,
                              "exit_reason":exit_reason}
                    exits.append(exit_row)
                    send_email(
                        f"ASX RSI EXIT: {sym} — {exit_reason}",
                        f"{item['company']} (ASX: {sym}) exit signal.\n\n"
                        f"Exit reason: {exit_reason}\n"
                        f"Exit price: A${last_close:.3f}\n"
                        f"RSI(10): {last_rsi:.2f}\n"
                        f"Entry date: {sig['entry_date']}\n"
                        f"Entry price: A${sig['entry_price']:.3f}\n"
                        f"Holding trading days: {held}\n"
                    )
                    del active[sym]
                    row["active"]=False
                else:
                    sig["holding_trading_days"]=held
                    sig["latest_price"]=last_close
                    sig["latest_rsi10"]=last_rsi

            # New entry only if market regime is ON and no signal already active.
            if sym not in active and market["above_sma200"]:
                if (last_rsi<CONFIG["entry_rsi_below"] and
                    last_close>=CONFIG.get("minimum_price",0) and
                    avgvol>=CONFIG.get("minimum_average_volume_20d",0)):
                    sig={"symbol":sym,"company":item["company"],"ticker":ticker,
                         "entry_date":d,"entry_price":last_close,"entry_rsi10":last_rsi,
                         "holding_trading_days":0,"latest_price":last_close,
                         "latest_rsi10":last_rsi}
                    active[sym]=sig; entries.append(sig); row["active"]=True
                    send_email(
                        f"ASX RSI ENTRY: {sym} RSI(10) {last_rsi:.1f}",
                        f"{item['company']} (ASX: {sym}) entry signal.\n\n"
                        f"ASX 200 filter: ON\n"
                        f"ASX 200 close: {market['close']:.2f}\n"
                        f"ASX 200 SMA(200): {market['sma200']:.2f}\n"
                        f"Stock price: A${last_close:.3f}\n"
                        f"RSI(10): {last_rsi:.2f}\n\n"
                        f"Entry rule: RSI(10) below {CONFIG['entry_rsi_below']} while ASX 200 is above its 200-day SMA.\n"
                        f"Exit: RSI(10) above {CONFIG['exit_rsi_above']} or after {CONFIG['max_holding_trading_days']} trading days."
                    )

            results.append(row)
        time.sleep(.25)

    state={"active_signals":active,"last_scan_date":market["date"],"updated_at":now_iso()}
    save_json(DATA/"state.json",state)
    dashboard={
        "generated_at":now_iso(),
        "market":market,
        "stats":{"universe":len(universe),"scanned":scanned,"entries_today":len(entries),
                 "exits_today":len(exits),"active":len(active),"errors":errors},
        "entries_today":entries,
        "exits_today":exits,
        "active_signals":sorted(active.values(),key=lambda x:x["symbol"]),
        "stocks":sorted(results,key=lambda x:x["rsi10"])
    }
    save_json(DATA/"scanner.json",dashboard)
    print(dashboard["stats"])

if __name__=="__main__": main()
