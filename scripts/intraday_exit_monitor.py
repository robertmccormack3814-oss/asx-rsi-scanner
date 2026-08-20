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
    delta = close.diff(); gain = delta.clip(lower=0.0); loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    safe_loss = avg_loss.where(avg_loss != 0, np.nan); rs = avg_gain / safe_loss
    rsi = (100.0 - (100.0 / (1.0 + rs))).astype("float64")
    rsi.loc[(avg_loss == 0) & (avg_gain > 0)] = 100.0; rsi.loc[(avg_gain == 0) & (avg_loss > 0)] = 0.0; rsi.loc[(avg_gain == 0) & (avg_loss == 0)] = 50.0
    return rsi


def send_email(subject, body):
    username=os.getenv("SMTP_USERNAME","").strip(); password=os.getenv("SMTP_APP_PASSWORD","").strip()
    if not username or not password: print("SMTP secrets missing; email skipped."); return False
    message=EmailMessage(); message["From"]=username; message["To"]=CONFIG["alert_email"]; message["Subject"]=subject; message.set_content(body)
    with smtplib.SMTP("smtp.gmail.com",587,timeout=30) as server: server.starttls(); server.login(username,password); server.send_message(message)
    return True


def is_market_monitor_window(n):
    if n.weekday()>=5:return False
    m=n.hour*60+n.minute; return MARKET_OPEN_HOUR*60<=m<=MARKET_CLOSE_HOUR*60+10


def one_history(df,ticker,multi):
    try:
        x=df[ticker].copy() if multi else df.copy()
        if "Close" not in x.columns:return pd.DataFrame()
        x["Close"]=pd.to_numeric(x["Close"],errors="coerce"); return x.dropna(subset=["Close"])
    except Exception:return pd.DataFrame()


def prior_closes(daily_hist,now_sydney):
    close=pd.to_numeric(daily_hist["Close"],errors="coerce").dropna().copy(); today=now_sydney.date(); keep=[]
    for idx in close.index:
        ts=pd.Timestamp(idx)
        try:d=ts.tz_convert(SYDNEY).date() if ts.tzinfo is not None else ts.date()
        except Exception:d=ts.date()
        keep.append(d<today)
    return close[pd.Series(keep,index=close.index)]


def provisional_daily_rsi(daily_hist,latest_price,now_sydney):
    if daily_hist.empty or latest_price is None:return None
    close=prior_closes(daily_hist,now_sydney)
    if len(close)<CONFIG["rsi_period"]+1:return None
    provisional=pd.concat([close,pd.Series([float(latest_price)],index=[pd.Timestamp(now_sydney.replace(tzinfo=None))])])
    rs=rsi_wilder(provisional,CONFIG["rsi_period"]).dropna(); return float(rs.iloc[-1]) if not rs.empty else None


def estimate_price_for_rsi(daily_hist,now_sydney,target=40.0):
    """Solve today's hypothetical close that makes the provisional Wilder RSI equal target."""
    close=prior_closes(daily_hist,now_sydney)
    if len(close)<CONFIG["rsi_period"]+1:return None
    last=float(close.iloc[-1]); lo=max(0.000001,last*0.20); hi=max(last*3.0,last+1.0)
    def f(p):
        s=pd.concat([close,pd.Series([float(p)],index=[pd.Timestamp(now_sydney.replace(tzinfo=None))])]); r=rsi_wilder(s,CONFIG["rsi_period"]).dropna()
        return float(r.iloc[-1]) if not r.empty else None
    rlo,rhi=f(lo),f(hi)
    if rlo is None or rhi is None or target<rlo or target>rhi:return None
    for _ in range(45):
        mid=(lo+hi)/2; rm=f(mid)
        if rm is None:return None
        if rm<target:lo=mid
        else:hi=mid
    return (lo+hi)/2


def signal_key(sig):return f"{sig.get('symbol','')}|{sig.get('entry_date','')}"
def trade_key(t):return (t.get("symbol"),t.get("entry_date"),t.get("exit_date"))
def holding_days(a,b,existing=0):
    try:return max(int(existing or 0),int(np.busday_count(str(a),str(b))))
    except Exception:return int(existing or 0)


def refresh_dashboard(dashboard,active,completed,new_exits,live_updates):
    today=datetime.now(SYDNEY).date().isoformat(); rsi_exits=sum(str(t.get("exit_reason","" )).startswith("RSI(10) rose above") for t in completed); time_exits=sum("time exit" in str(t.get("exit_reason","")).lower() for t in completed); resolved=len(completed)
    dashboard["generated_at"]=now_iso(); dashboard["completed_trades"]=completed
    # Persist the live target data directly into active signals as well as stock rows.
    for sym,sig in active.items():
        if sym in live_updates:
            u=live_updates[sym]
            sig["latest_price"]=u["price"]; sig["latest_rsi10"]=u["rsi10"]
            sig["rsi40_target_price"]=u.get("rsi40_target_price"); sig["rsi40_move_pct"]=u.get("rsi40_move_pct")
            sig["target_updated_at"]=now_iso()
    dashboard["active_signals"]=sorted(active.values(),key=lambda x:x.get("symbol",""))
    stats=dashboard.setdefault("stats",{}); stats["active"]=len(active); stats["completed_trades"]=resolved; stats["rsi_exits"]=rsi_exits; stats["time_exits"]=time_exits; stats["rsi_exit_rate_pct"]=round(rsi_exits/resolved*100,1) if resolved else None; stats["exits_today"]=sum(t.get("exit_date")==today for t in completed)
    prior=dashboard.get("exits_today",[]); by_key={trade_key(t):t for t in prior if t.get("exit_date")==today}
    for t in new_exits:by_key[trade_key(t)]=t
    dashboard["exits_today"]=list(by_key.values())
    for row in dashboard.get("stocks",[]):
        sym=row.get("symbol")
        if sym in live_updates:
            u=live_updates[sym]; row["price"]=u["price"]; row["rsi10"]=u["rsi10"]; row["rsi40_target_price"]=u.get("rsi40_target_price"); row["rsi40_move_pct"]=u.get("rsi40_move_pct"); row["date"]=today
        row["active"]=sym in active
    return dashboard


def main():
    now_sydney=datetime.now(SYDNEY); market_live=is_market_monitor_window(now_sydney)
    print(f"Sydney time: {now_sydney.isoformat()} | market_live={market_live}")
    state=load_json(DATA/"state.json",{"active_signals":{},"completed_trades":[]}); active=state.get("active_signals",{}); completed=state.get("completed_trades",[]); completed_keys={trade_key(t) for t in completed}; dashboard=load_json(DATA/"scanner.json",{})
    if not active:print("No active signals to monitor.");return
    monitor_state=load_json(DATA/"intraday_state.json",{"alerted":{},"latest":{},"updated_at":None}); alerted=monitor_state.get("alerted",{}); latest_state=monitor_state.get("latest",{})
    active_keys={signal_key(s) for s in active.values()}; alerted={k:v for k,v in alerted.items() if k in active_keys}; latest_state={k:v for k,v in latest_state.items() if k in active_keys}
    signals=list(active.values()); crossings=0; checked=0; targets_ready=0; new_exits=[]; live_updates={}
    for start in range(0,len(signals),BATCH_SIZE):
        batch=signals[start:start+BATCH_SIZE]; tickers=[s["ticker"] for s in batch]; multi=len(tickers)>1
        try:
            # Six months gives ample history for RSI10 and avoids sparse 3-month edge cases.
            daily_df=yf.download(tickers,period="6mo",interval="1d",group_by="ticker",auto_adjust=False,threads=True,progress=False)
            intraday_df=yf.download(tickers,period="1d",interval="5m",group_by="ticker",auto_adjust=False,threads=True,progress=False) if market_live else pd.DataFrame()
        except Exception as exc:print("Batch download error:",exc);continue
        for sig in batch:
            ticker=sig["ticker"]; sym=sig["symbol"]
            if sym not in active:continue
            key=signal_key(sig); daily_hist=one_history(daily_df,ticker,multi)
            if daily_hist.empty:print(f"{sym}: missing daily data");continue
            intraday_hist=one_history(intraday_df,ticker,multi) if market_live and not intraday_df.empty else pd.DataFrame()
            # During market hours use the latest 5m price. Premarket/after hours use the last daily close solely to prepare today's target.
            latest_price=float(intraday_hist["Close"].iloc[-1]) if not intraday_hist.empty else float(daily_hist["Close"].iloc[-1])
            live_rsi=provisional_daily_rsi(daily_hist,latest_price,now_sydney)
            if live_rsi is None:continue
            target=estimate_price_for_rsi(daily_hist,now_sydney,float(CONFIG["exit_rsi_above"])); target_move=((target-latest_price)/latest_price*100) if target and latest_price else None
            if target is not None:targets_ready+=1
            checked+=1; entry_price=float(sig.get("entry_price") or 0); gain_pct=((latest_price-entry_price)/entry_price)*100 if entry_price else 0
            live_updates[sym]={"price":latest_price,"rsi10":live_rsi,"rsi40_target_price":target,"rsi40_move_pct":target_move}
            latest_state[key]={"symbol":sym,"entry_date":sig.get("entry_date"),"checked_at":now_iso(),"price":latest_price,"provisional_rsi10":live_rsi,"gain_pct":gain_pct,"rsi40_target_price":target,"rsi40_move_pct":target_move,"market_live":market_live}
            threshold=float(CONFIG["exit_rsi_above"])
            # Never record an exit from a daily-close proxy outside live ASX hours.
            if market_live and not intraday_hist.empty and live_rsi>threshold:
                exit_date=now_sydney.date().isoformat(); held=holding_days(sig.get("entry_date"),exit_date,sig.get("holding_trading_days",0)); exit_trade={**sig,"exit_date":exit_date,"exit_price":latest_price,"exit_rsi10":live_rsi,"holding_trading_days":held,"exit_reason":f"RSI(10) rose above {threshold}","gain_pct":gain_pct,"exit_source":"intraday_monitor","exit_observed_at":now_iso()}; tkey=trade_key(exit_trade)
                if tkey not in completed_keys:completed.append(exit_trade);completed_keys.add(tkey);new_exits.append(exit_trade)
                del active[sym];latest_state.pop(key,None);alerted.pop(key,None)
                body=f"{sig.get('company',sym)} (ASX: {sym}) has crossed above RSI(10) {threshold:.0f}.\n\nOFFICIAL EXIT RECORDED IMMEDIATELY\nExit price: A${latest_price:.3f}\nRSI(10): {live_rsi:.2f}\nEntry price: A${entry_price:.3f}\nGain from entry: {gain_pct:+.2f}%\nEntry date: {sig.get('entry_date','—')}\nHolding trading days: {held}\nObserved: {now_sydney.strftime('%d/%m/%Y %I:%M %p')} Sydney time\n\nThis intraday RSI crossing is now the official scanner exit and has been written to the performance ledger."
                try:send_email(f"ASX RSI EXIT >40: {sym} — {gain_pct:+.2f}%",body)
                except Exception as exc:print(f"{sym}: email failed after exit was recorded: {exc}")
                crossings+=1;print(f"EXIT {sym}: provisional daily RSI {live_rsi:.2f}; {gain_pct:+.2f}%")
            else:print(f"{sym}: RSI {live_rsi:.2f}; est RSI40 price {target if target else '—'}")
        time.sleep(.5)
    state["active_signals"]=active;state["completed_trades"]=completed;state["updated_at"]=now_iso();save_json(DATA/"state.json",state);dashboard=refresh_dashboard(dashboard,active,completed,new_exits,live_updates);save_json(DATA/"scanner.json",dashboard)
    remaining={signal_key(s) for s in active.values()};monitor_state={"alerted":{k:v for k,v in alerted.items() if k in remaining},"latest":{k:v for k,v in latest_state.items() if k in remaining},"updated_at":now_iso(),"checked":checked,"targets_ready":targets_ready,"new_crossing_alerts":crossings,"market_live":market_live};save_json(DATA/"intraday_state.json",monitor_state);print(f"Checked {checked} active signals; targets ready {targets_ready}; recorded {crossings} immediate RSI exits.")

if __name__=="__main__":main()
