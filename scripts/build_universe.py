import io, re
import pandas as pd
import requests
from common import DATA, save_json, now_iso

HEADERS={"User-Agent":"Mozilla/5.0 Chrome/151 Safari/537.36"}
ASX_ISIN_URL="https://www.asx.com.au/content/dam/asx/issuers/ISIN.xls"

EXCLUDE_NAME_TERMS=[
    " ETF","ETF ","EXCHANGE TRADED FUND","EXCHANGE TRADED PRODUCT",
    "MANAGED FUND","INDEX FUND","ACTIVE ETF","BETASHARES","ISHARES ",
    "GLOBAL X ","VANECK ","VANGUARD "
]
EXCLUDE_DESC_TERMS=[
    "EXCHANGE TRADED FUND","EXCHANGE TRADED PRODUCT","MANAGED FUND",
    "STRUCTURED PRODUCT","WARRANT","OPTION","RIGHT","BOND","NOTE"
]

def norm(c): return re.sub(r"\s+"," ",str(c)).strip().lower()
def code(v): return re.sub(r"[^A-Z0-9]","",str(v or "").strip().upper())

def find_col(df, words):
    for c in df.columns:
        if any(w in norm(c) for w in words): return c
    return None

def main():
    r=requests.get(ASX_ISIN_URL,headers=HEADERS,timeout=60); r.raise_for_status()
    raw=r.content; frames=[]
    try: frames.append(pd.read_excel(io.BytesIO(raw),engine="xlrd"))
    except Exception: pass
    for enc in ("utf-8-sig","utf-8","latin1"):
        try: text=raw.decode(enc)
        except Exception: continue
        for sep in ("\t",",","|"):
            try:
                df=pd.read_csv(io.StringIO(text),sep=sep)
                if len(df)>10 and len(df.columns)>=2: frames.append(df)
            except Exception: pass
    if not frames: raise RuntimeError("Could not parse official ASX directory.")
    df=max(frames,key=len)
    cc=find_col(df,["asx code","security code","code"])
    nc=find_col(df,["company name","issuer name","issuer","name"])
    dc=find_col(df,["security description","description","security type"])
    if cc is None or nc is None:
        raise RuntimeError(f"Required ASX columns not found: {list(map(str,df.columns))}")
    rows={}
    for _,r0 in df.iterrows():
        sym=code(r0.get(cc)); name=str(r0.get(nc) or "").strip()
        desc=str(r0.get(dc) or "").strip() if dc is not None else ""
        if not re.fullmatch(r"[A-Z0-9]{3}",sym): continue
        if not name or name.lower()=="nan": continue
        un=" "+name.upper()+" "; ud=" "+desc.upper()+" "
        if any(x in un for x in EXCLUDE_NAME_TERMS): continue
        if desc and any(x in ud for x in EXCLUDE_DESC_TERMS): continue
        rows[sym]={
            "symbol":sym,
            "ticker":sym+".AX",
            "company":name,
            "description":desc or None,
            "source":"Official ASX ISIN directory",
            "discovered_at":now_iso()
        }
    universe=sorted(rows.values(),key=lambda x:x["symbol"])
    if len(universe)<500: raise RuntimeError(f"Universe unexpectedly small: {len(universe)}")
    save_json(DATA/"universe.json",universe)
    print(f"ASX universe: {len(universe)}")

if __name__=="__main__": main()
