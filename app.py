# Run locally:  streamlit run app.py
# Cloud: set DATABASE_URL in Streamlit secrets to keep data between restarts.
from io import StringIO

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from sqlalchemy import create_engine

st.set_page_config(page_title="Stock Screener", layout="wide")
st.title("Stock Screener")


# ---------- Database (Postgres if configured, otherwise local SQLite) ----------
@st.cache_resource
def get_engine():
    url = None
    try:
        url = st.secrets.get("DATABASE_URL")
    except Exception:
        pass
    if url:
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return create_engine(url, pool_pre_ping=True), "Postgres (persistent)"
    return create_engine("sqlite:///stocks.db"), "local SQLite (temporary on cloud)"


engine, db_label = get_engine()


# ---------- Symbols ----------
@st.cache_data(ttl=86400)
def sp500_symbols():
    r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    df = pd.read_html(StringIO(r.text))[0]
    return sorted(df["Symbol"].str.replace(".", "-", regex=False).tolist())


@st.cache_data(ttl=86400)
def all_us_symbols():
    base = "https://www.nasdaqtrader.com/dynamic/SymDir/"
    n = pd.read_csv(base + "nasdaqlisted.txt", sep="|")
    o = pd.read_csv(base + "otherlisted.txt", sep="|")
    n = n[(n["Test Issue"] == "N") & (n["ETF"] == "N")]["Symbol"]
    o = o[(o["Test Issue"] == "N") & (o["ETF"] == "N")]["ACT Symbol"]
    syms = pd.concat([n, o]).dropna().astype(str)
    syms = syms[~syms.str.contains(r"[\$\.]|File Creation")]
    return sorted(set(syms))


# ---------- Data ----------
def download_data(symbols, batch_size=100):
    rows = []
    bar = st.progress(0.0, text="Downloading...")
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        try:
            data = yf.download(batch, period="1mo", auto_adjust=False,
                               group_by="ticker", progress=False, threads=True)
        except Exception:
            continue
        for t in batch:
            try:
                d = data[t].dropna()
            except KeyError:
                continue
            if d.empty:
                continue
            d = d.reset_index()
            d["Ticker"] = t
            rows.append(d[["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]])
        bar.progress(min((i + batch_size) / len(symbols), 1.0),
                     text=f"Downloaded {min(i + batch_size, len(symbols))}/{len(symbols)}")
    bar.empty()
    if not rows:
        st.error("Download failed (Yahoo may be rate-limiting). Try again later.")
        return False
    prices = pd.concat(rows, ignore_index=True)
    prices["Date"] = pd.to_datetime(prices["Date"]).dt.tz_localize(None)
    prices.to_sql("daily_prices", engine, if_exists="replace", index=False,
                  chunksize=5000)
    return True


@st.cache_data(ttl=300)
def read_prices():
    try:
        return pd.read_sql("SELECT * FROM daily_prices", engine,
                           parse_dates=["Date"])
    except Exception:
        return None


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Rules")
    with st.form("rules"):
        days = st.slider("Look-back (trading days)", 2, 20, 5)
        mode = st.radio("Range type", ["High - Low (full range)", "Close to Close"])
        max_range = st.number_input("Max range in $", 0.0, 1000.0, 2.0, 0.25)
        min_price = st.number_input("Min last price $", 0.0, 100000.0, 1.0, 0.5)
        max_price = st.number_input("Max last price $ (0 = no limit)",
                                    0.0, 100000.0, 0.0, 1.0)
        min_vol = st.number_input("Min avg daily volume", 0, 100_000_000,
                                  100_000, 50_000)
        name_filter = st.text_input("Ticker contains (optional)").upper()
        st.form_submit_button("Apply")

    st.divider()
    st.header("Data")
    universe = st.selectbox("Stocks to scan", ["S&P 500", "All US stocks"])
    if universe == "All US stocks":
        st.caption("About 6,000 symbols. Slow, and may crash free cloud hosting.")
    if st.button("Download / refresh data"):
        try:
            symbols = sp500_symbols() if universe == "S&P 500" else all_us_symbols()
        except Exception as e:
            st.error(f"Could not load the symbol list: {e}")
            symbols = []
        if symbols:
            with st.spinner(f"Scanning {len(symbols)} symbols..."):
                ok = download_data(symbols)
            if ok:
                read_prices.clear()
                st.success("Data saved.")
                st.rerun()
    st.caption(f"Storage: {db_label}")

# ---------- Screening ----------
prices = read_prices()
if prices is None or prices.empty:
    st.info("No data yet. Click **Download / refresh data** in the sidebar.")
    st.stop()

st.caption(f"Data up to {prices['Date'].max().date()} - "
           f"{prices['Ticker'].nunique()} stocks stored")

recent = prices.sort_values("Date").groupby("Ticker").tail(days)
g = recent.groupby("Ticker")
res = pd.DataFrame({
    "week_low": g["Low"].min(),
    "week_high": g["High"].max(),
    "close_low": g["Close"].min(),
    "close_high": g["Close"].max(),
    "last_close": g["Close"].last(),
    "avg_volume": g["Volume"].mean(),
    "days": g["Close"].count(),
})
res = res[res["days"] >= max(2, days - 1)]
if mode.startswith("High"):
    res["range"] = res["week_high"] - res["week_low"]
else:
    res["range"] = res["close_high"] - res["close_low"]

res = res[(res["range"] <= max_range)
          & (res["last_close"] >= min_price)
          & (res["avg_volume"] >= min_vol)]
if max_price > 0:
    res = res[res["last_close"] <= max_price]
if name_filter:
    res = res[res.index.str.contains(name_filter)]

res = res.sort_values("range")[
    ["last_close", "week_low", "week_high", "range", "avg_volume"]].round(2)
res["avg_volume"] = res["avg_volume"].astype(int)

st.subheader(f"{len(res)} stocks matched")
st.dataframe(res, use_container_width=True)
st.download_button("Download results (CSV)", res.to_csv().encode(),
                   "results.csv", "text/csv")

# ---------- Details ----------
if len(res):
    st.subheader("Details")
    t = st.selectbox("Pick a stock", res.index.tolist())
    hist = prices[prices["Ticker"] == t].sort_values("Date")
    st.line_chart(hist.set_index("Date")[["Close", "High", "Low"]])
    st.dataframe(hist.drop(columns="Ticker").round(2), use_container_width=True)
    try:
        info = yf.Ticker(t).info
        c1, c2, c3 = st.columns(3)
        c1.metric("Company", info.get("longName", "-"))
        c2.metric("Sector", info.get("sector", "-"))
        c3.metric("Price", info.get("currentPrice", "-"))
        c1.metric("Market cap", f"{info.get('marketCap', 0):,}")
        c2.metric("P/E", info.get("trailingPE", "-"))
        c3.metric("Dividend yield", info.get("dividendYield", "-"))
        c1.metric("52w low", info.get("fiftyTwoWeekLow", "-"))
        c2.metric("52w high", info.get("fiftyTwoWeekHigh", "-"))
    except Exception:
        st.warning("Could not load company details from Yahoo.")
