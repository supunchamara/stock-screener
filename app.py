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


# ---------- Rule defaults and reset ----------
DEFAULTS = {"days": 5, "mode": "High - Low (full range)", "max_range": 2.0,
            "min_price": 1.0, "max_price": 0.0, "min_vol": 100_000,
            "name_filter": ""}
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def reset_rules():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Rules")
    with st.form("rules"):
        days = st.slider("Look-back (trading days)", 2, 20, key="days")
        mode = st.radio("Range type", ["High - Low (full range)", "Close to Close"],
                        key="mode")
        max_range = st.number_input("Max range in $", min_value=0.0,
                                    max_value=1000.0, step=0.25, key="max_range")
        min_price = st.number_input("Min last price $", min_value=0.0,
                                    max_value=100000.0, step=0.5, key="min_price")
        max_price = st.number_input("Max last price $ (0 = no limit)",
                                    min_value=0.0, max_value=100000.0, step=1.0,
                                    key="max_price")
        min_vol = st.number_input("Min avg daily volume", min_value=0,
                                  max_value=100_000_000, step=50_000,
                                  key="min_vol")
        name_filter = st.text_input("Ticker contains (optional)",
                                    key="name_filter").upper()
        b1, b2 = st.columns(2)
        with b1:
            st.form_submit_button("Apply")
        with b2:
            st.form_submit_button("Clear", on_click=reset_rules)

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
    hist = prices[prices["Ticker"] == t].sort_values("Date").tail(days)
    lo, hi = hist["Low"].min(), hist["High"].max()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Lowest price", f"${lo:.2f}")
    m2.metric("Highest price", f"${hi:.2f}")
    m3.metric("Range (high - low)", f"${hi - lo:.2f}")
    m4.metric("Last close", f"${hist['Close'].iloc[-1]:.2f}")
    st.caption("Closing price for each day")
    st.line_chart(hist.set_index("Date")["Close"])
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
