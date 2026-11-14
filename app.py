"""Streamlit front end for the stock forecaster.

Run locally with:  streamlit run app.py

Do not import keras / tensorflow in this file: see the import note in stock_utils.py.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import news
import scenarios as sc
from stock_utils import (HORIZON, LOOKBACK, Artifacts, DataDownloadError, PriceArrays, coverage,
                         download_stock_data, forecast_path, load_artifacts, normalize_ticker,
                         prepare, recent_shock, summary_table, walk_forward)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

DEFAULT_START = date(2012, 1, 1)
MIN_ROWS = LOOKBACK + 1
DATA_TTL_SECONDS = 15 * 60          # how long a download is reused before asking Yahoo again
BACKTEST_START_FRACTION = 0.8       # training never used the last 20% of any series
NEWS_TTL_SECONDS = 30 * 60
EARNINGS_WARNING_DAYS = 14          # calendar days: warn when earnings are this close
EXAMPLE_PATHS = 25
CURRENCY_BY_SUFFIX = {".NS": "₹", ".BO": "₹", ".L": "£", ".TO": "C$", ".AX": "A$",
                      ".DE": "€", ".PA": "€", ".T": "¥", ".HK": "HK$"}

st.set_page_config(page_title="Stock Market Predictor", page_icon="📈", layout="wide")


# ----------------------------------------------------------------------------
# Cached work
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def get_artifacts() -> Artifacts:
    return load_artifacts()


@st.cache_data(ttl=DATA_TTL_SECONDS, max_entries=32, show_spinner="Downloading prices from Yahoo Finance...")
def get_prices(ticker: str, start: date) -> tuple[pd.DataFrame, str]:
    df = download_stock_data(ticker, str(start))
    return df, df.attrs.get("source", "yahoo")


@st.cache_data(ttl=DATA_TTL_SECONDS, max_entries=32, show_spinner="Simulating scenarios...")
def get_forecast(ticker: str, start: date, last_date: pd.Timestamp, _pa: PriceArrays):
    """Forecast table and simulated paths. Cached per (ticker, start, last_date)."""
    fc = forecast_path(get_artifacts(), _pa)
    paths = sc.simulate_paths(_pa, fc[["Lower", "Predicted", "Upper"]].to_numpy())
    return fc, paths


@st.cache_data(ttl=DATA_TTL_SECONDS, max_entries=32, show_spinner="Checking the track record...")
def get_track_record(ticker: str, start: date, last_date: pd.Timestamp, _pa: PriceArrays):
    art = get_artifacts()
    start_pos = max(LOOKBACK, int(len(_pa) * BACKTEST_START_FRACTION))
    bt = walk_forward(art, _pa, start_pos=start_pos)
    if bt is None:
        return None
    calibration = sc.calibration_table(art, _pa, start_pos=start_pos)
    return summary_table(bt), coverage(bt), len(bt["origins"]), calibration[0] if calibration else None


@st.cache_data(ttl=NEWS_TTL_SECONDS, max_entries=64, show_spinner=False)
def get_news(ticker: str, today: date) -> tuple[list[news.Headline], date | None]:
    """Headlines and next earnings date. Both are empty when Yahoo has nothing or fails."""
    return news.fetch_headlines(ticker), news.fetch_next_earnings(ticker, today)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def currency_symbol(ticker: str) -> str:
    for suffix, symbol in CURRENCY_BY_SUFFIX.items():
        if ticker.endswith(suffix):
            return symbol
    return "$" if "." not in ticker and not ticker.startswith("^") else ""


def forecast_chart(fc: pd.DataFrame, history: pd.Series, title: str, actual: pd.Series | None = None,
                   example_paths: np.ndarray | None = None) -> go.Figure:
    fig = go.Figure()
    if example_paths is not None:
        for i, path in enumerate(example_paths):
            fig.add_trace(go.Scatter(x=fc.index, y=path, mode="lines", hoverinfo="skip",
                                     line=dict(color="rgba(90,90,90,0.25)", width=1),
                                     name="Example scenarios", legendgroup="scenarios",
                                     showlegend=i == 0))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["Upper"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["Lower"], fill="tonexty",
                             fillcolor="rgba(255,0,0,0.15)", line=dict(width=0), name="80% range"))
    fig.add_trace(go.Scatter(x=history.index, y=history, name="History", line=dict(color="green")))
    if actual is not None:
        fig.add_trace(go.Scatter(x=actual.index, y=actual, name="Actual", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["Predicted"], name="Forecast (middle)", line=dict(color="red")))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", hovermode="x unified")
    return fig


def moving_average_chart(close: pd.Series, windows: tuple[int, ...]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close, name="Close", line=dict(color="green")))
    for window, colour in zip(windows, ("red", "blue")):
        fig.add_trace(go.Scatter(x=close.index, y=close.rolling(window).mean(),
                                 name=f"MA{window}", line=dict(color=colour)))
    title = "Price vs " + " vs ".join(f"MA{w}" for w in windows)
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price")
    return fig


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
def forecast_tab(ticker: str, cur: str, close: pd.Series, fc_full: pd.DataFrame, paths: np.ndarray) -> None:
    n_days = st.slider("Trading days to forecast", 1, HORIZON, HORIZON)
    fc = fc_full.iloc[:n_days]
    last_date, last_price = close.index[-1], float(close.iloc[-1])
    end_price = float(fc["Predicted"].iloc[-1])

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Last close ({last_date.date()})", f"{cur}{last_price:,.2f}")
    m2.metric(f"Forecast in {n_days} trading days", f"{cur}{end_price:,.2f}",
              f"{(end_price / last_price - 1) * 100:+.1f}%")
    m3.metric("80% range", f"{cur}{fc['Lower'].iloc[-1]:,.2f} – {cur}{fc['Upper'].iloc[-1]:,.2f}")

    show_paths = st.toggle("Show example scenarios", value=True,
                           help="Possible futures built from this stock's own past moves, including its "
                                "sharp falls and jumps. They are illustrations, not predictions.")
    st.plotly_chart(
        forecast_chart(fc, close.iloc[-250:], f"{ticker}: forecast from {last_date.date()}",
                       example_paths=paths[:EXAMPLE_PATHS, :n_days] if show_paths else None),
        width="stretch")
    st.caption("The red line is the middle of the range, not a promise. It is smooth because the "
               "timing of future jumps cannot be known; the grey lines show what real paths can look like.")
    st.download_button("Download forecast (CSV)", fc.round(2).to_csv(),
                       file_name=f"{ticker}_forecast_{last_date.date()}.csv", mime="text/csv")


def plan_tab(ticker: str, cur: str, last_price: float, paths: np.ndarray) -> None:
    st.write("Test a trade idea before you place it. Every answer comes from "
             f"{len(paths):,} simulated futures for {ticker}.")
    c1, c2, c3, c4 = st.columns(4)
    days = c1.slider("Holding period (trading days)", 5, HORIZON, 60, key=f"days_{ticker}")
    amount = c2.number_input(f"Amount to invest ({cur or 'currency'})", min_value=0.0, value=50000.0,
                             step=1000.0, key=f"amount_{ticker}")
    target = c3.number_input("Target price", min_value=0.01, value=round(last_price * 1.10, 2),
                             key=f"target_{ticker}")
    stop = c4.number_input("Stop-loss price", min_value=0.01, value=round(last_price * 0.95, 2),
                           key=f"stop_{ticker}")

    st.markdown(f"#### 1. Will it reach {cur}{target:,.2f}?")
    a, b = st.columns(2)
    a.metric(f"Chance of closing above it on day {days}", f"{sc.prob_above(paths, target, days):.0%}")
    b.metric(f"Chance of touching it at any time within {days} days",
             f"{sc.prob_touch(paths, target, days, last_price):.0%}")

    st.markdown("#### 2. Target or stop-loss: which comes first?")
    if stop < last_price < target:
        race = sc.target_vs_stop(paths, target, stop, days)
        a, b, c = st.columns(3)
        a.metric(f"Target {cur}{target:,.2f} first", f"{race['target_first']:.0%}")
        b.metric(f"Stop-loss {cur}{stop:,.2f} first", f"{race['stop_first']:.0%}")
        c.metric("Neither within the period", f"{race['neither']:.0%}")
        gain, loss = target / last_price - 1, 1 - stop / last_price
        expected = race["target_first"] * gain - race["stop_first"] * loss
        st.caption(f"Reward if the target is hit: {gain:+.1%}. Loss if the stop is hit: {-loss:.1%}. "
                   f"Weighted by these chances, the two outcomes average {expected:+.1%} "
                   "(ignoring the cases where neither is hit, and ignoring costs and taxes).")
    else:
        st.info("Enter a stop-loss below the last close and a target above it to see this comparison.")

    st.markdown(f"#### 3. What could {cur}{amount:,.0f} be worth after {days} trading days?")
    out = sc.outcome_summary(paths, days, last_price, amount)
    a, b, c, d = st.columns(4)
    a.metric("Bad case (1 in 20)", f"{cur}{out['bad_case_value']:,.0f}",
             f"{out['bad_case_value'] - amount:+,.0f}")
    b.metric("Typical case", f"{cur}{out['typical_value']:,.0f}", f"{out['typical_value'] - amount:+,.0f}")
    c.metric("Good case (1 in 20)", f"{cur}{out['good_case_value']:,.0f}",
             f"{out['good_case_value'] - amount:+,.0f}")
    d.metric("Chance of ending at a loss", f"{out['prob_loss']:.0%}")

    st.markdown("#### 4. How much should I invest?")
    max_loss = st.number_input(f"The most I am willing to lose in this period ({cur or 'currency'})",
                               min_value=0.0, value=5000.0, step=500.0, key=f"maxloss_{ticker}")
    limit = sc.max_position(paths, days, last_price, max_loss)
    if limit is None:
        st.success("Even the bad case is not a loss over this period, so this rule sets no limit.")
    else:
        st.metric("Largest amount that keeps the bad-case loss within your limit", f"{cur}{limit:,.0f}",
                  help="Bad case = the outcome that only 1 in 20 simulated futures is worse than.")
        st.caption(f"About {limit / last_price:,.1f} shares at the last close. In roughly 1 case in 20 the "
                   "loss would be larger than your limit.")

    st.caption("These are estimates from simulated closing prices. They ignore brokerage, taxes and "
               "price gaps, and they are only as reliable as the Track record tab shows.")


def track_record_tab(ticker: str, start: date, art: Artifacts, pa: PriceArrays, close: pd.Series) -> None:
    if len(close) <= LOOKBACK + HORIZON + 1:
        st.info(f"A track record needs more than {LOOKBACK + HORIZON + 1} trading days of history.")
        return
    t0 = len(close) - 1 - HORIZON
    past = forecast_path(art, pa, t=t0)
    actual = close.iloc[t0 + 1:]
    mape = float(np.mean(np.abs(past["Predicted"].to_numpy() - actual.to_numpy()) / actual.to_numpy()) * 100)
    st.plotly_chart(
        forecast_chart(past, close.iloc[max(0, t0 - 150): t0 + 1],
                       f"Forecast made on {close.index[t0].date()} using only data up to that day "
                       f"(average error {mape:.1f}%)", actual=actual),
        width="stretch")

    record = get_track_record(ticker, start, close.index[-1], pa)
    if record is None:
        return
    table, covered, n_forecasts, calibration = record

    st.markdown("#### Forecast error")
    st.write(f"Average error (MAPE, %) over {n_forecasts} forecasts made across the most recent 20% of "
             "the history. *Naive* assumes the price does not change; the forecast line is only useful "
             "where it beats that row.")
    st.dataframe(table, width="stretch")
    st.write(f"The real price was inside the 80% range **{covered:.0f}%** of the time (ideal: 80%).")

    if calibration is not None:
        st.markdown("#### Were the probabilities right?")
        st.write("Each row is a chance the tool stated for the price finishing below some level. The "
                 "columns show how often that really happened, in %. Numbers close to the row label "
                 "mean the probabilities in *Plan a trade* can be trusted for this stock; numbers far "
                 "off mean they cannot.")
        st.dataframe(calibration, width="stretch")
        st.caption(f"Based on {n_forecasts} overlapping past forecasts, so differences of 10 points "
                   "or so can be chance.")


def context_warnings(ticker: str, pa: PriceArrays, earnings: date | None) -> None:
    """Banners for situations the price-only model cannot account for."""
    shock = recent_shock(pa)
    if shock is not None:
        st.warning(
            f"**Unusually sharp move.** {ticker} moved {shock['total_move']:+.1%} over the last "
            f"{shock['window']} trading days, including {shock['largest_daily_move']:+.1%} on "
            f"{shock['largest_move_date'].date()}. Moves like this are usually driven by news, which "
            "the model does not see. Read the *News & events* tab and treat the forecast range as "
            "probably too narrow.")
    if earnings is not None and (earnings - date.today()).days <= EARNINGS_WARNING_DAYS:
        st.warning(
            f"**Earnings due on {earnings:%d %b %Y}.** Results often move a stock sharply in either "
            "direction, and the forecast does not account for them.")


def news_tab(ticker: str, headlines: list[news.Headline], earnings: date | None,
             forecast_end: pd.Timestamp) -> None:
    st.markdown("#### Next earnings date")
    if earnings is None:
        st.write("Yahoo Finance lists no upcoming earnings date for this symbol. Indices and many "
                 "non-US stocks have none, so check the company's own announcements.")
    else:
        inside = "inside" if earnings <= forecast_end.date() else "after"
        st.write(f"**{earnings:%d %b %Y}** ({(earnings - date.today()).days} days from today), "
                 f"{inside} the {HORIZON}-day forecast period.")

    st.markdown("#### Recent headlines")
    if not headlines:
        st.info("Yahoo Finance returned no headlines for this symbol right now.")
    for item in headlines:
        title = item.title.replace("[", "(").replace("]", ")")
        when = f"{item.published:%d %b %Y}" if item.published else "date unknown"
        st.markdown(f"- [{title}]({item.url})  \n  {item.publisher} · {when}")
    st.caption("Headlines and dates come from Yahoo Finance and are shown as provided; coverage is "
               "thinner for non-US stocks. They are context for you to read. The forecast does not "
               "use them.")


def main() -> None:
    st.title("Stock Market Predictor")

    try:
        art = get_artifacts()
    except Exception as exc:
        logging.exception("Model loading failed")
        st.error(f"Could not load the model: {exc}")
        st.stop()

    col1, col2 = st.columns(2)
    raw_ticker = col1.text_input("Stock symbol", "GOOG",
                                 help="Yahoo Finance symbol, e.g. GOOG, AAPL, TSLA, TCS.NS, RELIANCE.NS, ^NSEI")
    start = col2.date_input("History start date", DEFAULT_START, min_value=date(1990, 1, 1),
                            max_value=date.today() - timedelta(days=200))

    try:
        ticker = normalize_ticker(raw_ticker)
        data, source = get_prices(ticker, start)
        pa = prepare(data)
    except (ValueError, DataDownloadError) as exc:
        st.error(str(exc))
        st.stop()

    if len(pa) < MIN_ROWS:
        st.error(f"{ticker} has only {len(pa)} trading days since {start}; "
                 f"at least {MIN_ROWS} are needed. Choose an earlier start date.")
        st.stop()

    close = pd.Series(pa.close, index=pa.index, name="Close")
    last_date = close.index[-1]
    if source == "cache":
        st.warning(f"Yahoo Finance is not responding, so saved data is shown. "
                   f"Last available day: {last_date.date()}.")
    st.caption(f"{len(close):,} trading days, {close.index[0].date()} to {last_date.date()} · "
               f"adjusted for splits and dividends · source: Yahoo Finance")

    cur = currency_symbol(ticker)
    fc, paths = get_forecast(ticker, start, last_date, pa)
    headlines, earnings = get_news(ticker, date.today())
    context_warnings(ticker, pa, earnings)

    tabs = st.tabs(["Forecast", "Plan a trade", "Track record", "News & events",
                    "Moving averages", "Price data"])
    with tabs[0]:
        forecast_tab(ticker, cur, close, fc, paths)
    with tabs[1]:
        plan_tab(ticker, cur, float(close.iloc[-1]), paths)
    with tabs[2]:
        track_record_tab(ticker, start, art, pa, close)
    with tabs[3]:
        news_tab(ticker, headlines, earnings, fc.index[-1])
    with tabs[4]:
        for windows in ((50,), (50, 100), (100, 200)):
            st.plotly_chart(moving_average_chart(close, windows), width="stretch")
    with tabs[5]:
        st.dataframe(data.sort_index(ascending=False), width="stretch")

    st.caption(f"Model trained on {len(art.trained_on)} US and Indian stocks (prices up to "
               f"{art.data_until}, with the most recent 20% held back for testing). "
               "Educational project, not financial advice: stock prices cannot be predicted "
               "reliably, so read the range rather than the single line.")


main()
