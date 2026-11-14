# Stock Market Predictor

A Streamlit app that downloads live daily prices from Yahoo Finance, forecasts the next
1–100 trading days with an LSTM as a range, lets you test a trade idea against thousands of
simulated futures, and shows its own track record for the stock you choose.

> **Educational project, not financial advice.** Stock prices cannot be predicted reliably.
> The measured accuracy is listed under [Results](#results); read it before trusting any number.

## Features

**Forecast**
- Live prices for any Yahoo Finance symbol (`GOOG`, `AAPL`, `TCS.NS`, `^NSEI`, ...), adjusted for splits and dividends
- Forecast for 1–100 trading days: a middle line plus an 80% range, downloadable as CSV
- Example scenarios drawn on the chart, built from the stock's own past moves including its crashes and jumps

**Plan a trade** (answers come from 5,000 simulated futures)
- Chance of the price closing above, or touching, a target within a holding period
- Target versus stop-loss: which is likely to be reached first
- What an invested amount could be worth in a bad, typical and good case
- The largest amount to invest for a loss limit you choose

**Track record** (the app audits itself for the chosen stock)
- A forecast made 100 trading days ago compared with what actually happened
- Error table against a "price does not change" baseline
- Calibration table: when the tool stated an x% chance, how often it really happened

**News & events** (context the price-only model cannot see)
- Recent headlines for the chosen stock and its next earnings date, from Yahoo Finance
- A warning when earnings are due within 14 days
- A warning when the stock has moved unusually sharply in the last 10 trading days

**Also:** 50 / 100 / 200-day moving averages, retries and an on-disk cache, and a visible
warning when saved data is shown instead of live data.

## Project structure

```
app.py                      Streamlit user interface
stock_utils.py              Data download, features, forecasting, backtesting
scenarios.py                Scenario simulation, trade-planning maths, calibration
news.py                     Headlines and earnings dates from Yahoo Finance
train.py                    Training pipeline (creates the files in models/)
stock.ipynb                 Model report: results, charts, examples
models/stock_lstm.keras     Trained model (must be committed)
models/model_meta.json      Post-processing constants and training metadata (must be committed)
models/metrics.json         Test results written by train.py
tests/                      Offline unit tests
requirements.txt            Runtime dependencies (used by Streamlit Cloud)
requirements-dev.txt        Adds Jupyter, matplotlib and pytest
.streamlit/config.toml      Streamlit settings
.github/workflows/ci.yml    Runs the tests on every push
```

## Run locally

Requires Python 3.13 (the version the pinned packages were tested on).

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub, including the `models/` folder:
   ```bash
   git init
   git add .
   git commit -m "Stock market predictor"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
2. Go to https://share.streamlit.io, sign in with GitHub and choose **Create app**.
3. Select the repository, branch `main`, and main file path `app.py`.
4. Open **Advanced settings** and set the Python version to **3.13**.
5. Click **Deploy**. The first build takes a few minutes because TensorFlow is large.

No secrets or API keys are needed.

**Yahoo rate limits.** Streamlit Cloud apps share IP addresses, and Yahoo sometimes throttles
them. The app retries, reuses each download for 15 minutes, and falls back to the last saved
copy with a warning. If a symbol has never been loaded on that server and Yahoo is throttling,
the app shows an error and works again on a later attempt.

## How the model works

| | |
|---|---|
| Input | 100 trading days of six features: returns, shape of the price path, size of each day's move, volume, daily high–low range, and whether the stock is calmer or more turbulent than its own normal. All are divided by the stock's recent volatility. |
| Output | For each of the next 100 trading days: the 10%, 50% and 90% quantiles of the price, predicted in one step |
| Network | LSTM(64) → LSTM(32) → Dense(64) → 300 outputs, quantile (pinball) loss; three copies with different random seeds are averaged |
| Training data | 24 stocks from 2012: 16 US and 8 Indian, across technology, banking, energy, retail, industry and healthcare |
| Split (by time, per stock) | 70% train, 10% validation, 20% test, with 100-day gaps so nothing overlaps |
| Tuned on validation only | How far to pull the middle line towards "no change" (result: halfway), and a scale factor so the range covers 80% |

Because the model sees volatility-scaled features rather than raw prices, it works at price
levels it never saw in training and for symbols it was not trained on.

**Scenarios.** The model gives three numbers per day; trade questions need whole price paths.
`scenarios.py` stitches together random 10-day blocks of the stock's own past returns, then
stretches the paths so their 10%, 50% and 90% levels match the model on every day. The paths
keep the stock's real crashes and jumps, and every number in the app comes from the same set.

## Results

Average percentage error (MAPE) over 1,552 forecasts on the held-out last 20% of all 24 stocks.
"Naive" assumes the price does not change.

| Days ahead | LSTM model | Naive | Drift (average growth) |
|---|---|---|---|
| 1 | 1.28% | 1.29% | 1.28% |
| 5 | 3.06% | 3.08% | 3.05% |
| 20 | 5.98% | 6.00% | 6.04% |
| 50 | 9.78% | 9.82% | 10.08% |
| 100 | 13.33% | 13.38% | 14.21% |

- The middle line is as accurate as the naive baseline, not meaningfully better: the margin is
  a few hundredths of a percentage point, and at 100 days the model wins on 12 of 24 stocks.
- The 80% range contained the real price 75.6% of the time overall, and between 61% and 91%
  for individual stocks. Check the *Track record* tab for the stock you care about.
- Direction is not predictable from price history. The value of the app is the range, the
  probabilities and their track record, not the single line.

## Retrain the model

```bash
pip install -r requirements-dev.txt
python train.py                   # about 15 minutes on a CPU
jupyter notebook stock.ipynb      # optional: Run All to refresh the report
```

`train.py` overwrites the three files in `models/` and prints the test results. Commit them
and push; Streamlit Cloud redeploys automatically. Retrain with the same `keras` version that
is pinned in `requirements.txt`, or update the pin in the same commit. To change the training
stocks, edit `UNIVERSE` in `train.py`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The tests need no network access. GitHub Actions runs them on every push.

## Maintenance notes

- **Import order.** `yfinance` must be imported before TensorFlow/Keras. TensorFlow bundles its
  own libcurl, and if it loads first every Yahoo request fails with `ImpersonateError`.
  `stock_utils.py` enforces the order and a test guards it; do not add `import keras` or
  `import tensorflow` at the top of `app.py`.
- **yfinance version.** Yahoo's API is unofficial and changes occasionally. If downloads start
  failing everywhere, raise the `yfinance` version in `requirements.txt`.
- **Forecast dates** are Monday–Friday business days; exchange holidays are not excluded.
- **News** is shown for the user to read; the forecast does not use it. Yahoo's coverage is
  thinner and less relevant for non-US stocks, and earnings dates are often missing for them.
  If headlines or dates are unavailable the tab says so and the rest of the app is unaffected.
- **Sharp-move warning.** It fires when one day moved at least 3.5 times the stock's normal
  daily move, or 10 days moved at least 2.5 times the normal amount (roughly 1 day in 12 for
  a typical stock). The thresholds are constants at the top of `stock_utils.py`.
- **Trade planning** uses daily closing prices and ignores brokerage, taxes and price gaps.

## License

No license file is included yet. Add one (MIT is a common choice) before publishing if you
want others to be able to reuse the code.
