# ASX RSI(10) Strategy Scanner

Rules implemented:

- Market: ASX ordinary-company universe from the official ASX ISIN directory.
- Market regime: S&P/ASX 200 (`^AXJO`) must close above its 200-day simple moving average for NEW entries.
- Entry: stock RSI(10) < 30.
- Exit: RSI(10) > 40 OR 10 trading sessions after entry.
- Existing active signals are managed even if the ASX 200 later falls below its 200-day SMA.
- Email is sent on a new entry and again on its exit.
- State is persisted in `data/state.json`, preventing the same active signal from generating a new entry email every day.

## Timing

The scheduled scan runs once each weekday after the Australian market close. This is intentionally an end-of-day strategy: signals are calculated from completed daily bars.

## Setup

1. Create a fresh GitHub repo, e.g. `asx-rsi-scanner`.
2. Upload the project contents.
3. Add repository secrets:
   - `SMTP_USERNAME`
   - `SMTP_APP_PASSWORD`
4. Settings -> Pages -> Source: GitHub Actions.
5. Actions -> Scan ASX RSI strategy -> Run workflow.
6. After the first successful run, GitHub Pages hosts the dashboard.

## Optional filters

`config.json` includes:
- `minimum_price`
- `minimum_average_volume_20d`

Both default to zero, meaning no liquidity/price filter is imposed yet.

This is a mechanical signal scanner, not financial advice.
