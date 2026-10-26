# QQE Signal Trading

Upbit KRW-BTC bot using QQE MOD signals on six-hour bars. Live trading and backtests share the same strategy kernel.

## Usage

```sh
uv sync
cp .env.example .env
uv run python autotrade_coin.py
```

On PowerShell, use `Copy-Item .env.example .env`. Configure `.env` for your market and position size. `DRY_RUN=1` uses a separate paper balance; `DRY_RUN=0` submits real orders and requires Upbit API keys. Telegram settings are optional.

State and logs are written to `out/`; paper state is in `out/paper/`. Stop older instances before upgrading. For a halt or unresolved order, check exchange records before editing state or removing a stop file. Positions and pending orders must stay with their recorded market.

### Backtests

```sh
uv run python upbit_candle_fetcher.py --market KRW-BTC --days 365 --outdir coin_hist_1h
uv run python backtest_coin.py
```

Use `--update` to extend the candle dataset and `--compare` to compare backtest settings. Signals fill at the next bar's open, with fees and slippage.

### Checks

```sh
uv run python test_backtest.py
uv run python test_engine.py
uv run python test_trade.py
uv run python test_upbit.py
bash check.sh
```

With `coin_hist_1h/KRW-BTC.parquet` collected:

```sh
uv run python test_kernel.py
uv run python test_mutation_gate.py
```

## Sources

- [QQE MOD — Mihkel00](https://www.tradingview.com/script/TpUW4muw-QQE-MOD/)
- [Upbit API](https://docs.upbit.com/)
