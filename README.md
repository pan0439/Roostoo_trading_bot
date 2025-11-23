# Roostoo_trading_bot
SMA20/50 crossover trading strategy that trades multiple crypto assets on Roostoo, using Horus for historical price data.

Features

Multi-asset trading across all coins supported on both Roostoo & Horus

SMA20 / SMA50 crossover signals

Batch execution to respect rate limits (e.g., 15 assets per batch)

Rebalancing with threshold bands to reduce noise trading

Market or limit-at-best order support

State persistence to avoid double-processing candles

Graceful shutdown with SIGINT/SIGTERM handling

Automatic timestamp alignment to 15-minute candle closes

The bot automatically:

Fetches all tradeable symbols from Roostoo

Cross-matches them with Horus-supported assets

Produces a final list of assets to trade (ALL_ASSETS)

These assets are divided into batches (BATCH_SIZE = 15) for efficient loop execution.
