# SMA Crossover Multi-Asset Trading Bot

This project implements an automated **SMA20/50 crossover** trading
strategy that trades multiple crypto assets on **Roostoo**, using
**Horus** for historical price data.\

## Features

-   **Multi-asset trading** across all coins supported on both Roostoo &
    Horus
-   **SMA20 / SMA50 crossover signals**
-   **Batch execution** to respect rate limits (e.g., 15 assets per
    batch)
-   **Rebalancing with threshold bands** to reduce noise trading
-   **Market or limit-at-best order support**
-   **State persistence** to avoid double-processing candles
-   **Graceful shutdown** with SIGINT/SIGTERM handling
-   **Automatic timestamp alignment** to 15-minute candle closes

## Strategy Logic

### 1. Asset Universe

The bot automatically: - Fetches all tradeable symbols from
**Roostoo** - Cross-matches them with Horus-supported assets\
- Produces a final list of assets to trade (`ALL_ASSETS`)

These assets are divided into batches (`BATCH_SIZE = 15`) for efficient
loop execution.

### 2. Candle Timing

The strategy operates on **15-minute bars**:

-   Waits for each candle to fully close
-   Adds \~60 seconds buffer (`RUN_OFFSET_SEC`)\
-   Then executes the next batch

### 3. Signal Generation (SMA Crossover)

For each asset: - Fetch the last \~60 candles\
- Compute: - **SMA20** (fast) - **SMA50** (slow) - Generate signal: -
**BUY** → SMA20 \> SMA50\
- **SELL** → SMA20 \< SMA50\
- **HOLD** → Otherwise

### 4. Portfolio Allocation

If multiple assets have BUY signals:

-   Allocate equal weights across BUY-signal assets\
-   Max per-coin weight capped at **1/3 of total portfolio**\
-   If no BUY signals → stay 100% in USD

### 5. Rebalancing Logic

Trades trigger only if:

    abs(diff) / target_value > 0.10

This prevents unnecessary small trades.

### 6. Order Execution

Supports: - **Market orders** (default) - **Limit-at-best** orders if
preferred

Includes: - Fee estimation (0.1%) - Slippage padding (5 bps) - Dust +
min-notional filtering

## Main Loop

1.  Wait for next candle close\
2.  Select current batch\
3.  Collect market data\
4.  Generate SMA signals\
5.  Compute allocation\
6.  Rebalance\
7.  Move to next batch\
8.  Sleep until next cycle
