from dotenv import load_dotenv
from pathlib import Path
import os
import requests
import time
import hmac
import hashlib
import math
from datetime import datetime, timezone, timedelta
import json, os, time, math
import signal
from typing import Optional

try:
    base_path = Path(__file__).resolve().parent
except NameError:
    base_path = Path.cwd()

env_path = base_path / "Roostoo.env"
load_dotenv(env_path)

API_KEY = os.getenv("Roostoo_API_KEY")
SECRET_KEY = os.getenv("Roostoo_API_SECRET")
Horus_api_key = os.getenv("Horus_API_KEY")

HORUS_BASE_URL = "https://api-horus.com"
BASE_URL = "https://mock-api.roostoo.com"

# --- Roostoo ---

def _get_timestamp():
    """Return a 13-digit millisecond timestamp as string."""
    return str(int(time.time() * 1000))

def _get_signed_headers(payload: Optional[dict] = None):
    """
    Generate signed headers and totalParams for RCL_TopLevelCheck endpoints.
    """
    if payload is None:
        payload = {}
    else:
        payload = dict(payload)
    payload['timestamp'] = _get_timestamp()
    sorted_keys = sorted(payload.keys())
    total_params = "&".join(f"{k}={payload[k]}" for k in sorted_keys)

    signature = hmac.new(
        SECRET_KEY.encode('utf-8'),
        total_params.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    headers = {
        'RST-API-KEY': API_KEY,
        'MSG-SIGNATURE': signature
    }

    return headers, payload, total_params

# ------------------------------
# Public Endpoints
# ------------------------------

def check_server_time():
    """Check API server time."""
    url = f"{BASE_URL}/v3/serverTime"
    try:
        res = requests.get(url)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error checking server time: {e}")
        return None

def get_exchange_info():
    """Get exchange trading pairs and info."""
    url = f"{BASE_URL}/v3/exchangeInfo"
    try:
        res = requests.get(url)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting exchange info: {e}")
        return None

def get_ticker(pair=None):
    """Get ticker for one or all pairs."""
    url = f"{BASE_URL}/v3/ticker"
    params = {'timestamp': _get_timestamp()}
    if pair:
        params['pair'] = pair
    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting ticker: {e}")
        return None

# ------------------------------
# Signed Endpoints
# ------------------------------

def get_balance():
    """Get wallet balances (RCL_TopLevelCheck)."""
    url = f"{BASE_URL}/v3/balance"
    headers, payload, _ = _get_signed_headers({})
    try:
        res = requests.get(url, headers=headers, params=payload)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting balance: {e}")
        print(f"Response text: {e.response.text if e.response else 'N/A'}")
        return None

def get_pending_count():
    """Get total pending order count."""
    url = f"{BASE_URL}/v3/pending_count"
    headers, payload, _ = _get_signed_headers({})
    try:
        res = requests.get(url, headers=headers, params=payload)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting pending count: {e}")
        print(f"Response text: {e.response.text if e.response else 'N/A'}")
        return None

def place_order(pair_or_coin, side, quantity, price=None, order_type=None):
    """Place a LIMIT or MARKET order."""
    url = f"{BASE_URL}/v3/place_order"
    pair = f"{pair_or_coin}/USD" if "/" not in pair_or_coin else pair_or_coin

    if order_type is None:
        order_type = "LIMIT" if price is not None else "MARKET"

    if order_type == 'LIMIT' and price is None:
        print("Error: LIMIT orders require 'price'.")
        return None

    payload = {
        'pair': pair,
        'side': side.upper(),
        'type': order_type.upper(),
        'quantity': str(quantity)
    }
    if order_type == 'LIMIT':
        payload['price'] = str(price)

    headers, _, total_params = _get_signed_headers(payload)
    headers['Content-Type'] = 'application/x-www-form-urlencoded'

    try:
        res = requests.post(url, headers=headers, data=total_params)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error placing order: {e}")
        print(f"Response text: {e.response.text if e.response else 'N/A'}")
        return None

def query_order(order_id=None, pair=None, pending_only=None):
    """Query order history or pending orders."""
    url = f"{BASE_URL}/v3/query_order"
    payload = {}
    if order_id:
        payload['order_id'] = str(order_id)
    elif pair:
        payload['pair'] = pair
        if pending_only is not None:
            payload['pending_only'] = 'TRUE' if pending_only else 'FALSE'

    headers, _, total_params = _get_signed_headers(payload)
    headers['Content-Type'] = 'application/x-www-form-urlencoded'

    try:
        res = requests.post(url, headers=headers, data=total_params)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying order: {e}")
        print(f"Response text: {e.response.text if e.response else 'N/A'}")
        return None

def cancel_order(order_id=None, pair=None):
    """Cancel specific or all pending orders."""
    url = f"{BASE_URL}/v3/cancel_order"
    payload = {}
    if order_id:
        payload['order_id'] = str(order_id)
    elif pair:
        payload['pair'] = pair

    headers, _, total_params = _get_signed_headers(payload)
    headers['Content-Type'] = 'application/x-www-form-urlencoded'

    try:
        res = requests.post(url, headers=headers, data=total_params)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error canceling order: {e}")
        print(f"Response text: {e.response.text if e.response else 'N/A'}")
        return None
    
# --- Horus ---

def UTC_convert(l):
    '''
    l = [year, month, day, hour, minute, second]
    HK is UTC + 8h
    UTC timestamp (seconds)
    '''
    return datetime(*l, tzinfo=timezone.utc).timestamp()

def UTC_to_HK_convert(t):
    '''Convert Unix UTC timestamp to Hong Kong local datetime'''
    hk_tz = timezone(timedelta(hours=8))
    dt_utc = datetime.fromtimestamp(t, tz=timezone.utc)
    dt_hk = dt_utc.astimezone(hk_tz)
    return dt_hk.strftime('%Y-%m-%d %H:%M:%S')

def get_historical_price(asset="BTC", interval="1d", start=None, end=None, format="json"):
    '''
    asset: coin symbol (e.g., BTC)
    interval: [1d, 1h, 15m]
    start: Unix timestamp (seconds, inclusive)
    end: Unix timestamp (seconds, exclusive)
    format: [json, csv]
    '''
    base = f"{HORUS_BASE_URL}/market/price"
    params = [f"asset={asset}", f"interval={interval}", f"format={format}"]
    
    if start:
        params.append(f"start={start}")
    if end:
        params.append(f"end={end}")
    
    url = base + "?" + "&".join(params)
    headers = {"X-API-Key": Horus_api_key}
    res = requests.get(url, headers=headers)
    return res.json()

def get_all_tradeable_coins_Roostoo():
    '''all tradeable in Roostoo'''
    res = get_exchange_info()
    if not isinstance(res, dict):
        return []

    trade_pairs = res.get("TradePairs")
    if not trade_pairs:
        return []

    if isinstance(trade_pairs, dict):
        symbols = trade_pairs.keys()
    else:
        symbols = trade_pairs

    tradeable = []
    for pair in symbols:
        if not isinstance(pair, str) or "/" not in pair:
            continue
        tradeable.append(pair.split("/")[0])
    return tradeable

def compare_tradeable(l, _coins_Horus):
    if not l:
        return []
    coin_set = set(_coins_Horus or [])
    return [i for i in l if i in coin_set]

# --- preparation ---

_tradeable_Roostoo = get_all_tradeable_coins_Roostoo()
_coins_Horus = [
    "BTC", "ETH", "XRP", "BNB", "SOL", "DOGE", "TRX", "ADA", "XLM", "WBTC",
    "SUI", "HBAR", "LINK", "BCH", "WBETH", "UNI", "AVAX", "SHIB", "TON", "LTC",
    "DOT", "PEPE", "AAVE", "ONDO", "TAO", "WLD", "APT", "NEAR", "ARB", "ICP",
    "ETC", "FIL", "TRUMP", "OP", "ALGO", "POL", "BONK", "ENA", "ENS", "VET",
    "SEI", "RENDER", "FET", "ATOM", "VIRTUAL", "SKY", "BNSOL", "RAY", "TIA",
    "JTO", "JUP", "QNT", "FORM", "INJ", "STX"
    ]
_tradeable_both = compare_tradeable(_tradeable_Roostoo, _coins_Horus)

# --- strategy ---

ALL_ASSETS = [
 'BTC','ETH','BNB','SOL','ADA','AVAX','DOT','LINK',
 'XRP','LTC','DOGE','TRX','FET','ARB','SUI'
]

QUOTE = "USD"
BATCH_SIZE = 5                 # 36 -> 3 batches of 12
INTERVAL = "15m"                # {"15m": 900, "1h": 3600, "1d": 86400}
RUN_OFFSET_SEC = 60             # run ~60s after candle close (rate limit)
POLLING_SEC = 30                # check loop cadence (don’t set < 10s)

# Trading params
SMA_FAST = 20
SMA_SLOW = 50
BARS_NEEDED = max(SMA_FAST, SMA_SLOW) + 10
API_COOLDOWN_SEC = 0.5

FEE_RATE = 0.001               # 0.1%
SLIPPAGE_BPS = 5               # 0.05% for safety checks
REBALANCE_BAND = 0.10          # only trade if >10% off target
MIN_NOTIONAL_USD = 20
MAX_PORTFOLIO_FRAC_PER_COIN = 1/3
USE_MARKET_ORDERS = True       # False -> limit-at-best

# State persistence
STATE_PATH = "./runner_state.json"

_INTERVAL_SECONDS = {"15m": 900, "1h": 3600, "1d": 86400}[INTERVAL]
_shutdown = False

def _log(*args):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts} UTC]", *args, flush=True)

def _sleep_brief():
    time.sleep(API_COOLDOWN_SEC)

def _now_utc_ms():
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)

def _with_backoff(func, *args, **kwargs):
    delay = 1.0
    for _ in range(5):
        try:
            res = func(*args, **kwargs)
            if isinstance(res, dict) and res.get("Success") is False:
                time.sleep(delay); delay = min(delay*2, 32)
                continue
            return res
        except Exception:
            time.sleep(delay); delay = min(delay*2, 32)
    return {"Success": False, "ErrMsg": "max retries/backoff exceeded"}

def _ceiling_to_interval(ts_sec, step_sec):
    return ts_sec - (ts_sec % step_sec) + step_sec

def _last_n_window(n_bars, step_sec):
    now = int(time.time())
    end = _ceiling_to_interval(now, step_sec)
    start = end - n_bars * step_sec
    return start, end

def _sma(prices, n):
    if len(prices) < n: return None
    return sum(prices[-n:]) / n

def _round_to_precision(value, decimals):
    if decimals is None: return float(value)
    factor = 10 ** decimals
    return math.floor(float(value) * factor) / factor

def _estimate_market_fill_price(ticker_price, side):
    return ticker_price * (1 + SLIPPAGE_BPS/10000.0) if side == "BUY" else ticker_price * (1 - SLIPPAGE_BPS/10000.0)

def _effective_price_with_fee(px, side):
    return px * (1 + FEE_RATE) if side == "BUY" else px * (1 - FEE_RATE)

def _pairify(assets, quote="USD"):
    return [f"{a}/{quote}" for a in assets]

# ---------- State (persist last bar timestamps + batch index) ----------
def _load_state():
    if not os.path.exists(STATE_PATH):
        return {"batch_index": 0, "last_bar_ts_by_asset": {}}
    try:
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"batch_index": 0, "last_bar_ts_by_asset": {}}

def _save_state(state):
    try:
        with open(STATE_PATH, "w") as f:
            json.dump(state, f)
    except Exception as e:
        _log("WARN: failed to save state:", e)

# ---------- Market data ----------
def _safe_get_latest_price(asset, last_n=BARS_NEEDED):
    start, end = _last_n_window(last_n, _INTERVAL_SECONDS)
    bars = _with_backoff(get_historical_price, asset=asset, interval=INTERVAL, start=start, end=end, format="json")
    _sleep_brief()
    if not bars or (isinstance(bars, dict) and bars.get("Success") is False):
        return None, None, None
    try:
        bars = sorted(bars, key=lambda x: x.get("timestamp", 0))
    except Exception:
        pass
    closes = [float(b["price"]) for b in bars if "price" in b]
    ts = [int(b["timestamp"]) for b in bars if "timestamp" in b and "price" in b]
    if not closes: return None, None, None
    return ts, closes, closes[-1]

def _new_bar_available(asset, state):
    ts, closes, _ = _safe_get_latest_price(asset)
    if not ts: return False
    last_bar_ts = ts[-1]
    if state["last_bar_ts_by_asset"].get(asset) == last_bar_ts:
        return False
    state["last_bar_ts_by_asset"][asset] = last_bar_ts
    return True

def should_run_this_cycle(assets, state):
    updated = [_new_bar_available(a, state) for a in assets]
    return any(updated)

# ---------- Exchange / portfolio ----------
def _get_exchange_rules():
    info = _with_backoff(get_exchange_info); _sleep_brief()
    if isinstance(info, dict) and info.get("Success") is False: return {}
    rules = {}
    for pair, meta in (info.get("TradePairs") or {}).items():
        rules[pair] = {
            "price_prec": meta.get("PricePrecision", 2),
            "amt_prec": meta.get("AmountPrecision", 5),
            "min_order": meta.get("MiniOrder", 0)
        }
    return rules

def _portfolio_snapshot(assets):
    bal = _with_backoff(get_balance); _sleep_brief()
    spot = (bal or {}).get("SpotWallet", {})
    usd_free = float(spot.get("USD", {}).get("Free", 0.0))
    ref_prices, holdings = {}, {}
    for asset in assets:
        _, _, px = _safe_get_latest_price(asset); _sleep_brief()
        ref_prices[asset] = px
        holdings[asset] = float(spot.get(asset, {}).get("Free", 0.0))
    total_value_usd = usd_free + sum((holdings[a] * ref_prices[a]) for a in assets if holdings[a] > 0 and ref_prices[a])
    return {"usd_free": usd_free, "holdings": holdings, "prices": ref_prices, "equity_usd": total_value_usd}

def _get_ma_signal(asset):
    ts, closes, last_price = _safe_get_latest_price(asset); _sleep_brief()
    if last_price is None: return "HOLD", None, None, None
    f, s = _sma(closes, SMA_FAST), _sma(closes, SMA_SLOW)
    if f is None or s is None: return "HOLD", last_price, f, s
    if f > s: return "BUY", last_price, f, s
    if f < s: return "SELL", last_price, f, s
    return "HOLD", last_price, f, s

def _target_allocation(assets, signals):
    buys = [a for a in assets if signals.get(a) == "BUY"]
    if not buys: return {a: 0.0 for a in assets}
    w_each = min(MAX_PORTFOLIO_FRAC_PER_COIN, 1.0/len(buys))
    return {a: (w_each if a in buys else 0.0) for a in assets}

def _qty_from_usd(usd, px, amt_prec):
    if not px or px <= 0: return 0.0
    raw = usd / px
    return _round_to_precision(raw, amt_prec)

# ---------- Execution ----------
def _place_market(pair, side, qty):
    if qty <= 0: return None
    coin = pair.split("/")[0]
    _, _, last_px = _safe_get_latest_price(coin); _sleep_brief()
    notional = qty * (last_px or 0)
    if notional < MIN_NOTIONAL_USD: return None
    return _with_backoff(place_order, pair, side, qty)

def _place_limit_at_best(pair, side, qty, price_prec):
    t = _with_backoff(get_ticker, pair); _sleep_brief()
    dat = (((t or {}).get("Data") or {}).get(pair) or {})
    best = dat.get("MinAsk") if side == "BUY" else dat.get("MaxBid")
    if not best: return {"Success": False, "ErrMsg": "No best price"}
    price = _round_to_precision(best, price_prec)
    return _with_backoff(place_order, pair, side, qty, price=price)

# ---------- One batch pass ----------
def run_sma_crossover_once_batch(assets, pairs):
    signals, ref = {}, {}
    for a in assets:
        sig, last_px, f, s = _get_ma_signal(a)
        signals[a] = sig
        ref[a] = {"last": last_px, "sma20": f, "sma50": s}

    pf = _portfolio_snapshot(assets)
    equity = pf["equity_usd"]
    if equity <= 0:
        return {"Success": False, "ErrMsg": "No equity to trade.", "Signals": signals, "Snapshot": pf}

    target_w = _target_allocation(assets, signals)
    rules = _get_exchange_rules()

    actions = []
    for asset, pair in zip(assets, pairs):
        price = pf["prices"].get(asset)
        if not price: continue

        cur_qty = pf["holdings"].get(asset, 0.0)
        cur_val = cur_qty * price
        tgt_val = target_w[asset] * equity

        if tgt_val == 0:
            if cur_val > MIN_NOTIONAL_USD:
                amt_prec = rules.get(pair, {}).get("amt_prec", 5)
                sell_qty = _round_to_precision(cur_qty, amt_prec)
                if sell_qty > 0:
                    actions.append({"pair": pair, "side": "SELL", "qty": sell_qty})
            continue

        diff = tgt_val - cur_val
        if abs(diff) / max(tgt_val, 1) < REBALANCE_BAND:
            continue

        amt_prec = rules.get(pair, {}).get("amt_prec", 5)
        if diff > 0:
            buy_qty = _qty_from_usd(diff, price, amt_prec)
            if buy_qty > 0:
                actions.append({"pair": pair, "side": "BUY", "qty": buy_qty})
        else:
            sell_qty = _qty_from_usd(-diff, price, amt_prec)
            sell_qty = min(sell_qty, cur_qty)
            sell_qty = _round_to_precision(sell_qty, amt_prec)
            if sell_qty > 0:
                actions.append({"pair": pair, "side": "SELL", "qty": sell_qty})

    fills = []
    for a in [x for x in actions if x["side"] == "SELL"] + [x for x in actions if x["side"] == "BUY"]:
        coin = a["pair"].split("/")[0]
        _, _, px_ref = _safe_get_latest_price(coin); _sleep_brief()
        if not px_ref:
            fills.append({**a, "error": "No price reference"}); continue

        price_prec = rules.get(a["pair"], {}).get("price_prec", 2)
        px_est = _estimate_market_fill_price(px_ref, a["side"])
        eff_px = _effective_price_with_fee(px_est, a["side"])
        if eff_px * a["qty"] < MIN_NOTIONAL_USD:  # dust guard with fees
            continue

        r = _place_market(a["pair"], a["side"], a["qty"]) if USE_MARKET_ORDERS \
            else _place_limit_at_best(a["pair"], a["side"], a["qty"], price_prec)
        _sleep_brief()

        if r and r.get("Success"):
            fills.append({"pair": a["pair"], "side": a["side"], "qty": a["qty"], "px_est": px_est})
        else:
            fills.append({"pair": a["pair"], "side": a["side"], "qty": a["qty"], "error": r})

    return {"Success": True, "Signals": signals, "Ref": ref, "ActionsPlanned": actions, "Fills": fills}

# ---------- Orchestrator ----------
def _batches_from_assets(all_assets, batch_size):
    return [all_assets[i:i+batch_size] for i in range(0, len(all_assets), batch_size)]

def _next_run_time(after_sec=RUN_OFFSET_SEC):
    """
    Next time we *should attempt* a run: next candle close + RUN_OFFSET_SEC.
    Loop will check more often (POLLING_SEC), but this helps target the window.
    """
    now = int(time.time())
    next_close = _ceiling_to_interval(now, _INTERVAL_SECONDS)
    return next_close + after_sec

def run_once_for_batch(batch_assets, state, batch_index):
    srv = _with_backoff(check_server_time); _sleep_brief()
    ex = _with_backoff(get_exchange_info); _sleep_brief()
    if not ex or ex.get("IsRunning") is not True:
        return {"Success": False, "ErrMsg": "Exchange not running", "batch_index": batch_index}

    if not should_run_this_cycle(batch_assets, state):
        return {
            "Success": True, "Skipped": True, "Reason": "No new candle for batch",
            "batch_index": batch_index, "assets": batch_assets,
            "ts_utc_ms": _now_utc_ms(), "interval": INTERVAL
        }

    pairs = _pairify(batch_assets, QUOTE)
    res = run_sma_crossover_once_batch(batch_assets, pairs)

    return {
        "ts_utc_ms": _now_utc_ms(),
        "interval": INTERVAL,
        "batch_index": batch_index,
        "assets": batch_assets,
        "signals": res.get("Signals", {}),
        "actions": res.get("ActionsPlanned", []),
        "fills": res.get("Fills", []),
        "skipped": False,
        "Success": res.get("Success", True),
        "ErrMsg": res.get("ErrMsg", "")
    }

# ---------- Graceful shutdown ----------
def _handle_sig(signum, frame):
    global _shutdown
    _shutdown = True
    _log(f"Received signal {signum}. Shutting down after this loop...")

signal.signal(signal.SIGINT, _handle_sig)
signal.signal(signal.SIGTERM, _handle_sig)

# ---------- Main continuous loop ----------
def main():
    if len(ALL_ASSETS) == 0:
        raise ValueError("ALL_ASSETS is empty. Populate your 36-coin list.")

    state = _load_state()
    batches = _batches_from_assets(ALL_ASSETS, BATCH_SIZE)
    num_batches = len(batches)
    batch_index = int(state.get("batch_index", 0)) % num_batches
    no_assets = len(ALL_ASSETS)
    
    _log(f"Runner start | interval={INTERVAL} | ALL_ASSETS = {no_assets} |batches={num_batches} x {BATCH_SIZE} | offset={RUN_OFFSET_SEC}s")
    next_target = _next_run_time()

    while not _shutdown:
        now = time.time()
        if now >= next_target:
            batch_assets = batches[batch_index]
            _log(f"Running batch {batch_index}/{num_batches-1}: {batch_assets}")
            try:
                out = run_once_for_batch(batch_assets, state, batch_index)
                _log("Result:", out)
            except Exception as e:
                _log("ERROR during batch run:", e)
                out = {"Success": False, "ErrMsg": str(e)}

            batch_index = (batch_index + 1) % num_batches

            state["batch_index"] = batch_index
            _save_state(state)

            next_target = _next_run_time()

        time.sleep(POLLING_SEC)

    _log("Runner stopped.")

if __name__ == "__main__":
    main()
