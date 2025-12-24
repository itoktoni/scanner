# engine/generic_json_strategy.py
import os
import json
import pandas as pd
import numpy as np
import yfinance as yf

from .engine_common import add_common_indicators, add_fundamental_indicators

TECHNICAL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "technical")

# Cache for fundamental data to avoid repeated API calls
_fundamental_cache = {}
_fundamental_cache_timestamps = {}
import time

# Cache timeout (24 hours)
CACHE_TIMEOUT = 24 * 60 * 60

def load_all_json_strategies():
    """
    Load semua file *.json di folder technical.
    Dipanggil setiap request supaya perubahan JSON langsung terbaca.
    """
    configs = {}
    for fname in os.listdir(TECHNICAL_DIR):
        if fname.lower().endswith(".json"):
            path = os.path.join(TECHNICAL_DIR, fname)
            try:
                with open(path, "r") as f:
                    content = f.read()
                if not content.strip():
                    print(f"Warning: Skipping empty file {fname}")
                    continue
                cfg = json.loads(content)
                name = cfg.get("name") or os.path.splitext(fname)[0]
                configs[name] = cfg
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON in {fname}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error loading {fname}: {e}")
                continue
    return configs


def get_fundamental_data(ticker):
    """
    Get fundamental data for a ticker, with caching to avoid repeated API calls.
    """
    # Remove .JK suffix if present to get the base ticker symbol
    base_ticker = ticker.replace(".JK", "") if ticker.endswith(".JK") else ticker

    # Check cache first with timestamp
    current_time = time.time()
    if base_ticker in _fundamental_cache:
        # Check if cache is still valid (less than 24 hours old)
        if base_ticker in _fundamental_cache_timestamps:
            cache_time = _fundamental_cache_timestamps[base_ticker]
            if current_time - cache_time < CACHE_TIMEOUT:
                return _fundamental_cache[base_ticker]

    try:
        # Get fundamental data from yfinance
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info

        # Check if info is None or not a dict
        if info is None:
            print(f"Warning: No fundamental data available for {ticker} (None)")
            _fundamental_cache[base_ticker] = {}
            _fundamental_cache_timestamps[base_ticker] = current_time
            return {}
        elif not isinstance(info, dict):
            print(f"Warning: Invalid fundamental data format for {ticker} (type: {type(info)})")
            _fundamental_cache[base_ticker] = {}
            _fundamental_cache_timestamps[base_ticker] = current_time
            return {}

        # Cache the data with timestamp
        _fundamental_cache[base_ticker] = info
        _fundamental_cache_timestamps[base_ticker] = current_time
        return info
    except TypeError as e:
        if "argument of type 'NoneType' is not iterable" in str(e):
            print(f"Warning: yfinance internal error for {ticker} (NoneType iteration)")
        else:
            print(f"Error fetching fundamental data for {ticker}: {e}")
        # Return empty dict if there's an error
        _fundamental_cache[base_ticker] = {}
        _fundamental_cache_timestamps[base_ticker] = current_time
        return {}
    except Exception as e:
        print(f"Error fetching fundamental data for {ticker}: {e}")
        # Return empty dict if there's an error
        _fundamental_cache[base_ticker] = {}
        _fundamental_cache_timestamps[base_ticker] = current_time
        return {}


def calc_indicators(df: pd.DataFrame, ticker=None, required_indicators=None) -> pd.DataFrame:
    df = add_common_indicators(df, required_indicators)

    # If ticker is provided, add fundamental indicators
    if ticker:
        fundamental_data = get_fundamental_data(ticker)
        df = add_fundamental_indicators(df, fundamental_data)

    return df


def build_env(df: pd.DataFrame) -> dict:
    """
    Build environment dict untuk eval:
    - Semua kolom df dengan nama apa adanya (MA20, RSI14, dll).
    - Juga versi UPPERCASE sebagai jaga-jaga.
    """
    env = {}
    for col in df.columns:
        env[col] = df[col]
        env[col.upper()] = df[col]
    return env


def extract_required_indicators(rules: list[str]) -> set:
    """
    Extract required indicators from strategy rules.
    """
    indicators = set()
    if not rules:
        return indicators

    # Common indicator patterns
    common_indicators = [
        "MA", "EMA", "VMA", "ATR", "RSI", "MACD", "STOCH", "BB",
        "VWAP", "VWMA", "PERCENT_SPIKE", "VOLUME_SPIKE", "RVOL",
        "THREE_RED_CANDLES", "THREE_GREEN_CANDLES", "DOJI", "HAMMER",
        "HANGING_MAN", "SHOOTING_STAR", "MORNING_STAR", "EVENING_STAR",
        "HH", "LL", "FIB", "SUPPORT", "RESISTANCE", "TRADE_FREQUENCY",
        "HIGH_VOL_DAYS",
        # Fundamental indicators
        "TRAILING_PE", "FORWARD_PE", "PEG_RATIO", "PRICE_TO_BOOK",
        "PRICE_TO_SALES", "PRICE_TO_CF", "RETURN_ON_EQUITY", "RETURN_ON_ASSETS",
        "PROFIT_MARGINS", "GROSS_MARGINS", "OPERATING_MARGINS", "DEBT_TO_EQUITY",
        "CURRENT_RATIO", "QUICK_RATIO", "TOTAL_CASH_PS", "DIVIDEND_YIELD",
        "PAYOUT_RATIO", "DIVIDEND_RATE", "REVENUE_GROWTH", "EARNINGS_GROWTH",
        "REVENUE_PS", "OPERATING_CF", "FREE_CASHFLOW"
    ]

    for rule in rules:
        for indicator in common_indicators:
            if indicator in rule:
                # Extract full indicator name (e.g., MA20, RSI14, TRAILING_PE, etc.)
                import re
                # For technical indicators, match with optional digits
                if indicator in ["MA", "EMA", "VMA", "ATR", "RSI", "MACD", "STOCH", "BB",
                                "VWAP", "VWMA", "PERCENT_SPIKE", "VOLUME_SPIKE", "RVOL",
                                "THREE_RED_CANDLES", "THREE_GREEN_CANDLES", "DOJI", "HAMMER",
                                "HANGING_MAN", "SHOOTING_STAR", "MORNING_STAR", "EVENING_STAR",
                                "HH", "LL", "FIB", "SUPPORT", "RESISTANCE", "TRADE_FREQUENCY",
                                "HIGH_VOL_DAYS"]:
                    pattern = rf'\b{indicator}\d*'  # Match indicator followed by optional digits
                else:
                    # For fundamental indicators, match exact name
                    pattern = rf'\b{indicator}\b'
                matches = re.findall(pattern, rule)
                indicators.update(matches)

    return indicators


def build_entry_mask(df: pd.DataFrame, rules: list[str]) -> pd.Series:
    """
    Terima list rules string, misalnya:
      "CLOSE > MA20", "RSI14 >= 45", "VOLUME >= VMA20"
    Kembalikan mask boolean (AND semua rule).
    """
    if not rules:
        return pd.Series(False, index=df.index)

    # Extract required indicators to optimize calculation
    required_indicators = extract_required_indicators(rules)
    df = calc_indicators(df.copy(), required_indicators=required_indicators)
    env = build_env(df)
    mask = pd.Series(True, index=df.index)

    for rule in rules:
        cond = eval(rule, {}, env)
        mask = mask & cond

    return mask


def get_entry_sl_tp_row(df: pd.DataFrame, config: dict, ticker=None):
    """
    SCAN: cek hanya bar terakhir.
    - ENTRY: pakai config["entry"]
    - TP/SL: pakai formula di config["tp"][0] dan config["sl"][0]
    """
    df = calc_indicators(df.copy(), ticker)
    if df.empty:
        return None

    rules = config.get("entry", [])
    if not rules:
        return None

    env_all = build_env(df)
    mask = pd.Series(True, index=df.index)
    for rule in rules:
        cond = eval(rule, {}, env_all)
        mask = mask & cond

    if not bool(mask.iloc[-1]):
        return None

    last_idx = df.index[-1]
    row = df.iloc[-1]

    entry_price = float(row["CLOSE"])
    atr = float(row.get("ATR14", np.nan))
    if np.isnan(atr):
        return None

    # Environment untuk TP/SL
    env = {
        "PRICE": entry_price,
        "ATR14": atr
    }
    for col in df.columns:
        env[col] = row[col]
        env[col.upper()] = row[col]

    tp_rules = config.get("tp", [])
    sl_rules = config.get("sl", [])

    if tp_rules:
        tp = float(eval(tp_rules[0], {}, env))
    else:
        tp = entry_price + atr * 3
    if sl_rules:
        sl = float(eval(sl_rules[0], {}, env))
    else:
        sl = entry_price - atr * 2

    return {
        "entry_date": last_idx,
        "entry_price": entry_price,
        "stop_loss": sl,
        "take_profit": tp,
    }


def backtest(df: pd.DataFrame, amount: float, config: dict,
             start_date=None, end_date=None, ticker=None):
    """
    BACKTEST generic:
    - ENTRY: pakai config["entry"].
    - TP/SL: pakai config["tp"][0] dan config["sl"][0], dieval di tiap entry bar.
    - max_hold_days: pakai config["max_hold_days"] (default 3).
    - Supports average down/up, partial take profit, and dynamic position sizing features.
    """
    df = calc_indicators(df.copy(), ticker)
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

    if df.empty:
        return [], {
            "total_trades": 0,
            "winrate": 0.0,
            "avg_holding_days": 0.0,
            "net_profit": 0.0,
            "final_equity": float(amount),
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "risk_reward": None,
            "best_roi_pct": 0.0,
            "exit_reason_count": {}
        }

    rules = config.get("entry", [])
    env_all = build_env(df)
    mask_entry = pd.Series(True, index=df.index)
    for rule in rules:
        cond = eval(rule, {}, env_all)
        mask_entry = mask_entry & cond

    max_hold_days = config.get("max_hold_days", 3)
    avg_down = config.get("avg_down", None)  # Threshold for averaging down
    # Handle multiple averaging down thresholds
    if isinstance(avg_down, list):
        # Multiple thresholds - will be handled as multiple averaging opportunities
        pass

    avg_up = config.get("avg_up", None)      # Threshold for averaging up
    # Handle multiple averaging up thresholds
    if isinstance(avg_up, list):
        # Multiple thresholds - will be handled as multiple averaging opportunities
        pass
    partial_tp_levels = config.get("partial_tp", [])             # Partial take profit levels
    partial_tp_ratios = config.get("partial_tp_ratios", [])      # Ratios for partial take profit
    # Handle take profit logic: if only one TP level, use it as full TP
    # If multiple TP levels, treat them as partial TP
    if len(config.get("tp", [])) == 1 and not partial_tp_levels:
        # Use single TP as full take profit
        pass
    elif len(config.get("tp", [])) > 1 and not partial_tp_levels:
        # Convert multiple TP levels to partial TP with equal distribution
        partial_tp_levels = config.get("tp", [])
        equal_ratio = 1.0 / len(partial_tp_levels)
        partial_tp_ratios = [equal_ratio] * len(partial_tp_levels)
        # Clear tp array since we're using partial TP
        tp = []
    elif partial_tp_levels and not partial_tp_ratios:
        # If partial TP levels are specified but no ratios, distribute equally
        equal_ratio = 1.0 / len(partial_tp_levels)
        partial_tp_ratios = [equal_ratio] * len(partial_tp_levels)
    amount_expr = config.get("amount", None)     # Dynamic position sizing expression

    trades = []
    in_position = False
    current_trade = None

    # Check for trailing stop configuration
    trailing_stop_config = config.get("ts", None)
    avg_down_count = 0                                   # Track averaging down次数
    avg_up_count = 0                                     # Track averaging up次数

    for i in range(len(df)):
        idx = df.index[i]
        row = df.iloc[i]

        if not in_position:
            if mask_entry.iloc[i]:
                entry_price = row["CLOSE"]
                atr = row.get("ATR14", np.nan)
                if np.isnan(atr):
                    continue

                env = {
                    "PRICE": float(entry_price),
                    "ATR14": float(atr),
                }
                for col in df.columns:
                    env[col] = row[col]
                    env[col.upper()] = row[col]

                tp_rules = config.get("tp", [])
                sl_rules = config.get("sl", [])

                if tp_rules:
                    tp = float(eval(tp_rules[0], {}, env))
                else:
                    tp = float(entry_price + atr * 3)

                # Handle stop loss - support for multiple levels
                sl_levels = sl_rules if isinstance(sl_rules, list) else [sl_rules] if sl_rules else []
                sl_value = None
                if sl_levels:
                    # Evaluate the first stop loss expression
                    env_sl = {
                        "PRICE": float(entry_price),
                        "ATR14": float(atr),
                        "ENTRY_PRICE": float(entry_price),
                    }
                    for col in df.columns:
                        env_sl[col] = row[col]
                        env_sl[col.upper()] = row[col]

                    sl_value = float(eval(sl_levels[0], {}, env_sl))
                else:
                    sl_value = float(entry_price - atr * 2)

                # Calculate position size - either fixed amount or dynamic
                if amount_expr:
                    # Evaluate dynamic amount expression
                    env = {
                        "PRICE": float(entry_price),
                        "ATR14": float(atr),
                        "FIXED_AMOUNT": float(amount),
                        "RSI14": float(row.get("RSI14", 50)),
                        "VOLUME": float(row.get("VOLUME", 0)),
                        "VMA20": float(row.get("VMA20", 0)),
                    }
                    for col in df.columns:
                        env[col] = row[col]
                        env[col.upper()] = row[col]

                    try:
                        dynamic_amount = float(eval(amount_expr, {}, env))
                        qty = dynamic_amount // entry_price
                    except:
                        # Fallback to fixed amount if dynamic calculation fails
                        qty = amount // entry_price
                else:
                    qty = amount // entry_price

                if qty <= 0:
                    continue

                in_position = True
                current_trade = {
                    "entry_date": idx,
                    "entry_price": float(entry_price),
                    "stop_loss": float(sl),
                    "take_profit": float(tp),
                    "qty": float(qty),
                    "bars_held": 0,
                    "initial_qty": float(qty),              # Store initial quantity
                    "avg_down_count": 0,                    # Track averaging down次数
                    "avg_up_count": 0,                      # Track averaging up次数
                    "partial_tp_executed": [False] * len(partial_tp_levels) if partial_tp_levels else [],  # Track partial TP execution
                    "highest_price": float(entry_price),     # Track highest price for trailing stop
                    "trailing_stop_level": None,            # Current trailing stop level
                }

                # Initialize trailing stop if configured
                if trailing_stop_config:
                    # Parse trailing stop configuration (e.g., "2%" or "0.02")
                    if isinstance(trailing_stop_config, str) and trailing_stop_config.endswith('%'):
                        ts_percent = float(trailing_stop_config[:-1]) / 100
                    else:
                        ts_percent = float(trailing_stop_config) if trailing_stop_config else 0.02

                    current_trade["trailing_stop_percent"] = ts_percent
                    current_trade["trailing_stop_level"] = float(entry_price * (1 - ts_percent))
        else:
            high = row["High"]
            low = row["Low"]
            current_trade["bars_held"] += 1

            # Update trailing stop level if configured
            if current_trade.get("trailing_stop_level") is not None:
                # Update highest price seen since entry
                current_trade["highest_price"] = max(current_trade["highest_price"], float(row["HIGH"]))

                # Update trailing stop level based on highest price
                ts_percent = current_trade["trailing_stop_percent"]
                new_trailing_level = current_trade["highest_price"] * (1 - ts_percent)

                # Only move trailing stop up, never down
                if new_trailing_level > current_trade["trailing_stop_level"]:
                    current_trade["trailing_stop_level"] = new_trailing_level

            exit_price = None
            exit_reason = None
            exit_qty = current_trade["qty"]  # Default to exiting all positions

            # Check for partial take profit levels
            if partial_tp_levels and partial_tp_ratios:
                for i, (tp_level, tp_ratio) in enumerate(zip(partial_tp_levels, partial_tp_ratios)):
                    # Evaluate the TP level expression
                    env = {
                        "PRICE": float(row["CLOSE"]),
                        "ATR14": float(row.get("ATR14", np.nan)),
                    }
                    for col in df.columns:
                        env[col] = row[col]
                        env[col.upper()] = row[col]

                    actual_tp_level = float(eval(tp_level, {}, env))

                    # If high crosses this partial TP level and it hasn't been executed yet
                    if high >= actual_tp_level and not current_trade["partial_tp_executed"][i]:
                        # Execute partial take profit
                        partial_qty = current_trade["initial_qty"] * tp_ratio
                        # Reduce position size
                        current_trade["qty"] -= partial_qty
                        current_trade["partial_tp_executed"][i] = True

                        # Record partial exit
                        partial_pnl = (actual_tp_level - current_trade["entry_price"]) * partial_qty
                        partial_roi = (actual_tp_level / current_trade["entry_price"] - 1) * 100

                        trades.append({
                            "entry_date": current_trade["entry_date"],
                            "close_date": idx,
                            "entry_price": current_trade["entry_price"],
                            "exit_price": float(actual_tp_level),
                            "pnl": float(partial_pnl),
                            "roi_pct": float(partial_roi),
                            "hold_period": int((idx - current_trade["entry_date"]).days),
                            "exit_reason": f"Partial TP {i+1}",
                            "stop_loss": current_trade["stop_loss"],
                            "take_profit": current_trade["take_profit"],
                        })

                        # If all shares are sold, exit position
                        if current_trade["qty"] <= 0:
                            in_position = False
                            current_trade = None
                            break

                        # Continue to check other exit conditions

            # Check for averaging down opportunity
            avg_down_thresholds = avg_down if isinstance(avg_down, list) else [avg_down] if avg_down else []

            for i, threshold in enumerate(avg_down_thresholds):
                if threshold and not exit_price and current_trade["avg_down_count"] < len(avg_down_thresholds):  # Allow multiple averaging downs
                    # Evaluate the averaging down threshold expression
                    env = {
                        "PRICE": float(row["CLOSE"]),
                        "ATR14": float(row.get("ATR14", np.nan)),
                    }
                    for col in df.columns:
                        env[col] = row[col]
                        env[col.upper()] = row[col]

                    avg_down_price = float(eval(threshold, {}, env))

                    # If current price is below the averaging down threshold
                    if row["CLOSE"] <= avg_down_price:
                        # Add to position (use dynamic amount if specified, otherwise double initial position)
                        if amount_expr:
                            # Evaluate dynamic amount for averaging
                            env_dyn = {
                                "PRICE": float(row["CLOSE"]),
                                "ATR14": float(row.get("ATR14", np.nan)),
                                "FIXED_AMOUNT": float(amount),
                                "RSI14": float(row.get("RSI14", 50)),
                                "VOLUME": float(row.get("VOLUME", 0)),
                                "VMA20": float(row.get("VMA20", 0)),
                            }
                            for col in df.columns:
                                env_dyn[col] = row[col]
                                env_dyn[col.upper()] = row[col]

                            try:
                                dynamic_amount = float(eval(amount_expr, {}, env_dyn))
                                additional_qty = dynamic_amount // row["CLOSE"]
                            except:
                                # Fallback to initial position size
                                additional_qty = current_trade["initial_qty"]
                        else:
                            additional_qty = current_trade["initial_qty"]

                        new_total_qty = current_trade["qty"] + additional_qty
                        new_entry_price = ((current_trade["entry_price"] * current_trade["qty"]) +
                                          (row["CLOSE"] * additional_qty)) / new_total_qty

                        current_trade["qty"] = new_total_qty
                        current_trade["entry_price"] = new_entry_price
                        current_trade["avg_down_count"] += 1

                        # Adjust stop loss (tighten it slightly)
                        current_trade["stop_loss"] = new_entry_price - (float(row.get("ATR14", np.nan)) * 1.5)

                        # Continue to next iteration without exiting
                        continue

            # Check for averaging up opportunity
            avg_up_thresholds = avg_up if isinstance(avg_up, list) else [avg_up] if avg_up else []

            for i, threshold in enumerate(avg_up_thresholds):
                if threshold and not exit_price and current_trade["avg_up_count"] < len(avg_up_thresholds):  # Allow multiple averaging ups
                    # Evaluate the averaging up threshold expression
                    env = {
                        "PRICE": float(row["CLOSE"]),
                        "ATR14": float(row.get("ATR14", np.nan)),
                    }
                    for col in df.columns:
                        env[col] = row[col]
                        env[col.upper()] = row[col]

                    avg_up_price = float(eval(threshold, {}, env))

                    # If current price is above the averaging up threshold
                    if row["CLOSE"] >= avg_up_price:
                        # Add to position (use dynamic amount if specified, otherwise double initial position)
                        if amount_expr:
                            # Evaluate dynamic amount for averaging
                            env_dyn = {
                                "PRICE": float(row["CLOSE"]),
                                "ATR14": float(row.get("ATR14", np.nan)),
                                "FIXED_AMOUNT": float(amount),
                                "RSI14": float(row.get("RSI14", 50)),
                                "VOLUME": float(row.get("VOLUME", 0)),
                                "VMA20": float(row.get("VMA20", 0)),
                            }
                            for col in df.columns:
                                env_dyn[col] = row[col]
                                env_dyn[col.upper()] = row[col]

                            try:
                                dynamic_amount = float(eval(amount_expr, {}, env_dyn))
                                additional_qty = dynamic_amount // row["CLOSE"]
                            except:
                                # Fallback to initial position size
                                additional_qty = current_trade["initial_qty"]
                        else:
                            additional_qty = current_trade["initial_qty"]

                        new_total_qty = current_trade["qty"] + additional_qty
                        new_entry_price = ((current_trade["entry_price"] * current_trade["qty"]) +
                                          (row["CLOSE"] * additional_qty)) / new_total_qty

                        current_trade["qty"] = new_total_qty
                        current_trade["entry_price"] = new_entry_price
                        current_trade["avg_up_count"] += 1

                        # Continue to next iteration without exiting
                        continue

            # Check standard exit conditions if no partial TP executed
            if not exit_price and not partial_tp_levels:
                # Check trailing stop first
                trailing_stop_triggered = False
                if current_trade.get("trailing_stop_level") is not None:
                    # If current price is below trailing stop level
                    if low <= current_trade["trailing_stop_level"]:
                        exit_price = current_trade["trailing_stop_level"]
                        exit_reason = "Trailing Stop Hit"
                        trailing_stop_triggered = True

                # If trailing stop not triggered, check regular stop loss levels
                if not trailing_stop_triggered:
                    # Handle multiple stop loss levels
                    sl_rules = config.get("sl", [])
                    sl_levels = sl_rules if isinstance(sl_rules, list) else [sl_rules] if sl_rules else []
                    sl_hit = False

                    # Check if any stop loss level is hit
                    for i, sl_expr in enumerate(sl_levels):
                        if sl_expr:
                            # Evaluate the stop loss expression
                            env = {
                                "PRICE": float(row["CLOSE"]),
                                "ATR14": float(row.get("ATR14", np.nan)),
                                "ENTRY_PRICE": float(current_trade["entry_price"]),
                            }
                            for col in df.columns:
                                env[col] = row[col]
                                env[col.upper()] = row[col]

                            sl_level = float(eval(sl_expr, {}, env))

                            # If low crosses this stop loss level
                            if low <= sl_level:
                                exit_price = sl_level
                                exit_reason = f"SL Hit (Level {i+1})"
                                sl_hit = True
                                break

                    if not sl_hit:
                        if high >= current_trade["take_profit"]:
                            exit_price = current_trade["take_profit"]
                            exit_reason = "TP Hit"
                        elif current_trade["bars_held"] >= max_hold_days:
                            exit_price = row["Close"]
                            exit_reason = "MaxHold Exit"

            if exit_price is not None:
                pnl = (exit_price - current_trade["entry_price"]) * current_trade["qty"]
                roi_pct = (exit_price / current_trade["entry_price"] - 1) * 100

                trades.append({
                    "entry_date": current_trade["entry_date"],
                    "close_date": idx,
                    "entry_price": current_trade["entry_price"],
                    "exit_price": float(exit_price),
                    "pnl": float(pnl),
                    "roi_pct": float(roi_pct),
                    "hold_period": int((idx - current_trade["entry_date"]).days),
                    "exit_reason": exit_reason,
                    "stop_loss": current_trade["stop_loss"],
                    "take_profit": current_trade["take_profit"],
                })
                in_position = False
                current_trade = None

    total_trades = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    net_profit = float(sum(t["pnl"] for t in trades))
    final_equity = float(amount + net_profit)

    winrate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
    avg_hold = float(np.mean([t["hold_period"] for t in trades])) if trades else 0.0
    avg_win_pct = float(np.mean([t["roi_pct"] for t in wins])) if wins else 0.0
    avg_loss_pct = float(np.mean([t["roi_pct"] for t in losses])) if losses else 0.0
    best_roi = float(max([t["roi_pct"] for t in trades])) if trades else 0.0
    rr = float(abs(avg_win_pct / avg_loss_pct)) if avg_loss_pct < 0 else None

    exit_reason_count = {}
    for t in trades:
        exit_reason_count[t["exit_reason"]] = exit_reason_count.get(t["exit_reason"], 0) + 1

    summary = {
        "total_trades": total_trades,
        "winrate": winrate,
        "avg_holding_days": avg_hold,
        "net_profit": net_profit,
        "final_equity": final_equity,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "risk_reward": rr,
        "best_roi_pct": best_roi,
        "exit_reason_count": exit_reason_count,
    }
    return trades, summary