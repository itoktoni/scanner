# technical/generic_json_strategy.py
import os
import json
import pandas as pd
import numpy as np

from .engine_common import add_common_indicators

TECHNICAL_DIR = os.path.dirname(__file__)


def load_all_json_strategies():
    """
    Load semua file *.json di folder technical.
    Dipanggil setiap request supaya perubahan JSON langsung terbaca.
    """
    configs = {}
    for fname in os.listdir(TECHNICAL_DIR):
        if fname.lower().endswith(".json"):
            path = os.path.join(TECHNICAL_DIR, fname)
            with open(path, "r") as f:
                cfg = json.load(f)
            name = cfg.get("name") or os.path.splitext(fname)[0]
            configs[name] = cfg
    return configs


def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    return add_common_indicators(df)


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


def build_entry_mask(df: pd.DataFrame, rules: list[str]) -> pd.Series:
    """
    Terima list rules string, misalnya:
      "CLOSE > MA20", "RSI14 >= 45", "VOLUME >= VMA20"
    Kembalikan mask boolean (AND semua rule).
    """
    if not rules:
        return pd.Series(False, index=df.index)

    df = calc_indicators(df.copy())
    env = build_env(df)
    mask = pd.Series(True, index=df.index)

    for rule in rules:
        cond = eval(rule, {}, env)
        mask = mask & cond

    return mask


def get_entry_sl_tp_row(df: pd.DataFrame, config: dict):
    """
    SCAN: cek hanya bar terakhir.
    - ENTRY: pakai config["entry"]
    - TP/SL: pakai formula di config["tp"][0] dan config["sl"][0]
    """
    df = calc_indicators(df.copy())
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
             start_date=None, end_date=None):
    """
    BACKTEST generic:
    - ENTRY: pakai config["entry"].
    - TP/SL: pakai config["tp"][0] dan config["sl"][0], dieval di tiap entry bar.
    - max_hold_days: pakai config["max_hold_days"] (default 3).
    """
    df = calc_indicators(df.copy())
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

    trades = []
    in_position = False
    current_trade = None

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

                if sl_rules:
                    sl = float(eval(sl_rules[0], {}, env))
                else:
                    sl = float(entry_price - atr * 2)

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
                }
        else:
            high = row["High"]
            low = row["Low"]
            current_trade["bars_held"] += 1

            exit_price = None
            exit_reason = None

            if low <= current_trade["stop_loss"]:
                exit_price = current_trade["stop_loss"]
                exit_reason = "SL Hit"
            elif high >= current_trade["take_profit"]:
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
