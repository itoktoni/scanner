# technical/trend.py
import pandas as pd
import numpy as np

NAME = "Trend"
DESCRIPTION = """
Sinyal entry ketika:
- price > 70
- price >= MA20
- price >= MA200
- MA20 >= MA200
- volume > MA(volume,20) * 0.5
- volume <= MA(volume,20) * 2
- RSI 14 antara 50 s/d 90
Stop loss = price - ATR(14) * 2
Take profit = price + ATR(14) * 3
""".strip()


def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Moving average harga & volume
    df["MA20"] = close.rolling(20).mean()
    df["MA200"] = close.rolling(200).mean()
    df["VOL_MA20"] = volume.rolling(20).mean()

    # === RSI 14 (aman 1D) ===
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    roll_up = gain.rolling(window=14).mean()
    roll_down = loss.rolling(window=14).mean()

    rs = roll_up / roll_down
    df["RSI14"] = 100 - (100 / (1 + rs))

    # === ATR 14 ===
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    return df


def entry_filter(df: pd.DataFrame) -> pd.Series:
    """
    Mask True/False semua bar yang memenuhi kriteria entry.
    Dipakai untuk BACKTEST (historical).
    """
    cond = (
        (df["Close"] > 70) &
        (df["Close"] >= df["MA20"]) &
        (df["Close"] >= df["MA200"]) &
        (df["MA20"] >= df["MA200"]) &
        (df["Volume"] > df["VOL_MA20"] * 0.5) &
        (df["Volume"] <= df["VOL_MA20"] * 2) &
        (df["RSI14"] >= 50) &
        (df["RSI14"] <= 90)
    )
    return cond


def get_entry_sl_tp_row(df: pd.DataFrame):
    """
    SCAN: cek hanya bar TERAKHIR.
    Jika bar terakhir memenuhi kriteria entry, hitung SL/TP dan return dict.
    Jika tidak memenuhi, return None (ticker tidak muncul di scan).
    """
    df = calc_indicators(df.copy())
    if df.empty:
        return None

    last_idx = df.index[-1]
    row = df.iloc[-1]

    cond = (
        (row["Close"] > 70) and
        (row["Close"] >= row["MA20"]) and
        (row["Close"] >= row["MA200"]) and
        (row["MA20"] >= row["MA200"]) and
        (row["Volume"] > row["VOL_MA20"] * 0.5) and
        (row["Volume"] <= row["VOL_MA20"] * 2) and
        (row["RSI14"] >= 50) and
        (row["RSI14"] <= 90)
    )

    if not cond:
        return None

    entry_price = float(row["Close"])
    atr = float(row["ATR14"])
    if np.isnan(atr):
        return None

    sl = entry_price - atr * 2
    tp = entry_price + atr * 3

    return {
        "entry_date": last_idx,
        "entry_price": entry_price,
        "stop_loss": sl,
        "take_profit": tp,
    }


def backtest(df: pd.DataFrame, amount: float, start_date=None, end_date=None):
    """
    BACKTEST:
    - Entry: setiap bar yang memenuhi entry_filter (historical).
    - Exit: kena SL atau TP (cek high/low berikutnya).
    - Satu posisi jalan sekaligus per ticker.
    Return list semua trade + summary.
    """
    df = calc_indicators(df.copy())
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

    trades = []
    in_position = False
    current_trade = None

    mask_entry = entry_filter(df)

    for i in range(len(df)):
        idx = df.index[i]
        row = df.iloc[i]

        if not in_position:
            if mask_entry.iloc[i]:
                entry_price = row["Close"]
                atr = row["ATR14"]
                if np.isnan(atr):
                    continue

                sl = entry_price - atr * 2
                tp = entry_price + atr * 3
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
                }
        else:
            high = row["High"]
            low = row["Low"]

            exit_price = None
            exit_reason = None

            # Cek SL dulu, lalu TP
            if low <= current_trade["stop_loss"]:
                exit_price = current_trade["stop_loss"]
                exit_reason = "SL Hit"
            elif high >= current_trade["take_profit"]:
                exit_price = current_trade["take_profit"]
                exit_reason = "TP Hit"

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
