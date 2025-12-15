# technical/engine_common.py
import pandas as pd
import numpy as np

def add_common_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan indikator umum:
    - OPEN, HIGH, LOW, CLOSE, VOLUME
    - MA5, MA10, MA20, MA50, MA100, MA200
    - VMA5, VMA10, VMA20
    - RSI14
    - ATR14
    - HH3, LL3, HH20, LL20
    """
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Harga dasar (UPPER)
    df["OPEN"] = df["Open"]
    df["HIGH"] = df["High"]
    df["LOW"] = df["Low"]
    df["CLOSE"] = df["Close"]
    df["VOLUME"] = df["Volume"]

    # MA Harga
    for win in [5, 10, 20, 50, 100, 200]:
        df[f"MA{win}"] = close.rolling(win).mean()

    # Volume MA
    for win in [5, 10, 20]:
        df[f"VMA{win}"] = volume.rolling(win).mean()

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.rolling(window=14).mean()
    roll_down = loss.rolling(window=14).mean()
    rs = roll_up / roll_down
    df["RSI14"] = 100 - (100 / (1 + rs))

    # ATR 14
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    # High/Low rolling
    df["HH3"] = high.rolling(3).max()
    df["LL3"] = low.rolling(3).min()
    df["HH20"] = high.rolling(20).max()
    df["LL20"] = low.rolling(20).min()

    # Support / Resistance sederhana (berbasis 20 hari)
    df["SUPPORT1"] = df["LL20"]
    df["RESIST1"] = df["HH20"]

    return df
