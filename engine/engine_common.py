# engine/engine_common.py
import pandas as pd
import numpy as np

def add_common_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menambahkan indikator teknikal umum ke dalam DataFrame untuk scanning/backtest.

    DAFTAR INDIKATOR (Nama Kolom):

    --- Harga & Volume Dasar ---
    - OPEN            : Harga pembukaan (Open)
    - HIGH            : Harga tertinggi (High)
    - LOW             : Harga terendah (Low)
    - CLOSE           : Harga penutupan (Close)
    - PRICE           : Alias untuk CLOSE
    - VOLUME          : Volume perdagangan

    --- Trend & Volume Average ---
    - MA5, MA10, MA20, MA50, MA100, MA200 :
      Simple Moving Average harga Close periode N.
    - VMA5, VMA10, VMA20 :
      Simple Moving Average Volume periode N.

    --- Momentum ---
    - RSI14           : Relative Strength Index (14).
    - MACD_LINE       : EMA12 - EMA26.
    - MACD_SIGNAL     : EMA9 dari MACD Line.
    - MACD_HISTOGRAM  : MACD Line - Signal Line.
    - STOCH_K         : Stochastic %K (Fast).
    - STOCH_D         : Stochastic %D (Slow / MA3 dari %K).

    --- Volatilitas & Bands ---
    - ATR14           : Average True Range (14). Ukuran volatilitas rata-rata.
    - BB_UPPER        : Bollinger Band Atas (MA20 + 2*StdDev).
    - BB_LOWER        : Bollinger Band Bawah (MA20 - 2*StdDev).
    - BB_MIDDLE       : Bollinger Band Tengah (MA20).
    - BB_WIDTH        : Lebar Band (Upper - Lower).
    - BB_PERCENT_B    : Posisi harga relatif dlm Band (0=Bawah, 1=Atas).

    --- Support & Resistance Dinamis ---
    - HH3, LL3        : Highest High / Lowest Low 3 hari terakhir.
    - HH20, LL20      : Highest High / Lowest Low 20 hari terakhir.
    - SUPPORT         : Support dinamis (MA20 - 1.5 * ATR14).
    - RESISTANCE      : Resistance dinamis (MA20 + 1.5 * ATR14).

    --- Return & Statistik ---
    - PREVIOUS_PRICE        : Harga Close kemarin.
    - PRICE_RETURN_1D_PCT   : % Return 1 hari.
    - PRICE_RETURN_7D_PCT   : % Return 7 hari.
    - SPIKE       : Jumlah hari dgn kenaikan >= 10% dlm 30 hari terakhir.

    --- Advanced ---
    - VWMA            : Volume Weighted Moving Average (20). Rata-rata harga berbobot volume.
    - VWAP            : Proxy untuk Daily Chart (di-set sama dengan VWMA).
    - FIB_236, FIB_382... : Level Retracement Fibonacci dari range High-Low 50 hari terakhir.
    """

    # 1. Copy data agar aman
    df = df.copy()
    if df.empty:
        return df

    # 2. Mapping Standard Columns (Huruf Besar)
    # Asumsi input yfinance punya kolom: Open, High, Low, Close, Volume
    df["OPEN"] = df["Open"]
    df["HIGH"] = df["High"]
    df["LOW"] = df["Low"]
    df["CLOSE"] = df["Close"]
    df["PRICE"] = df["Close"]   # Alias
    df["VOLUME"] = df["Volume"]

    # Variabel lokal untuk perhitungan
    close = df["CLOSE"]
    high = df["HIGH"]
    low = df["LOW"]
    volume = df["VOLUME"]

    # 3. Previous Price & Returns
    df["PREVIOUS_PRICE"] = close.shift(1)

    periods = [1, 7, 30, 50, 100]
    for p in periods:
        df[f"PRICE_RETURN_{p}D_PCT"] = close.pct_change(p) * 100

    # 4. Frequency Spike (Karakter Saham)
    # Hitung berapa kali naik >= 10% dalam 30 hari terakhir
    daily_pct = df["PRICE_RETURN_1D_PCT"]
    spike_mask = daily_pct >= 1
    df["SPIKE"] = spike_mask.rolling(30).sum()

    # 5. Moving Averages (Price & Volume)
    ma_windows = [5, 10, 20, 50, 100, 200]
    for w in ma_windows:
        df[f"MA{w}"] = close.rolling(w).mean()

    vma_windows = [5, 10, 20]
    for w in vma_windows:
        df[f"VMA{w}"] = volume.rolling(w).mean()

    # 6. Volatility (ATR & Bollinger Bands)
    # ATR 14
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    # Bollinger Bands (20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_MIDDLE"] = bb_mid
    df["BB_UPPER"] = bb_mid + (bb_std * 2)
    df["BB_LOWER"] = bb_mid - (bb_std * 2)
    df["BB_WIDTH"] = df["BB_UPPER"] - df["BB_LOWER"]
    # Handle division by zero
    df["BB_PERCENT_B"] = (close - df["BB_LOWER"]) / df["BB_WIDTH"].replace(0, np.nan)

    # 7. Momentum (RSI, MACD, Stochastic)
    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD_LINE"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD_LINE"].ewm(span=9, adjust=False).mean()
    df["MACD_HISTOGRAM"] = df["MACD_LINE"] - df["MACD_SIGNAL"]

    # Stochastic (14, 3)
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    stoch_k = 100 * ((close - low14) / (high14 - low14).replace(0, np.nan))
    df["STOCH_K"] = stoch_k
    df["STOCH_D"] = stoch_k.rolling(3).mean()

    # 8. Support / Resistance & High/Low Rolling
    for w in [3, 20]:
        df[f"HH{w}"] = high.rolling(w).max()
        df[f"LL{w}"] = low.rolling(w).min()

    # Fibonacci Retracement (50 Days Range)
    hh50 = high.rolling(50).max()
    ll50 = low.rolling(50).min()
    range50 = hh50 - ll50
    df["FIB_236"] = hh50 - (range50 * 0.236)
    df["FIB_382"] = hh50 - (range50 * 0.382)
    df["FIB_50"]  = hh50 - (range50 * 0.5)
    df["FIB_618"] = hh50 - (range50 * 0.618)

    # Dynamic Support/Resistance (ATR Based)
    # Support lantai yang naik turun ikut volatilitas
    df["SUPPORT"] = df["MA20"] - (df["ATR14"] * 1.5)
    df["RESISTANCE"] = df["MA20"] + (df["ATR14"] * 1.5)

    # 9. VWAP / VWMA
    # Typical Price
    tp = (high + low + close) / 3
    # VWMA (Rolling 20) -> Valid untuk chart Daily
    vp_sum = (tp * volume).rolling(20).sum()
    v_sum = volume.rolling(20).sum()
    df["VWMA"] = vp_sum / v_sum.replace(0, np.nan)

    # VWAP Proxy (disamakan dengan VWMA agar tidak misleading di chart daily)
    df["VWAP"] = df["VWMA"]

    # 10. Final Cleanup (Isi NaN dengan 0 agar engine eval aman)
    # Fill Forward dulu (untuk data yg bolong dikit), lalu Fill 0 (untuk awal data)
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)

    return df
