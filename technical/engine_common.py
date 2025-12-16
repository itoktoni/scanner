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
    - PREVIOUS_PRICE
    - 1 DAY PRICE RETURN %
    - 7 DAY PRICE RETURN %
    - 30 DAY PRICE RETURN %
    - 50 DAY PRICE RETURN %
    - 100 DAY PRICE RETURN %
    - FREQUENCY SPIKE (price INCREASE >= 10%) / 1 day
    - FIBONACCI RETRACEMENT LEVELS (23.6%, 38.2%, 50%, 61.8%)
    - MACD (12, 26, 9)
    - BOLLINGER BANDS (20, 2)
    - STOCHASTIC OSCILLATOR (14, 3)
    """
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Harga dasar (UPPER)
    # OPEN: Harga pembukaan
    # HIGH: Harga tertinggi
    # LOW: Harga terendah
    # CLOSE: Harga penutupan
    # VOLUME: Volume perdagangan
    df["OPEN"] = df["Open"]
    df["HIGH"] = df["High"]
    df["LOW"] = df["Low"]
    df["CLOSE"] = df["Close"]
    df["PRICE"] = df["Close"]
    df["VOLUME"] = df["Volume"]

    # Previous day's closing price
    # PREVIOUS_PRICE: Harga penutupan hari sebelumnya
    df["PREVIOUS_PRICE"] = close.shift(1)

    # Price return percentages for different periods
    # PRICE_RETURN_1D_PCT: Persentase perubahan harga dalam 1 hari
    # PRICE_RETURN_7D_PCT: Persentase perubahan harga dalam 7 hari
    # PRICE_RETURN_30D_PCT: Persentase perubahan harga dalam 30 hari
    # PRICE_RETURN_50D_PCT: Persentase perubahan harga dalam 50 hari
    # PRICE_RETURN_100D_PCT: Persentase perubahan harga dalam 100 hari
    df["PRICE_RETURN_1D_PCT"] = close.pct_change(1) * 100
    df["PRICE_RETURN_7D_PCT"] = close.pct_change(7) * 100
    df["PRICE_RETURN_30D_PCT"] = close.pct_change(30) * 100
    df["PRICE_RETURN_50D_PCT"] = close.pct_change(50) * 100
    df["PRICE_RETURN_100D_PCT"] = close.pct_change(100) * 100

    # Frequency spike: Count of times price increased >= 10% in 1 day over a rolling window
    # First calculate daily percentage change
    # FREQUENCY_SPIKE: Frekuensi kenaikan harga >= 10% dalam 1 hari selama periode 30 hari
    daily_pct_change = close.pct_change(1) * 100
    # Create boolean mask for spikes (>= 10% increase)
    spike_mask = daily_pct_change >= 10
    # Count frequency of spikes over a 30-day rolling window (can be adjusted)
    df["FREQUENCY_SPIKE"] = spike_mask.rolling(30).sum()

    # Fibonacci Retracement Levels (calculated over last 50 days)
    # Level-level retracement Fibonacci berdasarkan harga tertinggi dan terendah 50 hari terakhir
    # FIB_236: Level retracement 23.6%
    # FIB_382: Level retracement 38.2%
    # FIB_50: Level retracement 50%
    # FIB_618: Level retracement 61.8%
    # Calculate highest high and lowest low over the past 50 days
    highest_high = high.rolling(50).max()
    lowest_low = low.rolling(50).min()
    price_range = highest_high - lowest_low

    # Fibonacci levels
    df["FIB_236"] = highest_high - (price_range * 0.236)
    df["FIB_382"] = highest_high - (price_range * 0.382)
    df["FIB_50"] = highest_high - (price_range * 0.5)
    df["FIB_618"] = highest_high - (price_range * 0.618)

    # MACD (12, 26, 9)
    # Indikator yang menunjukkan momentum dengan menghitung selisih antara dua EMA
    # MACD_LINE: Garis MACD (selisih EMA 12 dan EMA 26)
    # MACD_SIGNAL: Garis sinyal (EMA 9 dari garis MACD)
    # MACD_HISTOGRAM: Histogram (selisih antara garis MACD dan garis sinyal)
    ema_12 = close.ewm(span=12).mean()
    ema_26 = close.ewm(span=26).mean()
    df["MACD_LINE"] = ema_12 - ema_26
    df["MACD_SIGNAL"] = df["MACD_LINE"].ewm(span=9).mean()
    df["MACD_HISTOGRAM"] = df["MACD_LINE"] - df["MACD_SIGNAL"]

    # Bollinger Bands (20, 2)
    # Menunjukkan volatilitas harga dengan pita atas dan bawah
    # BB_MIDDLE: Garis tengah (SMA 20)
    # BB_UPPER: Pita atas (SMA 20 + 2 * deviasi standar)
    # BB_LOWER: Pita bawah (SMA 20 - 2 * deviasi standar)
    # BB_WIDTH: Lebar pita Bollinger
    # BB_PERCENT_B: Posisi harga dalam pita Bollinger (%B)
    df["BB_MIDDLE"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_UPPER"] = df["BB_MIDDLE"] + (bb_std * 2)
    df["BB_LOWER"] = df["BB_MIDDLE"] - (bb_std * 2)
    # Bollinger Band Width
    df["BB_WIDTH"] = df["BB_UPPER"] - df["BB_LOWER"]
    # Bollinger Band %B
    df["BB_PERCENT_B"] = (close - df["BB_LOWER"]) / (df["BB_UPPER"] - df["BB_LOWER"])

    # Stochastic Oscillator (14, 3)
    # Indikator momentum yang membandingkan harga penutupan dengan range harga
    # STOCH_K: Garis %K (stochastic cepat)
    # STOCH_D: Garis %D (moving average 3 periode dari %K / stochastic lambat)
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    # Fast Stochastic
    df["STOCH_K"] = 100 * ((close - low_14) / (high_14 - low_14))
    # Slow Stochastic (3-period moving average of %K)
    df["STOCH_D"] = df["STOCH_K"].rolling(3).mean()

    # MA Harga
    # Moving averages untuk berbagai periode waktu
    # Digunakan untuk mengidentifikasi tren dan area support/resistance
    for win in [5, 10, 20, 50, 100, 200]:
        df[f"MA{win}"] = close.rolling(win).mean()

    # Volume MA
    # Moving average volume untuk mengidentifikasi aktivitas perdagangan
    for win in [5, 10, 20]:
        df[f"VMA{win}"] = volume.rolling(win).mean()

    # RSI 14
    # Relative Strength Index: Mengukur kekuatan tren dan kondisi overbought/oversold
    # Nilai 0-30: Oversold, 30-70: Netral, 70-100: Overbought
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.rolling(window=14).mean()
    roll_down = loss.rolling(window=14).mean()
    rs = roll_up / roll_down
    df["RSI14"] = 100 - (100 / (1 + rs))

    # ATR 14
    # Average True Range: Mengukur volatilitas pasar
    # Semakin tinggi nilai ATR, semakin besar volatilitas
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    # High/Low rolling
    # HH3: Harga tertinggi dalam 3 hari terakhir
    # LL3: Harga terendah dalam 3 hari terakhir
    # HH20: Harga tertinggi dalam 20 hari terakhir
    # LL20: Harga terendah dalam 20 hari terakhir
    df["HH3"] = high.rolling(3).max()
    df["LL3"] = low.rolling(3).min()
    df["HH20"] = high.rolling(20).max()
    df["LL20"] = low.rolling(20).min()

    # Support / Resistance yang lebih akurat (berbasis pengulangan harga terbanyak)
    # SUPPORT1: Level support berdasarkan harga terendah yang paling sering muncul dalam 20 hari
    # RESIST1: Level resistance berdasarkan harga tertinggi yang paling sering muncul dalam 20 hari
    # Metode ini mencari harga pembulatan atau level psikologis penting

    # Untuk support: cari harga terendah yang sering terjadi (cluster rendah)
    # Untuk resistance: cari harga tertinggi yang sering terjadi (cluster tinggi)

    # Pendekatan sederhana: gunakan moving average sebagai pendekatan support/resistance dinamis
    df["SUPPORT"] = df["MA20"] - (df["ATR14"] * 1.5)  # Support dinamis berdasarkan MA20 dan ATR
    df["RESISTANCE"] = df["MA20"] + (df["ATR14"] * 1.5)   # Resistance dinamis berdasarkan MA20 dan ATR

    return df