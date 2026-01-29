# engine/engine_common.py
import pandas as pd
import numpy as np

def add_common_indicators(df: pd.DataFrame, required_indicators=None, timeframe="1h") -> pd.DataFrame:
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
    - MA5             : Simple Moving Average harga Close periode 5.
    - MA10            : Simple Moving Average harga Close periode 10.
    - MA20            : Simple Moving Average harga Close periode 20.
    - MA50            : Simple Moving Average harga Close periode 50.
    - MA100           : Simple Moving Average harga Close periode 100.
    - MA200           : Simple Moving Average harga Close periode 200.
    - VMA5            : Simple Moving Average Volume periode 5.
    - VMA10           : Simple Moving Average Volume periode 10.
    - VMA20           : Simple Moving Average Volume periode 20.

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
    - HH3             : Highest High 3 hari terakhir.
    - LL3             : Lowest Low 3 hari terakhir.
    - HH20            : Highest High 20 hari terakhir.
    - LL20            : Lowest Low 20 hari terakhir.
    - SUPPORT         : Support dinamis (MA20 - 1.5 * ATR14).
    - RESISTANCE      : Resistance dinamis (MA20 + 1.5 * ATR14).

    --- Return & Statistik ---
    - PREVIOUS_PRICE        : Harga Close kemarin.
    - PRICE_RETURN_1D_PCT   : % Return 1 hari.
    - PRICE_RETURN_7D_PCT   : % Return 7 hari.
    - PRICE_RETURN_30D_PCT  : % Return 30 hari.
    - PRICE_RETURN_50D_PCT  : % Return 50 hari.
    - PRICE_RETURN_100D_PCT : % Return 100 hari.
    - SPIKE       : Jumlah hari dgn kenaikan >= 10% dlm 30 hari terakhir.

    --- Frekuensi Perdagangan ---
    - TRADE_FREQUENCY  : Frekuensi perdagangan berdasarkan volume relatif thd rata-rata.
                         Nilai > 100 berarti volume lebih tinggi dari rata-rata.
    - HIGH_VOL_DAYS    : Jumlah hari dengan volume tinggi (>150% rata-rata) dalam 30 hari terakhir.

    --- Advanced ---
    - VWMA            : Volume Weighted Moving Average (20). Rata-rata harga berbobot volume.
    - VWAP            : Proxy untuk Daily Chart (di-set sama dengan VWMA).
    - FIB_236         : Level Retracement Fibonacci 23.6% dari range High-Low 50 hari terakhir.
    - FIB_382         : Level Retracement Fibonacci 38.2% dari range High-Low 50 hari terakhir.
    - FIB_50          : Level Retracement Fibonacci 50.0% dari range High-Low 50 hari terakhir.
    - FIB_618         : Level Retracement Fibonacci 61.8% dari range High-Low 50 hari terakhir.
    - THREE_RED_CANDLES : Numeric indicator (1/0) untuk candle merah 3 hari yang lalu. 1 = candle merah, 0 = candle hijau.
    - THREE_GREEN_CANDLES : Numeric indicator (1/0) untuk 3 candle hijau berturut-turut (bullish pattern). 1 = pola ditemukan, 0 = tidak ditemukan.
    - DOJI : Numeric indicator (1/0) untuk pola Doji (open ≈ close). 1 = pola ditemukan, 0 = tidak ditemukan.
    - HAMMER : Numeric indicator (1/0) untuk pola Hammer (bullish reversal). 1 = pola ditemukan, 0 = tidak ditemukan.
    - HANGING_MAN : Numeric indicator (1/0) untuk pola Hanging Man (bearish reversal). 1 = pola ditemukan, 0 = tidak ditemukan.
    - SHOOTING_STAR : Numeric indicator (1/0) untuk pola Shooting Star (bearish reversal). 1 = pola ditemukan, 0 = tidak ditemukan.
    - MORNING_STAR : Numeric indicator (1/0) untuk pola Morning Star (bullish reversal). 1 = pola ditemukan, 0 = tidak ditemukan.
    - EVENING_STAR : Numeric indicator (1/0) untuk pola Evening Star (bearish reversal). 1 = pola ditemukan, 0 = tidak ditemukan.
    - ADX14           : Average Directional Index (14). Indikator kekuatan tren, > 25 menunjukkan tren kuat.
    - ADX_PLUS        : Positive Directional Indicator (+DI14). Mengukur kekuatan tren naik.
    - ADX_MINUS       : Negative Directional Indicator (-DI14). Mengukur kekuatan tren turun.

    --- Bullish Chart Patterns ---
    - BULLISH_ENGULFING : Bullish Engulfing Pattern (1 = found, 0 = not found).
    - PIERCING_LINE : Piercing Line Pattern (1 = found, 0 = not found).
    - INVERSE_HAMMER : Inverse Hammer Pattern (1 = found, 0 = not found).
    - BULLISH_HARAMI : Bullish Harami Pattern (1 = found, 0 = not found).
    - TWEZER_BOTTOM : Tweezer Bottom Pattern (1 = found, 0 = not found).
    - THREE_WHITE_SOLDIERS : Three White Soldiers Pattern (1 = found, 0 = not found).
    - RISING_THREE_METHODS : Rising Three Methods Pattern (1 = found, 0 = not found).
    - BULLISH_ABANDONED_BABY : Bullish Abandoned Baby Pattern (1 = found, 0 = not found).
    - BULLISH_KICKER : Bullish Kicker Pattern (1 = found, 0 = not found).
    - BULLISH_MARUBOZU : Bullish Marubozu Pattern (1 = found, 0 = not found).

    --- Divergence Patterns ---
    - BULLISH_DIVERGENCE : Bullish divergence pattern (1 = found, 0 = not found).
    - BEARISH_DIVERGENCE : Bearish divergence pattern (1 = found, 0 = not found).
    """

    # 1. Copy data agar aman
    df = df.copy()
    if df.empty:
        return df

    # Timeframe-based parameter adjustments
    def get_timeframe_multiplier(timeframe):
        """Get multiplier for indicator periods based on timeframe"""
        timeframe_map = {
            "5m": 0.25,    # 5 minutes = 1/4 hour
            "30m": 1.5,    # 30 minutes = 1.5x 15min, but we'll use 1.5x multiplier
            "1h": 1.0,     # 1 hour = base
            "4h": 4.0,     # 4 hours
            "6h": 6.0,     # 6 hours
            "12h": 12.0,   # 12 hours
            "1d": 24.0,    # 1 day = 24 hours
            "3d": 72.0,    # 3 days
            "7d": 168.0,   # 7 days
            "1m": 720.0    # 1 month ≈ 30 days
        }
        return timeframe_map.get(timeframe, 1.0)  # Default to 1.0 if timeframe not found

    timeframe_multiplier = get_timeframe_multiplier(timeframe)

    # 2. Handle MultiIndex columns if present (from yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        # Extract single-level columns for price data
        df_single = pd.DataFrame()
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df_single[col] = df[col].iloc[:, 0] if isinstance(df[col], pd.DataFrame) else df[col]
            elif col.lower() in df.columns:
                df_single[col] = df[col.lower()].iloc[:, 0] if isinstance(df[col.lower()], pd.DataFrame) else df[col.lower()]
        df = df_single

    # 3. Mapping Standard Columns (Huruf Besar)
    # Handle both uppercase and lowercase column names
    def get_col(df, name):
        if name in df.columns:
            return df[name]
        elif name.lower() in df.columns:
            return df[name.lower()]
        else:
            raise KeyError(f"Column {name} not found in DataFrame")

    df["OPEN"] = get_col(df, "Open")
    df["HIGH"] = get_col(df, "High")
    df["LOW"] = get_col(df, "Low")
    df["CLOSE"] = get_col(df, "Close")
    df["PRICE"] = df["CLOSE"]   # Alias
    df["VOLUME"] = get_col(df, "Volume")

    # Variabel lokal untuk perhitungan
    close = df["CLOSE"]
    high = df["HIGH"]
    low = df["LOW"]
    volume = df["VOLUME"]

    # Helper function to check if indicator is needed
    def need_indicator(indicator_prefix):
        if required_indicators is None:
            return True
        return any(ind.startswith(indicator_prefix) for ind in required_indicators)

    # Helper function to extract window from indicator name
    def get_window(indicator_name, default_window):
        if required_indicators is None:
            return default_window
        for ind in required_indicators:
            if ind.startswith(indicator_name):
                # Extract number from indicator name (e.g., MA20 -> 20)
                import re
                match = re.search(r'\d+', ind)
                if match:
                    return int(match.group())
        return default_window

    # 4. Previous Price & Returns (always calculate basic returns)
    df["PREVIOUS_PRICE"] = close.shift(1)

    # Adjust periods based on timeframe
    base_periods = [1, 7, 30, 50, 100]
    adjusted_periods = [int(p * timeframe_multiplier) for p in base_periods]
    for i, p in enumerate(adjusted_periods):
        original_p = base_periods[i]
        df[f"PRICE_RETURN_{original_p}D_PCT"] = close.pct_change(p, fill_method=None) * 100

    # 5. Frequency Spike (Karakter Saham)
    if need_indicator("PERCENT_SPIKE"):
        # Hitung berapa kali naik >= 1% dalam 30 hari terakhir
        daily_pct = df["PRICE_RETURN_1D_PCT"]
        spike_mask = daily_pct >= 1
        df["PERCENT_SPIKE"] = spike_mask.rolling(30).sum()

    # 5b. Volume Spike (VOLUME_SPIKE)
    if need_indicator("VOLUME_SPIKE"):
        # Volume hari ini dibagi rata-rata volume 30 hari
        volume_30_avg = volume.rolling(30).mean()
        df["VOLUME_SPIKE"] = volume / volume_30_avg.replace(0, np.nan)

    # 5c. Relative Volume (RVOL)
    if need_indicator("RVOL"):
        # Volume hari ini dibagi rata-rata volume 10 hari
        volume_10_avg = volume.rolling(10).mean()
        df["RVOL"] = volume / volume_10_avg.replace(0, np.nan)

    # 6. Moving Averages (Price & Volume)
    # Calculate only required moving averages
    ma_windows = []
    vma_windows = []

    if required_indicators is not None:
        # Extract required MA and VMA windows
        import re
        for ind in required_indicators:
            if ind.startswith('MA'):
                match = re.search(r'MA(\d+)', ind)
                if match:
                    ma_windows.append(int(match.group(1)))
            elif ind.startswith('VMA'):
                match = re.search(r'VMA(\d+)', ind)
                if match:
                    vma_windows.append(int(match.group(1)))

        # Remove duplicates and sort
        ma_windows = sorted(list(set(ma_windows)))
        vma_windows = sorted(list(set(vma_windows)))
    else:
        # Default windows if no specific requirements
        ma_windows = [5, 10, 20, 50, 100, 200]
        vma_windows = [5, 10, 20]

    # Adjust MA/VMA windows based on timeframe
    adjusted_ma_windows = [int(w * timeframe_multiplier) for w in ma_windows]
    adjusted_vma_windows = [int(w * timeframe_multiplier) for w in vma_windows]

    for i, w in enumerate(ma_windows):
        if need_indicator(f"MA{w}"):
            adjusted_w = adjusted_ma_windows[i]
            df[f"MA{w}"] = close.rolling(adjusted_w).mean()

    for i, w in enumerate(vma_windows):
        if need_indicator(f"VMA{w}"):
            adjusted_w = adjusted_vma_windows[i]
            df[f"VMA{w}"] = volume.rolling(adjusted_w).mean()

    # 7. Volatility (ATR & Bollinger Bands)
    if need_indicator("ATR") or need_indicator("BB") or need_indicator("SUPPORT") or need_indicator("RESISTANCE"):
        # ATR with timeframe adjustment
        atr_period = int(14 * timeframe_multiplier)
        tr1 = (high - low).abs()
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR14"] = tr.rolling(atr_period).mean()

    if need_indicator("BB"):
        # Bollinger Bands with timeframe adjustment
        bb_period = int(20 * timeframe_multiplier)
        bb_mid = close.rolling(bb_period).mean()
        bb_std = close.rolling(bb_period).std()
        df["BB_MIDDLE"] = bb_mid
        df["BB_UPPER"] = bb_mid + (bb_std * 2)
        df["BB_LOWER"] = bb_mid - (bb_std * 2)
        df["BB_WIDTH"] = df["BB_UPPER"] - df["BB_LOWER"]
        # Handle division by zero
        df["BB_PERCENT_B"] = (close - df["BB_LOWER"]) / df["BB_WIDTH"].replace(0, np.nan)

    # 8. Momentum (RSI, MACD, Stochastic)
    if need_indicator("RSI"):
        # RSI with timeframe adjustment
        rsi_period = int(14 * timeframe_multiplier)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(rsi_period).mean()
        avg_loss = loss.rolling(rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI14"] = 100 - (100 / (1 + rs))

    if need_indicator("MACD"):
        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["MACD_LINE"] = ema12 - ema26
        df["MACD_SIGNAL"] = df["MACD_LINE"].ewm(span=9, adjust=False).mean()
        df["MACD_HISTOGRAM"] = df["MACD_LINE"] - df["MACD_SIGNAL"]

    if need_indicator("STOCH"):
        # Stochastic with timeframe adjustment
        stoch_k_period = int(14 * timeframe_multiplier)
        stoch_d_period = max(3, int(3 * timeframe_multiplier))  # Ensure at least 3
        low_period = low.rolling(stoch_k_period).min()
        high_period = high.rolling(stoch_k_period).max()
        stoch_k = 100 * ((close - low_period) / (high_period - low_period).replace(0, np.nan))
        df["STOCH_K"] = stoch_k
        df["STOCH_D"] = stoch_k.rolling(stoch_d_period).mean()

    # 9. Support / Resistance & High/Low Rolling
    support_resistance_windows = []
    if required_indicators is not None:
        # Extract required HH/LL windows
        import re
        for ind in required_indicators:
            if ind.startswith('HH') or ind.startswith('LL'):
                match = re.search(r'[HL]{2}(\d+)', ind)
                if match:
                    support_resistance_windows.append(int(match.group(1)))
        support_resistance_windows = list(set(support_resistance_windows))
    else:
        support_resistance_windows = [3, 20]

    # Adjust support/resistance windows based on timeframe
    adjusted_sr_windows = [int(w * timeframe_multiplier) for w in support_resistance_windows]

    for i, w in enumerate(support_resistance_windows):
        if need_indicator(f"HH{w}"):
            adjusted_w = adjusted_sr_windows[i]
            df[f"HH{w}"] = high.rolling(adjusted_w).max()
        if need_indicator(f"LL{w}"):
            adjusted_w = adjusted_sr_windows[i]
            df[f"LL{w}"] = low.rolling(adjusted_w).min()

    # Fibonacci Retracement (50 Days Range)
    if need_indicator("FIB"):
        hh50 = high.rolling(50).max()
        ll50 = low.rolling(50).min()
        range50 = hh50 - ll50
        df["FIB_236"] = hh50 - (range50 * 0.236)
        df["FIB_382"] = hh50 - (range50 * 0.382)
        df["FIB_50"]  = hh50 - (range50 * 0.5)
        df["FIB_618"] = hh50 - (range50 * 0.618)

    # Dynamic Support/Resistance (ATR Based)
    if need_indicator("SUPPORT") or need_indicator("RESISTANCE"):
        # Support lantai yang naik turun ikut volatilitas
        if "MA20" not in df.columns and need_indicator("MA20"):
            df["MA20"] = close.rolling(20).mean()
        if "ATR14" not in df.columns and need_indicator("ATR14"):
            # ATR calculation (simplified)
            tr1 = (high - low).abs()
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df["ATR14"] = tr.rolling(14).mean()

        df["SUPPORT"] = df["MA20"] - (df["ATR14"] * 1.5)
        df["RESISTANCE"] = df["MA20"] + (df["ATR14"] * 1.5)

    # 10. VWAP / VWMA
    if need_indicator("VWAP") or need_indicator("VWMA"):
        # Typical Price
        tp = (high + low + close) / 3
        # VWMA with timeframe adjustment
        vwma_period = int(20 * timeframe_multiplier)
        vp_sum = (tp * volume).rolling(vwma_period).sum()
        v_sum = volume.rolling(vwma_period).sum()
        df["VWMA"] = vp_sum / v_sum.replace(0, np.nan)

        # VWAP Proxy (disamakan dengan VWMA agar tidak misleading di chart daily)
        df["VWAP"] = df["VWMA"]

    # 10. Candlestick Patterns
        # ... (kode sebelumnya)

    # 10. Candlestick Patterns
    if need_indicator("THREE_RED_CANDLES"):
        # UPDATE: Three Red Candles / Three Black Crows Pattern (Bearish)
        # Mendeteksi 3 candle merah berturut-turut dengan Low yang semakin rendah (Lower Low)

        # 1. Cek Candle Merah (Close < Open)
        is_red = df["CLOSE"] < df["OPEN"]

        # 2. Cek apakah 3 hari terakhir berturut-turut merah
        three_consecutive_red = is_red & is_red.shift(1) & is_red.shift(2)

        # 3. Cek Lower Low (Low hari ini < Low kemarin < Low 2 hari lalu)
        # Ini memastikan tren benar-benar sedang "turun ke bawah"
        descending_lows = (df["LOW"] < df["LOW"].shift(1)) & \
                          (df["LOW"].shift(1) < df["LOW"].shift(2))

        # Gabungkan kedua kondisi
        pattern_confirmed = three_consecutive_red & descending_lows

        # Convert ke integer (1 = Pola ditemukan hari ini, 0 = Tidak)
        df["THREE_RED_CANDLES"] = pattern_confirmed.fillna(False).astype(int)

    # ... (lanjut ke THREE_GREEN_CANDLES)


    if need_indicator("THREE_GREEN_CANDLES"):
        # Three Green Candles Pattern (Bullish)
        # Candle hijau = Close > Open
        is_green_candle = df["CLOSE"] > df["OPEN"]
        # Memeriksa apakah 3 candle berturut-turut hijau
        three_green_condition = is_green_candle & is_green_candle.shift(1) & is_green_candle.shift(2)
        # Konversi boolean ke integer (1 untuk True, 0 untuk False)
        df["THREE_GREEN_CANDLES"] = three_green_condition.astype(int)

    if need_indicator("DOJI"):
        # Doji Pattern
        # Doji terjadi ketika Open ≈ Close (selisih kecil)
        body_size = (df["CLOSE"] - df["OPEN"]).abs()
        average_body = body_size.rolling(10).mean()  # Rata-rata body size 10 hari terakhir
        # Doji jika body size < 10% dari rata-rata body size
        doji_condition = body_size < (average_body * 0.1)
        df["DOJI"] = doji_condition.astype(int)

    if need_indicator("HAMMER") or need_indicator("HANGING_MAN"):
        # Hammer Pattern (Bullish Reversal)
        # Hammer: panjang lower shadow jauh lebih besar dari body, upper shadow sangat kecil/tdk ada
        # Body kecil di bagian atas
        body_size = (df["CLOSE"] - df["OPEN"]).abs()
        upper_shadow = df["HIGH"] - df[["CLOSE", "OPEN"]].max(axis=1)
        lower_shadow = df[["CLOSE", "OPEN"]].min(axis=1) - df["LOW"]

        # Hammer conditions:
        # 1. Lower shadow >= 2x body size
        # 2. Upper shadow <= 0.5x body size
        # 3. Body kecil (< 20% dari range total)
        total_range = df["HIGH"] - df["LOW"]
        hammer_condition = (
            (lower_shadow >= body_size * 2) &
            (upper_shadow <= body_size * 0.5) &
            (body_size < total_range * 0.2)
        )
        df["HAMMER"] = hammer_condition.astype(int)

        # Hanging Man Pattern (Bearish Reversal)
        # Sama seperti hammer tapi terjadi di uptrend (kita bisa cek dengan MA200)
        # Untuk kesederhanaan, kita gunakan logika yang sama dengan hammer
        df["HANGING_MAN"] = hammer_condition.astype(int)

    if need_indicator("SHOOTING_STAR"):
        # Shooting Star Pattern (Bearish Reversal)
        # Kebalikan dari hammer: body kecil di bagian bawah
        # Upper shadow >= 2x body size
        # Lower shadow <= 0.5x body size
        body_size = (df["CLOSE"] - df["OPEN"]).abs()
        upper_shadow = df["HIGH"] - df[["CLOSE", "OPEN"]].max(axis=1)
        lower_shadow = df[["CLOSE", "OPEN"]].min(axis=1) - df["LOW"]

        total_range = df["HIGH"] - df["LOW"]
        shooting_star_condition = (
            (upper_shadow >= body_size * 2) &
            (lower_shadow <= body_size * 0.5) &
            (body_size < total_range * 0.2)
        )
        df["SHOOTING_STAR"] = shooting_star_condition.astype(int)

    if need_indicator("MORNING_STAR"):
        # Morning Star Pattern (Bullish Reversal - 3 candles)
        # 1. Large red candle
        # 2. Small body (gap down from 1st candle)
        # 3. Large green candle (gap up from 2nd candle)
        body_size = (df["CLOSE"] - df["OPEN"]).abs()
        first_candle_red = df["CLOSE"].shift(2) < df["OPEN"].shift(2)
        second_candle_small = body_size.shift(1) < (body_size.rolling(10).mean().shift(1) * 0.3)
        third_candle_green = df["CLOSE"] > df["OPEN"]

        morning_star_condition = first_candle_red & second_candle_small & third_candle_green
        df["MORNING_STAR"] = morning_star_condition.astype(int)

    if need_indicator("EVENING_STAR"):
        # Evening Star Pattern (Bearish Reversal - 3 candles)
        # 1. Large green candle
        # 2. Small body (gap up from 1st candle)
        # 3. Large red candle (gap down from 2nd candle)
        body_size = (df["CLOSE"] - df["OPEN"]).abs()
        first_candle_green = df["CLOSE"].shift(2) > df["OPEN"].shift(2)
        second_candle_small = body_size.shift(1) < (body_size.rolling(10).mean().shift(1) * 0.3)
        # Kebalikan dari morning star (third candle red)
        evening_star_condition_corrected = first_candle_green & second_candle_small & (df["CLOSE"] < df["OPEN"])
        df["EVENING_STAR"] = evening_star_condition_corrected.astype(int)

    # 11. Bullish Chart Patterns (10 Best Bullish Patterns)
    if need_indicator("BULLISH_ENGULFING"):
        # Bullish Engulfing Pattern: Small red candle followed by large green candle that "engulfs" it
        prev_candle_bearish = df["CLOSE"].shift(1) < df["OPEN"].shift(1)
        current_candle_bullish = df["CLOSE"] > df["OPEN"]
        engulfing_condition = (df["CLOSE"] > df["OPEN"].shift(1)) & (df["OPEN"] < df["CLOSE"].shift(1))
        df["BULLISH_ENGULFING"] = (prev_candle_bearish & current_candle_bullish & engulfing_condition).astype(int)

    if need_indicator("PIERCING_LINE"):
        # Piercing Line Pattern: Bearish candle followed by bullish candle that closes more than halfway up the bearish candle
        prev_candle_bearish = df["CLOSE"].shift(1) < df["OPEN"].shift(1)
        current_candle_bullish = df["CLOSE"] > df["OPEN"]
        # Current close should be above midpoint of previous candle
        prev_candle_midpoint = (df["OPEN"].shift(1) + df["CLOSE"].shift(1)) / 2
        piercing_condition = df["CLOSE"] > prev_candle_midpoint
        df["PIERCING_LINE"] = (prev_candle_bearish & current_candle_bullish & piercing_condition).astype(int)

    if need_indicator("INVERSE_HAMMER"):
        # Inverse Hammer Pattern: Small body at bottom, long upper shadow, little/no lower shadow
        body_size = (df["CLOSE"] - df["OPEN"]).abs()
        upper_shadow = df["HIGH"] - df[["CLOSE", "OPEN"]].max(axis=1)
        lower_shadow = df[["CLOSE", "OPEN"]].min(axis=1) - df["LOW"]
        total_range = df["HIGH"] - df["LOW"]

        # Inverse Hammer conditions:
        # 1. Upper shadow >= 2x body size
        # 2. Lower shadow <= 0.5x body size
        # 3. Body kecil (< 20% dari range total)
        # 4. Occurs after downtrend (close below recent average)
        inverse_hammer_condition = (
            (upper_shadow >= body_size * 2) &
            (lower_shadow <= body_size * 0.5) &
            (body_size < total_range * 0.2) &
            (df["CLOSE"] < df["CLOSE"].rolling(5).mean())
        )
        df["INVERSE_HAMMER"] = inverse_hammer_condition.astype(int)

    if need_indicator("BULLISH_HARAMI"):
        # Bullish Harami Pattern: Large bearish candle followed by small candle within the bearish candle's body
        prev_candle_bearish = df["CLOSE"].shift(1) < df["OPEN"].shift(1)
        prev_candle_large = (df["OPEN"].shift(1) - df["CLOSE"].shift(1)) > (df["HIGH"].shift(1) - df["LOW"].shift(1)) * 0.6
        current_candle_small = (df["HIGH"] - df["LOW"]) < (df["OPEN"].shift(1) - df["CLOSE"].shift(1)) * 0.5
        within_prev_body = (df["OPEN"] > df["CLOSE"].shift(1)) & (df["CLOSE"] < df["OPEN"].shift(1))
        df["BULLISH_HARAMI"] = (prev_candle_bearish & prev_candle_large & current_candle_small & within_prev_body).astype(int)

    if need_indicator("TWEZER_BOTTOM"):
        # Tweezer Bottom Pattern: Two candles with same low, first bearish, second bullish
        same_low = df["LOW"] == df["LOW"].shift(1)
        prev_candle_bearish = df["CLOSE"].shift(1) < df["OPEN"].shift(1)
        current_candle_bullish = df["CLOSE"] > df["OPEN"]
        df["TWEZER_BOTTOM"] = (same_low & prev_candle_bearish & current_candle_bullish).astype(int)

    if need_indicator("THREE_WHITE_SOLDIERS"):
        # Three White Soldiers Pattern: Three consecutive long bullish candles with higher closes
        candle1_bullish = df["CLOSE"].shift(2) > df["OPEN"].shift(2)
        candle2_bullish = df["CLOSE"].shift(1) > df["OPEN"].shift(1)
        candle3_bullish = df["CLOSE"] > df["OPEN"]
        higher_closes = (df["CLOSE"] > df["CLOSE"].shift(1)) & (df["CLOSE"].shift(1) > df["CLOSE"].shift(2))
        large_bodies = (
            ((df["CLOSE"] - df["OPEN"]) > (df["HIGH"] - df["LOW"]) * 0.6) &
            ((df["CLOSE"].shift(1) - df["OPEN"].shift(1)) > (df["HIGH"].shift(1) - df["LOW"].shift(1)) * 0.6) &
            ((df["CLOSE"].shift(2) - df["OPEN"].shift(2)) > (df["HIGH"].shift(2) - df["LOW"].shift(2)) * 0.6)
        )
        df["THREE_WHITE_SOLDIERS"] = (candle1_bullish & candle2_bullish & candle3_bullish & higher_closes & large_bodies).astype(int)

    if need_indicator("RISING_THREE_METHODS"):
        # Rising Three Methods Pattern: Long bullish candle, three small bearish candles, then another long bullish candle
        candle1_bullish = df["CLOSE"].shift(4) > df["OPEN"].shift(4)
        candle1_large = (df["CLOSE"].shift(4) - df["OPEN"].shift(4)) > (df["HIGH"].shift(4) - df["LOW"].shift(4)) * 0.7
        candle5_bullish = df["CLOSE"] > df["OPEN"]
        candle5_large = (df["CLOSE"] - df["OPEN"]) > (df["HIGH"] - df["LOW"]) * 0.7
        middle_candles_small = (
            ((df["HIGH"].shift(3) - df["LOW"].shift(3)) < (df["CLOSE"].shift(4) - df["OPEN"].shift(4)) * 0.5) &
            ((df["HIGH"].shift(2) - df["LOW"].shift(2)) < (df["CLOSE"].shift(4) - df["OPEN"].shift(4)) * 0.5) &
            ((df["HIGH"].shift(1) - df["LOW"].shift(1)) < (df["CLOSE"].shift(4) - df["OPEN"].shift(4)) * 0.5)
        )
        within_range = (
            (df["CLOSE"] > df["OPEN"].shift(4)) &
            (df["OPEN"].shift(3) < df["CLOSE"].shift(4)) &
            (df["CLOSE"].shift(3) > df["OPEN"].shift(4))
        )
        df["RISING_THREE_METHODS"] = (candle1_bullish & candle1_large & candle5_bullish & candle5_large & middle_candles_small & within_range).astype(int)

    if need_indicator("BULLISH_ABANDONED_BABY"):
        # Bullish Abandoned Baby Pattern: Bearish candle, doji gapping down, then bullish candle gapping up
        candle1_bearish = df["CLOSE"].shift(2) < df["OPEN"].shift(2)
        candle3_bullish = df["CLOSE"] > df["OPEN"]
        # Middle candle should be a doji
        middle_body_small = (df["CLOSE"].shift(1) - df["OPEN"].shift(1)).abs() < (df["HIGH"].shift(1) - df["LOW"].shift(1)) * 0.1
        # Gaps: candle2 opens below candle1 close, candle3 opens above candle2 close
        gap_down = df["OPEN"].shift(1) < df["CLOSE"].shift(2)
        gap_up = df["OPEN"] > df["CLOSE"].shift(1)
        df["BULLISH_ABANDONED_BABY"] = (candle1_bearish & candle3_bullish & middle_body_small & gap_down & gap_up).astype(int)

    if need_indicator("BULLISH_KICKER"):
        # Bullish Kicker Pattern: Bearish candle followed by bullish candle with gap up
        candle1_bearish = df["CLOSE"].shift(1) < df["OPEN"].shift(1)
        candle2_bullish = df["CLOSE"] > df["OPEN"]
        gap_up = df["OPEN"] > df["CLOSE"].shift(1)
        large_moves = (
            ((df["OPEN"].shift(1) - df["CLOSE"].shift(1)) > (df["HIGH"].shift(1) - df["LOW"].shift(1)) * 0.5) &
            ((df["CLOSE"] - df["OPEN"]) > (df["HIGH"] - df["LOW"]) * 0.5)
        )
        df["BULLISH_KICKER"] = (candle1_bearish & candle2_bullish & gap_up & large_moves).astype(int)

    if need_indicator("BULLISH_MARUBOZU"):
        # Bullish Marubozu Pattern: Long bullish candle with little or no shadows
        body_size = df["CLOSE"] - df["OPEN"]
        upper_shadow = df["HIGH"] - df["CLOSE"]
        lower_shadow = df["OPEN"] - df["LOW"]
        total_range = df["HIGH"] - df["LOW"]

        # Marubozu conditions:
        # 1. Large body (> 80% of total range)
        # 2. Very small or no shadows (< 5% of body size)
        # 3. Bullish (close > open)
        marubozu_condition = (
            (body_size > total_range * 0.8) &
            (upper_shadow < body_size * 0.05) &
            (lower_shadow < body_size * 0.05) &
            (df["CLOSE"] > df["OPEN"])
        )
        df["BULLISH_MARUBOZU"] = marubozu_condition.astype(int)

    # 11. Divergence Patterns (Bullish & Bearish)
    if need_indicator("BULLISH_DIVERGENCE") or need_indicator("BEARISH_DIVERGENCE"):
        # Ensure RSI14 is calculated first
        if "RSI14" not in df.columns:
            # Calculate RSI14 if not already present
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df["RSI14"] = 100 - (100 / (1 + rs))

        # Helper function to detect higher highs and lower lows
        def get_higher_highs(series, window=5):
            """Detect higher highs in a series"""
            return series > series.rolling(window).max().shift(1)

        def get_lower_lows(series, window=5):
            """Detect lower lows in a series"""
            return series < series.rolling(window).min().shift(1)

        def get_higher_lows(series, window=5):
            """Detect higher lows in a series"""
            return series > series.rolling(window).min().shift(1)

        def get_lower_highs(series, window=5):
            """Detect lower highs in a series"""
            return series < series.rolling(window).max().shift(1)

        # Bullish Divergence: Price makes lower lows while RSI makes higher lows
        if need_indicator("BULLISH_DIVERGENCE"):
            # Use timeframe-adjusted window for divergence detection
            divergence_window = max(3, int(5 * timeframe_multiplier))
            lookback_window = max(10, int(10 * timeframe_multiplier))

            # Enhanced bullish divergence detection
            # 1. Price is making lower lows (downtrend)
            price_lower_lows = get_lower_lows(low, divergence_window)

            # 2. RSI is making higher lows (potential reversal signal)
            rsi_higher_lows = get_higher_lows(df["RSI14"], divergence_window)

            # 3. Additional confirmation: RSI is in oversold territory or coming from oversold
            rsi_oversold_condition = (df["RSI14"] < 50) & (df["RSI14"].shift(1) < 50)

            # 4. Price is below recent average (confirming downtrend)
            price_below_ma = close < close.rolling(lookback_window).mean()

            # Combine conditions for more robust detection
            bullish_divergence = price_lower_lows & rsi_higher_lows & rsi_oversold_condition & price_below_ma
            df["BULLISH_DIVERGENCE"] = bullish_divergence.astype(int)

        # Bearish Divergence: Price makes higher highs while RSI makes lower highs
        if need_indicator("BEARISH_DIVERGENCE"):
            # Use timeframe-adjusted window for divergence detection
            divergence_window = max(3, int(5 * timeframe_multiplier))
            lookback_window = max(10, int(10 * timeframe_multiplier))

            # Enhanced bearish divergence detection
            # 1. Price is making higher highs (uptrend)
            price_higher_highs = get_higher_highs(high, divergence_window)

            # 2. RSI is making lower highs (potential reversal signal)
            rsi_lower_highs = get_lower_highs(df["RSI14"], divergence_window)

            # 3. Additional confirmation: RSI is in overbought territory or coming from overbought
            rsi_overbought_condition = (df["RSI14"] > 50) & (df["RSI14"].shift(1) > 50)

            # 4. Price is above recent average (confirming uptrend)
            price_above_ma = close > close.rolling(lookback_window).mean()

            # Combine conditions for more robust detection
            bearish_divergence = price_higher_highs & rsi_lower_highs & rsi_overbought_condition & price_above_ma
            df["BEARISH_DIVERGENCE"] = bearish_divergence.astype(int)

    # 12. ADX (Average Directional Index)
    if need_indicator("ADX"):
        # Check if we have enough data for ADX calculation with timeframe adjustment
        adx_period = int(14 * timeframe_multiplier)
        min_data_required = adx_period * 2  # Need at least 2x periods for ADX
        if len(df) < min_data_required:
            df["ADX14"] = 0.0
            df["ADX_PLUS"] = 0.0
            df["ADX_MINUS"] = 0.0
        else:
            # Calculate True Range (TR)
            tr1 = high - low
            tr2 = np.abs(high - close.shift(1))
            tr3 = np.abs(low - close.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Calculate Directional Movement (+DM and -DM)
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low

            # Positive Directional Movement (+DM)
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            plus_dm = pd.Series(plus_dm, index=high.index)

            # Negative Directional Movement (-DM)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            minus_dm = pd.Series(minus_dm, index=high.index)

            # Smoothed True Range and Directional Movements (Wilder's Method)
            atr_period = adx_period
            smoothed_tr = pd.Series(index=true_range.index, dtype=float)
            smoothed_plus_dm = pd.Series(index=plus_dm.index, dtype=float)
            smoothed_minus_dm = pd.Series(index=minus_dm.index, dtype=float)

            # Initialize first values
            smoothed_tr.iloc[atr_period-1] = true_range[:atr_period].sum()
            smoothed_plus_dm.iloc[atr_period-1] = plus_dm[:atr_period].sum()
            smoothed_minus_dm.iloc[atr_period-1] = minus_dm[:atr_period].sum()

            # Calculate remaining values using Wilder's smoothing method
            for i in range(atr_period, len(true_range)):
                smoothed_tr.iloc[i] = smoothed_tr.iloc[i-1] - (smoothed_tr.iloc[i-1] / atr_period) + true_range.iloc[i]
                smoothed_plus_dm.iloc[i] = smoothed_plus_dm.iloc[i-1] - (smoothed_plus_dm.iloc[i-1] / atr_period) + plus_dm.iloc[i]
                smoothed_minus_dm.iloc[i] = smoothed_minus_dm.iloc[i-1] - (smoothed_minus_dm.iloc[i-1] / atr_period) + minus_dm.iloc[i]

            # Calculate Directional Indicators
            plus_di = (smoothed_plus_dm / smoothed_tr) * 100
            minus_di = (smoothed_minus_dm / smoothed_tr) * 100

            # Calculate Directional Movement Index (DX)
            dx_divisor = (np.abs(plus_di) + np.abs(minus_di)).replace(0, np.nan)
            dx = (np.abs(plus_di - minus_di) / dx_divisor) * 100

            # Calculate ADX (Average Directional Index)
            adx = pd.Series(index=dx.index, dtype=float)
            adx.iloc[atr_period*2-1] = dx[atr_period:atr_period*2].mean()  # Initial ADX

            # Smoothed ADX calculation (Wilder's method)
            for i in range(adx_period*2, len(dx)):
                adx.iloc[i] = ((adx.iloc[i-1] * (adx_period - 1)) + dx.iloc[i]) / adx_period

            # Assign to DataFrame
            df["ADX14"] = adx
            df["ADX_PLUS"] = plus_di
            df["ADX_MINUS"] = minus_di

    # 12. Frekuensi Perdagangan (Trading Frequency)
    if need_indicator("TRADE_FREQUENCY") or need_indicator("HIGH_VOL_DAYS"):
        # Mengukur seberapa sering saham diperdagangkan berdasarkan volume
        tf_period = int(20 * timeframe_multiplier)
        avg_volume = volume.rolling(tf_period).mean()
        # TRADE_FREQUENCY: Rasio volume saat ini terhadap rata-rata volume (dalam %)
        df["TRADE_FREQUENCY"] = (volume / avg_volume.replace(0, np.nan)) * 100
        # HIGH_VOL_DAYS: Hari dengan volume tinggi (>150% dari rata-rata 20 hari)
        high_vol_mask = volume > (avg_volume * 1.5)
        high_vol_period = int(30 * timeframe_multiplier)
        df["HIGH_VOL_DAYS"] = high_vol_mask.rolling(high_vol_period).sum()

    # 13. Final Cleanup (Isi NaN dengan 0 agar engine eval aman)
    # Fill Forward dulu (untuk data yg bolong dikit), lalu Fill 0 (untuk awal data)
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)

    return df


def add_fundamental_indicators(df: pd.DataFrame, ticker_info: dict) -> pd.DataFrame:
    """
    Menambahkan indikator fundamental ke dalam DataFrame untuk scanning/backtest.

    Parameter:
    - df: DataFrame harga historis
    - ticker_info: Dictionary informasi fundamental dari yfinance

    DAFTAR INDIKATOR FUNDAMENTAL (Nama Kolom):

    --- Valuasi (Valuation) ---
    - TRAILING_PE      : Price to Earnings Ratio (TTM). Semakin kecil, semakin "murah".
    - FORWARD_PE       : Forward P/E. Estimasi P/E berdasarkan proyeksi laba analis.
    - PEG_RATIO        : Price/Earnings to Growth. Dibawah 1.0 biasanya undervalued.
    - PRICE_TO_BOOK    : PBV (Price to Book Value). < 1.0 biasanya "salah harga".
    - PRICE_TO_SALES   : Price to Sales (P/S). Untuk perusahaan belum untung tapi omzet besar.
    - PRICE_TO_CF      : Price to Cash Flow Ratio (P/CF). Rasio harga saham terhadap arus kas per lembar.

    --- Profitabilitas (Profitability) ---
    - RETURN_ON_EQUITY : ROE. Efisiensi manajemen mengelola uang investor (%).
    - RETURN_ON_ASSETS : ROA. Efisiensi penggunaan aset perusahaan (%).
    - PROFIT_MARGINS   : Net Profit Margin (NPM). Laba bersih dari total pendapatan (%).
    - GROSS_MARGINS    : Gross Profit Margin (GPM). Laba kotor dari total pendapatan (%).
    - OPERATING_MARGINS: Operating Margin (OPM). Laba operasional dari total pendapatan (%).

    --- Kesehatan Keuangan (Financial Health) ---
    - DEBT_TO_EQUITY   : DER. Rasio total hutang terhadap total modal (>1.0 berarti hutang > modal sendiri).
    - CURRENT_RATIO    : Aset Lancar/Hutang Lancar. Kemampuan bayar hutang jangka pendek (>1.0 aman).
    - QUICK_RATIO      : Acid Test Ratio. Seperti Current Ratio tapi tanpa inventory (lebih ketat).
    - TOTAL_CASH_PS    : Jumlah uang tunai per lembar saham.

    --- Dividen (Dividends) ---
    - DIVIDEND_YIELD   : Imbal hasil dividen tahunan terhadap harga saham (%).
    - PAYOUT_RATIO     : Persentase laba bersih yang dibagikan sebagai dividen (%).
    - DIVIDEND_RATE    : Estimasi nominal dividen (dalam Rupiah) per tahun.

    --- Pertumbuhan & Efisiensi (Growth & Efficiency) ---
    - REVENUE_GROWTH   : Pertumbuhan pendapatan (omzet) YoY (%).
    - EARNINGS_GROWTH  : Pertumbuhan laba bersih YoY (%).
    - REVENUE_PS       : Pendapatan per lembar saham.

    --- Arus Kas (Cash Flow) ---
    - OPERATING_CF     : Arus kas bersih dari operasi bisnis inti.
    - FREE_CASHFLOW    : Free Cash Flow (FCF). Sisa uang tunai setelah Capex.
    """

    # Copy data agar aman
    df = df.copy()
    if df.empty:
        return df

    # Validate ticker_info
    if ticker_info is None or not isinstance(ticker_info, dict):
        print(f"Warning: Invalid fundamental data format, using defaults")
        ticker_info = {}

    # Mapping fundamental indicators dari yfinance info
    fundamentals_map = {
        # Valuasi
        'TRAILING_PE': 'trailingPE',
        'FORWARD_PE': 'forwardPE',
        'PEG_RATIO': 'trailingPegRatio',  # Using trailingPegRatio as proxy for pegRatio
        'PRICE_TO_BOOK': 'priceToBook',
        'PRICE_TO_SALES': 'priceToSalesTrailing12Months',
        # Note: EV_TO_EBITDA not directly available from yfinance

        # Profitabilitas
        'RETURN_ON_EQUITY': 'returnOnEquity',
        'RETURN_ON_ASSETS': 'returnOnAssets',
        'PROFIT_MARGINS': 'profitMargins',
        'GROSS_MARGINS': 'grossMargins',
        'OPERATING_MARGINS': 'operatingMargins',

        # Kesehatan Keuangan
        'CURRENT_RATIO': 'currentRatio',
        'QUICK_RATIO': 'quickRatio',  # Not directly available from yfinance
        'TOTAL_CASH_PS': 'totalCashPerShare',

        # Dividen
        'DIVIDEND_YIELD': 'dividendYield',  # Usually expressed as decimal, convert to percentage
        'PAYOUT_RATIO': 'payoutRatio',
        'DIVIDEND_RATE': 'dividendRate',

        # Pertumbuhan
        'REVENUE_GROWTH': 'revenueGrowth',
        'EARNINGS_GROWTH': 'earningsGrowth',
        'REVENUE_PS': 'revenuePerShare',

        # Arus Kas
        'OPERATING_CF': 'operatingCashflow',
        'FREE_CASHFLOW': 'freeCashflow'
    }

    # Add fundamental indicators to DataFrame
    for indicator_col, info_key in fundamentals_map.items():
        if info_key in ticker_info and ticker_info[info_key] is not None:
            # Special handling for dividend yield (usually comes as decimal, convert to percentage)
            if info_key == 'dividendYield':
                df[indicator_col] = ticker_info[info_key] * 100
            else:
                df[indicator_col] = ticker_info[info_key]
        else:
            # Set default value if not available
            df[indicator_col] = 0.0

    # Calculate derived fundamentals if raw data is available
    # Debt to Equity Ratio = Total Debt / Total Equity
    if 'totalDebt' in ticker_info and 'totalStockholderEquity' in ticker_info:
        total_debt = ticker_info.get('totalDebt', 0)
        total_equity = ticker_info.get('totalStockholderEquity', 1)  # Avoid division by zero
        if total_equity != 0:
            df['DEBT_TO_EQUITY'] = total_debt / total_equity
        else:
            df['DEBT_TO_EQUITY'] = 0.0
    elif 'debtToEquity' in ticker_info:
        # Use directly provided debtToEquity if available
        df['DEBT_TO_EQUITY'] = ticker_info.get('debtToEquity', 0.0)
    else:
        df['DEBT_TO_EQUITY'] = 0.0

    # Price to Cash Flow Ratio (P/CF) = Market Price per Share / Cash Flow per Share
    # We'll approximate this using operating cash flow and shares outstanding
    if 'operatingCashflow' in ticker_info and 'sharesOutstanding' in ticker_info and len(df) > 0:
        operating_cf = ticker_info.get('operatingCashflow', 0)
        shares_outstanding = ticker_info.get('sharesOutstanding', 1)
        current_price = df['CLOSE'].iloc[-1] if len(df) > 0 else 0

        if shares_outstanding != 0:
            cf_per_share = operating_cf / shares_outstanding
            if cf_per_share != 0:
                df['PRICE_TO_CF'] = current_price / cf_per_share
            else:
                df['PRICE_TO_CF'] = 0.0
        else:
            df['PRICE_TO_CF'] = 0.0
    else:
        df['PRICE_TO_CF'] = 0.0

    # Final Cleanup (Fill NaN with 0)
    df.fillna(0, inplace=True)

    return df