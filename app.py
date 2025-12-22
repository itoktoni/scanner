# app.py
from flask import Flask, render_template, request, jsonify
import os
import json
import datetime as dt
import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from engine.generic_json_strategy import (
    load_all_json_strategies,
    get_entry_sl_tp_row,
    backtest as json_backtest,
)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TICKERS_JSON_PATH = os.path.join(BASE_DIR, "tickers_id.json")
SYARIAH_TICKERS_PATH = os.path.join(BASE_DIR, "syariah_tickers.json")


def load_all_tickers():
    """
    Load semua emiten IDX dari tickers_id.json.
    Format baru: ["ANTM", "BBRI", ...]
    Di-convert ke list of dict untuk dipakai di template:
    [{"symbol": "ANTM", "name": "ANTM"}, ...]
    """
    if not os.path.exists(TICKERS_JSON_PATH):
        return []
    with open(TICKERS_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # data = list of symbols (string)
    return [{"symbol": s, "name": s} for s in data]

def load_syariah_tickers():
    """
    Load daftar saham syariah dari syariah_tickers.json.
    """
    if not os.path.exists(SYARIAH_TICKERS_PATH):
        return []
    with open(SYARIAH_TICKERS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_tickers(raw: str | None, all_tickers: list[dict]):
    """
    Jika raw tidak kosong → parse manual.
    Jika raw kosong → gunakan semua ticker dari tickers_id.json.
    """
    if raw and raw.strip():
        parts = [p.strip().upper() for p in raw.split(",") if p.strip()]
        return [p + ".JK" for p in parts]

    # fallback: semua ticker dari file JSON
    symbols = [item["symbol"].upper() for item in all_tickers]
    return [s + ".JK" for s in symbols]


def get_price_data(ticker, period="2y"):
    print(f"[*********************100%***********************]  1 of 1 {ticker} completed")
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, level=1, axis=1)
    return df


def determine_optimal_period(config):
    """
    Determine the optimal data period based on strategy requirements.
    """
    # Default to 2 years
    period = "2y"
    
    # Extract all rules from the strategy
    rules = config.get("entry", []) + config.get("tp", []) + config.get("sl", [])
    
    # Look for indicators that require longer data periods
    max_window = 0
    
    # Check for moving averages
    import re
    for rule in rules:
        # Find MA, VMA, etc. with window sizes
        ma_matches = re.findall(r'(?:MA|VMA|HH|LL)(\d+)', rule)
        for match in ma_matches:
            max_window = max(max_window, int(match))
        
        # Check for other indicators with known windows
        if 'ATR14' in rule or 'RSI14' in rule:
            max_window = max(max_window, 14)
        if 'BB_' in rule:
            max_window = max(max_window, 20)
        if 'MACD' in rule:
            max_window = max(max_window, 26)  # MACD needs at least 26 periods
        if 'STOCH' in rule:
            max_window = max(max_window, 14)
        if 'FIB_' in rule:
            max_window = max(max_window, 50)
        if 'PERCENT_SPIKE' in rule or 'HIGH_VOL_DAYS' in rule:
            max_window = max(max_window, 30)
    
    # Add buffer for calculations and ensure we have enough data
    if max_window > 0:
        # Convert to appropriate period string
        if max_window <= 30:
            period = "3mo"  # 3 months is usually enough for short windows
        elif max_window <= 100:
            period = "6mo"  # 6 months for medium windows
        elif max_window <= 200:
            period = "1y"   # 1 year for long windows
        else:
            period = "2y"   # 2 years for very long windows
    
    return period


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        configs = load_all_json_strategies()
        all_tickers = load_all_tickers()
        syariah_tickers = load_syariah_tickers()
        return render_template(
            "index.html",
            strategies=configs,
            all_tickers=all_tickers,
            syariah_tickers=syariah_tickers
        )

    configs = load_all_json_strategies()
    all_tickers = load_all_tickers()
    syariah_tickers = load_syariah_tickers()

    tickers_raw = request.form.get("tickers")
    strat_name = request.form.get("strategy")
    action = request.form.get("action")  # scan / backtest
    amount = float(request.form.get("amount") or 0)
    bt_period = request.form.get("bt_period")

    tickers = parse_tickers(tickers_raw, all_tickers)
    if not tickers:
        return jsonify({"error": "Tidak ada ticker tersedia."}), 400

    if strat_name not in configs:
        return jsonify({"error": "Strategy tidak ditemukan."}), 400

    config = configs[strat_name]

    rows_scan = []
    backtest_trades_all = []
    summary_all = []

    # Process stocks in parallel
    def process_stock(t):
        try:
            # Determine optimal period based on strategy requirements
            period = determine_optimal_period(config)
            df = get_price_data(t, period=period)
            if df.empty:
                return None, None, None

            last_price = float(df["Close"].iloc[-1])

            # SCAN: kondisi sekarang per ticker
            entry_info = get_entry_sl_tp_row(df, config, t)  # Pass ticker info
            scan_result = None
            backtest_results = None
            summary_result = None
            
            if entry_info:
                entry_price = entry_info["entry_price"]
                sl = entry_info["stop_loss"]
                tp = entry_info["take_profit"]

                est_profit_pct = (tp / entry_price - 1) * 100
                est_loss_pct = (sl / entry_price - 1) * 100
                est_profit_rp = amount * est_profit_pct / 100
                est_loss_rp = amount * est_loss_pct / 100

                ticker_symbol = t.replace(".JK", "")
                # Tambahkan * untuk ticker non-syariah
                display_ticker = ticker_symbol + ("" if ticker_symbol in syariah_tickers else "*")
                
                scan_result = {
                    "ticker": display_ticker,
                    "price": last_price,
                    "entry_price": entry_price,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "est_profit_pct": est_profit_pct,
                    "est_profit_rp": est_profit_rp,
                    "est_loss_pct": est_loss_pct,
                    "est_loss_rp": est_loss_rp,
                    "pnl": None,
                    "entry_date": entry_info["entry_date"].strftime("%Y-%m-%d")
                    if entry_info.get("entry_date") is not None else None,
                    "close_date": None,
                    "hold_period": None,
                }

            # BACKTEST
            if action == "backtest":
                today = df.index[-1].date()
                if bt_period == "1m":
                    start = today - dt.timedelta(days=30)
                elif bt_period == "3m":
                    start = today - dt.timedelta(days=90)
                elif bt_period == "6m":
                    start = today - dt.timedelta(days=180)
                elif bt_period == "12m":
                    start = today - dt.timedelta(days=365)
                else:
                    start = today - dt.timedelta(days=365*2)

                trades, summary = json_backtest(
                    df,
                    amount=amount,
                    config=config,
                    start_date=start,
                    end_date=today,
                    ticker=t  # Pass ticker info
                )
                
                ticker_symbol = t.replace(".JK", "")
                # Tambahkan * untuk ticker non-syariah
                display_ticker = ticker_symbol + ("" if ticker_symbol in syariah_tickers else "*")
                
                backtest_trades = []
                for tr in trades:
                    backtest_trades.append({
                        "ticker": display_ticker,
                        "entry_date": tr["entry_date"].strftime("%Y-%m-%d"),
                        "close_date": tr["close_date"].strftime("%Y-%m-%d"),
                        "entry_price": tr["entry_price"],
                        "exit_price": tr["exit_price"],
                        "pnl": tr["pnl"],
                        "roi_pct": tr["roi_pct"],
                        "hold_period": tr["hold_period"],
                        "exit_reason": tr["exit_reason"],
                        "stop_loss": tr["stop_loss"],
                        "take_profit": tr["take_profit"],
                    })
                
                backtest_results = backtest_trades
                summary_result = summary
            
            return scan_result, backtest_results, summary_result
        except Exception as e:
            print(f"Error processing {t}: {e}")
            return None, None, None

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        future_to_ticker = {executor.submit(process_stock, t): t for t in tickers}
        
        # Track progress
        total_tickers = len(tickers)
        completed = 0
        
        # Collect results
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                scan_result, backtest_results, summary_result = future.result()
                if scan_result:
                    rows_scan.append(scan_result)
                if backtest_results:
                    backtest_trades_all.extend(backtest_results)
                if summary_result:
                    summary_all.append(summary_result)
                
                # Update progress
                completed += 1
                progress_percent = (completed / total_tickers) * 100
                print(f"Progress: {completed}/{total_tickers} ({progress_percent:.1f}%)")
            except Exception as e:
                print(f"Error getting result for {ticker}: {e}")
            today = df.index[-1].date()
            if bt_period == "1m":
                start = today - dt.timedelta(days=30)
            elif bt_period == "3m":
                start = today - dt.timedelta(days=90)
            elif bt_period == "6m":
                start = today - dt.timedelta(days=180)
            elif bt_period == "12m":
                start = today - dt.timedelta(days=365)
            else:
                start = today - dt.timedelta(days=365*2)

            trades, summary = json_backtest(
                df,
                amount=amount,
                config=config,
                start_date=start,
                end_date=today,
                ticker=t  # Pass ticker info
            )

            ticker_symbol = t.replace(".JK", "")
            # Tambahkan * untuk ticker non-syariah
            display_ticker = ticker_symbol + ("" if ticker_symbol in syariah_tickers else "*")
            
            for tr in trades:
                backtest_trades_all.append({
                    "ticker": display_ticker,
                    "entry_date": tr["entry_date"].strftime("%Y-%m-%d"),
                    "close_date": tr["close_date"].strftime("%Y-%m-%d"),
                    "entry_price": tr["entry_price"],
                    "exit_price": tr["exit_price"],
                    "pnl": tr["pnl"],
                    "roi_pct": tr["roi_pct"],
                    "hold_period": tr["hold_period"],
                    "exit_reason": tr["exit_reason"],
                    "stop_loss": tr["stop_loss"],
                    "take_profit": tr["take_profit"],
                })

            summary_all.append(summary)

    return jsonify({
        "rows": rows_scan,
        "backtest_trades": backtest_trades_all,
        "summary": summary_all,
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=True)