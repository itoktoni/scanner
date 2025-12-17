# app.py
from flask import Flask, render_template, request, jsonify
import os
import json
import datetime as dt
import yfinance as yf
import pandas as pd

from technical.generic_json_strategy import (
    load_all_json_strategies,
    get_entry_sl_tp_row,
    backtest as json_backtest,
)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TICKERS_JSON_PATH = os.path.join(BASE_DIR, "tickers_id.json")


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
    df = yf.download(ticker, period=period)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, level=1, axis=1)
    return df


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        configs = load_all_json_strategies()
        all_tickers = load_all_tickers()
        return render_template(
            "index.html",
            strategies=configs,
            all_tickers=all_tickers,
        )

    configs = load_all_json_strategies()
    all_tickers = load_all_tickers()

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

    for t in tickers:
        df = get_price_data(t, period="2y")
        if df.empty:
            continue

        last_price = float(df["Close"].iloc[-1])

        # SCAN: kondisi sekarang per ticker
        entry_info = get_entry_sl_tp_row(df, config)
        if entry_info:
            entry_price = entry_info["entry_price"]
            sl = entry_info["stop_loss"]
            tp = entry_info["take_profit"]

            est_profit_pct = (tp / entry_price - 1) * 100
            est_loss_pct = (sl / entry_price - 1) * 100
            est_profit_rp = amount * est_profit_pct / 100
            est_loss_rp = amount * est_loss_pct / 100

            rows_scan.append({
                "ticker": t.replace(".JK", ""),
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
            })

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
            )

            for tr in trades:
                backtest_trades_all.append({
                    "ticker": t.replace(".JK", ""),
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
