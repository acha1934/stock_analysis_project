"""Stock analysis tools operating on the indian_stocks.csv dataset.

Each function is a self-contained, validated interface with typed
parameters, docstrings, and error-handling wrappers — the "Interface
Library" from the research paper's Phase 1.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

from config import settings

logger = logging.getLogger(__name__)

# ── Dataset loader (cached) ────────────────────────────────────────
_df_cache: pd.DataFrame | None = None


def _load_data() -> pd.DataFrame:
    """Load and cache the stock dataset."""
    global _df_cache
    if _df_cache is None:
        path = settings.dataset_path
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")
        _df_cache = pd.read_csv(path, parse_dates=["Date"])
        logger.info("Loaded dataset: %s | rows=%d | tickers=%d",
                     path, len(_df_cache), _df_cache["Ticker"].nunique())
    return _df_cache


# ── Tool functions ─────────────────────────────────────────────────

def get_stock_price(ticker: str) -> dict[str, Any]:
    """Fetch the latest closing price for a given ticker.

    Args:
        ticker: Stock ticker symbol with .NS suffix (e.g., 'TCS.NS', 'RELIANCE.NS').

    Returns:
        Dict with ticker, latest close price, open, high, low, volume, and date.
    """
    df = _load_data()
    stock_df = df[df["Ticker"] == ticker]
    if stock_df.empty:
        available = ", ".join(df["Ticker"].unique()[:10])
        return {"error": f"No data found for ticker '{ticker}'. Available tickers: {available}, ..."}
    latest = stock_df.iloc[-1]
    return {
        "ticker": ticker,
        "date": str(latest["Date"].date()),
        "close_price": round(float(latest["Close"]), 2),
        "open_price": round(float(latest["Open"]), 2),
        "high": round(float(latest["High"]), 2),
        "low": round(float(latest["Low"]), 2),
        "volume": int(latest["Volume"]),
    }


def compare_returns(ticker1: str, ticker2: str, start: str, end: str) -> dict[str, Any]:
    """Compare percentage returns between two stock tickers over a date range.

    Args:
        ticker1: First stock ticker (e.g., 'TCS.NS').
        ticker2: Second stock ticker (e.g., 'RELIANCE.NS').
        start: Start date in 'YYYY-MM-DD' format.
        end: End date in 'YYYY-MM-DD' format.

    Returns:
        Dict with start/end prices and return percentage for both tickers,
        plus which one performed better.
    """
    df = _load_data()
    results = {}

    for ticker in [ticker1, ticker2]:
        stock_df = df[(df["Ticker"] == ticker) & (df["Date"] >= start) & (df["Date"] <= end)]
        if stock_df.empty:
            results[ticker] = {"error": f"No data for {ticker} in range {start} to {end}"}
            continue
        start_price = float(stock_df.iloc[0]["Close"])
        end_price = float(stock_df.iloc[-1]["Close"])
        ret_pct = ((end_price - start_price) / start_price) * 100
        results[ticker] = {
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
            "return_pct": round(ret_pct, 2),
        }

    # Determine winner
    r1 = results.get(ticker1, {}).get("return_pct")
    r2 = results.get(ticker2, {}).get("return_pct")
    if r1 is not None and r2 is not None:
        if r1 > r2:
            results["winner"] = f"{ticker1} outperformed {ticker2} by {round(r1 - r2, 2)}%"
        elif r2 > r1:
            results["winner"] = f"{ticker2} outperformed {ticker1} by {round(r2 - r1, 2)}%"
        else:
            results["winner"] = f"{ticker1} and {ticker2} performed equally"

    return results


def plot_stock_trend(ticker: str, start: str, end: str) -> dict[str, Any]:
    """Generate a line chart of closing prices over a date range.

    Args:
        ticker: Stock ticker symbol (e.g., 'TCS.NS').
        start: Start date in 'YYYY-MM-DD' format.
        end: End date in 'YYYY-MM-DD' format.

    Returns:
        Dict with chart file path and summary statistics (min, max, avg close).
    """
    df = _load_data()
    stock_df = df[(df["Ticker"] == ticker) & (df["Date"] >= start) & (df["Date"] <= end)]
    if stock_df.empty:
        return {"error": f"No data for {ticker} in range {start} to {end}"}

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(stock_df["Date"], stock_df["Close"], linewidth=1.5, color="#1f77b4")
    ax.set_title(f"{ticker} Closing Price Trend ({start} to {end})", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price (INR)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    safe_ticker = ticker.replace(".", "_")
    chart_path = settings.charts_dir / f"{safe_ticker}_{start}_{end}.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    return {
        "chart_path": str(chart_path),
        "ticker": ticker,
        "start_date": start,
        "end_date": end,
        "min_close": round(float(stock_df["Close"].min()), 2),
        "max_close": round(float(stock_df["Close"].max()), 2),
        "avg_close": round(float(stock_df["Close"].mean()), 2),
        "data_points": len(stock_df),
    }


def plot_comparison(ticker1: str, ticker2: str, start: str, end: str) -> dict[str, Any]:
    """Plot and visually compare the normalised closing price trends of two stocks on the same chart.

    Normalises both series to 100 at the start date so the relative
    performance is directly comparable regardless of absolute price difference.

    Args:
        ticker1: First stock ticker (e.g., 'TCS.NS').
        ticker2: Second stock ticker (e.g., 'INFY.NS').
        start: Start date in 'YYYY-MM-DD' format.
        end: End date in 'YYYY-MM-DD' format.

    Returns:
        Dict with chart file path, return percentages, and winner.
    """
    df = _load_data()
    results = {}
    series = {}

    for ticker in [ticker1, ticker2]:
        stock_df = df[
            (df["Ticker"] == ticker) & (df["Date"] >= start) & (df["Date"] <= end)
        ].copy()
        if stock_df.empty:
            return {"error": f"No data for {ticker} in range {start} to {end}"}
        stock_df = stock_df.sort_values("Date")
        base = float(stock_df.iloc[0]["Close"])
        stock_df["Normalised"] = stock_df["Close"] / base * 100
        ret_pct = round((float(stock_df.iloc[-1]["Close"]) - base) / base * 100, 2)
        results[ticker] = {"return_pct": ret_pct}
        series[ticker] = stock_df

    # Build dual-line chart
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#1f77b4", "#ff7f0e"]
    for (ticker, s), color in zip(series.items(), colors):
        ax.plot(s["Date"], s["Normalised"], linewidth=2, label=ticker, color=color)

    ax.axhline(100, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(
        f"Normalised Price Comparison: {ticker1} vs {ticker2}\n({start} to {end})",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Normalised Price (Base = 100)", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    safe = f"{ticker1.replace('.','_')}_vs_{ticker2.replace('.','_')}_{start}_{end}"
    chart_path = settings.charts_dir / f"{safe}.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    r1 = results[ticker1]["return_pct"]
    r2 = results[ticker2]["return_pct"]
    winner = (
        f"{ticker1} outperformed {ticker2} by {round(r1 - r2, 2)}%"
        if r1 > r2
        else f"{ticker2} outperformed {ticker1} by {round(r2 - r1, 2)}%"
        if r2 > r1
        else "Both stocks performed equally"
    )

    return {
        "chart_path": str(chart_path),
        "ticker1": ticker1,
        "ticker2": ticker2,
        "ticker1_return_pct": r1,
        "ticker2_return_pct": r2,
        "winner": winner,
        "start_date": start,
        "end_date": end,
    }


def compare_multiple(tickers: list[str], start: str, end: str) -> dict[str, Any]:
    """Compare returns and plot normalised trends for 2–5 stocks on the same chart.

    Normalises all series to 100 at start date so relative performance is
    directly comparable regardless of absolute price differences. Also returns
    a ranked table of returns.

    Args:
        tickers: List of 2–5 stock ticker symbols (e.g., ['TCS.NS', 'INFY.NS', 'WIPRO.NS']).
        start: Start date in 'YYYY-MM-DD' format.
        end: End date in 'YYYY-MM-DD' format.

    Returns:
        Dict with chart_path, ranked returns for each ticker, and winner.
    """
    if len(tickers) < 2:
        return {"error": "Please provide at least 2 tickers to compare."}
    if len(tickers) > 5:
        tickers = tickers[:5]  # cap at 5 for readability

    df = _load_data()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    returns = {}
    series = {}

    for ticker in tickers:
        stock_df = df[
            (df["Ticker"] == ticker) & (df["Date"] >= start) & (df["Date"] <= end)
        ].copy()
        if stock_df.empty:
            return {"error": f"No data for {ticker} in range {start} to {end}"}
        stock_df = stock_df.sort_values("Date")
        base = float(stock_df.iloc[0]["Close"])
        stock_df["Normalised"] = stock_df["Close"] / base * 100
        ret_pct = round((float(stock_df.iloc[-1]["Close"]) - base) / base * 100, 2)
        returns[ticker] = ret_pct
        series[ticker] = stock_df

    # ── Build multi-line chart ─────────────────────────────────────
    fig, (ax_chart, ax_bar) = plt.subplots(
        1, 2, figsize=(16, 6),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    # Left: normalised trend lines
    for (ticker, s), color in zip(series.items(), colors):
        ax_chart.plot(s["Date"], s["Normalised"], linewidth=2,
                      label=f"{ticker} ({returns[ticker]:+.1f}%)", color=color)
    ax_chart.axhline(100, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_chart.set_title(
        f"Normalised Price Comparison ({start} to {end})",
        fontsize=13, fontweight="bold",
    )
    ax_chart.set_xlabel("Date", fontsize=11)
    ax_chart.set_ylabel("Normalised Price (Base = 100)", fontsize=11)
    ax_chart.legend(fontsize=10, loc="upper left")
    ax_chart.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    # Right: return % bar chart (ranked)
    ranked = sorted(returns.items(), key=lambda x: x[1], reverse=True)
    bar_labels = [t for t, _ in ranked]
    bar_values = [v for _, v in ranked]
    bar_colors = [colors[tickers.index(t)] for t in bar_labels]
    bars = ax_bar.barh(bar_labels, bar_values, color=bar_colors, edgecolor="white", height=0.5)
    ax_bar.axvline(0, color="grey", linewidth=0.8)
    ax_bar.set_title("Total Return (%)", fontsize=12, fontweight="bold")
    ax_bar.set_xlabel("Return %", fontsize=10)
    ax_bar.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    # Label each bar
    for bar, val in zip(bars, bar_values):
        ax_bar.text(
            val + (0.3 if val >= 0 else -0.3),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.1f}%",
            va="center", ha="left" if val >= 0 else "right",
            fontsize=9, fontweight="bold",
        )
    ax_bar.grid(True, axis="x", alpha=0.3)

    fig.tight_layout(pad=2)

    safe = "_vs_".join(t.replace(".", "_") for t in tickers)
    chart_path = settings.charts_dir / f"multi_{safe}_{start}_{end}.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    winner = ranked[0][0]
    loser = ranked[-1][0]
    ranked_str = ", ".join(f"{t} ({v:+.1f}%)" for t, v in ranked)

    return {
        "chart_path": str(chart_path),
        "tickers": tickers,
        "returns": returns,
        "ranked": ranked_str,
        "winner": f"{winner} was the best performer",
        "loser": f"{loser} was the weakest performer",
        "start_date": start,
        "end_date": end,
    }


def list_available_tickers() -> list[str]:
    """Return all stock tickers available in the dataset."""
    df = _load_data()
    return df["Ticker"].unique().tolist()


# ── Tool registry (used by retriever and agent) ────────────────────
# Each entry has: name, description, function reference, and parameter schema
# in OpenAI function-calling format so the LLM can reason about them.

TOOL_REGISTRY: list[dict[str, Any]] = [
    {
        "name": "get_stock_price",
        "description": (
            "Fetch the latest closing price for a given Indian stock ticker. "
            "Returns the most recent price, open, high, low, volume, and date."
        ),
        "function": get_stock_price,
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol with .NS suffix (e.g., 'TCS.NS', 'RELIANCE.NS', 'INFY.NS')",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "compare_returns",
        "description": (
            "Compare the percentage returns of two Indian stocks over a specified date range. "
            "Useful for comparing performance between two companies."
        ),
        "function": compare_returns,
        "parameters": {
            "type": "object",
            "properties": {
                "ticker1": {"type": "string", "description": "First stock ticker (e.g., 'TCS.NS')"},
                "ticker2": {"type": "string", "description": "Second stock ticker (e.g., 'RELIANCE.NS')"},
                "start": {"type": "string", "description": "Start date in YYYY-MM-DD format"},
                "end": {"type": "string", "description": "End date in YYYY-MM-DD format"},
            },
            "required": ["ticker1", "ticker2", "start", "end"],
        },
    },
    {
        "name": "plot_stock_trend",
        "description": (
            "Generate a line chart showing the closing price trend of a SINGLE Indian stock "
            "over a date range. Use this only for one stock. Saves the chart as a PNG file."
        ),
        "function": plot_stock_trend,
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., 'TCS.NS')"},
                "start": {"type": "string", "description": "Start date in YYYY-MM-DD format"},
                "end": {"type": "string", "description": "End date in YYYY-MM-DD format"},
            },
            "required": ["ticker", "start", "end"],
        },
    },
    {
        "name": "plot_comparison",
        "description": (
            "Plot and visually compare TWO Indian stocks on the same chart over a date range. "
            "Use this when the user asks to compare, visualise, or show a graph of exactly two stocks. "
            "Normalises both to base 100 so relative performance is easy to read."
        ),
        "function": plot_comparison,
        "parameters": {
            "type": "object",
            "properties": {
                "ticker1": {"type": "string", "description": "First stock ticker (e.g., 'TCS.NS')"},
                "ticker2": {"type": "string", "description": "Second stock ticker (e.g., 'INFY.NS')"},
                "start": {"type": "string", "description": "Start date in YYYY-MM-DD format"},
                "end": {"type": "string", "description": "End date in YYYY-MM-DD format"},
            },
            "required": ["ticker1", "ticker2", "start", "end"],
        },
    },
    {
        "name": "compare_multiple",
        "description": (
            "Compare and visually plot 2 to 5 Indian stocks together on one chart. "
            "Use this when the user mentions THREE OR MORE stocks, or uses words like 'all', 'multiple', 'several', 'a bunch of'. "
            "Shows normalised trend lines AND a ranked return bar chart side by side."
        ),
        "function": compare_multiple,
        "parameters": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of 2–5 stock tickers (e.g., ['TCS.NS', 'INFY.NS', 'WIPRO.NS'])",
                },
                "start": {"type": "string", "description": "Start date in YYYY-MM-DD format"},
                "end": {"type": "string", "description": "End date in YYYY-MM-DD format"},
            },
            "required": ["tickers", "start", "end"],
        },
    },
]


def list_tools() -> list[dict[str, Any]]:
    """Return tool metadata for the API endpoint and retriever indexing."""
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"]["properties"],
        }
        for t in TOOL_REGISTRY
    ]
