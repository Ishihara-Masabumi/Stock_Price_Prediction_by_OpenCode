from __future__ import annotations

from typing import Any

from agents import function_tool

from .data_sources import (
    MarketDataError,
    chart,
    fmt_number,
    fmt_percent,
    moving_average,
    percent_change,
    quote_summary,
    search_symbol,
    unix_to_iso,
)
from .company_master import resolve_from_master
from .forecast_model import build_multi_horizon_forecast, build_quant_forecast
from .ir_scraper import get_ir_data
from .news_analyzer import get_geopolitical_analysis


@function_tool
def resolve_company(company_name: str) -> dict[str, Any]:
    """Resolve a company name into a Yahoo Finance ticker candidate."""
    master_matches = resolve_from_master(company_name)
    if len(master_matches) == 1:
        record = master_matches[0]
        return {
            "input": company_name,
            "ticker": record.ticker,
            "company_name": record.canonical_name,
            "exchange": record.exchange,
            "quote_type": "EQUITY",
            "resolution_status": "resolved_from_company_master",
            "source": "local company master",
        }
    if len(master_matches) > 1:
        return {
            "input": company_name,
            "resolution_status": "needs_user_confirmation",
            "candidates": [
                {
                    "ticker": record.ticker,
                    "company_name": record.canonical_name,
                    "exchange": record.exchange,
                    "source": "local company master",
                }
                for record in master_matches
            ],
        }

    try:
        candidate = search_symbol(company_name)
        return {
            "input": company_name,
            "ticker": candidate.symbol,
            "company_name": candidate.short_name,
            "exchange": candidate.exchange,
            "quote_type": candidate.quote_type,
            "resolution_status": "resolved_from_yahoo_search",
            "source": "Yahoo Finance search",
        }
    except MarketDataError as exc:
        return {"input": company_name, "error": str(exc), "source": "Yahoo Finance search"}


@function_tool
def get_financial_data(ticker: str) -> dict[str, Any]:
    """Fetch valuation, profitability, balance sheet, and profile data for a ticker."""
    try:
        summary = quote_summary(ticker)
    except MarketDataError as exc:
        return {"ticker": ticker, "error": str(exc), "source": "Yahoo Finance quoteSummary"}

    price = summary.get("price", {})
    detail = summary.get("summaryDetail", {})
    stats = summary.get("defaultKeyStatistics", {})
    financial = summary.get("financialData", {})
    profile = summary.get("assetProfile", {})

    return {
        "ticker": ticker,
        "company_name": price.get("longName") or price.get("shortName"),
        "currency": price.get("currency"),
        "sector": profile.get("sector"),
        "industry": profile.get("industry"),
        "market_cap": fmt_number(price.get("marketCap"), 0),
        "trailing_pe": fmt_number(detail.get("trailingPE")),
        "forward_pe": fmt_number(detail.get("forwardPE")),
        "price_to_book": fmt_number(stats.get("priceToBook")),
        "dividend_yield": fmt_percent(detail.get("dividendYield")),
        "profit_margin": fmt_percent(financial.get("profitMargins")),
        "operating_margin": fmt_percent(financial.get("operatingMargins")),
        "return_on_equity": fmt_percent(financial.get("returnOnEquity")),
        "revenue_growth": fmt_percent(financial.get("revenueGrowth")),
        "earnings_growth": fmt_percent(financial.get("earningsGrowth")),
        "total_debt": fmt_number(financial.get("totalDebt"), 0),
        "total_cash": fmt_number(financial.get("totalCash"), 0),
        "free_cashflow": fmt_number(financial.get("freeCashflow"), 0),
        "data_timestamp": _extract_financial_timestamp(summary),
        "source_quality": {
            "rank": 4,
            "category": "financial_data_service",
            "note": "Prototype fallback. Prefer company official IR for final earnings figures.",
        },
        "source": "Yahoo Finance quoteSummary",
    }


@function_tool
def get_price_trend(ticker: str, range_: str = "1y") -> dict[str, Any]:
    """Fetch price trend, moving averages, volume, and recent returns for a ticker."""
    try:
        payload = chart(ticker, range_=range_, interval="1d")
    except MarketDataError as exc:
        return {"ticker": ticker, "error": str(exc), "source": "Yahoo Finance chart"}

    timestamps = payload.get("timestamp") or []
    quote = (payload.get("indicators", {}).get("quote") or [{}])[0]
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []
    valid_closes = [float(value) for value in closes if isinstance(value, (int, float))]
    if not valid_closes:
        return {"ticker": ticker, "error": "No close prices returned", "source": "Yahoo Finance chart"}

    latest = valid_closes[-1]
    change_1m = percent_change(valid_closes[-22] if len(valid_closes) >= 22 else None, latest)
    change_3m = percent_change(valid_closes[-66] if len(valid_closes) >= 66 else None, latest)
    change_6m = percent_change(valid_closes[-132] if len(valid_closes) >= 132 else None, latest)
    change_1y = percent_change(valid_closes[0], latest)
    avg_volume_20d = None
    valid_volumes = [int(value) for value in volumes if isinstance(value, (int, float))]
    if len(valid_volumes) >= 20:
        avg_volume_20d = sum(valid_volumes[-20:]) / 20

    return {
        "ticker": ticker,
        "range": range_,
        "latest_date": unix_to_iso(timestamps[-1] if timestamps else None),
        "latest_close": round(latest, 4),
        "return_1m": _fmt_change(change_1m),
        "return_3m": _fmt_change(change_3m),
        "return_6m": _fmt_change(change_6m),
        "return_range": _fmt_change(change_1y),
        "ma_50": _round_optional(moving_average(valid_closes, 50)),
        "ma_200": _round_optional(moving_average(valid_closes, 200)),
        "avg_volume_20d": round(avg_volume_20d) if avg_volume_20d else None,
        "data_timestamp": unix_to_iso(timestamps[-1] if timestamps else None),
        "source_quality": {
            "rank": 4,
            "category": "financial_data_service",
            "note": "Market data fallback. Prefer exchange or licensed market data in production.",
        },
        "source": "Yahoo Finance chart",
    }


@function_tool
def get_macro_data() -> dict[str, Any]:
    """Fetch market proxies for rates, FX, oil, gold, and broad equity sentiment."""
    tickers = {
        "us_10y_yield_proxy": "^TNX",
        "usd_jpy": "JPY=X",
        "wti_crude_oil": "CL=F",
        "gold": "GC=F",
        "s_and_p_500": "^GSPC",
        "nasdaq_100": "^NDX",
        "nikkei_225": "^N225",
    }
    results: dict[str, Any] = {
        "source": "Yahoo Finance chart",
        "source_quality": {
            "rank": 4,
            "category": "financial_data_service",
            "note": "Macro market proxies. Prefer central bank, government, and exchange data where available.",
        },
        "items": {},
    }
    for name, symbol in tickers.items():
        try:
            payload = chart(symbol, range_="6mo", interval="1d")
            quote = (payload.get("indicators", {}).get("quote") or [{}])[0]
            closes = [float(value) for value in (quote.get("close") or []) if isinstance(value, (int, float))]
            timestamps = payload.get("timestamp") or []
            latest = closes[-1] if closes else None
            results["items"][name] = {
                "symbol": symbol,
                "latest_date": unix_to_iso(timestamps[-1] if timestamps else None),
                "latest": round(latest, 4) if latest is not None else None,
                "return_1m": _fmt_change(percent_change(closes[-22] if len(closes) >= 22 else None, latest)),
                "return_6m": _fmt_change(percent_change(closes[0] if closes else None, latest)),
            }
        except MarketDataError as exc:
            results["items"][name] = {"symbol": symbol, "error": str(exc)}
    return results


@function_tool
def get_quant_forecast(ticker: str) -> dict[str, Any]:
    """Return numeric 3-month forecast prices, ranges, model weights, and probabilities."""
    try:
        return build_quant_forecast(ticker)
    except MarketDataError as exc:
        return {"ticker": ticker, "error": str(exc), "source": "deterministic numeric forecast model"}


@function_tool
def get_multi_horizon_forecast(ticker: str) -> dict[str, Any]:
    """Return uncalibrated direction scores for 1, 5, 21, and 63 trading day horizons."""
    try:
        return build_multi_horizon_forecast(ticker)
    except MarketDataError as exc:
        return {"ticker": ticker, "error": str(exc), "source": "deterministic multi-horizon forecast model"}


@function_tool
def get_company_ir(ticker: str) -> dict[str, Any]:
    """Fetch IR (Investor Relations) data directly from the company's official website.
    Includes recent filings, earnings reports, and financial highlights."""
    return get_ir_data(ticker)


@function_tool
def get_geopolitical_news() -> dict[str, Any]:
    """Fetch and analyze recent geopolitical, political, and economic news that may affect stock prices.
    Covers international politics, central bank policies, trade disputes, conflicts, and commodity markets."""
    return get_geopolitical_analysis()


def _fmt_change(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value * 100:.2f}%"


def _round_optional(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _extract_financial_timestamp(summary: dict[str, Any]) -> str | None:
    calendar = summary.get("calendarEvents", {})
    candidates = [calendar.get("earningsDate"), summary.get("price", {}).get("regularMarketTime")]
    for value in candidates:
        if isinstance(value, list) and value:
            value = value[0]
        if isinstance(value, dict):
            value = value.get("raw")
        if isinstance(value, (int, float)):
            return unix_to_iso(value)
    return None
