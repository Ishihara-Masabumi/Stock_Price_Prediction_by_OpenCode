from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class MarketDataError(RuntimeError):
    """Raised when a remote market data endpoint cannot be read."""


@dataclass(frozen=True)
class QuoteCandidate:
    symbol: str
    short_name: str | None
    exchange: str | None
    quote_type: str | None


USER_AGENT = "stock-price-prediction-agent/0.1"
DEFAULT_TIMEOUT_SECONDS = 12


def get_json(url: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise MarketDataError(f"Failed to read {url}: {exc}") from exc

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise MarketDataError(f"Invalid JSON returned from {url}") from exc
    if not isinstance(data, dict):
        raise MarketDataError(f"Unexpected JSON shape returned from {url}")
    return data


def search_symbol(company_name: str) -> QuoteCandidate:
    query = quote(company_name)
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=8&newsCount=0"
    data = get_json(url)
    quotes = data.get("quotes") or []
    if not isinstance(quotes, list) or not quotes:
        raise MarketDataError(f"No ticker candidates found for {company_name!r}")

    equity_quotes = [
        quote_data
        for quote_data in quotes
        if isinstance(quote_data, dict)
        and quote_data.get("symbol")
        and quote_data.get("quoteType") in {"EQUITY", "ETF", "INDEX"}
    ]
    selected = equity_quotes[0] if equity_quotes else quotes[0]
    if not isinstance(selected, dict) or not selected.get("symbol"):
        raise MarketDataError(f"No usable ticker candidate found for {company_name!r}")

    return QuoteCandidate(
        symbol=str(selected["symbol"]),
        short_name=selected.get("shortname") or selected.get("longname"),
        exchange=selected.get("exchange"),
        quote_type=selected.get("quoteType"),
    )


def quote_summary(symbol: str) -> dict[str, Any]:
    modules = ",".join(
        [
            "price",
            "summaryDetail",
            "defaultKeyStatistics",
            "financialData",
            "assetProfile",
            "calendarEvents",
        ]
    )
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{quote(symbol)}?modules={modules}"
    data = get_json(url)
    result = (
        data.get("quoteSummary", {})
        .get("result", [{}])
    )
    if not result or not isinstance(result[0], dict):
        raise MarketDataError(f"No quote summary returned for {symbol}")
    return result[0]


def chart(symbol: str, range_: str = "1y", interval: str = "1d") -> dict[str, Any]:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol)}"
        f"?range={quote(range_)}&interval={quote(interval)}"
    )
    data = get_json(url)
    result = data.get("chart", {}).get("result", [{}])
    if not result or not isinstance(result[0], dict):
        raise MarketDataError(f"No chart data returned for {symbol}")
    return result[0]


def raw_value(value: Any) -> Any:
    if isinstance(value, dict) and "raw" in value:
        return value["raw"]
    return value


def fmt_number(value: Any, digits: int = 2) -> str | None:
    value = raw_value(value)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return f"{value:,.{digits}f}"
    return str(value)


def fmt_percent(value: Any, digits: int = 2) -> str | None:
    value = raw_value(value)
    if value is None or not isinstance(value, (int, float)):
        return None
    return f"{value * 100:.{digits}f}%"


def moving_average(values: list[float], window: int) -> float | None:
    clean_values = [value for value in values if isinstance(value, (int, float))]
    if len(clean_values) < window:
        return None
    return sum(clean_values[-window:]) / window


def percent_change(start: float | None, end: float | None) -> float | None:
    if start in (None, 0) or end is None:
        return None
    return (end - start) / start


def unix_to_iso(timestamp: int | float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()


def unix_to_iso_datetime(timestamp: int | float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def now_unix() -> int:
    return int(time.time())


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def sample_stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def normal_cdf(value: float, mean_value: float = 0.0, stdev: float = 1.0) -> float:
    if stdev <= 0:
        return 1.0 if value >= mean_value else 0.0
    z = (value - mean_value) / (stdev * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
