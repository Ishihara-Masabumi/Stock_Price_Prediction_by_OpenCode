from __future__ import annotations

import math
from typing import Any

from .data_sources import (
    MarketDataError,
    chart,
    clamp,
    fmt_number,
    fmt_percent,
    moving_average,
    normal_cdf,
    percent_change,
    quote_summary,
    raw_value,
    sample_stdev,
    unix_to_iso,
    unix_to_iso_datetime,
)


MODEL_VERSION = "deterministic_weighted_v1"
MULTI_HORIZON_MODEL_VERSION = "deterministic_multi_horizon_v1"
HORIZON_TRADING_DAYS = 63
FORECAST_INTERVAL_LEVEL = 0.80
FORECAST_INTERVAL_Z = 1.2815515655446004
MARKET_TIMEZONE_TOKYO = "Asia/Tokyo"

COMPONENT_WEIGHTS = {
    "price_technical_model": 0.40,
    "financial_earnings_model": 0.25,
    "macro_model": 0.20,
    "news_event_model": 0.15,
}

HORIZON_CONFIGS = {
    "next_trading_day": {
        "label": "明日",
        "horizon_trading_days": 1,
        "technical": 0.70,
        "financial": 0.05,
        "macro": 0.15,
        "news": 0.10,
    },
    "5_trading_days": {
        "label": "来週",
        "horizon_trading_days": 5,
        "technical": 0.55,
        "financial": 0.10,
        "macro": 0.20,
        "news": 0.15,
    },
    "21_trading_days": {
        "label": "来月",
        "horizon_trading_days": 21,
        "technical": 0.35,
        "financial": 0.25,
        "macro": 0.25,
        "news": 0.15,
    },
    "63_trading_days": {
        "label": "3か月後",
        "horizon_trading_days": 63,
        "technical": 0.25,
        "financial": 0.35,
        "macro": 0.25,
        "news": 0.15,
    },
}

SOURCE_QUALITY_RANKING = [
    {"rank": 1, "category": "company_official_ir", "description": "Company official IR and securities filings"},
    {"rank": 2, "category": "central_bank_government_exchange", "description": "Central banks, government agencies, exchanges"},
    {"rank": 3, "category": "wire_service", "description": "Reuters, AP, and comparable news wires"},
    {"rank": 4, "category": "financial_data_service", "description": "Established financial data services"},
    {"rank": 5, "category": "general_news_or_tech_media", "description": "General news and technology media"},
]

OFFICIAL_IR_FINANCIALS = {
    "7203.T": {
        "revenue_guidance_growth": 0.006,
        "operating_profit_guidance_growth": -0.203,
        "net_income_guidance_growth": -0.220,
        "forward_eps": 251.25,
        "annual_dividend": 100.0,
        "source": "company_official_ir",
        "source_rank": 1,
        "source_date": "2026-05-08",
        "note": "Toyota official IR structured fallback supplied for prototype modeling.",
    }
}


def build_quant_forecast(ticker: str) -> dict[str, Any]:
    price_payload = chart(ticker, range_="1y", interval="1d")
    summary_error = None
    try:
        summary = quote_summary(ticker)
    except MarketDataError as exc:
        summary = {}
        summary_error = str(exc)
    macro_payload = _build_macro_snapshot()

    timestamps = price_payload.get("timestamp") or []
    quote = (price_payload.get("indicators", {}).get("quote") or [{}])[0]
    closes = [float(value) for value in quote.get("close", []) if isinstance(value, (int, float))]
    if len(closes) < 30:
        raise MarketDataError(f"Not enough price history to build a forecast for {ticker}")

    current_price = closes[-1]
    price_timestamp = timestamps[-1] if timestamps else None
    daily_returns = [
        math.log(closes[index] / closes[index - 1])
        for index in range(1, len(closes))
        if closes[index - 1] > 0 and closes[index] > 0
    ]
    daily_vol = sample_stdev(daily_returns[-HORIZON_TRADING_DAYS:]) or 0.02
    annualized_vol = daily_vol * math.sqrt(252)
    horizon_vol = daily_vol * math.sqrt(HORIZON_TRADING_DAYS)

    technical = _technical_component(closes)
    official_ir = OFFICIAL_IR_FINANCIALS.get(ticker)
    financial = _financial_component(summary, current_price, official_ir)
    macro = _macro_component(macro_payload)
    news = _news_component()
    component_scores = {
        "price_technical_model": technical,
        "financial_earnings_model": financial,
        "macro_model": macro,
        "news_event_model": news,
    }
    available_component_weight = sum(
        COMPONENT_WEIGHTS[name]
        for name, score in component_scores.items()
        if score.get("used_in_quant_model")
    )

    expected_return = (
        COMPONENT_WEIGHTS["price_technical_model"] * technical["expected_return"]
        + COMPONENT_WEIGHTS["financial_earnings_model"] * financial["expected_return"]
        + COMPONENT_WEIGHTS["macro_model"] * macro["expected_return"]
        + COMPONENT_WEIGHTS["news_event_model"] * news["expected_return"]
    )
    expected_return = clamp(expected_return, -0.35, 0.35)

    base_price = current_price * (1 + expected_return)
    low_price = current_price * math.exp(math.log(max(0.01, 1 + expected_return)) - FORECAST_INTERVAL_Z * horizon_vol)
    high_price = current_price * math.exp(math.log(max(0.01, 1 + expected_return)) + FORECAST_INTERVAL_Z * horizon_vol)

    score_based_probability_up = 1 - normal_cdf(0.0, expected_return, horizon_vol)
    score_based_down_10 = normal_cdf(-0.10, expected_return, horizon_vol)
    score_based_up_10 = 1 - normal_cdf(0.10, expected_return, horizon_vol)
    score_based_bear_weight = clamp(score_based_down_10, 0.05, 0.85)
    score_based_bull_weight = clamp(score_based_up_10, 0.05, 0.85)
    if score_based_bull_weight + score_based_bear_weight > 0.92:
        scale = 0.92 / (score_based_bull_weight + score_based_bear_weight)
        score_based_bull_weight *= scale
        score_based_bear_weight *= scale
    score_based_base_weight = 1 - score_based_bull_weight - score_based_bear_weight

    financial_timestamp = financial.get("source_date") or _financial_timestamp(summary)
    signal_strength = clamp(abs(expected_return) / 0.08, 0.0, 1.0)
    return {
        "ticker": ticker,
        "model_version": MODEL_VERSION,
        "model_note": (
            "Deterministic numeric model scaffold. Replace calibration constants with a separately "
            "backtested model before production use."
        ),
        "current_price": _round_price(current_price),
        "forecast_price_base": _round_price(base_price),
        "forecast_range_low": _round_price(low_price),
        "forecast_range_high": _round_price(high_price),
        "expected_return_pct": round(expected_return * 100, 2),
        "forecast_interval": {
            "level": FORECAST_INTERVAL_LEVEL,
            "method": "historical_volatility",
            "annualized_volatility": round(annualized_vol, 4),
            "horizon_trading_days": HORIZON_TRADING_DAYS,
            "z_score": round(FORECAST_INTERVAL_Z, 4),
            "low": _round_price(low_price),
            "high": _round_price(high_price),
            "low_return_pct": round(((low_price / current_price) - 1) * 100, 2),
            "high_return_pct": round(((high_price / current_price) - 1) * 100, 2),
            "calibrated": False,
            "note": "Range uses realized volatility, not a statistically validated prediction interval.",
        },
        "score_based_probability_up": round(score_based_probability_up, 4),
        "score_based_down_10pct": round(score_based_down_10, 4),
        "calibrated": False,
        "calibration_method": "none",
        "calibration_note": (
            "Scores are uncalibrated. Backtest and calibrate with logistic regression or isotonic regression "
            "before interpreting them as true probabilities."
        ),
        "scenario_weights": {
            "bull": round(score_based_bull_weight, 4),
            "base": round(score_based_base_weight, 4),
            "bear": round(score_based_bear_weight, 4),
        },
        "component_weights": COMPONENT_WEIGHTS,
        "component_scores": component_scores,
        "confidence": {
            "data_completeness": round(available_component_weight, 4),
            "model_validation": 0.0,
            "signal_strength": round(signal_strength, 4),
            "overall": "low",
            "note": "Overall is low because this scaffold is not backtested or probability-calibrated.",
        },
        "quantitative_inputs": [
            name for name, score in component_scores.items() if score.get("used_in_quant_model")
        ],
        "qualitative_only_inputs": [
            "news_events",
        ]
        + ([] if financial.get("used_in_quant_model") else ["company_financial_ir"]),
        "data_timestamp": {
            "stock_price": {
                "trading_date": unix_to_iso(price_timestamp),
                "market_timezone": MARKET_TIMEZONE_TOKYO,
                "price_type": "daily_close",
                "provider_timestamp": None,
            },
            "financials": financial_timestamp,
            "macro": {
                "latest_observation": macro_payload["latest_timestamp"],
                "market_timezone": "UTC",
                "price_type": "daily_close_or_latest_provider_value",
            },
            "news_cutoff": None,
        },
        "source_quality_ranking": SOURCE_QUALITY_RANKING,
        "preferred_sources": [
            {
                "rank": 1,
                "name": "Company official IR",
                "usage": "Use for earnings, guidance, dividends, buybacks, and management commentary.",
            },
            {
                "rank": 2,
                "name": "Central bank, government, and exchange data",
                "usage": "Use for policy rates, FX reference information, market holidays, and official statistics.",
            },
            {
                "rank": 4,
                "name": "Yahoo Finance chart and quoteSummary",
                "usage": "Used by this prototype for market prices and financial data fallback.",
            },
        ],
        "source_caveats": [
            "Financial figures from Yahoo Finance are a fallback and should be superseded by company official IR.",
            "News event probability is neutral until a backtested event model or curated news score is connected.",
            "Macro series use market proxies where official central bank/government series are not yet integrated.",
        ]
        + ([f"Financial fallback unavailable: {summary_error}"] if summary_error else []),
    }


def build_multi_horizon_forecast(ticker: str) -> dict[str, Any]:
    price_payload = chart(ticker, range_="1y", interval="1d")
    summary_error = None
    try:
        summary = quote_summary(ticker)
    except MarketDataError as exc:
        summary = {}
        summary_error = str(exc)
    macro_payload = _build_macro_snapshot()

    timestamps = price_payload.get("timestamp") or []
    quote = (price_payload.get("indicators", {}).get("quote") or [{}])[0]
    closes = [float(value) for value in quote.get("close", []) if isinstance(value, (int, float))]
    if len(closes) < 30:
        raise MarketDataError(f"Not enough price history to build multi-horizon forecasts for {ticker}")

    current_price = closes[-1]
    price_timestamp = timestamps[-1] if timestamps else None
    official_ir = OFFICIAL_IR_FINANCIALS.get(ticker)
    financial = _financial_component(summary, current_price, official_ir)
    macro = _macro_component(macro_payload)
    news = _news_component()
    technical_risk_flags = _technical_risk_flags(macro_payload)
    forecasts = []

    for horizon, config in HORIZON_CONFIGS.items():
        horizon_days = int(config["horizon_trading_days"])
        technical = _technical_component_for_horizon(closes, horizon_days)
        risk_adjustment = _risk_adjustment_for_horizon(technical_risk_flags, horizon)
        expected_return = (
            config["technical"] * technical["expected_return"]
            + config["financial"] * financial["expected_return"]
            + config["macro"] * macro["expected_return"]
            + config["news"] * news["expected_return"]
            + risk_adjustment
        )
        expected_return = clamp(expected_return, -0.25, 0.25)
        horizon_vol = _horizon_volatility(closes, horizon_days)
        score_up = 1 - normal_cdf(0.0, expected_return, horizon_vol)
        direction = "up" if score_up >= 0.5 else "down"
        direction_score = score_up if direction == "up" else 1 - score_up
        primary_reasons = _direction_reasons(direction, technical, financial, macro, risk_adjustment)
        applied_risks = [
            flag
            for flag in technical_risk_flags
            if horizon in flag["applies_to_horizons"]
        ]
        forecasts.append(
            {
                "horizon": horizon,
                "label": config["label"],
                "horizon_trading_days": horizon_days,
                "direction": direction,
                "direction_score": round(direction_score, 4),
                "score_up": round(score_up, 4),
                "judgment": _score_judgment(direction_score),
                "calibrated": False,
                "model_version": MULTI_HORIZON_MODEL_VERSION,
                "model_type": "rule_based_weighted_score",
                "expected_return_score": round(expected_return, 4),
                "horizon_volatility": round(horizon_vol, 4),
                "risk_adjustment": round(risk_adjustment, 4),
                "primary_reasons": primary_reasons,
                "risk_flags": applied_risks,
                "weights": {
                    "technical": config["technical"],
                    "financial": config["financial"],
                    "macro": config["macro"],
                    "news": config["news"],
                },
                "score_note": "Uncalibrated direction score, not a true probability.",
            }
        )

    financial_timestamp = financial.get("source_date") or _financial_timestamp(summary)
    return {
        "ticker": ticker,
        "current_price": _round_price(current_price),
        "as_of": unix_to_iso(price_timestamp),
        "data_timestamp": {
            "stock_price": {
                "trading_date": unix_to_iso(price_timestamp),
                "market_timezone": MARKET_TIMEZONE_TOKYO,
                "price_type": "daily_close",
                "provider_timestamp": None,
            },
            "financials": financial_timestamp,
            "macro": {
                "latest_observation": macro_payload["latest_timestamp"],
                "market_timezone": "UTC",
                "price_type": "daily_close_or_latest_provider_value",
            },
            "news_cutoff": None,
        },
        "direction_forecasts": forecasts,
        "technical_risk_flags": technical_risk_flags,
        "calibrated": False,
        "calibration_note": (
            "Each horizon uses separate hand-set weights. Scores are not calibrated probabilities "
            "until backtested and calibrated per horizon."
        ),
        "source_caveats": [
            "The four horizons are separate rule-based scorers, not scaled versions of the 3-month model.",
            "News score is neutral until a backtested event model is connected.",
        ]
        + ([f"Financial fallback unavailable: {summary_error}"] if summary_error else []),
    }


def _technical_component(closes: list[float]) -> dict[str, Any]:
    latest = closes[-1]
    ret_1m = percent_change(closes[-22] if len(closes) >= 22 else None, latest) or 0.0
    ret_3m = percent_change(closes[-66] if len(closes) >= 66 else None, latest) or 0.0
    ma_50 = moving_average(closes, 50)
    ma_200 = moving_average(closes, 200)
    ma_signal = 0.0
    if ma_50 and ma_200 and ma_200 > 0:
        ma_signal = clamp((ma_50 / ma_200) - 1, -0.15, 0.15)

    expected_return = clamp(0.35 * ret_1m + 0.25 * ret_3m + 0.40 * ma_signal, -0.18, 0.18)
    return {
        "expected_return": round(expected_return, 4),
        "used_in_quant_model": True,
        "return_1m": round(ret_1m, 4),
        "return_3m": round(ret_3m, 4),
        "ma_50": _round_optional(ma_50),
        "ma_200": _round_optional(ma_200),
        "rationale": "Momentum plus 50/200-day moving-average spread.",
    }


def _technical_component_for_horizon(closes: list[float], horizon_days: int) -> dict[str, Any]:
    latest = closes[-1]
    if horizon_days <= 1:
        ret_fast = percent_change(closes[-2] if len(closes) >= 2 else None, latest) or 0.0
        ret_slow = percent_change(closes[-6] if len(closes) >= 6 else None, latest) or 0.0
        ma_fast = moving_average(closes, 5)
        ma_slow = moving_average(closes, 20)
        expected_return = clamp(0.55 * ret_fast + 0.30 * ret_slow + 0.15 * _ma_spread(ma_fast, ma_slow), -0.08, 0.08)
        rationale = "1-day scorer: same-day/weekly momentum plus 5/20-day moving-average spread."
    elif horizon_days <= 5:
        ret_fast = percent_change(closes[-6] if len(closes) >= 6 else None, latest) or 0.0
        ret_slow = percent_change(closes[-22] if len(closes) >= 22 else None, latest) or 0.0
        ma_fast = moving_average(closes, 10)
        ma_slow = moving_average(closes, 50)
        expected_return = clamp(0.45 * ret_fast + 0.35 * ret_slow + 0.20 * _ma_spread(ma_fast, ma_slow), -0.10, 0.10)
        rationale = "5-day scorer: short momentum plus 10/50-day moving-average spread."
    elif horizon_days <= 21:
        ret_fast = percent_change(closes[-22] if len(closes) >= 22 else None, latest) or 0.0
        ret_slow = percent_change(closes[-66] if len(closes) >= 66 else None, latest) or 0.0
        ma_fast = moving_average(closes, 20)
        ma_slow = moving_average(closes, 100)
        expected_return = clamp(0.40 * ret_fast + 0.30 * ret_slow + 0.30 * _ma_spread(ma_fast, ma_slow), -0.14, 0.14)
        rationale = "21-day scorer: monthly/quarterly momentum plus 20/100-day moving-average spread."
    else:
        ret_fast = percent_change(closes[-22] if len(closes) >= 22 else None, latest) or 0.0
        ret_slow = percent_change(closes[-66] if len(closes) >= 66 else None, latest) or 0.0
        ma_fast = moving_average(closes, 50)
        ma_slow = moving_average(closes, 200)
        expected_return = clamp(0.35 * ret_fast + 0.25 * ret_slow + 0.40 * _ma_spread(ma_fast, ma_slow), -0.18, 0.18)
        rationale = "63-day scorer: 1-month/3-month momentum plus 50/200-day moving-average spread."

    return {
        "expected_return": round(expected_return, 4),
        "used_in_quant_model": True,
        "return_fast": round(ret_fast, 4),
        "return_slow": round(ret_slow, 4),
        "ma_fast": _round_optional(ma_fast),
        "ma_slow": _round_optional(ma_slow),
        "rationale": rationale,
    }


def _financial_component(
    summary: dict[str, Any],
    current_price: float,
    official_ir: dict[str, Any] | None,
) -> dict[str, Any]:
    if official_ir:
        revenue_growth = official_ir["revenue_guidance_growth"]
        operating_growth = official_ir["operating_profit_guidance_growth"]
        net_income_growth = official_ir["net_income_guidance_growth"]
        annual_dividend = official_ir["annual_dividend"]
        dividend_yield = annual_dividend / current_price if current_price > 0 else None
        growth_score = 0.20 * revenue_growth + 0.35 * operating_growth + 0.35 * net_income_growth
        dividend_support = min((dividend_yield or 0.0) * 0.60, 0.035)
        expected_return = clamp(growth_score + dividend_support, -0.12, 0.08)
        return {
            "expected_return": round(expected_return, 4),
            "used_in_quant_model": True,
            "source": official_ir["source"],
            "source_rank": official_ir["source_rank"],
            "source_date": official_ir["source_date"],
            "revenue_guidance_growth": round(revenue_growth, 4),
            "operating_profit_guidance_growth": round(operating_growth, 4),
            "net_income_guidance_growth": round(net_income_growth, 4),
            "forward_eps": official_ir["forward_eps"],
            "annual_dividend": annual_dividend,
            "dividend_yield": round(dividend_yield, 4) if dividend_yield is not None else None,
            "rationale": "Official IR guidance: weak profit outlook partly offset by dividend support.",
        }

    if not summary:
        return {
            "expected_return": 0.0,
            "used_in_quant_model": False,
            "source": None,
            "source_rank": None,
            "source_date": None,
            "revenue_growth": None,
            "earnings_growth": None,
            "operating_margin": None,
            "forward_pe": None,
            "price_to_book": None,
            "rationale": "Neutral because financial data was unavailable.",
        }

    detail = summary.get("summaryDetail", {})
    stats = summary.get("defaultKeyStatistics", {})
    financial = summary.get("financialData", {})
    revenue_growth = _as_float(raw_value(financial.get("revenueGrowth")))
    earnings_growth = _as_float(raw_value(financial.get("earningsGrowth")))
    operating_margin = _as_float(raw_value(financial.get("operatingMargins")))
    forward_pe = _as_float(raw_value(detail.get("forwardPE")))
    price_to_book = _as_float(raw_value(stats.get("priceToBook")))

    growth_score = clamp(((revenue_growth or 0.0) + (earnings_growth or 0.0)) / 2, -0.15, 0.20)
    margin_score = clamp((operating_margin or 0.0) - 0.08, -0.08, 0.12)
    valuation_penalty = 0.0
    if forward_pe and forward_pe > 20:
        valuation_penalty -= min((forward_pe - 20) / 250, 0.08)
    if price_to_book and price_to_book > 2.5:
        valuation_penalty -= min((price_to_book - 2.5) / 50, 0.04)

    expected_return = clamp(0.45 * growth_score + 0.35 * margin_score + valuation_penalty, -0.12, 0.14)
    return {
        "expected_return": round(expected_return, 4),
        "used_in_quant_model": True,
        "source": "Yahoo Finance quoteSummary",
        "source_rank": 4,
        "source_date": _financial_timestamp(summary),
        "revenue_growth": fmt_percent(revenue_growth),
        "earnings_growth": fmt_percent(earnings_growth),
        "operating_margin": fmt_percent(operating_margin),
        "forward_pe": fmt_number(forward_pe),
        "price_to_book": fmt_number(price_to_book),
        "rationale": "Growth and margin score, reduced by valuation pressure.",
    }


def _macro_component(macro_payload: dict[str, Any]) -> dict[str, Any]:
    items = macro_payload["items"]
    usd_jpy_1m = items.get("usd_jpy", {}).get("return_1m_raw") or 0.0
    nikkei_1m = items.get("nikkei_225", {}).get("return_1m_raw") or 0.0
    oil_1m = items.get("wti_crude_oil", {}).get("return_1m_raw") or 0.0
    rates_1m = items.get("us_10y_yield_proxy", {}).get("return_1m_raw") or 0.0
    expected_return = clamp(0.25 * usd_jpy_1m + 0.35 * nikkei_1m - 0.15 * oil_1m - 0.25 * rates_1m, -0.10, 0.10)
    oil_phrase = "falling-oil tailwind" if oil_1m < 0 else "rising-oil headwind"
    return {
        "expected_return": round(expected_return, 4),
        "used_in_quant_model": True,
        "usd_jpy_1m": round(usd_jpy_1m, 4),
        "nikkei_1m": round(nikkei_1m, 4),
        "oil_1m": round(oil_1m, 4),
        "us_10y_yield_proxy_1m": round(rates_1m, 4),
        "rationale": f"FX, equity-market, and {oil_phrase}, partly offset or amplified by yield moves.",
    }


def _technical_risk_flags(macro_payload: dict[str, Any]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    nikkei = macro_payload["items"].get("nikkei_225", {})
    nikkei_latest = nikkei.get("latest")
    nikkei_1m = nikkei.get("return_1m_raw") or 0.0
    nikkei_6m = nikkei.get("return_6m_raw") or 0.0

    if (
        isinstance(nikkei_latest, (int, float))
        and nikkei_latest >= 70000
        or nikkei_1m >= 0.10
        or nikkei_6m >= 0.25
    ):
        flags.append(
            {
                "name": "profit_taking_risk",
                "label": "利益確定売り注意",
                "severity": "medium",
                "reason": (
                    "Broad Japanese equity market proxy is elevated or has risen quickly, "
                    "so short-term profit-taking pressure is explicitly deducted from direction scores."
                ),
                "observed_value": f"nikkei_latest={nikkei_latest}, return_1m={_fmt_change(nikkei_1m)}, return_6m={_fmt_change(nikkei_6m)}",
                "applies_to_horizons": ["next_trading_day", "5_trading_days", "21_trading_days"],
                "adjustments": [
                    {"horizon": "next_trading_day", "value": -0.015},
                    {"horizon": "5_trading_days", "value": -0.012},
                    {"horizon": "21_trading_days", "value": -0.006},
                    {"horizon": "63_trading_days", "value": 0.0},
                ],
            }
        )
    return flags


def _risk_adjustment_for_horizon(flags: list[dict[str, Any]], horizon: str) -> float:
    adjustment = 0.0
    for flag in flags:
        adjustments = flag.get("adjustments") or []
        for item in adjustments:
            if not isinstance(item, dict) or item.get("horizon") != horizon:
                continue
            value = item.get("value", 0.0)
            if isinstance(value, (int, float)):
                adjustment += float(value)
    return adjustment


def _direction_reasons(
    direction: str,
    technical: dict[str, Any],
    financial: dict[str, Any],
    macro: dict[str, Any],
    risk_adjustment: float,
) -> list[str]:
    reasons: list[str] = []
    if technical["expected_return"] < -0.005:
        reasons.append(f"テクニカルが弱い: expected_return={technical['expected_return']}")
    elif technical["expected_return"] > 0.005:
        reasons.append(f"テクニカルが強い: expected_return={technical['expected_return']}")

    if financial["expected_return"] < -0.005:
        reasons.append(f"財務・業績モデルが重い: expected_return={financial['expected_return']}")
    elif financial["expected_return"] > 0.005:
        reasons.append(f"財務・業績モデルが支援: expected_return={financial['expected_return']}")

    if macro["expected_return"] < -0.005:
        reasons.append(f"マクロが逆風: expected_return={macro['expected_return']}")
    elif macro["expected_return"] > 0.005:
        reasons.append(f"マクロが支援: expected_return={macro['expected_return']}")

    if risk_adjustment < 0:
        reasons.append(f"利益確定売り注意を減点: adjustment={round(risk_adjustment, 4)}")

    if not reasons:
        reasons.append("主要シグナルが拮抗しているため中立寄り")
    reasons.append(f"最終方向: {'上昇' if direction == 'up' else '下降'}")
    return reasons


def _horizon_volatility(closes: list[float], horizon_days: int) -> float:
    daily_returns = [
        math.log(closes[index] / closes[index - 1])
        for index in range(1, len(closes))
        if closes[index - 1] > 0 and closes[index] > 0
    ]
    window = max(20, min(len(daily_returns), horizon_days * 4))
    daily_vol = sample_stdev(daily_returns[-window:]) or 0.02
    return max(daily_vol * math.sqrt(horizon_days), 0.0001)


def _ma_spread(ma_fast: float | None, ma_slow: float | None) -> float:
    if ma_fast is None or ma_slow in (None, 0):
        return 0.0
    return clamp((ma_fast / ma_slow) - 1, -0.15, 0.15)


def _score_judgment(direction_score: float) -> str:
    if direction_score >= 0.70:
        return "strong"
    if direction_score >= 0.60:
        return "slightly_strong"
    return "neutral"


def _news_component() -> dict[str, Any]:
    return {
        "expected_return": 0.0,
        "used_in_quant_model": False,
        "event_score": 0.0,
        "rationale": "Neutral until a backtested news-event classifier is connected.",
    }


def _build_macro_snapshot() -> dict[str, Any]:
    tickers = {
        "us_10y_yield_proxy": "^TNX",
        "usd_jpy": "JPY=X",
        "wti_crude_oil": "CL=F",
        "gold": "GC=F",
        "s_and_p_500": "^GSPC",
        "nasdaq_100": "^NDX",
        "nikkei_225": "^N225",
    }
    items: dict[str, Any] = {}
    latest_timestamp = None
    for name, symbol in tickers.items():
        try:
            payload = chart(symbol, range_="6mo", interval="1d")
            quote = (payload.get("indicators", {}).get("quote") or [{}])[0]
            closes = [float(value) for value in (quote.get("close") or []) if isinstance(value, (int, float))]
            timestamps = payload.get("timestamp") or []
            latest = closes[-1] if closes else None
            item_timestamp = timestamps[-1] if timestamps else None
            if item_timestamp and (latest_timestamp is None or item_timestamp > latest_timestamp):
                latest_timestamp = item_timestamp
            ret_1m = percent_change(closes[-22] if len(closes) >= 22 else None, latest)
            ret_6m = percent_change(closes[0] if closes else None, latest)
            items[name] = {
                "symbol": symbol,
                "latest_date": unix_to_iso(item_timestamp),
                "latest": round(latest, 4) if latest is not None else None,
                "return_1m_raw": ret_1m,
                "return_6m_raw": ret_6m,
                "return_1m": _fmt_change(ret_1m),
                "return_6m": _fmt_change(ret_6m),
            }
        except MarketDataError as exc:
            items[name] = {"symbol": symbol, "error": str(exc)}
    return {"items": items, "latest_timestamp": unix_to_iso_datetime(latest_timestamp)}


def _financial_timestamp(summary: dict[str, Any]) -> str | None:
    financial = summary.get("financialData", {})
    calendar = summary.get("calendarEvents", {})
    for value in [
        financial.get("financialCurrency"),
        calendar.get("earningsDate"),
        summary.get("price", {}).get("regularMarketTime"),
    ]:
        if isinstance(value, list) and value:
            value = value[0]
        raw = raw_value(value)
        if isinstance(raw, (int, float)):
            return unix_to_iso(raw)
    return None


def _fmt_change(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value * 100:.2f}%"


def _round_optional(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _round_price(value: float) -> float:
    if value >= 100:
        return round(value, 1)
    return round(value, 2)


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None
