from __future__ import annotations

from typing import Literal

from agents import Agent, AgentOutputSchema, WebSearchTool
from pydantic import BaseModel, Field

from .tools import (
    get_company_ir,
    get_financial_data,
    get_geopolitical_news,
    get_macro_data,
    get_multi_horizon_forecast,
    get_price_trend,
    get_quant_forecast,
    resolve_company,
)


class ForecastPrice(BaseModel):
    current_price: float
    forecast_price_base: float
    forecast_range_low: float
    forecast_range_high: float
    expected_return_pct: float
    model_version: str
    model_note: str


class ForecastInterval(BaseModel):
    level: float = Field(ge=0.0, le=1.0)
    method: Literal["historical_volatility", "scenario_range"]
    annualized_volatility: float | None = None
    horizon_trading_days: int
    z_score: float | None = None
    low: float
    high: float
    low_return_pct: float
    high_return_pct: float
    calibrated: bool
    note: str


class ConfidenceBreakdown(BaseModel):
    data_completeness: float = Field(ge=0.0, le=1.0)
    model_validation: float = Field(ge=0.0, le=1.0)
    signal_strength: float = Field(ge=0.0, le=1.0)
    overall: Literal["low", "medium", "high"]
    note: str


class Scenario(BaseModel):
    label: Literal["bull", "base", "bear"]
    scenario_weight: float = Field(ge=0.0, le=1.0)
    expected_direction: Literal["up", "flat", "down"]
    rationale: str
    probability_source: str


class ProbabilityModel(BaseModel):
    score_based_probability_up: float = Field(ge=0.0, le=1.0)
    score_based_down_10pct: float = Field(ge=0.0, le=1.0)
    calibrated: bool
    calibration_method: Literal["none", "logistic_regression", "isotonic_regression"]
    calibration_note: str
    component_weights: list["ComponentWeight"]
    component_scores: list["ComponentScore"]
    explanation: str


class DirectionForecast(BaseModel):
    horizon: Literal["next_trading_day", "5_trading_days", "21_trading_days", "63_trading_days"]
    label: str
    horizon_trading_days: int
    direction: Literal["up", "down"]
    direction_score: float = Field(ge=0.0, le=1.0)
    score_up: float = Field(ge=0.0, le=1.0)
    judgment: Literal["strong", "slightly_strong", "neutral"]
    calibrated: bool
    model_version: str
    model_type: Literal["rule_based_weighted_score"]
    expected_return_score: float
    horizon_volatility: float
    risk_adjustment: float
    primary_reasons: list[str]
    risk_flags: list["TechnicalRiskFlag"]
    score_note: str


class TechnicalRiskFlag(BaseModel):
    name: Literal["profit_taking_risk"]
    label: str
    severity: Literal["low", "medium", "high"]
    reason: str
    observed_value: str
    applies_to_horizons: list[str]
    adjustments: list["RiskAdjustment"]


class RiskAdjustment(BaseModel):
    horizon: Literal["next_trading_day", "5_trading_days", "21_trading_days", "63_trading_days"]
    value: float


class ComponentWeight(BaseModel):
    name: Literal["price_technical_model", "financial_earnings_model", "macro_model", "news_event_model"]
    weight: float = Field(ge=0.0, le=1.0)


class ComponentScore(BaseModel):
    name: Literal["price_technical_model", "financial_earnings_model", "macro_model", "news_event_model"]
    expected_return: float
    used_in_quant_model: bool
    rationale: str
    details: list[str]


class StockPriceTimestamp(BaseModel):
    trading_date: str | None = None
    market_timezone: Literal["Asia/Tokyo"]
    price_type: Literal["daily_close"]
    provider_timestamp: str | None = None


class MacroTimestamp(BaseModel):
    latest_observation: str | None = None
    market_timezone: str
    price_type: str


class DataTimestamp(BaseModel):
    stock_price: StockPriceTimestamp
    financials: str | None = None
    macro: MacroTimestamp
    news_cutoff: str | None = None


class SourceQuality(BaseModel):
    rank: int
    category: str
    description: str | None = None
    usage: str | None = None
    examples: list[str] = Field(default_factory=list)


class IRDocument(BaseModel):
    title: str
    url: str
    date: str | None = None
    category: str = "unknown"


class IRInfo(BaseModel):
    ticker: str
    company_name: str
    ir_url: str | None = None
    documents: list[IRDocument] = Field(default_factory=list)
    financial_highlights: dict[str, str] = Field(default_factory=dict)
    note: str | None = None


class GeopoliticalEvent(BaseModel):
    title: str
    source: str
    published: str | None = None
    summary: str | None = None
    relevance_score: float
    category: str


class GeopoliticalInfo(BaseModel):
    summary: str
    risk_level: str
    affected_sectors: list[str] = Field(default_factory=list)
    key_events: list[str] = Field(default_factory=list)
    items: list[GeopoliticalEvent] = Field(default_factory=list)


class StockForecast(BaseModel):
    company: str
    ticker: str
    horizon: str
    as_of: str
    overall_view: Literal["bullish", "neutral", "bearish"]
    confidence: ConfidenceBreakdown
    forecast_price: ForecastPrice
    forecast_interval: ForecastInterval
    probability_model: ProbabilityModel
    direction_forecasts: list[DirectionForecast]
    ir_data: IRInfo
    geopolitical: GeopoliticalInfo
    data_timestamp: DataTimestamp
    source_quality_ranking: list[SourceQuality]
    quantitative_inputs: list[str]
    qualitative_only_inputs: list[str]
    summary: str
    price_trend: list[str]
    financial_factors: list[str]
    macro_factors: list[str]
    news_factors: list[str]
    geopolitical_factors: list[str]
    ir_factors: list[str]
    upside_factors: list[str]
    downside_factors: list[str]
    scenarios: list[Scenario]
    data_sources: list[str]
    caveats: list[str]


INSTRUCTIONS = """
You are a careful equity research agent. The user provides a target company and forecast horizon.

Workflow:
1. Resolve the company name to a ticker.
2. Fetch financial data for the resolved ticker.
3. Fetch price trend data for the resolved ticker.
4. Fetch macro data (exchange rates, interest rates, oil, gold, indices).
5. Fetch IR (Investor Relations) data from the company's official website.
6. Fetch the numeric forecast with get_quant_forecast.
7. Fetch the multi-horizon direction forecast with get_multi_horizon_forecast.
8. Fetch geopolitical and international news analysis.
9. Use web search for recent company-specific news and broad geopolitical, election, central bank,
   diplomacy, war, regulation, commodity, and supply-chain events that could affect the stock.
10. Produce a structured forecast combining all data sources.

Rules:
- This is analysis, not financial advice.
- Do not provide buy, sell, or hold instructions.
- Ground every conclusion in the retrieved data or clearly label uncertainty.
- Prefer scenarios over point predictions.
- Do not invent or alter numeric forecast values. Copy current_price, forecast_price_base,
  forecast_range_low, forecast_range_high, expected_return_pct, forecast_interval,
  score_based_probability_up, score_based_down_10pct, scenario_weights, component_weights,
  component_scores, confidence, quantitative_inputs, qualitative_only_inputs, model_version,
  and data_timestamp from get_quant_forecast exactly. Copy direction_forecasts from
  get_multi_horizon_forecast exactly.
- LLM-generated text may explain the numeric model, but must not override its scores or prices.
- If the numeric model returns an error, say the numeric forecast is unavailable instead of estimating a price.
- Scenario weights must come from get_quant_forecast.scenario_weights, not from a neat 25/50/25 split.
- The score_based_* values are uncalibrated scores, not validated probabilities. Do not describe them
  as "the chance of being correct" or "true probability" unless calibrated is true.
- For direction_forecasts, display direction_score as an uncalibrated score, not a probability.
  If score_up >= 0.5, direction must be up; otherwise direction must be down. Do not convert the
  63-trading-day result into shorter horizons.
  Preserve primary_reasons, risk_flags, risk_adjustment, expected_return_score, and horizon_volatility.
  If a risk flag says 利益確定売り注意, your explanation must not contradict the direction score.
- Convert get_quant_forecast.component_weights and component_scores into the required output lists without changing values.
- Put the numeric model's confidence object into confidence exactly. confidence.overall is a quality label,
  not a percentage probability that the forecast is correct.
- Rank source quality in this order: company official IR; central bank/government/exchange;
  Reuters/AP-like wire services; trusted financial data services; general news/technology media.
- For earnings and guidance, prefer company official IR over news articles or financial portals.
- Mention stale, missing, conflicting, or unofficial data.
- Keep the output concise and useful for a human analyst.
- Use Japanese when the user writes Japanese; otherwise use the user's language.
- When analyzing IR data, highlight key financial metrics, recent filings, and any guidance or outlook.
- When analyzing geopolitical news, assess the potential impact on the stock price and sector.
- Consider the combined effect of IR fundamentals, macro indicators, technical analysis, and geopolitical factors.
"""


def build_agent(model: str = "gpt-5.5") -> Agent:
    return Agent(
        name="Stock Price Prediction Agent",
        instructions=INSTRUCTIONS,
        model=model,
        tools=[
            resolve_company,
            get_financial_data,
            get_price_trend,
            get_macro_data,
            get_company_ir,
            get_quant_forecast,
            get_multi_horizon_forecast,
            get_geopolitical_news,
            WebSearchTool(search_context_size="medium"),
        ],
        output_type=AgentOutputSchema(StockForecast, strict_json_schema=False),
    )
