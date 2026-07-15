from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data_sources import chart, quote_summary, moving_average, percent_change, fmt_number, fmt_percent, raw_value, unix_to_iso, MarketDataError
from .forecast_model import build_quant_forecast, build_multi_horizon_forecast
from .ir_scraper import get_ir_data
from .news_analyzer import get_geopolitical_analysis
from .company_master import resolve_from_master
from .llm_analyzer import analyze_stock


def collect_all_data(company_name: str) -> dict:
    master_matches = resolve_from_master(company_name)
    if master_matches:
        ticker = master_matches[0].ticker
        canonical = master_matches[0].canonical_name
    else:
        print(f"Error: Company '{company_name}' not found in master.", file=sys.stderr)
        sys.exit(1)

    print(f"=== {canonical} ({ticker}) ===\n", file=sys.stderr)

    print("[1/5] Fetching price trend data...", file=sys.stderr)
    try:
        price_payload = chart(ticker, range_="1y", interval="1d")
        timestamps = price_payload.get("timestamp") or []
        quote_data = (price_payload.get("indicators", {}).get("quote") or [{}])[0]
        closes = [float(v) for v in quote_data.get("close", []) if isinstance(v, (int, float))]
        volumes = [int(v) for v in quote_data.get("volume", []) if isinstance(v, (int, float))]
        
        latest_price = closes[-1] if closes else None
        change_1m = percent_change(closes[-22] if len(closes) >= 22 else None, latest_price)
        change_3m = percent_change(closes[-66] if len(closes) >= 66 else None, latest_price)
        change_6m = percent_change(closes[-132] if len(closes) >= 132 else None, latest_price)
        change_1y = percent_change(closes[0], latest_price)
        ma_50 = moving_average(closes, 50)
        ma_200 = moving_average(closes, 200)
        avg_vol = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else None
        
        price_data = {
            "latest_price": round(latest_price, 1) if latest_price else None,
            "latest_date": unix_to_iso(timestamps[-1]) if timestamps else None,
            "return_1m": round(change_1m * 100, 2) if change_1m else None,
            "return_3m": round(change_3m * 100, 2) if change_3m else None,
            "return_6m": round(change_6m * 100, 2) if change_6m else None,
            "return_1y": round(change_1y * 100, 2) if change_1y else None,
            "ma_50": round(ma_50, 1) if ma_50 else None,
            "ma_200": round(ma_200, 1) if ma_200 else None,
            "avg_volume_20d": int(avg_vol) if avg_vol else None,
        }
        print(f"  Price: {price_data['latest_price']}", file=sys.stderr)
    except MarketDataError as e:
        price_data = {"error": str(e)}
        print(f"  Error: {e}", file=sys.stderr)

    print("[2/5] Fetching financial data...", file=sys.stderr)
    try:
        summary = quote_summary(ticker)
        price_info = summary.get("price", {})
        detail = summary.get("summaryDetail", {})
        stats = summary.get("defaultKeyStatistics", {})
        financial = summary.get("financialData", {})
        profile = summary.get("assetProfile", {})
        
        fin_data = {
            "market_cap": fmt_number(price_info.get("marketCap"), 0),
            "trailing_pe": fmt_number(detail.get("trailingPE")),
            "forward_pe": fmt_number(detail.get("forwardPE")),
            "price_to_book": fmt_number(stats.get("priceToBook")),
            "dividend_yield": fmt_percent(detail.get("dividendYield")),
            "profit_margin": fmt_percent(financial.get("profitMargins")),
            "operating_margin": fmt_percent(financial.get("operatingMargins")),
            "revenue_growth": fmt_percent(financial.get("revenueGrowth")),
            "earnings_growth": fmt_percent(financial.get("earningsGrowth")),
            "sector": profile.get("sector"),
            "industry": profile.get("industry"),
        }
        print(f"  P/E: {fin_data['trailing_pe']}", file=sys.stderr)
    except MarketDataError as e:
        fin_data = {"error": str(e)}
        print(f"  Error: {e}", file=sys.stderr)

    print("[3/5] Fetching macro data...", file=sys.stderr)
    macro_tickers = {
        "usd_jpy": "JPY=X",
        "us_10y_yield": "^TNX",
        "wti_crude_oil": "CL=F",
        "gold": "GC=F",
        "s_and_p_500": "^GSPC",
        "nikkei_225": "^N225",
    }
    macro_data = {}
    for name, symbol in macro_tickers.items():
        try:
            payload = chart(symbol, range_="6mo", interval="1d")
            q = (payload.get("indicators", {}).get("quote") or [{}])[0]
            c = [float(v) for v in (q.get("close") or []) if isinstance(v, (int, float))]
            latest = c[-1] if c else None
            ret_1m = percent_change(c[-22] if len(c) >= 22 else None, latest)
            ret_6m = percent_change(c[0] if c else None, latest)
            macro_data[name] = {
                "latest": round(latest, 2) if latest else None,
                "return_1m": round(ret_1m * 100, 2) if ret_1m else None,
                "return_6m": round(ret_6m * 100, 2) if ret_6m else None,
            }
            print(f"  {name}: {macro_data[name]['latest']}", file=sys.stderr)
        except MarketDataError as e:
            macro_data[name] = {"error": str(e)}

    print("[4/5] Fetching IR data...", file=sys.stderr)
    ir_data = get_ir_data(ticker)
    ir_summary = {
        "ir_url": ir_data.get("ir_url"),
        "documents_count": len(ir_data.get("documents", [])),
        "documents": ir_data.get("documents", [])[:10],
    }
    print(f"  Documents: {ir_summary['documents_count']}", file=sys.stderr)

    print("[5/5] Fetching geopolitical news...", file=sys.stderr)
    geo_data = get_geopolitical_analysis()
    print(f"  Risk level: {geo_data.get('risk_level', 'unknown')}", file=sys.stderr)

    print("[6/6] Building forecasts...", file=sys.stderr)
    try:
        quant_forecast = build_quant_forecast(ticker)
        print(f"  Forecast (3M): {quant_forecast.get('forecast_price_base')}", file=sys.stderr)
    except Exception as e:
        quant_forecast = {"error": str(e)}
        print(f"  Error: {e}", file=sys.stderr)

    try:
        multi_forecast = build_multi_horizon_forecast(ticker)
        for f in multi_forecast.get("direction_forecasts", []):
            print(f"  {f['label']}: {f['direction']} ({f['direction_score']*100:.0f}%)", file=sys.stderr)
    except Exception as e:
        multi_forecast = {"error": str(e)}
        print(f"  Error: {e}", file=sys.stderr)

    return {
        "ticker": ticker,
        "company_name": canonical,
        "price_data": price_data,
        "financial_data": fin_data,
        "macro_data": macro_data,
        "ir_data": ir_summary,
        "geopolitical_data": geo_data,
        "quant_forecast": quant_forecast,
        "multi_horizon_forecast": multi_forecast,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock price prediction with LLM analysis")
    parser.add_argument("company", help="Company name (e.g., 'トヨタ自動車', 'Apple')")
    parser.add_argument("--json", action="store_true", help="Output raw JSON data only (no LLM analysis)")
    parser.add_argument("--model", default=None, help="LLM model name (default: OPENAI_MODEL env or gpt-4o)")
    args = parser.parse_args()

    if args.model:
        import os
        os.environ["OPENAI_MODEL"] = args.model

    data = collect_all_data(args.company)
    
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print("\n=== LLM Analysis ===\n", file=sys.stderr)
        try:
            analysis = analyze_stock(data)
            print(analysis)
        except ValueError as e:
            print(f"\nError: {e}", file=sys.stderr)
            print("\nFalling back to JSON output:\n", file=sys.stderr)
            print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
