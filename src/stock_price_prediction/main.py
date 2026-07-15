from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from agents import Runner, trace
from openai import APIError, AuthenticationError, OpenAIError, RateLimitError

from .agent import build_agent


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the stock price prediction agent.")
    parser.add_argument("company", help="Target company name, for example 'トヨタ自動車' or 'Apple'.")
    parser.add_argument("--horizon", default="今後3か月", help="Forecast horizon.")
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-5.5"),
        help="OpenAI model name. Defaults to OPENAI_MODEL or gpt-5.5.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the structured forecast as JSON.",
    )
    return parser.parse_args()


def load_dotenv(path: str | os.PathLike[str] = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> None:
    args = parse_args()
    agent = build_agent(model=args.model)
    prompt = (
        f"対象会社: {args.company}\n"
        f"予測期間: {args.horizon}\n"
        "以下の情報を収集して、株価見通しを分析してください：\n"
        "1. 会社のIR（インベスター relations）情報\n"
        "2. 過去数年分の株価推移データ\n"
        "3. 経済指標（為替、金利、原油価格、金価格）\n"
        "4. 国際情勢（政治、経済、外交のニュース）\n"
        "これらの情報を総合的に判断し、1日後、1週間後、1か月後の株価を予測してください。"
    )
    try:
        with trace("stock_price_prediction", metadata={"company": args.company, "horizon": args.horizon}):
            result = Runner.run_sync(agent, prompt)
    except RateLimitError as exc:
        print(_format_rate_limit_error(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    except AuthenticationError as exc:
        print(
            "OpenAI API authentication failed. Check that OPENAI_API_KEY is set correctly.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    except APIError as exc:
        print(f"OpenAI API request failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except OpenAIError as exc:
        print(f"OpenAI SDK error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    output = result.final_output
    if args.json and hasattr(output, "model_dump"):
        print(json.dumps(output.model_dump(), ensure_ascii=False, indent=2))
    elif hasattr(output, "direction_forecasts"):
        _print_human_readable_output(output)
    else:
        print(output)


def _format_rate_limit_error(exc: RateLimitError) -> str:
    error = getattr(exc, "body", None)
    code = None
    if isinstance(error, dict):
        code = error.get("code")
        nested = error.get("error")
        if isinstance(nested, dict):
            code = nested.get("code") or code

    if code == "insufficient_quota":
        return (
            "OpenAI API quota is insufficient for this key. "
            "Check your OpenAI billing, usage limits, or switch OPENAI_API_KEY to a key with available quota."
        )
    return f"OpenAI API rate limit error: {exc}"


def _print_human_readable_output(output: object) -> None:
    direction_ja = {
        "up": "上昇",
        "down": "下降",
    }
    judgment_ja = {
        "strong": "強い",
        "slightly_strong": "やや強い",
        "neutral": "中立",
    }

    company = getattr(output, "company", "")
    ticker = getattr(output, "ticker", "")
    forecast_price = getattr(output, "forecast_price", None)
    current_price = getattr(forecast_price, "current_price", None)
    if company and ticker:
        print(f"{company}（{ticker}）")
    if current_price is not None:
        print(f"基準株価：{current_price:,.0f}")
    print()

    ir_data = getattr(output, "ir_data", None)
    if ir_data and hasattr(ir_data, "ir_url") and ir_data.ir_url:
        print(f"IR情報: {ir_data.ir_url}")
        ir_docs = getattr(ir_data, "documents", [])
        if ir_docs:
            print(f"  最近の開示: {len(ir_docs)}件")
        print()

    geopolitical = getattr(output, "geopolitical", None)
    if geopolitical and hasattr(geopolitical, "risk_level"):
        risk_ja = {"low": "低", "medium": "中", "high": "高"}
        print(f"国際情勢リスク: {risk_ja.get(geopolitical.risk_level, geopolitical.risk_level)}")
        key_events = getattr(geopolitical, "key_events", [])
        if key_events:
            print(f"  主要イベント: {key_events[0]}")
        print()

    for forecast in getattr(output, "direction_forecasts", []):
        direction = direction_ja.get(forecast.direction, forecast.direction)
        judgment = judgment_ja.get(forecast.judgment, forecast.judgment)
        score_pct = forecast.direction_score * 100
        print(f"{forecast.label}の株価予測：{direction}スコア {score_pct:.0f}%　判定：{judgment}")
        reasons = getattr(forecast, "primary_reasons", [])
        if reasons:
            print(f"  理由：{' / '.join(reasons[:3])}")
        risk_flags = getattr(forecast, "risk_flags", [])
        if risk_flags:
            labels = [flag.label for flag in risk_flags]
            print(f"  注意：{' / '.join(labels)}")

    print()
    print("※スコアは未校正モデルによる方向スコアであり、確率ではありません。")
    print("※このアプリは投資助言ではありません。分析・研究用途の参考情報です。")
    summary = getattr(output, "summary", "")
    if summary:
        print()
        print(summary)


if __name__ == "__main__":
    main()
