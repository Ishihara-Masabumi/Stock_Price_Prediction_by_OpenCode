from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI


ANALYSIS_PROMPT = """あなたは経験豊富な株式アナリストです。以下のデータを基に、株価予測分析を行ってください。

## 対象企業
- 会社名: {company_name}
- ティッカー: {ticker}

## 株価データ
{price_json}

## 財務データ
{financial_json}

## 経済指標
{macro_json}

## IR情報
{ir_json}

## 国際情勢
{geo_json}

## 予測モデル結果
{forecast_json}

---

以下の観点から、1日後・1週間後・1か月後の株価予測を日本語で分析してください：

1. **テクニカル分析**: トレンド、移動平均、モメンタム
2. **ファンダメンタル分析**: 財務データ、業績予想
3. **マクロ経済**: 為替、金利、原油価格の影響
4. **国際情勢**: 地政学リスクの影響
5. **IR情報**: 決算内容、業績予想の評価

各期間について以下を出力してください：
- 方向（上昇/下降/横ばい）
- 主な理由（2-3点）
- リスク要因
- 注目ポイント

結論として、投資判断の参考となる総合評価を述べてください。
免責事項も含めてください。
"""


def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", None)
    
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Please set the environment variable: export OPENAI_API_KEY=your-key"
        )
    
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def analyze_stock(data: dict[str, Any]) -> str:
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    
    client = get_client()
    
    prompt = ANALYSIS_PROMPT.format(
        company_name=data.get("company_name", "N/A"),
        ticker=data.get("ticker", "N/A"),
        price_json=json.dumps(data.get("price_data", {}), ensure_ascii=False, indent=2),
        financial_json=json.dumps(data.get("financial_data", {}), ensure_ascii=False, indent=2),
        macro_json=json.dumps(data.get("macro_data", {}), ensure_ascii=False, indent=2),
        ir_json=json.dumps(data.get("ir_data", {}), ensure_ascii=False, indent=2),
        geo_json=json.dumps(data.get("geopolitical_data", {}), ensure_ascii=False, indent=2),
        forecast_json=json.dumps(data.get("quant_forecast", {}), ensure_ascii=False, indent=2),
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "あなたは株式アナリストです。データに基づいて客観的な分析を行います。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=2000,
    )
    
    return response.choices[0].message.content or "分析結果を取得できませんでした。"
