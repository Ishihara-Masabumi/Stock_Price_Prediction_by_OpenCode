from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json
from datetime import datetime, timezone


USER_AGENT = "Mozilla/5.0 (compatible; StockPredictionAgent/1.0)"
DEFAULT_TIMEOUT = 15

NEWS_SOURCES = {
    "reuters": "https://www.reuters.com/arc/outboundfeeds/v3/all/rss.xml",
    "bbc_business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "nikkei": "https://www.nikkei.com/rss/",
}

GEOPOLITICAL_KEYWORDS = [
    "war", "conflict", "sanctions", "tariff", "trade war", "geopolitical",
    "election", "president", "prime minister", "central bank", "fed", "boj", "ecb",
    "interest rate", "inflation", "recession", "gdp", "unemployment",
    "oil", "crude", "energy", "commodity", "gold", "copper",
    "supply chain", "semiconductor", "chip", "ai", "artificial intelligence",
    "china", "russia", "ukraine", "taiwan", "middle east", "north korea",
    "nuclear", "missile", "military", "defense",
    "regulation", "antitrust", "compliance", "policy",
    "currency", "forex", "dollar", "yen", "euro", "yuan",
]

JAPANESE_KEYWORDS = [
    "戦争", "紛争", "制裁", "関税", "貿易戦争", "地政学",
    "選挙", "大統領", "首相", "中央銀行", "日銀", " Fed", "ECB",
    "金利", "物価", "インフレ", "不景気", "リセッション", "GDP", "失業率",
    "原油", "石油", "エネルギー", "資源", "金", "銅",
    "サプライチェーン", "半導体", "チップ", "AI", "人工知能",
    "中国", "ロシア", "ウクライナ", "台湾", "中東", "北朝鮮",
    "核", "ミサイル", "軍事", "防衛",
    "規制", "独占禁止", "ポリシー",
    "為替", "ドル", "円", "ユーロ", "人民元",
]


@dataclass
class NewsItem:
    title: str
    link: str
    source: str
    published: str | None = None
    summary: str | None = None
    relevance_score: float = 0.0
    category: str = "general"


@dataclass
class GeopoliticalAnalysis:
    items: list[NewsItem] = field(default_factory=list)
    summary: str = ""
    risk_level: str = "low"
    affected_sectors: list[str] = field(default_factory=list)
    key_events: list[str] = field(default_factory=list)
    error: str | None = None


def _fetch_rss(url: str, source_name: str) -> list[NewsItem]:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/xml, text/xml"})
    try:
        with urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
            content = response.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, TimeoutError):
        return []

    try:
        import feedparser
        feed = feedparser.parse(content)
    except Exception:
        return []

    items = []
    for entry in feed.entries[:20]:
        published = entry.get("published") or entry.get("updated") or None
        summary = entry.get("summary") or entry.get("description") or None
        if summary:
            summary = re.sub(r"<[^>]+>", "", summary)
            if len(summary) > 300:
                summary = summary[:300] + "..."
        items.append(
            NewsItem(
                title=entry.get("title", ""),
                link=entry.get("link", ""),
                source=source_name,
                published=published,
                summary=summary,
            )
        )
    return items


def _score_relevance(item: NewsItem) -> float:
    text = f"{item.title} {item.summary or ''}".lower()
    score = 0.0
    all_keywords = GEOPOLITICAL_KEYWORDS + JAPANESE_KEYWORDS
    for kw in all_keywords:
        if kw.lower() in text:
            score += 0.1
    if any(kw in text for kw in ["war", "conflict", "sanctions", "戦争", "紛争", "制裁"]):
        score += 0.3
    if any(kw in text for kw in ["fed", "boj", "ecb", "日銀", "central bank", "金利"]):
        score += 0.2
    if any(kw in text for kw in ["oil", "crude", "原油", "energy"]):
        score += 0.15
    if any(kw in text for kw in ["tariff", "trade", "関税", "貿易"]):
        score += 0.2
    return min(score, 1.0)


def _categorize_item(item: NewsItem) -> str:
    text = f"{item.title} {item.summary or ''}".lower()
    if any(kw in text for kw in ["war", "conflict", "military", "戦争", "軍事"]):
        return "conflict"
    if any(kw in text for kw in ["fed", "boj", "ecb", "interest rate", "金利", "中央銀行"]):
        return "monetary_policy"
    if any(kw in text for kw in ["oil", "crude", "energy", "原油", "エネルギー"]):
        return "commodity"
    if any(kw in text for kw in ["tariff", "trade", "sanctions", "関税", "貿易", "制裁"]):
        return "trade"
    if any(kw in text for kw in ["election", "president", "選挙", "大統領"]):
        return "political"
    if any(kw in text for kw in ["gdp", "inflation", "recession", "物価", "GDP"]):
        return "macroeconomic"
    return "general"


def fetch_geopolitical_news() -> GeopoliticalAnalysis:
    all_items: list[NewsItem] = []
    for source_name, url in NEWS_SOURCES.items():
        items = _fetch_rss(url, source_name)
        all_items.extend(items)

    for item in all_items:
        item.relevance_score = _score_relevance(item)
        item.category = _categorize_item(item)

    relevant_items = [item for item in all_items if item.relevance_score > 0.1]
    relevant_items.sort(key=lambda x: x.relevance_score, reverse=True)

    if not relevant_items:
        return GeopoliticalAnalysis(
            items=[],
            summary="No significant geopolitical events detected.",
            risk_level="low",
        )

    high_relevance = [item for item in relevant_items if item.relevance_score >= 0.4]
    risk_level = "low"
    if len(high_relevance) >= 3:
        risk_level = "high"
    elif len(high_relevance) >= 1:
        risk_level = "medium"

    categories = set(item.category for item in relevant_items)
    key_events = [item.title for item in relevant_items[:5]]

    summary_parts = [f"{len(relevant_items)} relevant news items found."]
    if high_relevance:
        summary_parts.append(f"{len(high_relevance)} high-relevance events detected.")
    summary_parts.append(f"Risk level: {risk_level}.")

    return GeopoliticalAnalysis(
        items=relevant_items[:15],
        summary=" ".join(summary_parts),
        risk_level=risk_level,
        affected_sectors=list(categories),
        key_events=key_events,
    )


def get_geopolitical_analysis() -> dict[str, Any]:
    analysis = fetch_geopolitical_news()
    return {
        "items": [
            {
                "title": item.title,
                "source": item.source,
                "published": item.published,
                "summary": item.summary,
                "relevance_score": round(item.relevance_score, 2),
                "category": item.category,
            }
            for item in analysis.items
        ],
        "summary": analysis.summary,
        "risk_level": analysis.risk_level,
        "affected_sectors": analysis.affected_sectors,
        "key_events": analysis.key_events,
        "source_quality": {
            "rank": 3,
            "category": "wire_service_and_news",
            "description": "RSS feeds from Reuters, BBC, Nikkei for geopolitical events",
        },
        "source": "rss_news_feeds",
    }
