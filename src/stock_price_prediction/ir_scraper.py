from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
DEFAULT_TIMEOUT = 15


@dataclass
class IRDocument:
    title: str
    url: str
    date: str | None = None
    category: str = "unknown"
    summary: str | None = None


@dataclass
class IRData:
    ticker: str
    company_name: str
    ir_url: str
    documents: list[IRDocument] = field(default_factory=list)
    financial_highlights: dict[str, Any] = field(default_factory=dict)
    recent_filings: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


KNOWN_IR_PAGES = {
    "7203.T": {
        "company_name": "Toyota Motor Corporation",
        "ir_url": "https://global.toyota/en/ir/",
        "domains": ["global.toyota"],
    },
    "6758.T": {
        "company_name": "Sony Group Corporation",
        "ir_url": "https://www.sony.com/en/SonyInfo/IR/",
        "domains": ["sony.com"],
    },
    "9984.T": {
        "company_name": "SoftBank Group Corp.",
        "ir_url": "https://group.softbank/ir",
        "domains": ["softbank"],
    },
    "7974.T": {
        "company_name": "Nintendo Co., Ltd.",
        "ir_url": "https://www.nintendo.co.jp/ir/",
        "domains": ["nintendo.co.jp"],
    },
    "8306.T": {
        "company_name": "Mitsubishi UFJ Financial Group",
        "ir_url": "https://www.mufg.jp/english/ir/",
        "domains": ["mufg.jp"],
    },
    "6861.T": {
        "company_name": "Keyence Corporation",
        "ir_url": "https://www.keyence.co.jp/corporate/ir/",
        "domains": ["keyence.co.jp"],
    },
    "9434.T": {
        "company_name": "SoftBank Corp.",
        "ir_url": "https://www.softbank.jp/corp/ir/",
        "domains": ["softbank.jp"],
    },
    "4063.T": {
        "company_name": "Shin-Etsu Chemical Co., Ltd.",
        "ir_url": "https://www.shinetsu.co.jp/en/ir/",
        "domains": ["shinetsu.co.jp"],
    },
    "8316.T": {
        "company_name": "Sumitomo Mitsui Financial Group",
        "ir_url": "https://www.smfg.co.jp/english/investor/",
        "domains": ["smfg.co.jp"],
    },
    "6501.T": {
        "company_name": "Hitachi, Ltd.",
        "ir_url": "https://www.hitachi.com/environment/investor/",
        "domains": ["hitachi.com"],
    },
    "7011.T": {
        "company_name": "Mitsubishi Heavy Industries, Ltd.",
        "ir_url": "https://www.mhi.com/investors",
        "domains": ["mhi.com"],
    },
}


def _fetch_html(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
    try:
        with urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
            return response.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc


def _extract_text(element: Any, max_length: int = 500) -> str | None:
    if element is None:
        return None
    text = element.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text if text else None


def _detect_document_category(title: str) -> str:
    title_lower = title.lower()
    if any(kw in title_lower for kw in ["決算", "earnings", "financial results", "quarterly"]):
        return "earnings"
    if any(kw in title_lower for kw in ["有価証券", "annual report", "securities report"]):
        return "annual_report"
    if any(kw in title_lower for kw in ["業績予想", "forecast", "guidance"]):
        return "guidance"
    if any(kw in title_lower for kw in ["配当", "dividend"]):
        return "dividend"
    if any(kw in title_lower for kw in ["自社株", "buyback", "repurchase"]):
        return "buyback"
    if any(kw in title_lower for kw in ["開示", "disclosure", "factbook"]):
        return "disclosure"
    return "other"


def scrape_ir_page(ir_url: str, ticker: str) -> IRData:
    ir_info = KNOWN_IR_PAGES.get(ticker)
    company_name = ir_info["company_name"] if ir_info else ticker

    try:
        html = _fetch_html(ir_url)
    except RuntimeError as exc:
        return IRData(
            ticker=ticker,
            company_name=company_name,
            ir_url=ir_url,
            error=str(exc),
        )

    soup = BeautifulSoup(html, "html.parser")
    documents: list[IRDocument] = []

    for link in soup.find_all("a", href=True):
        href = link.get("href", "")
        text = _extract_text(link, max_length=200)
        if not text or len(text) < 5:
            continue
        full_url = urljoin(ir_url, href)
        if not _is_relevant_ir_link(full_url, text):
            continue
        category = _detect_document_category(text)
        date_match = re.search(r"(\d{4}[./-]\d{1,2}[./-]\d{1,2})", text)
        documents.append(
            IRDocument(
                title=text,
                url=full_url,
                date=date_match.group(1) if date_match else None,
                category=category,
            )
        )

    financial_highlights = _extract_financial_highlights(soup)

    return IRData(
        ticker=ticker,
        company_name=company_name,
        ir_url=ir_url,
        documents=documents[:30],
        financial_highlights=financial_highlights,
    )


def _is_relevant_ir_link(url: str, text: str) -> bool:
    text_lower = text.lower()
    keywords = [
        "決算", "earnings", "financial", "report", "annual", "quarterly",
        "有価証券", "sec", "filing", "forecast", "guidance", "dividend",
        "配当", "ir", "investor", "factbook", "開示", "disclosure",
        "業績", "results", "presentation", "supplement", "medium",
        "長期", "コア", "esg", "サステナビリティ",
    ]
    return any(kw in text_lower for kw in keywords)


def _extract_financial_highlights(soup: BeautifulSoup) -> dict[str, Any]:
    highlights: dict[str, Any] = {}
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["th", "td"])
            if len(cells) >= 2:
                key = _extract_text(cells[0], max_length=100)
                value = _extract_text(cells[1], max_length=200)
                if key and value:
                    highlights[key] = value
        if len(highlights) >= 5:
            break
    return highlights


def get_ir_data(ticker: str) -> dict[str, Any]:
    ir_info = KNOWN_IR_PAGES.get(ticker)

    if not ir_info:
        return {
            "ticker": ticker,
            "company_name": ticker,
            "ir_url": None,
            "documents": [],
            "financial_highlights": {},
            "note": "IR page not configured. Add to KNOWN_IR_PAGES in ir_scraper.py.",
            "source_quality": {
                "rank": 1,
                "category": "company_official_ir",
                "description": "IR data from company website (not configured for this ticker)",
            },
        }

    ir_data = scrape_ir_page(ir_info["ir_url"], ticker)
    return {
        "ticker": ir_data.ticker,
        "company_name": ir_data.company_name,
        "ir_url": ir_data.ir_url,
        "documents": [
            {
                "title": doc.title,
                "url": doc.url,
                "date": doc.date,
                "category": doc.category,
            }
            for doc in ir_data.documents
        ],
        "financial_highlights": ir_data.financial_highlights,
        "error": ir_data.error,
        "source_quality": {
            "rank": 1,
            "category": "company_official_ir",
            "description": "Direct scraping from company IR website",
        },
        "source": "company_ir_website",
    }


def search_ir_page_by_ticker(ticker: str) -> str | None:
    ir_info = KNOWN_IR_PAGES.get(ticker)
    if ir_info:
        return ir_info["ir_url"]
    return None
