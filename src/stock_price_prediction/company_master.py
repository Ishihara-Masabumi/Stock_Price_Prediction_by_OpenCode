from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompanyMasterRecord:
    canonical_name: str
    ticker: str
    exchange: str
    aliases: tuple[str, ...]


COMPANY_MASTER = [
    CompanyMasterRecord(
        canonical_name="Toyota Motor Corporation",
        ticker="7203.T",
        exchange="Tokyo Stock Exchange",
        aliases=(
            "トヨタ",
            "トヨタ自動車",
            "toyota",
            "toyota motor",
            "toyota motor corporation",
            "TOYOTA MOTOR CORPORATION",
            "Toyota Motor",
        ),
    ),
    CompanyMasterRecord(
        canonical_name="Mitsubishi Heavy Industries, Ltd.",
        ticker="7011.T",
        exchange="Tokyo Stock Exchange",
        aliases=(
            "三菱重工",
            "三菱重工業",
            "mitsubishi heavy",
            "mitsubishi heavy industries",
            "MITSUBISHI HEAVY INDUSTRIES",
        ),
    ),
    CompanyMasterRecord(
        canonical_name="Sony Group Corporation",
        ticker="6758.T",
        exchange="Tokyo Stock Exchange",
        aliases=(
            "ソニー",
            "ソニーグループ",
            "sony",
            "sony group",
            "SONY GROUP CORPORATION",
        ),
    ),
    CompanyMasterRecord(
        canonical_name="SoftBank Group Corp.",
        ticker="9984.T",
        exchange="Tokyo Stock Exchange",
        aliases=(
            "ソフトバンクグループ",
            "ソフトバンク",
            "softbank group",
            "softbank",
        ),
    ),
    CompanyMasterRecord(
        canonical_name="Nintendo Co., Ltd.",
        ticker="7974.T",
        exchange="Tokyo Stock Exchange",
        aliases=(
            "任天堂",
            "nintendo",
            "NINTENDO CO., LTD.",
        ),
    ),
    CompanyMasterRecord(
        canonical_name="Hitachi, Ltd.",
        ticker="6501.T",
        exchange="Tokyo Stock Exchange",
        aliases=(
            "日立",
            "日立製作所",
            "hitachi",
            "HITACHI, LTD.",
        ),
    ),
    CompanyMasterRecord(
        canonical_name="Mitsubishi UFJ Financial Group",
        ticker="8306.T",
        exchange="Tokyo Stock Exchange",
        aliases=(
            "三菱UFJフィナンシャル・グループ",
            "三菱UFJ",
            "mufg",
            "MITSUBISHI UFJ FINANCIAL GROUP",
        ),
    ),
    CompanyMasterRecord(
        canonical_name="Keyence Corporation",
        ticker="6861.T",
        exchange="Tokyo Stock Exchange",
        aliases=(
            "キーエンス",
            "keyence",
            "KEYENCE CORPORATION",
        ),
    ),
    CompanyMasterRecord(
        canonical_name="Shin-Etsu Chemical Co., Ltd.",
        ticker="4063.T",
        exchange="Tokyo Stock Exchange",
        aliases=(
            "信越化学工業",
            "信越化学",
            "shin-etsu",
            "SHIN-ETSU CHEMICAL",
        ),
    ),
    CompanyMasterRecord(
        canonical_name="Sumitomo Mitsui Financial Group",
        ticker="8316.T",
        exchange="Tokyo Stock Exchange",
        aliases=(
            "三井住友フィナンシャルグループ",
            "三井住友",
            "smfg",
            "SUMITOMO MITSUI FINANCIAL GROUP",
        ),
    ),
    CompanyMasterRecord(
        canonical_name="Apple Inc.",
        ticker="AAPL",
        exchange="NASDAQ",
        aliases=(
            "アップル",
            "apple",
            "APPLE INC.",
        ),
    ),
    CompanyMasterRecord(
        canonical_name="Microsoft Corporation",
        ticker="MSFT",
        exchange="NASDAQ",
        aliases=(
            "マイクロソフト",
            "microsoft",
            "MICROSOFT CORPORATION",
        ),
    ),
    CompanyMasterRecord(
        canonical_name="NVIDIA Corporation",
        ticker="NVDA",
        exchange="NASDAQ",
        aliases=(
            "エヌビディア",
            "nvidia",
            "NVIDIA CORPORATION",
        ),
    ),
    CompanyMasterRecord(
        canonical_name="Alphabet Inc.",
        ticker="GOOGL",
        exchange="NASDAQ",
        aliases=(
            "アルファベット",
            "google",
            "alphabet",
            "ALPHABET INC.",
        ),
    ),
    CompanyMasterRecord(
        canonical_name="Amazon.com, Inc.",
        ticker="AMZN",
        exchange="NASDAQ",
        aliases=(
            "アマゾン",
            "amazon",
            "AMAZON.COM, INC.",
        ),
    ),
    CompanyMasterRecord(
        canonical_name="Tesla, Inc.",
        ticker="TSLA",
        exchange="NASDAQ",
        aliases=(
            "テスラ",
            "tesla",
            "TESLA, INC.",
        ),
    ),
]


def resolve_from_master(company_name: str) -> list[CompanyMasterRecord]:
    normalized = _normalize(company_name)
    exact_matches = [
        record
        for record in COMPANY_MASTER
        if normalized == _normalize(record.canonical_name)
        or any(normalized == _normalize(alias) for alias in record.aliases)
    ]
    if exact_matches:
        return exact_matches

    return [
        record
        for record in COMPANY_MASTER
        if normalized in _normalize(record.canonical_name)
        or any(normalized in _normalize(alias) for alias in record.aliases)
    ]


def _normalize(value: str) -> str:
    return " ".join(value.strip().casefold().split())
