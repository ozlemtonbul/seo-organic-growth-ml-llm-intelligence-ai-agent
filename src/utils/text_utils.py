from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse

import numpy as np
import pandas as pd


STOPWORDS = {
    "ve",
    "veya",
    "ile",
    "için",
    "bir",
    "bu",
    "şu",
    "o",
    "da",
    "de",
    "mi",
    "mı",
    "mu",
    "mü",
    "en",
    "çok",
    "daha",
    "gibi",
    "olan",
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "are",
    "is",
}


INFORMATIONAL_SIGNALS = [
    "nedir",
    "nasil",
    "nasıl",
    "ne-zaman",
    "neden",
    "rehber",
    "blog",
    "haber",
    "what-is",
    "how-to",
    "guide",
    "tips",
    "learn",
    "tutorial",
    "why",
    "when",
]


TRANSACTIONAL_SIGNALS = [
    "satin-al",
    "satın-al",
    "fiyat",
    "indirim",
    "kampanya",
    "siparis",
    "sipariş",
    "urun",
    "ürün",
    "buy",
    "price",
    "discount",
    "sale",
    "order",
    "shop",
    "cart",
    "checkout",
]


COMMERCIAL_SIGNALS = [
    "karsilastir",
    "karşılaştır",
    "en-iyi",
    "yorumlar",
    "degerlendirme",
    "değerlendirme",
    "vs",
    "best",
    "compare",
    "review",
    "top",
    "ranked",
]


NAVIGATIONAL_SIGNALS = [
    "giris",
    "giriş",
    "hesabim",
    "hesabım",
    "login",
    "account",
    "dashboard",
    "panel",
]


PAGE_TYPE_ALIASES = {
    "kategori": "category",
    "category": "category",
    "urun": "product",
    "ürün": "product",
    "product": "product",
    "blog": "blog",
    "article": "blog",
    "haber": "blog",
    "rehber": "blog",
    "makale": "blog",
    "icerik": "blog",
    "içerik": "blog",
    "yazi": "blog",
    "yazı": "blog",
    "informational": "informational",
    "other": "other",
}


def clean_text(value: Any) -> str:
    """
    Normalize a value into a clean single-line string.
    """
    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def normalize_slug(value: str) -> str:
    """
    Convert text into a lowercase URL-friendly slug.
    """
    normalized = clean_text(value).lower()

    replacements = str.maketrans(
        "çğıöşü",
        "cgiosu",
    )

    normalized = normalized.translate(
        replacements
    )

    normalized = re.sub(
        r"[^a-z0-9]+",
        "-",
        normalized,
    )

    return normalized.strip("-")


def get_url_path(url: str) -> str:
    """
    Return the path component of a URL.
    """
    text = clean_text(url)

    try:
        return urlparse(text).path or text

    except (
        TypeError,
        ValueError,
    ):
        return text


def get_slug(url: str) -> str:
    """
    Return the final path segment of a URL.
    """
    path = get_url_path(
        url
    ).rstrip("/")

    if not path:
        return ""

    return (
        path
        .split("/")[-1]
        .lower()
    )


def humanize_slug(url: str) -> str:
    """
    Convert a URL slug into a human-readable title.
    """
    slug = get_slug(url)

    slug = re.sub(
        r"[-_]+",
        " ",
        slug,
    )

    return re.sub(
        r"\s+",
        " ",
        slug,
    ).strip().title()


def clip_text(
    text: str,
    max_length: int,
) -> str:
    """
    Clip text without breaking the final word whenever possible.
    """
    cleaned = clean_text(text)

    if max_length < 1:
        return ""

    if len(cleaned) <= max_length:
        return cleaned

    clipped = (
        cleaned[
            : max_length - 1
        ]
        .rsplit(
            " ",
            1,
        )[0]
    )

    if not clipped:
        clipped = cleaned[
            : max_length - 1
        ]

    return clipped + "…"


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Divide two pandas Series while treating zero denominators as missing.
    """
    return numerator / denominator.replace(
        0,
        np.nan,
    )


def column_or_default(
    dataframe: pd.DataFrame,
    column: str,
    default: Any = "",
) -> pd.Series:
    """
    Return an existing DataFrame column or a default-valued Series.
    """
    if column in dataframe.columns:
        return dataframe[column]

    return pd.Series(
        [default] * len(dataframe),
        index=dataframe.index,
    )


def select_first_nonempty(
    values: Iterable[Any],
    fallback: str = "",
) -> str:
    """
    Return the first non-empty cleaned value.
    """
    for value in values:
        text = clean_text(
            value
        )

        if text:
            return text

    return fallback


def extract_terms(
    text: str,
    limit: int = 8,
) -> List[str]:
    """
    Extract the most frequent meaningful terms from text.
    """
    words = re.findall(
        r"[A-Za-zÇĞİÖŞÜçğıöşü0-9]+",
        clean_text(
            text
        ).lower(),
    )

    counts: Dict[str, int] = {}

    for word in words:
        if (
            len(word) < 3
            or word in STOPWORDS
        ):
            continue

        counts[word] = (
            counts.get(
                word,
                0,
            )
            + 1
        )

    ranked = sorted(
        counts.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    return [
        term
        for term, _ in ranked[
            :limit
        ]
    ]


def classify_keyword_intent(
    text: str,
) -> str:
    """
    Classify text into a high-level search intent.
    """
    value = clean_text(
        text
    ).lower()

    if any(
        signal in value
        for signal in NAVIGATIONAL_SIGNALS
    ):
        return "Navigational"

    if any(
        signal in value
        for signal in TRANSACTIONAL_SIGNALS
    ):
        return "Transactional"

    if any(
        signal in value
        for signal in COMMERCIAL_SIGNALS
    ):
        return "Commercial"

    if any(
        signal in value
        for signal in INFORMATIONAL_SIGNALS
    ):
        return "Informational"

    return "Uncategorized"


def infer_page_type(
    row: pd.Series,
) -> str:
    """
    Infer the SEO page type from explicit metadata,
    entity metadata, URL structure and editorial-content
    signals.

    Decision priority:

    1. Explicit page type supplied by the source
    2. Product/category entity metadata
    3. Strong product/category URL structures
    4. Blog/article/editorial URL structures
    5. Content/template metadata
    6. Shallow e-commerce page fallback
    7. Other

    The function intentionally keeps keyword intent and
    page type separate. An informational query can land on
    a category page without turning that page into a blog.
    """

    # ========================================================
    # 1. EXPLICIT PAGE TYPE
    # ========================================================

    explicit_page_type = clean_text(
        row.get(
            "page_type",
            "",
        )
    ).lower()

    if (
        explicit_page_type
        in PAGE_TYPE_ALIASES
    ):
        return PAGE_TYPE_ALIASES[
            explicit_page_type
        ]

    # ========================================================
    # 2. ENTITY METADATA
    # ========================================================

    product_name = clean_text(
        row.get(
            "product_name",
            "",
        )
    )

    category_name = clean_text(
        row.get(
            "category_name",
            "",
        )
    )

    if product_name:
        return "product"

    if category_name:
        return "category"

    # ========================================================
    # 3. URL / PATH
    # ========================================================

    url = clean_text(
        row.get(
            "page",
            "",
        )
    ).lower()

    path = get_url_path(
        url
    ).lower()

    # --------------------------------------------------------
    # PRODUCT URL SIGNALS
    # --------------------------------------------------------

    product_path_signals = [
        "/urun/",
        "/urunler/",
        "/ürün/",
        "/ürünler/",
        "/product/",
        "/products/",
        "/p/",
    ]

    if any(
        signal in path
        for signal in product_path_signals
    ):
        return "product"

    # --------------------------------------------------------
    # CATEGORY URL SIGNALS
    # --------------------------------------------------------

    category_path_signals = [
        "/kategori/",
        "/kategoriler/",
        "/category/",
        "/categories/",
        "/c/",
    ]

    if any(
        signal in path
        for signal in category_path_signals
    ):
        return "category"

    # ========================================================
    # 4. BLOG / ARTICLE / EDITORIAL CONTENT SIGNALS
    # ========================================================

    content_path_signals = [
        "/blog/",
        "/blogs/",
        "/article/",
        "/articles/",
        "/rehber/",
        "/rehberler/",
        "/haber/",
        "/haberler/",
        "/makale/",
        "/makaleler/",
        "/icerik/",
        "/icerikler/",
        "/içerik/",
        "/içerikler/",
        "/yazi/",
        "/yazilar/",
        "/yazı/",
        "/yazılar/",

        "blog-",
        "-blog-",

        "article-",
        "-article-",

        "rehber-",
        "-rehber-",

        "haber-",
        "-haber-",

        "makale-",
        "-makale-",

        "icerik-",
        "-icerik-",

        "içerik-",
        "-içerik-",

        "yazi-",
        "-yazi-",

        "yazı-",
        "-yazı-",
    ]

    if any(
        signal in path
        for signal in content_path_signals
    ):
        return "blog"

    # ========================================================
    # 5. CONTENT / TEMPLATE METADATA
    # ========================================================

    content_metadata = " ".join(
        [
            clean_text(
                row.get(
                    "content_type",
                    "",
                )
            ),
            clean_text(
                row.get(
                    "template",
                    "",
                )
            ),
            clean_text(
                row.get(
                    "page_template",
                    "",
                )
            ),
            clean_text(
                row.get(
                    "template_name",
                    "",
                )
            ),
        ]
    ).lower()

    content_metadata_signals = [
        "blog",
        "article",
        "editorial",
        "haber",
        "rehber",
        "makale",
        "icerik",
        "içerik",
        "yazi",
        "yazı",
    ]

    if any(
        signal in content_metadata
        for signal in content_metadata_signals
    ):
        return "blog"

    # ========================================================
    # 6. URL DEPTH FALLBACK
    # ========================================================

    depth = len(
        [
            part
            for part in path.split("/")
            if part
        ]
    )

    # Preserve the current e-commerce behaviour:
    # homepage and shallow landing URLs are considered
    # category/landing pages unless a stronger signal above
    # identified them as product or editorial content.
    if depth <= 1:
        return "category"

    # ========================================================
    # 7. OTHER
    # ========================================================

    return "other"