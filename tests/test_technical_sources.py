from __future__ import annotations

import pandas as pd

from src.extract.pagespeed_extractor import (
    enrich_crawl_with_pagespeed,
    select_pagespeed_urls,
)
from src.extract.technical_crawler import _extract_schema_types
from bs4 import BeautifulSoup


def test_extract_schema_types_reads_json_ld_types():
    soup = BeautifulSoup(
        '<html><script type="application/ld+json">'
        '{"@context":"https://schema.org","@type":["Product","Thing"]}'
        '</script></html>',
        "html.parser",
    )
    assert _extract_schema_types(soup) == ["Product", "Thing"]


def test_select_pagespeed_urls_prefers_business_opportunity():
    frame = pd.DataFrame(
        [
            {
                "page": "https://example.com/low",
                "PageOpportunityScore": 10,
                "CommerceScore": 5,
                "Revenue": 0,
                "Sessions": 10,
            },
            {
                "page": "https://example.com/high",
                "PageOpportunityScore": 95,
                "CommerceScore": 90,
                "Revenue": 10000,
                "Sessions": 1000,
            },
        ]
    )
    urls = select_pagespeed_urls(frame, max_urls=1)
    assert urls == ["https://example.com/high"]


def test_enrich_crawl_with_pagespeed_merges_by_normalized_url():
    crawl = pd.DataFrame(
        [{"url": "https://example.com/page/", "status_code": 200}]
    )
    psi = pd.DataFrame(
        [{"url": "https://example.com/page", "lcp": 3.2, "cls": 0.08}]
    )
    result = enrich_crawl_with_pagespeed(crawl, psi)
    assert len(result) == 1
    assert float(result.loc[0, "lcp"]) == 3.2
    assert float(result.loc[0, "cls"]) == 0.08
