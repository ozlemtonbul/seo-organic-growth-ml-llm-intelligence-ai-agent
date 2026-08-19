import pandas as pd

from src.features.technical_seo_intelligence import build_technical_seo_intelligence


def _pages():
    return pd.DataFrame([
        {
            "page": "https://example.com/category/shoes",
            "page_type": "category",
            "CommerceScore": 90,
            "PageOpportunityScore": 80,
            "Revenue": 50000,
            "Purchases": 50,
            "AddToCarts": 150,
        }
    ])


def test_technical_engine_is_transparent_when_crawl_is_missing():
    result = build_technical_seo_intelligence(pd.DataFrame(), _pages())
    assert len(result) == 1
    assert result.loc[0, "AuditStatus"] == "Not Audited"
    assert result.loc[0, "IssueType"] == "Technical Audit Data Missing"


def test_technical_engine_detects_high_impact_crawl_issues():
    crawl = pd.DataFrame([
        {
            "address": "https://example.com/category/shoes",
            "status_code": 404,
            "indexability": "Non-Indexable",
            "title_1": "",
            "meta_description_1": "",
            "h1_1": "",
            "canonical_link_element_1": "",
            "unique_inlinks": 0,
        }
    ])
    result = build_technical_seo_intelligence(crawl, _pages())
    issue_types = set(result["IssueType"])
    assert "4xx Client Error" in issue_types
    assert "Indexability Block" in issue_types
    assert "Missing Title" in issue_types
    assert result["BusinessPriorityScore"].max() >= 65


def test_technical_engine_detects_core_web_vitals_when_available():
    crawl = pd.DataFrame([
        {
            "url": "https://example.com/category/shoes",
            "status_code": 200,
            "canonical": "https://example.com/category/shoes",
            "title": "Shoes",
            "meta_description": "Shop shoes",
            "h1": "Shoes",
            "lcp": 3.2,
            "cls": 0.18,
            "inp": 280,
        }
    ])
    result = build_technical_seo_intelligence(crawl, _pages())
    issue_types = set(result["IssueType"])
    assert {"Poor LCP", "Poor CLS", "Poor INP"}.issubset(issue_types)
