from __future__ import annotations

from typing import Dict, Iterable, List, Optional
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd


SEVERITY_WEIGHTS: Dict[str, float] = {
    "Critical": 100.0,
    "High": 80.0,
    "Medium": 55.0,
    "Low": 30.0,
    "Info": 10.0,
}


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    available = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        match = available.get(candidate.lower())
        if match is not None:
            return str(match)
    return None


def _num(value, default: float = 0.0) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return default
    return float(numeric)


def _text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _normalize_url(value: str) -> str:
    text = _text(value)
    if not text:
        return ""
    try:
        parts = urlsplit(text)
        path = parts.path or "/"
        if path != "/":
            path = path.rstrip("/")
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))
    except ValueError:
        return text.rstrip("/").lower()


def _page_context_map(page_intelligence: pd.DataFrame) -> Dict[str, dict]:
    if page_intelligence is None or page_intelligence.empty or "page" not in page_intelligence.columns:
        return {}

    context: Dict[str, dict] = {}
    for _, row in page_intelligence.drop_duplicates("page").iterrows():
        key = _normalize_url(row.get("page", ""))
        if not key:
            continue
        context[key] = {
            "page_type": _text(row.get("page_type", "other")) or "other",
            "commerce_score": _num(row.get("CommerceScore", 0)),
            "seo_opportunity_score": _num(
                row.get("PageOpportunityScore", row.get("OpportunityScore", 0))
            ),
            "revenue": _num(row.get("Revenue", 0)),
            "purchases": _num(row.get("Purchases", 0)),
            "add_to_carts": _num(row.get("AddToCarts", 0)),
        }
    return context


def _priority_score(severity: str, context: dict) -> float:
    base = SEVERITY_WEIGHTS.get(severity, 10.0)
    commerce = min(max(_num(context.get("commerce_score", 0)), 0.0), 100.0)
    seo = min(max(_num(context.get("seo_opportunity_score", 0)), 0.0), 100.0)
    return round(min(100.0, base * 0.60 + commerce * 0.22 + seo * 0.18), 2)


def _priority_label(score: float) -> str:
    if score >= 80:
        return "P1 Critical"
    if score >= 65:
        return "P2 High"
    if score >= 45:
        return "P3 Medium"
    return "P4 Low"


def _issue_row(
    page: str,
    issue_type: str,
    severity: str,
    reason: str,
    fix: str,
    validation: str,
    team: str,
    context: dict,
    source_value: str = "",
    audit_status: str = "Audited",
) -> dict:
    score = _priority_score(severity, context)
    return {
        "page": page,
        "page_type": context.get("page_type", "other"),
        "IssueType": issue_type,
        "Severity": severity,
        "AuditStatus": audit_status,
        "BusinessPriorityScore": score,
        "PriorityTier": _priority_label(score),
        "WhyItMatters": reason,
        "RecommendedFix": fix,
        "ResponsibleTeam": team,
        "ValidationMethod": validation,
        "ObservedValue": source_value,
        "CommerceScore": context.get("commerce_score", 0.0),
        "SEOOpportunityScore": context.get("seo_opportunity_score", 0.0),
        "Revenue": context.get("revenue", 0.0),
        "Purchases": context.get("purchases", 0.0),
        "AddToCarts": context.get("add_to_carts", 0.0),
    }


def _not_audited_rows(page_intelligence: pd.DataFrame) -> pd.DataFrame:
    if page_intelligence is None or page_intelligence.empty or "page" not in page_intelligence.columns:
        return pd.DataFrame(
            [
                _issue_row(
                    page="",
                    issue_type="Technical Audit Data Missing",
                    severity="Info",
                    reason="A crawl/PageSpeed source is not configured, so crawl-dependent technical SEO checks cannot be verified.",
                    fix="Configure CRAWL_INPUT_FILE with a crawler export and optionally enrich it with PageSpeed/Core Web Vitals data.",
                    validation="Run the pipeline again and confirm technical audit rows are marked Audited.",
                    team="SEO / Development",
                    context={},
                    audit_status="Not Audited",
                )
            ]
        )

    context_map = _page_context_map(page_intelligence)
    rows: List[dict] = []
    for page in page_intelligence["page"].dropna().astype(str).drop_duplicates():
        context = context_map.get(_normalize_url(page), {})
        rows.append(
            _issue_row(
                page=page,
                issue_type="Technical Audit Data Missing",
                severity="Info",
                reason="This URL has SEO/business data, but crawl-dependent technical checks have not been verified because no crawl dataset is configured.",
                fix="Add this URL to a crawler export (for example Screaming Frog/Sitebulb) and configure CRAWL_INPUT_FILE; add PageSpeed data for Core Web Vitals checks.",
                validation="Re-run the pipeline and verify indexability, canonical, metadata, headings, schema, status code, internal links and performance checks.",
                team="SEO / Development",
                context=context,
                audit_status="Not Audited",
            )
        )
    return pd.DataFrame(rows)


def build_technical_seo_intelligence(
    crawl_dataframe: pd.DataFrame,
    page_intelligence: pd.DataFrame,
) -> pd.DataFrame:
    """Build actionable technical SEO issues with commercial prioritization.

    The engine accepts flexible crawler schemas. When no crawl source is
    configured, it returns transparent ``Not Audited`` rows instead of
    inventing technical findings from GSC/GA4 data.
    """
    if crawl_dataframe is None or crawl_dataframe.empty:
        return _not_audited_rows(page_intelligence)

    crawl = crawl_dataframe.copy()
    crawl.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in crawl.columns]
    context_map = _page_context_map(page_intelligence)

    url_col = _first_existing(crawl.columns, ["address", "url", "page", "final_url", "landing_page"])
    if url_col is None:
        return _not_audited_rows(page_intelligence)

    cols = {
        "status": _first_existing(crawl.columns, ["status_code", "status", "http_status", "response_code"]),
        "indexability": _first_existing(crawl.columns, ["indexability", "indexable", "index_status"]),
        "index_status": _first_existing(crawl.columns, ["indexability_status", "robots_index", "meta_robots"]),
        "canonical": _first_existing(crawl.columns, ["canonical_link_element_1", "canonical", "canonical_url"]),
        "title": _first_existing(crawl.columns, ["title_1", "title", "page_title"]),
        "title_len": _first_existing(crawl.columns, ["title_1_length", "title_length"]),
        "meta": _first_existing(crawl.columns, ["meta_description_1", "meta_description", "description"]),
        "meta_len": _first_existing(crawl.columns, ["meta_description_1_length", "meta_description_length"]),
        "h1": _first_existing(crawl.columns, ["h1_1", "h1"]),
        "h1_count": _first_existing(crawl.columns, ["h1_count", "h1_1_count"]),
        "word_count": _first_existing(crawl.columns, ["word_count", "words", "content_word_count"]),
        "schema": _first_existing(crawl.columns, ["structured_data_types", "schema_type", "schema", "json_ld_type"]),
        "inlinks": _first_existing(crawl.columns, ["unique_inlinks", "inlinks", "internal_inlinks"]),
        "alt_missing": _first_existing(crawl.columns, ["images_missing_alt_text", "missing_alt_text", "images_without_alt"]),
        "response_time": _first_existing(crawl.columns, ["response_time", "response_time_seconds", "time_to_first_byte"]),
        "lcp": _first_existing(crawl.columns, ["lcp", "largest_contentful_paint"]),
        "cls": _first_existing(crawl.columns, ["cls", "cumulative_layout_shift"]),
        "inp": _first_existing(crawl.columns, ["inp", "interaction_to_next_paint"]),
    }

    rows: List[dict] = []

    for _, source in crawl.iterrows():
        page = _text(source.get(url_col, ""))
        if not page:
            continue
        context = context_map.get(_normalize_url(page), {})
        page_type = str(context.get("page_type", "other")).lower()

        status = _num(source.get(cols["status"], 200), 200) if cols["status"] else 200
        if status >= 500:
            rows.append(_issue_row(page, "5xx Server Error", "Critical", "Server errors prevent users and crawlers from accessing the page.", "Resolve the server/application error and return HTTP 200 for valid indexable URLs.", "Re-crawl the URL and verify HTTP 200; inspect server logs and GSC indexing.", "Development / DevOps", context, str(int(status))))
        elif status >= 400:
            rows.append(_issue_row(page, "4xx Client Error", "High", "Broken URLs waste crawl demand and can remove ranking/sales landing pages from search.", "Restore the URL when it should exist or 301 redirect it to the closest relevant live page; update internal links.", "Re-crawl and verify the final URL is 200 with no broken internal links.", "Development / SEO", context, str(int(status))))
        elif 300 <= status < 400:
            rows.append(_issue_row(page, "Redirecting Internal URL", "Medium", "Internal redirects add crawl hops and can weaken navigation efficiency.", "Update internal links to point directly to the final canonical 200 URL and remove unnecessary redirect chains.", "Re-crawl and confirm internal links resolve directly to HTTP 200.", "Development / SEO", context, str(int(status))))

        indexability = " ".join([_text(source.get(cols["indexability"], "")) if cols["indexability"] else "", _text(source.get(cols["index_status"], "")) if cols["index_status"] else ""]).lower()
        if any(token in indexability for token in ["noindex", "non-index", "not indexable", "blocked by robots", "robots blocked"]):
            rows.append(_issue_row(page, "Indexability Block", "High", "A blocked/noindex landing page cannot reliably compete for organic visibility.", "Confirm whether the page should rank. If yes, remove unintended noindex/robots blocking and ensure it returns 200 with a self-consistent canonical.", "Check robots.txt/meta robots, use GSC URL Inspection, then re-crawl.", "SEO / Development", context, indexability))

        canonical = _text(source.get(cols["canonical"], "")) if cols["canonical"] else ""
        if status == 200 and not canonical:
            rows.append(_issue_row(page, "Missing Canonical", "Medium", "Missing canonicals can make duplicate or parameterized URL consolidation less explicit.", "Add a canonical pointing to the preferred indexable version of this page.", "Re-crawl and verify one valid canonical that returns HTTP 200.", "Development / SEO", context))
        elif canonical and _normalize_url(canonical) != _normalize_url(page) and status == 200:
            rows.append(_issue_row(page, "Canonical Points Elsewhere", "Medium", "A non-self canonical may cause this URL's ranking signals to consolidate into another page.", "Verify the canonical target is intentional. If this page should rank independently, use a self-referencing canonical.", "Re-crawl, compare canonical target and confirm Google-selected canonical in GSC.", "SEO / Development", context, canonical))

        title = _text(source.get(cols["title"], "")) if cols["title"] else ""
        title_len = _num(source.get(cols["title_len"], len(title)), len(title)) if title else 0
        if cols["title"] and not title:
            rows.append(_issue_row(page, "Missing Title", "High", "The title is a primary relevance and SERP-click signal.", "Write a unique title aligned to the page's primary search intent, entity and commercial purpose.", "Re-crawl and verify one non-empty unique title; monitor GSC CTR/ranking.", "SEO / Content", context))
        elif title and title_len > 65:
            rows.append(_issue_row(page, "Title Too Long", "Low", "Long titles may truncate and dilute the primary intent in search results.", "Rewrite the title so the main entity/keyword and differentiator appear early; keep it concise.", "Re-crawl title length and monitor GSC CTR.", "SEO / Content", context, f"{title_len:.0f} chars"))

        meta = _text(source.get(cols["meta"], "")) if cols["meta"] else ""
        meta_len = _num(source.get(cols["meta_len"], len(meta)), len(meta)) if meta else 0
        if cols["meta"] and not meta:
            rows.append(_issue_row(page, "Missing Meta Description", "Medium", "A missing description reduces control over the search snippet and click messaging.", "Write a unique description that reflects intent, benefit and relevant commercial context without keyword stuffing.", "Re-crawl and monitor GSC CTR after deployment.", "SEO / Content", context))
        elif meta and meta_len > 170:
            rows.append(_issue_row(page, "Meta Description Too Long", "Low", "Overlong descriptions may truncate and hide the most persuasive message.", "Shorten the description and put the key benefit/intent early.", "Re-crawl meta length and monitor CTR.", "SEO / Content", context, f"{meta_len:.0f} chars"))

        h1 = _text(source.get(cols["h1"], "")) if cols["h1"] else ""
        h1_count = _num(source.get(cols["h1_count"], 1 if h1 else 0), 1 if h1 else 0) if cols["h1_count"] else (1 if h1 else 0)
        if cols["h1"] and not h1:
            rows.append(_issue_row(page, "Missing H1", "Medium", "A clear H1 helps users and search systems understand the primary page topic.", "Add one descriptive H1 aligned with the page's main entity and search intent.", "Re-crawl and verify a descriptive H1 is rendered.", "SEO / Content / Development", context))
        elif h1_count > 1:
            rows.append(_issue_row(page, "Multiple H1s", "Low", "Multiple competing primary headings can make page hierarchy less clear.", "Keep one primary H1 and use H2/H3 for supporting sections where appropriate.", "Re-crawl heading structure.", "Content / Development", context, f"{h1_count:.0f}"))

        if cols["word_count"]:
            words = _num(source.get(cols["word_count"], 0))
            threshold = 250 if page_type in {"category", "blog", "informational"} else 100
            if words > 0 and words < threshold:
                rows.append(_issue_row(page, "Thin Content", "Medium", "The page may not provide enough unique, useful information for its search intent or AI-answer context.", "Expand useful intent coverage, entities, product/category guidance, FAQs and differentiating information; avoid filler text.", "Re-crawl word/content coverage and review ranking/GEO signals after publication.", "SEO / Content", context, f"{words:.0f} words"))

        if cols["schema"]:
            schema = _text(source.get(cols["schema"], ""))
            if not schema and page_type in {"product", "category", "blog", "informational"}:
                rows.append(_issue_row(page, "Structured Data Missing", "Medium", "Relevant structured data helps search and answer systems interpret page entities and eligible rich-result attributes.", "Implement valid schema appropriate to the page type (for example Product, BreadcrumbList, Article where genuinely applicable). Do not add unsupported markup.", "Validate with Schema.org/Google rich-result tools and re-crawl rendered structured data.", "Development / SEO", context))

        if cols["inlinks"]:
            inlinks = _num(source.get(cols["inlinks"], 0))
            if inlinks <= 0 and _normalize_url(page).count("/") > 2:
                rows.append(_issue_row(page, "Orphan / No Internal Inlinks", "High", "Pages with no internal links are harder for users and crawlers to discover and receive little internal authority.", "Link to the page contextually from relevant category, product, navigation or editorial pages using descriptive anchors.", "Re-crawl and verify at least one relevant internal inlink; review crawl depth.", "SEO / Content / Development", context, str(int(inlinks))))

        if cols["alt_missing"]:
            missing_alt = _num(source.get(cols["alt_missing"], 0))
            if missing_alt > 0:
                rows.append(_issue_row(page, "Images Missing Alt Text", "Low", "Missing alternative text reduces accessibility and image/entity context.", "Add concise, meaningful alt text to informative images; keep decorative images empty where appropriate.", "Re-crawl image alt coverage and spot-check rendered HTML/accessibility.", "Content / Development", context, str(int(missing_alt))))

        if cols["response_time"]:
            response_time = _num(source.get(cols["response_time"], 0))
            if response_time > 2.0:
                rows.append(_issue_row(page, "Slow Server Response", "Medium", "Slow responses can hurt crawl efficiency and user experience.", "Profile backend/CDN/cache/database bottlenecks and reduce server response time.", "Re-test with crawler and PageSpeed/Lighthouse under representative conditions.", "Development / DevOps", context, f"{response_time:.2f}s"))

        if cols["lcp"]:
            lcp = _num(source.get(cols["lcp"], 0))
            if lcp > 2.5:
                rows.append(_issue_row(page, "Poor LCP", "Medium", "Largest Contentful Paint above the good threshold can harm user experience and Core Web Vitals quality.", "Optimize the LCP resource, image delivery, critical CSS, caching and server response.", "Re-test PageSpeed/CrUX and verify LCP <= 2.5s where representative field data exists.", "Development / Performance", context, f"{lcp:.2f}s"))

        if cols["cls"]:
            cls = _num(source.get(cols["cls"], 0))
            if cls > 0.1:
                rows.append(_issue_row(page, "Poor CLS", "Medium", "Layout shifts degrade usability and Core Web Vitals quality.", "Reserve dimensions for media/ads, avoid late layout injection and stabilize fonts/components.", "Re-test PageSpeed/CrUX and verify CLS <= 0.1 where representative field data exists.", "Development / Performance", context, f"{cls:.3f}"))

        if cols["inp"]:
            inp = _num(source.get(cols["inp"], 0))
            if inp > 200:
                rows.append(_issue_row(page, "Poor INP", "Medium", "Slow interaction responsiveness can hurt user experience and Core Web Vitals quality.", "Reduce long main-thread tasks, optimize JavaScript/event handlers and defer non-critical work.", "Re-test PageSpeed/CrUX and verify INP <= 200ms where representative field data exists.", "Development / Performance", context, f"{inp:.0f}ms"))

    if not rows:
        return pd.DataFrame(columns=[
            "page", "page_type", "IssueType", "Severity", "AuditStatus",
            "BusinessPriorityScore", "PriorityTier", "WhyItMatters",
            "RecommendedFix", "ResponsibleTeam", "ValidationMethod",
            "ObservedValue", "CommerceScore", "SEOOpportunityScore",
            "Revenue", "Purchases", "AddToCarts",
        ])

    result = pd.DataFrame(rows)
    return result.sort_values(
        ["BusinessPriorityScore", "Severity", "page"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
