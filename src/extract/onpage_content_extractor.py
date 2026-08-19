from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.parse import urlsplit

import pandas as pd
import requests
from bs4 import BeautifulSoup


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OnPageCrawlConfig:
    workers: int = 4
    request_timeout: int = 15
    user_agent: str = (
        "Mozilla/5.0 (compatible; SEOOrganicGrowthIntelligence/1.0; "
        "+https://demo.example.com/)"
    )
    max_content_chars: int = 30000


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _walk_jsonld(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_jsonld(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_jsonld(child)


def _jsonld_objects(soup: BeautifulSoup) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text(" ", strip=True)
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        objects.extend(_walk_jsonld(payload))
    return objects


def _schema_types(objects: list[dict[str, Any]]) -> list[str]:
    found: set[str] = set()
    for obj in objects:
        schema_type = obj.get("@type")
        if isinstance(schema_type, str) and schema_type.strip():
            found.add(schema_type.strip())
        elif isinstance(schema_type, list):
            for item in schema_type:
                if item:
                    found.add(str(item).strip())
    return sorted(found)


def _extract_brand(objects: list[dict[str, Any]], soup: BeautifulSoup) -> str:
    for obj in objects:
        brand = obj.get("brand")
        if isinstance(brand, dict):
            name = _clean_text(brand.get("name"))
            if name:
                return name
        if isinstance(brand, str) and brand.strip():
            return brand.strip()

    og_site_name = soup.find("meta", attrs={"property": "og:site_name"})
    if og_site_name:
        value = _clean_text(og_site_name.get("content"))
        if value:
            return value

    return ""


def _extract_author(objects: list[dict[str, Any]], soup: BeautifulSoup) -> str:
    for obj in objects:
        author = obj.get("author")
        if isinstance(author, dict):
            name = _clean_text(author.get("name"))
            if name:
                return name
        if isinstance(author, list):
            names = []
            for item in author:
                if isinstance(item, dict):
                    name = _clean_text(item.get("name"))
                    if name:
                        names.append(name)
                elif item:
                    names.append(_clean_text(item))
            if names:
                return ", ".join(names)
        if isinstance(author, str) and author.strip():
            return author.strip()

    meta = soup.find("meta", attrs={"name": lambda x: x and x.lower() == "author"})
    return _clean_text(meta.get("content")) if meta else ""


def _extract_date_modified(objects: list[dict[str, Any]], soup: BeautifulSoup) -> str:
    for obj in objects:
        value = _clean_text(obj.get("dateModified") or obj.get("datePublished"))
        if value:
            return value

    for attrs in (
        {"property": "article:modified_time"},
        {"property": "article:published_time"},
        {"name": "last-modified"},
    ):
        meta = soup.find("meta", attrs=attrs)
        if meta:
            value = _clean_text(meta.get("content"))
            if value:
                return value
    return ""


def _extract_faq(objects: list[dict[str, Any]], soup: BeautifulSoup) -> str:
    faq_pairs: list[str] = []

    for obj in objects:
        schema_type = obj.get("@type")
        types = (
            {schema_type}
            if isinstance(schema_type, str)
            else set(schema_type or [])
            if isinstance(schema_type, list)
            else set()
        )

        if "Question" in types:
            question = _clean_text(obj.get("name") or obj.get("text"))
            accepted = obj.get("acceptedAnswer")
            answer = ""
            if isinstance(accepted, dict):
                answer = _clean_text(accepted.get("text"))
            if question or answer:
                faq_pairs.append(f"{question} {answer}".strip())

    if faq_pairs:
        return " | ".join(faq_pairs)[:12000]

    # Visible FAQ-like headings are useful even without FAQPage schema.
    visible_questions = []
    for tag in soup.find_all(["h2", "h3", "h4", "strong"]):
        text = _clean_text(tag.get_text(" ", strip=True))
        low = text.lower()
        if (
            text.endswith("?")
            or any(
                token in low
                for token in (
                    "sık sorulan",
                    "sik sorulan",
                    "nasıl ",
                    "nasil ",
                    "nedir",
                    "what is",
                    "how to",
                    "frequently asked",
                )
            )
        ):
            visible_questions.append(text)

    return " | ".join(visible_questions[:30])


def extract_onpage_signals_from_html(
    url: str,
    html: str,
    status_code: int = 200,
    final_url: str | None = None,
    max_content_chars: int = 30000,
) -> dict[str, Any]:
    soup = BeautifulSoup(html or "", "html.parser")
    objects = _jsonld_objects(soup)
    schemas = _schema_types(objects)

    title = _clean_text(soup.title.get_text(" ", strip=True) if soup.title else "")

    meta_tag = soup.find(
        "meta",
        attrs={"name": lambda x: x and x.lower() == "description"},
    )
    meta_description = _clean_text(meta_tag.get("content")) if meta_tag else ""

    h1_tag = soup.find("h1")
    h1 = _clean_text(h1_tag.get_text(" ", strip=True)) if h1_tag else ""

    # Strip non-content elements before extracting visible text.
    content_soup = BeautifulSoup(html or "", "html.parser")
    for tag in content_soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    content = _clean_text(content_soup.get_text(" ", strip=True))
    content = content[:max_content_chars]
    word_count = len(content.split()) if content else 0

    return {
        "page": url,
        "final_url": final_url or url,
        "status_code": int(status_code),
        "title": title,
        "h1": h1,
        "meta_description": meta_description,
        "content": content,
        "content_word_count": int(word_count),
        "schema_type": ", ".join(schemas),
        "faq": _extract_faq(objects, soup),
        "brand": _extract_brand(objects, soup),
        "author": _extract_author(objects, soup),
        "date_modified": _extract_date_modified(objects, soup),
        "crawl_error": "",
        "onpage_crawled_at": datetime.now(timezone.utc).isoformat(),
    }


def _crawl_one(url: str, config: OnPageCrawlConfig) -> dict[str, Any]:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": config.user_agent,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.7",
        }
    )

    try:
        response = session.get(
            url,
            timeout=config.request_timeout,
            allow_redirects=True,
        )
        content_type = response.headers.get("Content-Type", "").lower()

        if "text/html" not in content_type:
            return {
                "page": url,
                "final_url": response.url,
                "status_code": int(response.status_code),
                "title": "",
                "h1": "",
                "meta_description": "",
                "content": "",
                "content_word_count": 0,
                "schema_type": "",
                "faq": "",
                "brand": "",
                "author": "",
                "date_modified": "",
                "crawl_error": f"Non-HTML content type: {content_type}",
                "onpage_crawled_at": datetime.now(timezone.utc).isoformat(),
            }

        return extract_onpage_signals_from_html(
            url=url,
            html=response.text,
            status_code=response.status_code,
            final_url=response.url,
            max_content_chars=config.max_content_chars,
        )

    except requests.RequestException as exc:
        return {
            "page": url,
            "final_url": "",
            "status_code": 0,
            "title": "",
            "h1": "",
            "meta_description": "",
            "content": "",
            "content_word_count": 0,
            "schema_type": "",
            "faq": "",
            "brand": "",
            "author": "",
            "date_modified": "",
            "crawl_error": str(exc),
            "onpage_crawled_at": datetime.now(timezone.utc).isoformat(),
        }


def crawl_onpage_urls(
    urls: Iterable[str],
    config: OnPageCrawlConfig | None = None,
) -> pd.DataFrame:
    config = config or OnPageCrawlConfig()

    clean_urls = []
    seen: set[str] = set()
    for value in urls:
        url = _clean_text(value)
        if not url or url in seen:
            continue
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            continue
        seen.add(url)
        clean_urls.append(url)

    if not clean_urls:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max(1, int(config.workers))) as executor:
        futures = {
            executor.submit(_crawl_one, url, config): url
            for url in clean_urls
        }

        total = len(futures)
        completed = 0

        for future in as_completed(futures):
            completed += 1
            url = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:  # safety net per URL
                logger.exception("Unexpected on-page extraction failure for %s", url)
                rows.append(
                    {
                        "page": url,
                        "final_url": "",
                        "status_code": 0,
                        "crawl_error": str(exc),
                        "onpage_crawled_at": datetime.now(timezone.utc).isoformat(),
                    }
                )

            if completed % 50 == 0 or completed == total:
                logger.info(
                    "On-page extraction progress: %d/%d",
                    completed,
                    total,
                )

    return pd.DataFrame(rows)
