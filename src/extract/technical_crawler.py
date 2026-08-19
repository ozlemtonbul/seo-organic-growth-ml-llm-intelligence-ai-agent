from __future__ import annotations

import json
import time
from collections import Counter, deque
from typing import Dict, Iterable, List, Set
from urllib.parse import urldefrag, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config.logging_config import get_logger
from config.settings import SETTINGS

logger = get_logger(__name__)


def _normalize_url(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text, _ = urldefrag(text)
    parts = urlsplit(text)
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))


def _same_site(url: str, site_netloc: str) -> bool:
    try:
        return urlsplit(url).netloc.lower() == site_netloc.lower()
    except ValueError:
        return False


def _extract_schema_types(soup: BeautifulSoup) -> List[str]:
    found: Set[str] = set()

    def collect(value) -> None:
        if isinstance(value, dict):
            schema_type = value.get("@type")
            if isinstance(schema_type, str):
                found.add(schema_type)
            elif isinstance(schema_type, list):
                found.update(str(item) for item in schema_type if item)
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text(" ", strip=True)
        if not raw:
            continue
        try:
            collect(json.loads(raw))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    return sorted(found)


def _build_robots_parser(
    start_url: str,
    user_agent: str,
    request_timeout: int = 15,
) -> RobotFileParser | None:
    if not SETTINGS.crawl_respect_robots:
        return None

    parts = urlsplit(start_url)
    robots_url = urlunsplit((parts.scheme, parts.netloc, "/robots.txt", "", ""))
    parser = RobotFileParser()
    parser.set_url(robots_url)
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/plain,text/*;q=0.9,*/*;q=0.8",
    }

    try:
        response = requests.get(
            robots_url,
            headers=headers,
            timeout=request_timeout,
            allow_redirects=True,
        )
        if response.status_code in {401, 403}:
            parser.disallow_all = True
            return parser
        if 400 <= response.status_code < 500:
            parser.allow_all = True
            return parser
        response.raise_for_status()
        parser.parse(response.text.splitlines())
        logger.info("robots.txt loaded successfully from %s.", robots_url)
        return parser
    except requests.RequestException as exc:
        logger.warning(
            "robots.txt could not be fetched from %s: %s. "
            "Crawler will continue without a robots parser.",
            robots_url,
            exc,
        )
        return None

def crawl_website(
    start_url: str | None = None,
    max_pages: int | None = None,
    request_timeout: int | None = None,
    delay_seconds: float | None = None,
    user_agent: str | None = None,
    seed_urls: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Crawl same-domain HTML pages and return a technical SEO dataset.

    The crawler is intentionally conservative: it respects robots.txt when
    configured, limits page count, waits between requests and ignores external
    links and binary assets.
    """
    start_url = _normalize_url(start_url or SETTINGS.crawl_start_url or SETTINGS.gsc_site_url)
    if not start_url:
        logger.info("Technical crawler start URL is not configured. Skipping.")
        return pd.DataFrame()

    max_pages = int(max_pages or SETTINGS.crawl_max_pages)
    request_timeout = int(request_timeout or SETTINGS.crawl_request_timeout)
    delay_seconds = SETTINGS.crawl_delay_seconds if delay_seconds is None else float(delay_seconds)
    user_agent = user_agent or SETTINGS.crawl_user_agent

    site_netloc = urlsplit(start_url).netloc
    robots = _build_robots_parser(start_url, user_agent, request_timeout=request_timeout)

    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml"})

    normalized_seeds = []
    if seed_urls is not None:
        for value in seed_urls:
            candidate = _normalize_url(value)
            if candidate and _same_site(candidate, site_netloc):
                normalized_seeds.append(candidate)

    initial_urls = normalized_seeds or [start_url]
    queue = deque(initial_urls)
    queued: Set[str] = set(initial_urls)
    visited: Set[str] = set()
    rows: List[Dict[str, object]] = []
    discovered_inlinks: Counter[str] = Counter()

    logger.info("Starting technical crawl at %s (max pages: %d).", start_url, max_pages)

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        if robots is not None and not robots.can_fetch(user_agent, url):
            logger.debug("robots.txt blocked URL: %s", url)
            continue

        started = time.perf_counter()
        try:
            response = session.get(url, timeout=request_timeout, allow_redirects=True)
            elapsed = time.perf_counter() - started
        except requests.RequestException as exc:
            logger.warning("Crawler request failed for %s: %s", url, exc)
            rows.append({
                "url": url,
                "status_code": 0,
                "response_time": round(time.perf_counter() - started, 3),
                "crawl_error": str(exc),
            })
            continue

        final_url = _normalize_url(response.url)
        content_type = response.headers.get("Content-Type", "").lower()
        row: Dict[str, object] = {
            "url": url,
            "final_url": final_url,
            "status_code": int(response.status_code),
            "response_time": round(elapsed, 3),
            "content_type": content_type,
            "redirect_target": final_url if final_url and final_url != url else "",
        }

        if "text/html" not in content_type:
            rows.append(row)
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        meta_description_tag = soup.find("meta", attrs={"name": lambda x: x and x.lower() == "description"})
        meta_description = meta_description_tag.get("content", "").strip() if meta_description_tag else ""
        robots_meta_tag = soup.find("meta", attrs={"name": lambda x: x and x.lower() == "robots"})
        meta_robots = robots_meta_tag.get("content", "").strip() if robots_meta_tag else ""
        canonical_tag = soup.find("link", attrs={"rel": lambda x: x and "canonical" in [str(i).lower() for i in (x if isinstance(x, list) else [x])]})
        canonical = _normalize_url(urljoin(final_url or url, canonical_tag.get("href", ""))) if canonical_tag and canonical_tag.get("href") else ""
        h1s = [tag.get_text(" ", strip=True) for tag in soup.find_all("h1")]
        text = soup.get_text(" ", strip=True)
        word_count = len([token for token in text.split() if token])
        missing_alt = sum(1 for img in soup.find_all("img") if not str(img.get("alt", "")).strip())
        schema_types = _extract_schema_types(soup)

        internal_links: Set[str] = set()
        for anchor in soup.find_all("a", href=True):
            href = str(anchor.get("href", "")).strip()
            if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
                continue
            target = _normalize_url(urljoin(final_url or url, href))
            if not target or not _same_site(target, site_netloc):
                continue
            internal_links.add(target)
            discovered_inlinks[target] += 1
            suffix = urlsplit(target).path.lower()
            if suffix.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".pdf", ".zip", ".xml", ".css", ".js")):
                continue
            if target not in visited and target not in queued and len(queued) < max_pages * 5:
                queue.append(target)
                queued.add(target)

        indexable = "no" if "noindex" in meta_robots.lower() else "yes"
        row.update({
            "indexability": "Non-Indexable" if indexable == "no" else "Indexable",
            "meta_robots": meta_robots,
            "canonical": canonical,
            "title": title,
            "title_length": len(title),
            "meta_description": meta_description,
            "meta_description_length": len(meta_description),
            "h1": h1s[0] if h1s else "",
            "h1_count": len(h1s),
            "word_count": word_count,
            "structured_data_types": ", ".join(schema_types),
            "internal_outlinks": len(internal_links),
            "images_missing_alt_text": missing_alt,
        })
        rows.append(row)

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    result = pd.DataFrame(rows)
    if not result.empty:
        result["inlinks"] = result["url"].map(lambda value: int(discovered_inlinks.get(_normalize_url(value), 0)))

    logger.info("Technical crawl completed: %d rows.", len(result))
    return result
