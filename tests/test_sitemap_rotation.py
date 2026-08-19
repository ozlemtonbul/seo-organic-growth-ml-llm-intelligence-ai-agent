from pathlib import Path

import pandas as pd

from src.extract.sitemap_inventory import (
    build_url_inventory,
    mark_crawl_batch_completed,
    select_crawl_rotation_batch,
)


def test_url_inventory_combines_sitemap_and_gsc_sources():
    seo = pd.DataFrame({
        "page": [
            "https://example.com/a",
            "https://example.com/c",
        ]
    })
    result = build_url_inventory(
        ["https://example.com/a", "https://example.com/b"],
        seo,
    )
    assert set(result["url"]) == {
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
    }
    source = result.set_index("url").loc["https://example.com/a", "inventory_sources"]
    assert source == "gsc,sitemap"


def test_rotation_selects_never_crawled_urls_first(tmp_path: Path):
    inventory = pd.DataFrame({
        "url": [
            "https://example.com/a",
            "https://example.com/b",
            "https://example.com/c",
        ],
        "inventory_sources": ["sitemap", "sitemap", "sitemap"],
    })
    state_file = str(tmp_path / "rotation.json")
    mark_crawl_batch_completed(["https://example.com/a"], state_file)
    batch = select_crawl_rotation_batch(inventory, 2, state_file)
    assert set(batch["url"]) == {
        "https://example.com/b",
        "https://example.com/c",
    }


def test_rotation_state_moves_completed_urls_to_back(tmp_path: Path):
    inventory = pd.DataFrame({
        "url": [
            "https://example.com/a",
            "https://example.com/b",
        ],
        "inventory_sources": ["sitemap", "sitemap"],
    })
    state_file = str(tmp_path / "rotation.json")
    first = select_crawl_rotation_batch(inventory, 1, state_file)
    first_url = first.loc[0, "url"]
    mark_crawl_batch_completed([first_url], state_file)
    second = select_crawl_rotation_batch(inventory, 1, state_file)
    assert second.loc[0, "url"] != first_url
