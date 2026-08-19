from __future__ import annotations

import pandas as pd

from src.utils.text_utils import (
    classify_keyword_intent,
    clean_text,
    extract_terms,
    get_slug,
    get_url_path,
    humanize_slug,
    infer_page_type,
    normalize_slug,
    safe_divide,
    select_first_nonempty,
)


def test_clean_text_removes_extra_whitespace() -> None:
    assert clean_text("  sample   text \n value  ") == (
        "sample text value"
    )


def test_normalize_slug_supports_turkish_characters() -> None:
    assert normalize_slug(
        "Çocuk Ayakkabısı Ürünleri"
    ) == "cocuk-ayakkabisi-urunleri"


def test_get_url_path_removes_domain_and_query_string() -> None:
    assert get_url_path(
        "https://example.com/product/sample?source=organic"
    ) == "/product/sample"


def test_get_slug_returns_last_url_segment() -> None:
    assert get_slug(
        "https://example.com/category/sample-product/"
    ) == "sample-product"


def test_humanize_slug_returns_readable_text() -> None:
    assert humanize_slug(
        "https://example.com/sample-product-name"
    ) == "Sample Product Name"


def test_transactional_keyword_intent() -> None:
    assert classify_keyword_intent(
        "çocuk ayakkabısı fiyat satın al"
    ) == "Transactional"


def test_informational_keyword_intent() -> None:
    assert classify_keyword_intent(
        "çocuk ayakkabısı nasıl seçilir"
    ) == "Informational"


def test_infer_product_page_type() -> None:
    row = pd.Series(
        {
            "page": (
                "https://example.com/product/sample-product"
            )
        }
    )

    assert infer_page_type(row) == "product"


def test_extract_terms_excludes_stopwords() -> None:
    terms = extract_terms(
        "çocuk ayakkabısı ve çocuk ürünleri için "
        "ayakkabı rehberi",
        limit=5,
    )

    assert "çocuk" in terms
    assert "ve" not in terms
    assert "için" not in terms


def test_safe_divide_handles_zero_denominator() -> None:
    numerator = pd.Series(
        [10, 20]
    )

    denominator = pd.Series(
        [2, 0]
    )

    result = safe_divide(
        numerator,
        denominator,
    )

    assert result.iloc[0] == 5
    assert pd.isna(
        result.iloc[1]
    )


def test_select_first_nonempty_returns_first_value() -> None:
    assert select_first_nonempty(
        [
            "",
            None,
            "  selected value  ",
            "another value",
        ]
    ) == "selected value"