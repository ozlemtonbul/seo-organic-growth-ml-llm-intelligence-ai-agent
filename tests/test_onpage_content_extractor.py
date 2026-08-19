from __future__ import annotations

from src.extract.onpage_content_extractor import extract_onpage_signals_from_html


def test_extracts_core_onpage_geo_signals() -> None:
    html = """
    <html>
      <head>
        <title>Çocuk Sandaletleri | Demo Store</title>
        <meta name="description" content="Çocuk sandalet modelleri ve seçim rehberi.">
        <meta property="og:site_name" content="Demo Store">
        <script type="application/ld+json">
        {
          "@context": "https://schema.org",
          "@type": "Product",
          "brand": {"@type": "Brand", "name": "Demo Store"}
        }
        </script>
      </head>
      <body>
        <h1>Çocuk Sandaletleri</h1>
        <h2>Çocuk sandaleti nasıl seçilir?</h2>
        <p>Doğru numara ve kullanım amacına göre seçim yapılmalıdır.</p>
      </body>
    </html>
    """

    row = extract_onpage_signals_from_html(
        "https://demo.example.com/sandalet",
        html,
    )

    assert row["h1"] == "Çocuk Sandaletleri"
    assert row["meta_description"].startswith("Çocuk sandalet")
    assert "Product" in row["schema_type"]
    assert row["brand"] == "Demo Store"
    assert "nasıl seçilir" in row["faq"].lower()
    assert row["content_word_count"] > 0


def test_extracts_faq_author_and_modified_date_from_jsonld() -> None:
    html = """
    <html>
      <head>
        <script type="application/ld+json">
        {
          "@context": "https://schema.org",
          "@graph": [
            {
              "@type": "Article",
              "author": {"@type": "Person", "name": "SEO Team"},
              "dateModified": "2026-08-12"
            },
            {
              "@type": "Question",
              "name": "İlk adım ayakkabısı nasıl seçilir?",
              "acceptedAnswer": {
                "@type": "Answer",
                "text": "Ayağın yapısına ve doğru numaraya göre seçilir."
              }
            }
          ]
        }
        </script>
      </head>
      <body>
        <h1>İlk Adım Rehberi</h1>
        <p>Detaylı rehber içeriği.</p>
      </body>
    </html>
    """

    row = extract_onpage_signals_from_html(
        "https://demo.example.com/ilk-adim-rehberi",
        html,
    )

    assert row["author"] == "SEO Team"
    assert row["date_modified"] == "2026-08-12"
    assert "İlk adım ayakkabısı" in row["faq"]
