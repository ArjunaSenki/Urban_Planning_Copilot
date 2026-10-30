from landuse_copilot.documents import (
    DocumentValidationError,
    extract_passages,
    retrieve_passages,
)
from landuse_copilot.report import render_report


def test_text_extraction_preserves_traceable_section_numbers():
    passages = extract_passages(
        "zoning.txt",
        b"Section A permits homes.\n\nSection B protects the public park.",
    )
    assert passages == [
        ("section 1", "Section A permits homes."),
        ("section 2", "Section B protects the public park."),
    ]


def test_retrieval_returns_exact_source_and_location():
    documents = [
        (
            "zoning.txt",
            [
                ("section 1", "Residential use is permitted in this district."),
                ("section 2", "Public parks must retain open green space."),
            ],
        )
    ]
    results = retrieve_passages(documents, "district residential use")
    assert len(results) == 1
    assert results[0].document == "zoning.txt"
    assert results[0].location == "section 1"
    assert results[0].text == "Residential use is permitted in this district."


def test_unsupported_documents_are_reported():
    try:
        extract_passages("scan.png", b"image")
    except DocumentValidationError as error:
        assert "PDF, TXT, and DOCX" in str(error)
    else:
        raise AssertionError("An unsupported document should be rejected.")


def test_report_discloses_unavailable_change_and_legal_limitations():
    report = render_report([], {"available": False, "reason": "A second image is missing."}, [], [], "Test area")
    assert "A second image is missing" in report
    assert "not a legal determination" in report
    assert "No relevant, searchable passages" in report


def test_report_embeds_map_evidence_and_preserves_citations():
    image = {
        "name": "review.tif",
        "date": None,
        "crs": "EPSG:32631",
        "width": 4,
        "height": 4,
        "average_confidence": 0.7,
        "uncertainty_fraction": 0.2,
        "class_areas_m2": {"Vegetation": 100.0},
        "class_shares_pct": {"Vegetation": 100.0},
        "mean_class_scores": {"Vegetation": 0.7},
        "classification_map_png": "data:image/png;base64,preview",
    }
    passage = type(
        "RetrievedPassage",
        (),
        {
            "document": "zoning.pdf",
            "location": "page 3",
            "text": "Public parks must retain open green space.",
        },
    )()
    report = render_report([image], {"available": False, "reason": "No second image."}, [passage], [], "Test area")
    assert "![review.tif sampled RGB and land-cover classification overlay]" in report
    assert "data:image/png;base64,preview" in report
    assert "zoning.pdf, page 3" in report
