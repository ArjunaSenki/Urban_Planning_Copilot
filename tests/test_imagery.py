from datetime import date
from io import BytesIO

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from landuse_copilot.imagery import (
    ImageryValidationError,
    analyze_geotiff,
    compare_images,
    inspect_geotiff,
    parse_boundary_geojson,
    suggested_band_roles,
)

BANDS = {"red": 1, "green": 2, "blue": 3, "nir": 4, "swir1": 5, "unused": None}


def make_geotiff(
    *,
    origin=(500_000, 4_500_000),
    crs="EPSG:32631",
    vegetation=True,
    descriptions=("red", "green", "blue", "nir", "swir1"),
):
    values = np.empty((5, 4, 4), dtype="uint16")
    values[0] = 2_000
    values[1] = 3_000
    values[2] = 4_000
    values[3] = 6_000 if vegetation else 1_000
    values[4] = 4_000 if vegetation else 500
    buffer = BytesIO()
    with rasterio.open(
        buffer,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=5,
        dtype="uint16",
        crs=crs,
        transform=from_origin(*origin, 10, 10),
        nodata=0,
    ) as dataset:
        dataset.write(values)
        for band, description in enumerate(descriptions, start=1):
            dataset.set_band_description(band, description)
    return buffer.getvalue()


def analyze(data, acquired=date(2025, 1, 1)):
    return analyze_geotiff(data, BANDS, acquired)


def test_inspection_requires_georeferencing():
    with pytest.raises(ImageryValidationError, match="coordinate reference system"):
        inspect_geotiff(make_geotiff(crs=None))


def test_suggested_roles_only_use_explicit_descriptions():
    assert suggested_band_roles(("Blue", "Green", "Red", "NIR", None)) == {
        "red": 3,
        "green": 2,
        "blue": 1,
        "nir": 4,
        "swir1": None,
    }
    assert suggested_band_roles((None, None))["red"] is None


def test_classification_reports_areas_scores_and_uncertainty():
    result = analyze(make_geotiff())

    assert result["class_areas_m2"]["Vegetation"] == pytest.approx(1_600)
    assert sum(result["class_shares_pct"].values()) == pytest.approx(100)
    assert sum(result["mean_class_scores"].values()) == pytest.approx(1)
    assert result["uncertainty_fraction"] == 0
    assert result["sample_width"] == 4
    assert result["rgb_png"].startswith("data:image/png;base64,")
    assert result["classification_map_png"].startswith("data:image/png;base64,")


def test_change_requires_distinct_dates_and_exact_grid_alignment():
    earlier = analyze(make_geotiff(vegetation=True), date(2024, 1, 1))
    later = analyze(make_geotiff(vegetation=False), date(2025, 1, 1))

    result = compare_images(earlier, later)
    assert result["available"]
    assert result["changed_area_m2"] > 0
    assert result["change_map_png"].startswith("data:image/png;base64,")
    assert result["earlier"]["date"] == date(2024, 1, 1)
    assert result["later"]["date"] == date(2025, 1, 1)

    same_date = compare_images(earlier, analyze(make_geotiff(), date(2024, 1, 1)))
    assert not same_date["available"]
    assert "different acquisition dates" in same_date["reason"]

    misaligned = analyze(make_geotiff(origin=(500_100, 4_500_000)), date(2025, 1, 1))
    result = compare_images(earlier, misaligned)
    assert not result["available"]
    assert "geotransforms" in result["reason"]


def test_boundary_parser_rejects_non_polygon_geometry():
    point = b'{"type":"Point","coordinates":[0,0]}'
    with pytest.raises(ImageryValidationError, match="Polygon or MultiPolygon"):
        parse_boundary_geojson(point)

    polygon = b'{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]}'
    assert parse_boundary_geojson(polygon)["type"] == "Polygon"
