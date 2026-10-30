from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from landuse_copilot.documents import Passage
from landuse_copilot.eurosat_demo import classify_eurosat_image, render_demo_report


class FixtureModel:
    classes_ = np.array(["Forest", "Residential"])
    n_features_in_ = 65

    def predict_proba(self, features):
        assert features.shape == (1, 65)
        return np.array([[0.2, 0.8]])


def write_fixture(path: Path):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=64,
        height=64,
        count=13,
        dtype="uint16",
        crs="EPSG:32631",
        transform=from_origin(500_000, 4_500_000, 10, 10),
    ) as dataset:
        dataset.write(np.ones((13, 64, 64), dtype="uint16") * 1000)


def test_classifier_reads_folder_label_and_model_score(tmp_path):
    dataset_root = tmp_path / "EuroSAT_MS"
    image = dataset_root / "Residential" / "Residential_1.tif"
    image.parent.mkdir(parents=True)
    write_fixture(image)

    result = classify_eurosat_image(image, FixtureModel(), dataset_root)

    assert result["predicted_class"] == "Residential"
    assert result["model_score"] == pytest.approx(0.8)
    assert result["known_label"] == "Residential"
    assert result["is_correct"]
    assert result["metadata"]["channel_preview"].size == (64, 64)


def test_classifier_rejects_non_eurosat_shape(tmp_path):
    image = tmp_path / "small.tif"
    with rasterio.open(
        image,
        "w",
        driver="GTiff",
        width=32,
        height=32,
        count=13,
        dtype="uint16",
        crs="EPSG:32631",
        transform=from_origin(500_000, 4_500_000, 10, 10),
    ) as dataset:
        dataset.write(np.ones((13, 32, 32), dtype="uint16"))

    with pytest.raises(ValueError, match="expects a 64"):
        classify_eurosat_image(image, FixtureModel(), tmp_path)


def test_demo_report_marks_fictional_source_and_separates_workflows():
    report = render_demo_report(
        {
            "image_path": "/data/Residential_1.tif",
            "predicted_class": "Residential",
            "model_score": 0.8,
            "known_label": "Residential",
            "is_correct": True,
            "metadata": {"channels": 13},
        },
        Passage(
            "fictional_planning_excerpt.txt",
            "section 2",
            "FICTIONAL DEMO CONTENT — invented residential example.",
            1.0,
        ),
        fictional_source=True,
        test_accuracy=0.9,
        test_macro_f1=0.88,
    )

    assert "FICTIONAL DEMO CONTENT — NOT A REAL PLANNING DOCUMENT" in report
    assert "not a calibrated probability" in report
    assert "does not validate the separate spectral heuristic" in report
    assert "No temporal change result is produced" in report
