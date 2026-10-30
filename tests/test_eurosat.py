import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from landuse_copilot.eurosat import (
    Sample,
    _image_features,
    _spatial_group,
    _spatial_split,
    train_evaluate,
)


def test_feature_extraction_uses_channel_statistics_without_band_name_assumptions():
    buffer = MemoryFile()
    with buffer.open(
        driver="GTiff",
        width=2,
        height=2,
        count=2,
        dtype="uint16",
        crs="EPSG:32631",
        transform=from_origin(500_000, 4_500_000, 10, 10),
    ) as dataset:
        dataset.write(np.array([[[1, 2], [3, 4]], [[10, 20], [30, 40]]], dtype="uint16"))
        features, minima, maxima = _image_features(dataset)

    assert features.shape == (10,)
    assert np.allclose(features[:2], [2.5, 25])
    assert np.allclose(features[2:4], [np.std([1, 2, 3, 4]), np.std([10, 20, 30, 40])])
    assert minima.tolist() == [1, 10]
    assert maxima.tolist() == [4, 40]


def test_geographic_blocks_have_deterministic_single_split():
    first_group = _spatial_group(-3.71, 40.41)
    same_block = _spatial_group(-3.72, 40.42)
    assert first_group == same_block
    assert _spatial_split(first_group, 42) == _spatial_split(same_block, 42)
    assert _spatial_split(first_group, 42) == _spatial_split(first_group, 42)


def test_spatial_training_writes_metrics_and_confusion_matrix(tmp_path):
    labels = ["Forest", "River", "Residential"]
    samples = []
    features = []
    for group_index in range(120):
        group = (group_index, group_index // 10)
        for label_index, label in enumerate(labels):
            samples.append(Sample(Path(f"{label}_{group_index}.tif"), label, group))
            row = np.zeros(3, dtype="float32")
            row[label_index] = 1
            features.append(row)

    metadata = {
        "dataset_variant": "Synthetic test data",
        "sample_count": len(samples),
        "raster_dimensions": {"bands": 1, "height": 1, "width": 1},
        "band_order_note": "Test channels have no band identity.",
    }
    result = train_evaluate(
        samples,
        np.stack(features),
        labels,
        metadata,
        tmp_path,
        seed=42,
    )

    assert result["test"]["sample_count"] > 0
    assert result["test"]["accuracy"] == 1
    assert set(result["test"]["per_class"]) == set(labels)
    assert result["test"]["confusion_matrix"] == [[result["test"]["per_class"][label]["support"] if row == column else 0 for column, _ in enumerate(labels)] for row, label in enumerate(labels)]
    assert Path(result["artifacts"]["model"]).is_file()
    metrics = json.loads(Path(result["artifacts"]["metrics"]).read_text())
    assert metrics["split"]["seed"] == 42
    assert metrics["heuristic_comparison"]["status"] == "not_evaluated"
    assert Path(result["artifacts"]["test_confusion_matrix_csv"]).is_file()
