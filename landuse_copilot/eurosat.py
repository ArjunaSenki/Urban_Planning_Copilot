"""Reproducible, spatially blocked EuroSAT multispectral baseline."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import sklearn
from pyproj import Transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_QUANTILES = (10, 50, 90)
SPATIAL_BLOCK_DEGREES = 0.1
SPLIT_THRESHOLDS = (70, 85)
MODEL_NAME = "standardized multispectral patch-feature multinomial logistic regression"


class EuroSATValidationError(ValueError):
    """The local labeled dataset cannot support the configured baseline."""


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str
    group: tuple[int, int]


def _spatial_group(
    longitude: float, latitude: float, block_degrees: float = SPATIAL_BLOCK_DEGREES
) -> tuple[int, int]:
    return (
        int(np.floor(longitude / block_degrees)),
        int(np.floor(latitude / block_degrees)),
    )


def _spatial_split(group: tuple[int, int], seed: int) -> str:
    stable_group = f"{seed}:{group[0]}:{group[1]}".encode("ascii")
    bucket = int.from_bytes(hashlib.sha256(stable_group).digest()[:8], "big") % 100
    if bucket < SPLIT_THRESHOLDS[0]:
        return "train"
    if bucket < SPLIT_THRESHOLDS[1]:
        return "validation"
    return "test"


def _feature_names(band_count: int) -> list[str]:
    return [
        f"channel_{band + 1:02d}_{statistic}"
        for statistic in ("mean", "std", *(f"p{q}" for q in FEATURE_QUANTILES))
        for band in range(band_count)
    ]


def _image_features(dataset: rasterio.io.DatasetReader) -> tuple[np.ndarray, np.ndarray]:
    data = dataset.read(out_dtype="float32")
    if data.ndim != 3 or not np.isfinite(data).all():
        raise EuroSATValidationError(f"{dataset.name} contains missing or non-finite pixel values.")
    flattened = data.reshape(data.shape[0], -1)
    stats = [
        flattened.mean(axis=1, dtype=np.float64),
        flattened.std(axis=1, dtype=np.float64),
        *np.percentile(flattened, FEATURE_QUANTILES, axis=1),
    ]
    return np.concatenate(stats).astype("float32"), data.reshape(data.shape[0], -1).min(axis=1), data.max(axis=(1, 2))


def scan_dataset(root: Path, seed: int = 42) -> tuple[list[Sample], np.ndarray, list[str], dict[str, Any]]:
    """Validate all labeled rasters, extract compact patch features, and summarize metadata."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise EuroSATValidationError(f"Dataset directory does not exist: {root}")

    class_directories = sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
    if len(class_directories) < 2:
        raise EuroSATValidationError("Expected at least two class directories containing labeled TIFF images.")

    paths_by_class: dict[str, list[Path]] = {}
    for class_directory in class_directories:
        paths = sorted(
            path
            for path in class_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        )
        if paths:
            paths_by_class[class_directory.name] = paths
    if len(paths_by_class) < 2:
        raise EuroSATValidationError("Expected TIFF image files in at least two labeled class directories.")

    labels = sorted(paths_by_class)
    samples: list[Sample] = []
    feature_rows: list[np.ndarray] = []
    formats: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    band_counts: Counter[int] = Counter()
    dtypes: Counter[str] = Counter()
    crs_counts: Counter[str] = Counter()
    tag_keys: Counter[str] = Counter()
    pixel_min: np.ndarray | None = None
    pixel_max: np.ndarray | None = None
    reference_shape: tuple[int, int, int] | None = None
    reference_dtype: tuple[str, ...] | None = None
    reference_descriptions: tuple[str | None, ...] | None = None
    reference_tags: dict[str, str] | None = None
    transformers: dict[str, Transformer] = {}
    class_counts = {label: len(paths) for label, paths in paths_by_class.items()}
    total = sum(class_counts.values())
    processed = 0

    for label in labels:
        for path in paths_by_class[label]:
            try:
                with rasterio.open(path) as dataset:
                    if dataset.driver != "GTiff":
                        raise EuroSATValidationError(f"Expected GeoTIFF input: {path}")
                    if dataset.crs is None or dataset.transform.is_identity:
                        raise EuroSATValidationError(
                            f"{path} lacks usable georeferencing; spatial leakage controls require CRS and transform metadata."
                        )
                    current_shape = (dataset.count, dataset.height, dataset.width)
                    if reference_shape is None:
                        reference_shape = current_shape
                        reference_dtype = tuple(dataset.dtypes)
                        reference_descriptions = tuple(dataset.descriptions)
                        reference_tags = dataset.tags()
                    if current_shape != reference_shape:
                        raise EuroSATValidationError(
                            f"Inconsistent raster shape in {path}: {current_shape}, expected {reference_shape}."
                        )
                    if tuple(dataset.dtypes) != reference_dtype:
                        raise EuroSATValidationError(f"Inconsistent pixel data type in {path}.")
                    if tuple(dataset.descriptions) != reference_descriptions:
                        raise EuroSATValidationError(f"Inconsistent band descriptions in {path}.")

                    features, minima, maxima = _image_features(dataset)
                    pixel_min = minima if pixel_min is None else np.minimum(pixel_min, minima)
                    pixel_max = maxima if pixel_max is None else np.maximum(pixel_max, maxima)
                    crs_name = dataset.crs.to_string()
                    if crs_name not in transformers:
                        transformers[crs_name] = Transformer.from_crs(
                            dataset.crs, "EPSG:4326", always_xy=True
                        )
                    x, y = dataset.transform * (dataset.width / 2, dataset.height / 2)
                    longitude, latitude = transformers[crs_name].transform(x, y)
                    if not np.isfinite(longitude) or not np.isfinite(latitude):
                        raise EuroSATValidationError(f"Could not derive a spatial group for {path}.")
                    group = _spatial_group(longitude, latitude)
                    samples.append(Sample(path=path, label=label, group=group))
                    feature_rows.append(features)

                    formats[dataset.driver] += 1
                    dimensions[f"{dataset.width}x{dataset.height}"] += 1
                    band_counts[dataset.count] += 1
                    dtypes[",".join(dataset.dtypes)] += 1
                    crs_counts[crs_name] += 1
                    tag_keys.update(dataset.tags().keys())
                    processed += 1
                    if processed % 1000 == 0 or processed == total:
                        print(f"Scanned {processed:,}/{total:,} images", flush=True)
            except EuroSATValidationError:
                raise
            except (rasterio.errors.RasterioError, OSError, ValueError, OverflowError) as error:
                raise EuroSATValidationError(f"Could not read dataset image {path}: {error}") from error

    if pixel_min is None or pixel_max is None or reference_shape is None or reference_dtype is None:
        raise EuroSATValidationError("No readable labeled GeoTIFF images were found.")

    metadata = {
        "dataset_variant": (
            f"Local folder-labeled {reference_shape[0]}-band multispectral GeoTIFF patch dataset "
            "(EuroSAT_MS path)"
        ),
        "source_directory": str(root),
        "sample_count": len(samples),
        "class_counts": class_counts,
        "classes": labels,
        "raster_dimensions": {
            "bands": reference_shape[0],
            "height": reference_shape[1],
            "width": reference_shape[2],
        },
        "data_types": list(reference_dtype),
        "band_descriptions": list(reference_descriptions or ()),
        "band_order_note": (
            "Raster band descriptions are missing; channels are treated as ordered numeric channels only. "
            "Their Sentinel-2 band identities/order cannot be established from the available tags."
        ),
        "pixel_value_ranges_by_channel": [
            {"channel": channel + 1, "min": int(low), "max": int(high)}
            for channel, (low, high) in enumerate(zip(pixel_min, pixel_max))
        ],
        "driver_counts": dict(sorted(formats.items())),
        "dimension_counts": dict(sorted(dimensions.items())),
        "band_count_distribution": {str(key): value for key, value in sorted(band_counts.items())},
        "dtype_distribution": dict(sorted(dtypes.items())),
        "crs_counts": dict(sorted(crs_counts.items())),
        "metadata_tag_keys": sorted(tag_keys),
        "reference_image_tags": reference_tags or {},
        "software_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "acquisition_date_metadata": (
            "No acquisition date tags were found."
            if not any("DATE" in key.upper() or "TIME" in key.upper() for key in tag_keys)
            else "Date/time tag keys are present; inspect reference_image_tags before interpreting."
        ),
        "spatial_grouping": {
            "coordinate_source": "Raster affine transform and each raster's CRS, transformed to EPSG:4326",
            "block_size_degrees": SPATIAL_BLOCK_DEGREES,
            "assignment": "All images whose centers fall in a geographic block are assigned to one split.",
            "split_seed": seed,
        },
    }
    return samples, np.stack(feature_rows), labels, metadata


def _metric_summary(
    truth: np.ndarray, predicted: np.ndarray, labels: list[str]
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0
    )
    macro = precision_recall_fscore_support(
        truth, predicted, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        truth, predicted, labels=labels, average="weighted", zero_division=0
    )
    micro = precision_recall_fscore_support(
        truth, predicted, labels=labels, average="micro", zero_division=0
    )
    return {
        "sample_count": int(len(truth)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "overall": {
            "macro_precision": float(macro[0]),
            "macro_recall": float(macro[1]),
            "macro_f1": float(macro[2]),
            "weighted_precision": float(weighted[0]),
            "weighted_recall": float(weighted[1]),
            "weighted_f1": float(weighted[2]),
            "micro_precision": float(micro[0]),
            "micro_recall": float(micro[1]),
            "micro_f1": float(micro[2]),
        },
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
    }


def train_evaluate(
    samples: list[Sample],
    features: np.ndarray,
    labels: list[str],
    metadata: dict[str, Any],
    output_directory: Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit on training blocks, report validation/test metrics, and save local-only artifacts."""
    if len(samples) != len(features):
        raise EuroSATValidationError("Feature rows and labeled raster records do not match.")
    splits = np.array([_spatial_split(sample.group, seed) for sample in samples])
    target = np.array([sample.label for sample in samples])
    class_counts = {
        split: {
            label: int(np.count_nonzero((splits == split) & (target == label)))
            for label in labels
        }
        for split in ("train", "validation", "test")
    }
    if any(class_counts["train"][label] == 0 for label in labels):
        raise EuroSATValidationError(
            "The spatially blocked training split does not contain every class. "
            "Try a different --seed; do not switch to a pixel-random split."
        )
    if any(class_counts["test"][label] == 0 for label in labels):
        missing = [label for label in labels if class_counts["test"][label] == 0]
        raise EuroSATValidationError(
            "The spatially blocked test split has no examples for: "
            + ", ".join(missing)
            + ". Try a different --seed."
        )
    if not np.any(splits == "validation"):
        raise EuroSATValidationError("The spatially blocked validation split is empty; try another --seed.")

    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=seed,
            tol=1e-4,
        ),
    )
    train_mask = splits == "train"
    validation_mask = splits == "validation"
    test_mask = splits == "test"
    print(
        "Training a standardized multinomial logistic-regression baseline "
        f"on {int(train_mask.sum()):,} spatially grouped training images.",
        flush=True,
    )
    pipeline.fit(features[train_mask], target[train_mask])
    validation_predictions = pipeline.predict(features[validation_mask])
    test_predictions = pipeline.predict(features[test_mask])
    validation_metrics = _metric_summary(target[validation_mask], validation_predictions, labels)
    test_metrics = _metric_summary(target[test_mask], test_predictions, labels)

    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    from joblib import dump

    model_path = output_directory / "eurosat_ms_logistic_regression.joblib"
    dump(pipeline, model_path, compress=3)
    metrics = {
        "model": MODEL_NAME,
        "dataset": metadata,
        "split": {
            "method": "Deterministic geographic-block holdout, keeping each block wholly in one split",
            "seed": seed,
            "block_degrees": SPATIAL_BLOCK_DEGREES,
            "split_percentages_by_block_hash": {
                "train": SPLIT_THRESHOLDS[0],
                "validation": SPLIT_THRESHOLDS[1] - SPLIT_THRESHOLDS[0],
                "test": 100 - SPLIT_THRESHOLDS[1],
            },
            "image_counts_by_split_and_class": class_counts,
            "group_counts_by_split": {
                split: len({sample.group for sample, assigned in zip(samples, splits) if assigned == split})
                for split in ("train", "validation", "test")
            },
            "limitations": [
                "Geographic blocks reduce near-neighbor leakage, but do not identify original Sentinel-2 scenes or source acquisition groups.",
                "Adjacent blocks may still contain nearby or overlapping source acquisitions; source-scene metadata is unavailable.",
                "Class folders provide image-level scene labels, not pixel-level ground truth.",
                "Dataset CRS/affine metadata is retained for splitting but is not used as a model feature.",
            ],
        },
        "feature_definition": {
            "features_per_image": len(_feature_names(metadata["raster_dimensions"]["bands"])),
            "statistics_per_channel": ["mean", "standard deviation", "10th percentile", "50th percentile", "90th percentile"],
            "feature_names": _feature_names(metadata["raster_dimensions"]["bands"]),
            "channel_order": metadata["band_order_note"],
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "artifacts": {
            "model": str(model_path),
            "metrics": str(output_directory / "metrics.json"),
            "dataset_summary": str(output_directory / "dataset_summary.json"),
            "test_confusion_matrix_csv": str(output_directory / "test_confusion_matrix.csv"),
        },
        "heuristic_comparison": {
            "status": "not_evaluated",
            "reason": (
                "The four spectral-proxy classes do not form a complete, defensible mapping to all ten "
                "EuroSAT scene labels. Results from the upload heuristic are not scored against these labels."
            ),
        },
        "interpretation": (
            "This is a standalone image-level EuroSAT baseline. It is not applied to the app's uploaded imagery, "
            "paired-image change detection, or zoning evidence."
        ),
    }

    summary_path = output_directory / "dataset_summary.json"
    metrics_path = output_directory / "metrics.json"
    confusion_path = output_directory / "test_confusion_matrix.csv"
    summary_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    header = ",".join(["true_label\\predicted_label", *labels])
    rows = [header]
    for label, row in zip(labels, test_metrics["confusion_matrix"]):
        rows.append(",".join([label, *(str(value) for value in row)]))
    confusion_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return metrics
