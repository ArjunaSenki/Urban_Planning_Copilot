"""Inference helpers and a concise evidence report for the EuroSAT class demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from sklearn.pipeline import Pipeline

from landuse_copilot.documents import Passage
from landuse_copilot.eurosat import _image_features


def list_dataset_examples(dataset_root: Path, limit_per_class: int = 20) -> dict[str, list[Path]]:
    """Return a short deterministic list of existing TIFF examples for each class."""
    examples: dict[str, list[Path]] = {}
    if not dataset_root.is_dir():
        return examples
    for class_directory in sorted(
        path for path in dataset_root.iterdir() if path.is_dir() and not path.name.startswith(".")
    ):
        paths = [
            path
            for path in class_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        ]
        paths.sort(
            key=lambda path: (
                path.stem.rsplit("_", 1)[0],
                int(path.stem.rsplit("_", 1)[1]) if path.stem.rsplit("_", 1)[-1].isdigit() else -1,
                str(path),
            )
        )
        if paths:
            examples[class_directory.name] = paths[:limit_per_class]
    return examples


def classify_eurosat_image(
    image_path: Path,
    model: Pipeline,
    dataset_root: Path,
) -> dict[str, Any]:
    """Classify a compatible 64×64, 13-channel TIFF with the saved EuroSAT model."""
    path = image_path.expanduser().resolve()
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("Choose a local .tif or .tiff EuroSAT multispectral image.")
    if not path.is_file():
        raise ValueError(f"Image file does not exist: {path}")

    try:
        with rasterio.open(path) as dataset:
            if dataset.driver != "GTiff":
                raise ValueError("The selected file is not a GeoTIFF.")
            if (dataset.height, dataset.width, dataset.count) != (64, 64, 13):
                raise ValueError(
                    "The EuroSAT demo model expects a 64×64 image with 13 multispectral channels; "
                    f"this file is {dataset.height}×{dataset.width} with {dataset.count} channels."
                )
            if any(dtype != "uint16" for dtype in dataset.dtypes):
                raise ValueError("The EuroSAT demo model expects the dataset's unsigned 16-bit pixel type.")
            if dataset.crs is None or dataset.transform.is_identity:
                raise ValueError("The selected EuroSAT image has no usable CRS/georeferencing metadata.")
            features, _, _ = _image_features(dataset)
            image_metadata = {
                "width": dataset.width,
                "height": dataset.height,
                "channels": dataset.count,
                "dtypes": tuple(dataset.dtypes),
                "crs": dataset.crs.to_string(),
                "transform": tuple(dataset.transform),
                "descriptions": tuple(dataset.descriptions),
                "channel_preview": _channel_preview(dataset.read((1, 2, 3))),
            }
    except ValueError:
        raise
    except (rasterio.errors.RasterioError, OSError) as error:
        raise ValueError(f"Could not read the selected TIFF image: {error}") from error

    if not hasattr(model, "classes_") or not hasattr(model, "predict_proba"):
        raise ValueError("The saved EuroSAT artifact is not a compatible probability-scoring classifier.")
    expected_features = getattr(model, "n_features_in_", features.size)
    if features.size != expected_features:
        raise ValueError(
            f"The saved model expects {expected_features} features, but this TIFF produced {features.size}."
        )
    probabilities = model.predict_proba(features.reshape(1, -1))[0]
    best_index = int(np.argmax(probabilities))
    predicted_class = str(model.classes_[best_index])
    known_label: str | None = None
    resolved_root = dataset_root.expanduser().resolve()
    try:
        relative = path.relative_to(resolved_root)
    except ValueError:
        relative = None
    if relative is not None and len(relative.parts) >= 2:
        candidate = relative.parts[0]
        if candidate in {str(label) for label in model.classes_}:
            known_label = candidate

    return {
        "image_path": str(path),
        "predicted_class": predicted_class,
        "model_score": float(probabilities[best_index]),
        "known_label": known_label,
        "is_correct": predicted_class == known_label if known_label is not None else None,
        "classes": [str(label) for label in model.classes_],
        "probabilities": {
            str(label): float(probability)
            for label, probability in zip(model.classes_, probabilities)
        },
        "metadata": image_metadata,
    }


def _channel_preview(channels: np.ndarray) -> Image.Image:
    preview = np.moveaxis(channels, 0, -1).astype("float32")
    rendered = np.zeros_like(preview, dtype="uint8")
    for index in range(preview.shape[-1]):
        values = preview[:, :, index]
        low, high = np.percentile(values, (2, 98))
        if high <= low:
            high = low + 1
        rendered[:, :, index] = np.clip(
            (values - low) * (255 / (high - low)), 0, 255
        ).astype("uint8")
    return Image.fromarray(rendered)


def render_demo_report(
    prediction: dict[str, Any],
    passage: Passage,
    *,
    fictional_source: bool,
    test_accuracy: float | None = None,
    test_macro_f1: float | None = None,
) -> str:
    """Render a concise classifier and document-evidence demo report."""
    label = prediction["known_label"]
    lines = [
        "# EuroSAT land-cover classifier demo",
        "",
        "- **Classifier:** saved EuroSAT multispectral logistic-regression baseline.",
        "- **Separate workflow:** this trained dataset classifier is not the uploaded-image spectral-index heuristic.",
        f"- **Image:** `{prediction['image_path']}`",
        f"- **Prediction:** **{prediction['predicted_class']}**",
        f"- **Model score:** {prediction['model_score']:.1%} for the predicted class; this score is not a calibrated probability.",
        f"- **Known folder label:** {label if label is not None else 'Not available for this path.'}",
    ]
    if label is not None:
        lines.append(
            f"- **Example comparison:** {'prediction matches' if prediction['is_correct'] else 'prediction does not match'} "
            "the dataset folder label for this example. This one image is not an evaluation result."
        )
    lines.extend(
        [
            "",
            "## Planning-document retrieval",
            "",
            f"- **Source:** {passage.document}",
            f"- **Location:** {passage.location}",
            f"- **Retrieved excerpt:** {passage.text}",
        ]
    )
    if fictional_source:
        lines.extend(
            [
                "- **Status:** FICTIONAL DEMO CONTENT — NOT A REAL PLANNING DOCUMENT, ZONING RULE, OR CITATION.",
                "- All statements in this passage are invented for the class demonstration and must not be used for planning decisions.",
            ]
        )
    else:
        lines.append(
            "- This excerpt is document evidence for human review, not a legal conclusion or determination that a rule applies."
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- EuroSAT scene-label performance does not validate the separate spectral heuristic, uploaded-image classification, temporal change detection, or zoning findings.",
            "- No temporal change result is produced from this EuroSAT example.",
            "- The 13 input channels have no band descriptions in the dataset files; their channel-to-sensor-band identities are unknown.",
        ]
    )
    if test_accuracy is not None and test_macro_f1 is not None:
        lines.extend(
            [
                "",
                "## Dataset evaluation context",
                "",
                f"- Existing held-out spatial-block test accuracy: {test_accuracy:.1%}; macro F1: {test_macro_f1:.1%}.",
                "- These metrics describe the dataset baseline's held-out EuroSAT scene-label task, not this particular image or any planning workflow.",
            ]
        )
    return "\n".join(lines)
