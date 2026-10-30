#!/usr/bin/env python3
"""Train and evaluate the separate local EuroSAT multispectral baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from landuse_copilot.eurosat import EuroSATValidationError, scan_dataset, train_evaluate

DEFAULT_DATASET = Path(
    os.environ.get("EUROSAT_MS_DIR", str(REPOSITORY_ROOT / "EuroSAT_MS"))
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs" / "eurosat"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train an image-level multinomial logistic-regression baseline from labeled "
            "13-band EuroSAT GeoTIFF patches using deterministic geographic-block splits."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATASET, help="Labeled dataset root (default: EUROSAT_MS_DIR or ./EuroSAT_MS).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Output directory outside the source dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for deterministic geographic-block assignments.")
    args = parser.parse_args()

    data_directory = args.data_dir.expanduser().resolve()
    output_directory = args.output_dir.expanduser().resolve()
    if output_directory == data_directory or data_directory in output_directory.parents:
        parser.error("--output-dir must be outside the source dataset directory.")

    try:
        samples, features, labels, metadata = scan_dataset(data_directory, seed=args.seed)
        metadata["split_seed"] = args.seed
        print(
            json.dumps(
                {
                    "dataset_variant": metadata["dataset_variant"],
                    "source_directory": metadata["source_directory"],
                    "sample_count": metadata["sample_count"],
                    "class_counts": metadata["class_counts"],
                    "raster_dimensions": metadata["raster_dimensions"],
                    "data_types": metadata["data_types"],
                    "band_descriptions": metadata["band_descriptions"],
                    "pixel_value_ranges_by_channel": metadata["pixel_value_ranges_by_channel"],
                    "crs_counts": metadata["crs_counts"],
                    "metadata_tag_keys": metadata["metadata_tag_keys"],
                },
                indent=2,
            ),
            flush=True,
        )
        result = train_evaluate(
            samples,
            features,
            labels,
            metadata,
            output_directory,
            seed=args.seed,
        )
    except EuroSATValidationError as error:
        parser.exit(2, f"Dataset/training error: {error}\n")

    print("\nTest-set evaluation (spatially held-out blocks):")
    print(f"Accuracy: {result['test']['accuracy']:.4f}")
    for metric, value in result["test"]["overall"].items():
        print(f"{metric}: {value:.4f}")
    print("\nPer-class test metrics:")
    print("class,precision,recall,f1,support")
    for label, scores in result["test"]["per_class"].items():
        print(
            f"{label},{scores['precision']:.4f},{scores['recall']:.4f},"
            f"{scores['f1']:.4f},{scores['support']}"
        )
    print(f"\nMetrics: {result['artifacts']['metrics']}")
    print(f"Model: {result['artifacts']['model']}")
    print("The trained EuroSAT baseline is separate from the app's heuristic and temporal workflow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
