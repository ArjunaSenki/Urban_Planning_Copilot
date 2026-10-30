# Satellite Land-Use Review Copilot

A local-first Streamlit MVP for municipal planning teams to screen georeferenced imagery, compare suitable dated image pairs, retrieve passages from supplied planning documents, and export an evidence report for human review.

## Setup

Requires Python 3.11 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The app runs locally. It does not require credentials, paid services, a hosted model, or a network basemap. Uploaded imagery and documents are processed in the app process; the app does not intentionally send them to an external analysis service.

Run the focused automated tests with:

```bash
python -m pytest
```

## Review workflow

1. **Define an area.** Upload a GeoJSON Polygon/MultiPolygon, use the actual image footprint as the clearly labeled sample area, or analyze the full image without a boundary. GeoJSON boundaries must contain longitude/latitude coordinates in WGS 84 (EPSG:4326).
2. **Upload imagery.** Supply up to two georeferenced GeoTIFF (`.tif`/`.tiff`) images. The app displays their CRS, dimensions, band descriptions, and any recognized acquisition date. Confirm or enter each acquisition date, assign the red/green/blue/NIR band roles, optionally assign SWIR1, and set the reflectance divisor based on the product metadata. Descriptions are suggested only when their labels match known band names; otherwise assign bands yourself.
3. **Review classification.** The app shows an image preview, per-class area and share, an overlay, mean heuristic class scores, and the share of pixels below the visible uncertainty threshold. Classification is sampled to at most 512 pixels on the longest side.
4. **Review possible changes.** The comparison is enabled only for two differently dated images with matching CRS, dimensions, affine transform, assigned spectral roles, and reflectance divisor. Lower-confidence class flips are excluded from the estimated changed area and reported separately. An overlay, dates, transition classes, and confidence proxy are shown when comparison is possible.
5. **Supply planning documents.** Upload searchable PDF, DOCX, or UTF-8 TXT files and optionally provide search terms. Retrieved passages include the filename and PDF page, DOCX paragraph, or TXT section. Review the exact excerpts and download a Markdown report with embedded sampled map overlays.

## Separate EuroSAT multispectral training baseline

The local dataset was inspected at `./EuroSAT_MS`. Configure another local copy with the `EUROSAT_MS_DIR` environment variable or `--data-dir`; for example, use `/downloads/EuroSAT_MS` if that path is mounted in your environment. The exact current-workspace command is:

```bash
EUROSAT_MS_DIR="$PWD/EuroSAT_MS" python scripts/train_eurosat.py \
  --output-dir "$PWD/outputs/eurosat" \
  --seed 42
```

The script does not run when Streamlit starts. It reads class labels from the immediate class-directory names, inspects each TIFF as it is processed, and extracts compact per-image channel mean, standard deviation, and 10th/50th/90th percentile features. A standardized multinomial logistic-regression baseline is fit on CPU. The only new dependency is scikit-learn, used for a conventional scaler/classifier and established evaluation metrics.

The seed assigns entire 0.1-degree latitude/longitude blocks to train (70% of hashed blocks), validation (15%), or test (15%). All patches in a block stay together. The test metrics are calculated only on the held-out test blocks. Runs write `outputs/eurosat/dataset_summary.json`, `metrics.json`, `test_confusion_matrix.csv`, and `eurosat_ms_logistic_regression.joblib`. Both the source dataset and this output directory are excluded from Git.

The upload workflow remains a separate spectral-index heuristic. Its four output classes have no complete, defensible mapping to all ten EuroSAT scene labels, so its outputs are **not** scored against EuroSAT. EuroSAT scene-label classification does not establish accuracy for uploaded imagery, temporal change detection, or zoning evidence.

## Short class demo (2–3 minutes)

The demo requires the local dataset at `./EuroSAT_MS`, the already-trained model at `./outputs/eurosat/eurosat_ms_logistic_regression.joblib`, its `metrics.json`, and the declared Python dependencies. These dataset/model artifacts are ignored by Git; do not retrain the model just to launch the app. Alternatively, set `EUROSAT_MS_DIR` and `EUROSAT_MODEL_PATH` to existing local copies before starting Streamlit. If you have a real searchable planning document, upload it in the EuroSAT view; otherwise the app uses [`demo_data/fictional_planning_excerpt.txt`](./demo_data/fictional_planning_excerpt.txt), which is labeled fictional in the source text, interface, and report.

From the repository root, launch the app:

```bash
EUROSAT_MS_DIR="$PWD/EuroSAT_MS" \
EUROSAT_MODEL_PATH="$PWD/outputs/eurosat/eurosat_ms_logistic_regression.joblib" \
streamlit run app.py
```

In the app:

1. Select **EuroSAT Land-Cover Classifier** in the sidebar. The saved model loads only when you request a classification; the app never starts training.
2. Select a dataset class and local example, or switch to **Enter a local image path** and provide a 64×64, 13-channel, `uint16` GeoTIFF path.
3. Optionally upload a real planning document. If no real document is available, retain the clearly labeled fictional demo document and search for “residential development green space.”
4. Click **Classify image and retrieve planning evidence**. Review the predicted class, uncalibrated model score, known folder label when present, source passage and location, then download the short report.

**Speaking outline:** (1) 30–40 seconds: explain that EuroSAT’s class-folder labels are scene labels and show which local image is selected. (2) 45–60 seconds: classify it, compare the prediction to the known folder label, and explain that the model score is not a calibrated probability. (3) 30–40 seconds: inspect the retrieved document passage and location; if fictional, point out the repeated fictional warning, or upload a real document. (4) 20–30 seconds: clarify that this trained EuroSAT view is separate from the uploaded-image spectral heuristic; EuroSAT does not validate change detection or zoning evidence, and no EuroSAT temporal change is demonstrated.

## Supported inputs and limits

- **Imagery:** Georeferenced GeoTIFF only; up to two files, 150 MB each. Required assigned bands: red, green, blue, and NIR. SWIR1 is optional. Inputs must have a CRS, an affine transform, readable pixels, and no more than 100 million source pixels. The reflectance divisor defaults to `10000`; verify it against the source product's metadata.
- **Boundaries:** GeoJSON Polygon/MultiPolygon in WGS 84 longitude/latitude. The image-footprint sample option uses the uploaded imagery's own extent; it does not generate sample coordinates or imagery.
- **Documents:** Searchable PDF (20 MB each), DOCX (20 MB each), and UTF-8 TXT (20 MB each). Text extraction is local. Scanned PDFs without an embedded text layer require OCR, which is not included.
- **Classification:** A small, explicit spectral-index heuristic (NDWI, NDVI, and, when SWIR1 is assigned, NDBI), not a trained or validated land-cover model. Class scores are softmax-normalized heuristic scores, not calibrated statistical probabilities or proof of a class. A “built-up spectral proxy” is not a building detector.
- **Change detection:** A difference between spectral-proxy classes is a possible land-cover change, not proof of construction or a zoning violation. Matching geotransforms do not verify sensor-level co-registration, equivalent acquisition conditions, or radiometric calibration. No temporal result is provided without a suitable image pair and both acquisition dates.
- **Area estimates:** Calculated from the sampled pixels, CRS units, and rasterized boundary. They are approximate, especially for small sites or coarse imagery. A source CRS and reflectance divisor must be selected correctly.
- **Planning evidence:** Offline keyword-weighted retrieval over supplied text only. Search results are not a complete legal search, do not establish that a rule applies, and never yield a legal finding. No external documents, zoning rules, coordinates, citations, or acquisition dates are invented.
- **Local EuroSAT multispectral copy inspected:** 27,000 labeled TIFF patches in ten class folders: AnnualCrop (3,000), Forest (3,000), HerbaceousVegetation (3,000), Highway (2,500), Industrial (2,500), Pasture (2,000), PermanentCrop (2,500), Residential (3,000), River (2,500), and SeaLake (3,000). Every inspected image is a 64×64, 13-channel, unsigned-16-bit GeoTIFF. Raster affine transforms and UTM CRS metadata are present (EPSG:32627–32636); band descriptions are blank, and the files do not provide acquisition-date tags. The training output records actual per-channel value ranges and CRS counts.
- **EuroSAT limitations:** The class folders supply image-level scene labels, not pixel ground truth. Missing band descriptions prevent confirming channel-to-Sentinel-2-band identities/order from these files alone. Spatial-block splitting reduces neighboring-patch leakage, but the dataset contains no source-scene/acquisition identifiers; adjacent blocks can still have related source imagery. The separately trained baseline must not be treated as evidence about the upload heuristic, change detection, or zoning conclusions.

## Project layout

- `app.py` — Streamlit workflow and review interface.
- `landuse_copilot/imagery.py` — GeoTIFF validation, classification, and paired-image comparison.
- `landuse_copilot/documents.py` — document extraction and local passage retrieval.
- `landuse_copilot/report.py` — evidence-focused Markdown report.
- `landuse_copilot/eurosat.py` — separate dataset scan, feature extraction, spatial split, training, and evaluation.
- `scripts/train_eurosat.py` — explicit offline training/evaluation command; not run by the app.
- `tests/` — focused validation for raster processing, comparisons, retrieval, and report safeguards.
