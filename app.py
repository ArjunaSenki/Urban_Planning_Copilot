"""Streamlit interface for the local satellite land-use review MVP."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import folium
import joblib
import streamlit as st
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium

from landuse_copilot.documents import (
    DocumentValidationError,
    extract_passages,
    retrieve_passages,
)
from landuse_copilot.eurosat_demo import (
    classify_eurosat_image,
    list_dataset_examples,
    render_demo_report,
)
from landuse_copilot.imagery import (
    BAND_ALIASES,
    CLASS_COLORS,
    ImageryValidationError,
    analyze_geotiff,
    compare_images,
    inspect_geotiff,
    parse_boundary_geojson,
    suggested_band_roles,
)
from landuse_copilot.report import format_area, render_report

REPOSITORY_ROOT = Path(__file__).resolve().parent
EUROSAT_DATASET_ROOT = Path(
    os.environ.get("EUROSAT_MS_DIR", str(REPOSITORY_ROOT / "EuroSAT_MS"))
).expanduser()
EUROSAT_MODEL_PATH = Path(
    os.environ.get(
        "EUROSAT_MODEL_PATH",
        str(REPOSITORY_ROOT / "outputs" / "eurosat" / "eurosat_ms_logistic_regression.joblib"),
    )
).expanduser()
EUROSAT_METRICS_PATH = EUROSAT_MODEL_PATH.with_name("metrics.json")
FICTIONAL_PLANNING_DOCUMENT = (
    REPOSITORY_ROOT / "demo_data" / "fictional_planning_excerpt.txt"
)

st.set_page_config(page_title="Satellite Land-Use Review Copilot", page_icon="🛰️", layout="wide")

st.title("Satellite Land-Use Review Copilot")
st.caption(
    "A local-first evidence review tool for planning teams. Outputs are preliminary screening evidence, "
    "not legal conclusions or verified land-use findings."
)

with st.sidebar:
    st.header("1 · Define the area")
    boundary_upload = st.file_uploader(
        "Upload a GeoJSON Polygon or MultiPolygon",
        type=["geojson", "json"],
        key="boundary_upload",
        help="GeoJSON coordinates must use longitude/latitude (WGS 84 / EPSG:4326).",
    )
    area_choice = st.radio(
        "Area of interest",
        ("Uploaded boundary", "Image footprint (sample area)", "No boundary"),
        index=0 if boundary_upload else 1,
        help="The sample area is the uploaded image footprint, not a fictional location or generated image.",
    )
    area_label = st.text_input("Area label (optional)", placeholder="e.g., North district review")
    st.header("2 · Add dated imagery")
    image_uploads = st.file_uploader(
        "Upload one or two georeferenced GeoTIFF files",
        type=["tif", "tiff"],
        accept_multiple_files=True,
        key="image_uploads",
        help="A valid temporal comparison requires two different dates and exactly matching grids.",
    )
    st.header("3 · Add planning documents")
    document_uploads = st.file_uploader(
        "Upload zoning/planning PDFs, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="document_uploads",
    )
    retrieval_query = st.text_input(
        "Planning question or search terms",
        placeholder="e.g., permitted land use, open space, district designation",
        help="Optional offline keyword search. Only supplied document text is quoted.",
    )

    with st.sidebar:
        selected_view = st.radio(
            "Demo view",
            ("Planning review · spectral heuristic", "EuroSAT Land-Cover Classifier"),
            help="These are separate workflows. EuroSAT scores do not validate the planning-review heuristic.",
        )


    @st.cache_resource
    def _load_eurosat_model(model_path: str, modified_ns: int) -> Any:
        del modified_ns
        return joblib.load(model_path)


    @st.cache_data
    def _cached_eurosat_examples(dataset_root: str) -> dict[str, list[str]]:
        return {
            label: [str(path) for path in paths]
            for label, paths in list_dataset_examples(Path(dataset_root)).items()
        }


    if selected_view == "EuroSAT Land-Cover Classifier":
        st.header("EuroSAT Land-Cover Classifier")
        st.caption(
            "Trained 13-channel EuroSAT scene-label baseline. Separate from the planning-review "
            "spectral heuristic; this demo does not perform change detection."
        )
        st.warning(
            "EuroSAT results do not validate uploaded-image heuristic classifications, temporal change "
            "detection, or zoning conclusions. The class folder label is available only for images inside "
            "the configured local dataset."
        )

        metrics: dict[str, Any] = {}
        test_metrics: dict[str, Any] | None = None
        if EUROSAT_METRICS_PATH.is_file():
            try:
                metrics = json.loads(EUROSAT_METRICS_PATH.read_text(encoding="utf-8"))
                test_metrics = metrics["test"]
                st.caption(
                    f"Saved model: `{EUROSAT_MODEL_PATH}` · held-out spatial-block test "
                    f"accuracy {test_metrics['accuracy']:.1%}, macro F1 "
                    f"{test_metrics['overall']['macro_f1']:.1%} "
                    f"(n={test_metrics['sample_count']:,}). These are dataset-level metrics, "
                    "not a guarantee for this image."
                )
            except (OSError, ValueError, KeyError, TypeError) as error:
                st.error(f"Could not read the saved EuroSAT evaluation summary: {error}")
                metrics = {}
                test_metrics = None
        elif EUROSAT_MODEL_PATH.is_file():
            st.info("Saved evaluation metrics are unavailable; per-image model scores can still be shown.")
        else:
            st.error(
                f"Saved EuroSAT model not found at `{EUROSAT_MODEL_PATH}`. Run the separate training "
                "command documented in README.md; app startup never trains the model."
            )

        examples = _cached_eurosat_examples(str(EUROSAT_DATASET_ROOT.resolve()))
        source_mode = st.radio(
            "Image source",
            ("Select a local EuroSAT example", "Enter a local image path"),
            horizontal=True,
        )
        image_path: str | None = None
        if source_mode == "Select a local EuroSAT example":
            if examples:
                selected_class = st.selectbox("Dataset class folder", list(examples))
                example_paths = examples[selected_class]
                image_path = st.selectbox(
                    "Dataset image (sample list)",
                    example_paths,
                    help="A short list of existing examples from the selected local class folder.",
                )
            else:
                st.error(
                    f"No EuroSAT TIFF examples found under `{EUROSAT_DATASET_ROOT}`. Set "
                    "`EUROSAT_MS_DIR` or choose the local image path option."
                )
        else:
            image_path = st.text_input(
                "Local .tif/.tiff image path",
                placeholder="/path/to/EuroSAT_MS/Forest/Forest_1.tif",
                help="The image must be a georeferenced 64×64, 13-channel uint16 GeoTIFF.",
            )

        st.subheader("Planning-document retrieval")
        planning_upload = st.file_uploader(
            "Optional real planning document (PDF, DOCX, or TXT)",
            type=["pdf", "docx", "txt"],
            key="eurosat_demo_planning_document",
            help="If no real document is supplied, the demo uses an explicitly fictional excerpt.",
        )
        demo_query = st.text_input(
            "Retrieval query",
            value="residential development green space planning area",
            key="eurosat_demo_retrieval_query",
        )

        classify_clicked = st.button(
            "Classify image and retrieve planning evidence",
            type="primary",
            disabled=not image_path or not EUROSAT_MODEL_PATH.is_file(),
        )
        if classify_clicked and image_path:
            try:
                model_stat = EUROSAT_MODEL_PATH.stat()
                model = _load_eurosat_model(str(EUROSAT_MODEL_PATH), model_stat.st_mtime_ns)
                prediction = classify_eurosat_image(
                    Path(image_path),
                    model,
                    EUROSAT_DATASET_ROOT,
                )
                if planning_upload is not None:
                    document_name = planning_upload.name
                    document_passages = extract_passages(
                        document_name,
                        planning_upload.getvalue(),
                    )
                    fictional_source = False
                else:
                    document_name = FICTIONAL_PLANNING_DOCUMENT.name
                    document_passages = extract_passages(
                        document_name,
                        FICTIONAL_PLANNING_DOCUMENT.read_bytes(),
                    )
                    fictional_source = True
                retrieved = retrieve_passages(
                    [(document_name, document_passages)],
                    demo_query,
                    limit=3,
                )
                if not retrieved:
                    st.error(
                        "No passage matched this query. Try terms appearing in the document or upload "
                        "a relevant planning document."
                    )
                    st.stop()
                best_passage = retrieved[0]
                report = render_demo_report(
                    prediction,
                    best_passage,
                    fictional_source=fictional_source,
                    test_accuracy=test_metrics["accuracy"] if test_metrics else None,
                    test_macro_f1=test_metrics["overall"]["macro_f1"] if test_metrics else None,
                )
                document_source = (
                    planning_upload.name
                    if planning_upload is not None
                    else FICTIONAL_PLANNING_DOCUMENT.name
                )
                st.session_state["eurosat_demo_result"] = {
                    "prediction": prediction,
                    "passage": best_passage,
                    "fictional_source": fictional_source,
                    "report": report,
                    "image_path": str(Path(image_path).expanduser().resolve()),
                    "query": demo_query,
                    "document_source": document_source,
                }
            except (ValueError, OSError, KeyError, TypeError, ImportError, DocumentValidationError) as error:
                st.error(f"Could not complete the EuroSAT demo: {error}")

        demo_result = st.session_state.get("eurosat_demo_result")
        current_document_source = (
            planning_upload.name
            if planning_upload is not None
            else FICTIONAL_PLANNING_DOCUMENT.name
        )
        if (
            demo_result
            and demo_result["image_path"] == str(Path(image_path or "").expanduser().resolve())
            and demo_result["query"] == demo_query
            and demo_result["document_source"] == current_document_source
        ):
            prediction = demo_result["prediction"]
            st.divider()
            st.subheader("EuroSAT image classification")
            prediction_columns = st.columns(3)
            prediction_columns[0].metric("Predicted class", prediction["predicted_class"])
            prediction_columns[1].metric("Predicted-class model score", f"{prediction['model_score']:.1%}")
            prediction_columns[2].metric(
                "Known dataset label",
                prediction["known_label"] or "Not available",
            )
            st.caption(
                "The model score is not a calibrated probability. A matching folder label for one "
                "example is an illustration, not a held-out evaluation."
            )
            if prediction["known_label"]:
                if prediction["is_correct"]:
                    st.success("For this example, the predicted class matches its dataset folder label.")
                else:
                    st.warning("For this example, the prediction differs from its dataset folder label.")
            st.image(
                prediction["metadata"]["channel_preview"],
                caption=(
                    "Channels 1, 2, and 3 normalized for display only. "
                    "The dataset metadata does not identify their sensor-band identities."
                ),
                width=260,
            )
            st.caption(
                f"Source `{prediction['image_path']}` · "
                f"{prediction['metadata']['width']}×{prediction['metadata']['height']} · "
                f"{prediction['metadata']['channels']} channels · CRS "
                f"{prediction['metadata']['crs']}"
            )
            st.write("**Top class scores (not calibrated probabilities):**")
            st.dataframe(
                [
                    {"Class": label, "Model score": f"{score:.1%}"}
                    for label, score in sorted(
                        prediction["probabilities"].items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:3]
                ],
                hide_index=True,
                width="stretch",
            )

            st.subheader("Retrieved planning-document passage")
            if demo_result["fictional_source"]:
                st.error(
                    "**FICTIONAL DEMO CONTENT — NOT A REAL PLANNING DOCUMENT, ZONING RULE, OR CITATION.** "
                    "All statements below are invented for this demonstration."
                )
            else:
                st.info(
                    "Document excerpt for human review only. Retrieval does not establish that a rule applies "
                    "or provide a legal conclusion."
                )
            st.caption(
                f"Source: {demo_result['passage'].document} · "
                f"{demo_result['passage'].location}"
            )
            st.write(f"> {demo_result['passage'].text}")

            st.subheader("Short demo report")
            st.markdown(demo_result["report"])
            st.download_button(
                "Download EuroSAT demo report",
                data=demo_result["report"],
                file_name="eurosat-land-cover-demo.md",
                mime="text/markdown",
            )
        st.stop()

    st.info(
    "**Interpretation safeguard:** Uploaded-image land-cover classes use a transparent spectral-index heuristic, "
    "not the separately trained EuroSAT baseline. EuroSAT metrics do not validate this workflow, temporal "
    "change detection, or zoning evidence. Temporal changes require dated, grid-matched images and remain "
    "potential changes for human review."
)

boundary_geojson: dict[str, Any] | None = None
boundary_error: str | None = None
if boundary_upload is not None and area_choice == "Uploaded boundary":
    try:
        boundary_geojson = parse_boundary_geojson(boundary_upload.getvalue())
        st.sidebar.success("Boundary loaded.")
    except ImageryValidationError as error:
        boundary_error = str(error)
        st.sidebar.error(boundary_error)

if len(image_uploads) > 2:
    st.error("Upload no more than two images per review.")
if not image_uploads:
    st.subheader("Start a review")
    st.write(
        "Upload a georeferenced GeoTIFF to see its land-cover screening map. Add a second, "
        "co-registered image with a different acquisition date to enable change screening."
    )
    st.write(
        "The **image footprint (sample area)** uses the actual extent of your uploaded imagery. "
        "It does not create sample coordinates, dates, imagery, or planning rules."
    )


def _band_control(
    name: str,
    index: int,
    role: str,
    band_count: int,
    suggested: int | None,
    descriptions: tuple[str | None, ...],
) -> int | None:
    options: list[int | None] = [None, *range(1, band_count + 1)]
    default_index = options.index(suggested) if suggested in options else 0

    def format_option(option: int | None) -> str:
        if option is None:
            return "Not assigned"
        description = descriptions[option - 1]
        suffix = f" — {description}" if description else ""
        return f"Band {option}{suffix}"

    return st.selectbox(
        f"{role.upper()} band",
        options,
        index=default_index,
        format_func=format_option,
        key=f"band_{index}_{name}_{role}",
    )


image_inputs: list[dict[str, Any]] = []
if image_uploads and len(image_uploads) <= 2:
    st.subheader("Image metadata and band assignments")
    for image_index, uploaded_image in enumerate(image_uploads):
        image_bytes = uploaded_image.getvalue()
        with st.expander(f"{uploaded_image.name} · inspect before analysis", expanded=True):
            try:
                raster_info = inspect_geotiff(image_bytes)
            except ImageryValidationError as error:
                st.error(f"{uploaded_image.name}: {error}")
                continue
            st.caption(
                f"{raster_info.width:,} × {raster_info.height:,} pixels · "
                f"{raster_info.band_count} bands · CRS {raster_info.crs}"
            )
            st.write(
                "Band descriptions: "
                + (
                    ", ".join(
                        f"{number}: {description or 'unlabelled'}"
                        for number, description in enumerate(raster_info.descriptions, start=1)
                    )
                )
            )
            suggested = suggested_band_roles(raster_info.descriptions)
            columns = st.columns(5)
            band_map: dict[str, int | None] = {}
            for column, role in zip(columns, BAND_ALIASES):
                with column:
                    band_map[role] = _band_control(
                        uploaded_image.name,
                        image_index,
                        role,
                        raster_info.band_count,
                        suggested[role],
                        raster_info.descriptions,
                    )
            tagged_date: date | None = raster_info.acquisition_date
            selected_date = st.date_input(
                "Acquisition date (confirm image metadata or enter the known date)",
                value=tagged_date,
                key=f"date_{image_index}_{uploaded_image.name}",
                help="A temporal comparison is unavailable unless both image acquisition dates are supplied.",
            )
            reflectance_scale = st.number_input(
                "Reflectance divisor",
                min_value=0.0001,
                value=10000.0,
                step=100.0,
                format="%.4f",
                key=f"scale_{image_index}_{uploaded_image.name}",
                help=(
                    "Change this if needed to match your product metadata (e.g., 10000 for common "
                    "integer surface reflectance, or 1 for already-normalized data)."
                ),
            )
            image_inputs.append(
                {
                    "name": uploaded_image.name,
                    "bytes": image_bytes,
                    "date": selected_date,
                    "band_map": band_map,
                    "reflectance_scale": float(reflectance_scale),
                }
            )

run_review = st.button(
    "Analyze and prepare review report",
    type="primary",
    disabled=not image_uploads or len(image_uploads) > 2,
)

if run_review:
    if boundary_error:
        st.error("Fix the uploaded boundary or choose another area option before analysis.")
        st.stop()
    if area_choice == "Uploaded boundary" and boundary_upload is None:
        st.error("Upload a GeoJSON boundary or choose the image-footprint sample area.")
        st.stop()
    if area_choice != "Uploaded boundary":
        boundary_geojson = None

    with st.spinner("Validating inputs, sampling imagery, and retrieving document evidence…"):
        analyzed_images: list[dict[str, Any]] = []
        image_errors: list[str] = []
        for image_input in image_inputs:
            try:
                image = analyze_geotiff(
                    image_input["bytes"],
                    image_input["band_map"],
                    image_input["date"],
                    reflectance_scale=image_input["reflectance_scale"],
                    boundary_geojson=boundary_geojson,
                )
                image["name"] = image_input["name"]
                analyzed_images.append(image)
            except ImageryValidationError as error:
                image_errors.append(f"{image_input['name']}: {error}")

        if len(analyzed_images) == 2:
            comparison = compare_images(analyzed_images[0], analyzed_images[1])
        elif len(image_inputs) == 2:
            comparison = {
                "available": False,
                "reason": "A valid analysis could not be completed for both images.",
            }
        elif analyzed_images and not analyzed_images[0]["date"]:
            comparison = {
                "available": False,
                "reason": "The image acquisition date was not supplied; change detection needs two dated images.",
            }
        else:
            comparison = {
                "available": False,
                "reason": "Only one image was supplied; change detection needs two dated, aligned images.",
            }

        searchable_documents: list[tuple[str, list[tuple[str, str]]]] = []
        document_errors: list[str] = []
        for document_upload in document_uploads or []:
            try:
                extracted = extract_passages(document_upload.name, document_upload.getvalue())
                searchable_documents.append((document_upload.name, extracted))
            except DocumentValidationError as error:
                document_errors.append(str(error))

        query_parts = [retrieval_query.strip()]
        if area_label.strip():
            query_parts.append(area_label.strip())
        if comparison.get("available"):
            query_parts.extend(
                transition
                for transition in comparison["transition_counts"]
                for _ in range(1)
            )
        for image in analyzed_images:
            query_parts.extend(image["class_labels"])
        passages = retrieve_passages(searchable_documents, " ".join(query_parts))

        gaps: list[str] = []
        if area_choice == "No boundary":
            gaps.append("No area-of-interest boundary was supplied; statistics cover the full valid image footprint.")
        elif area_choice == "Image footprint (sample area)":
            gaps.append("The image footprint is used as a sample area, not as a verified planning boundary.")
        if not any(image.get("date") for image in analyzed_images):
            gaps.append("No acquisition date is available for the analyzed imagery.")
        elif any(not image.get("date") for image in analyzed_images):
            gaps.append("At least one analyzed image has no acquisition date.")
        if len(analyzed_images) < 2:
            gaps.append("A second valid, dated image is missing; temporal change detection is unavailable.")
        if len(analyzed_images) == 2 and not comparison.get("available"):
            gaps.append(f"Paired-image comparison unavailable: {comparison['reason']}")
        if not searchable_documents:
            gaps.append("No searchable planning/zoning document was supplied.")
        elif not passages:
            gaps.append("No supplied planning-document passage matched the search terms.")
        if any(not image["has_swir1"] for image in analyzed_images):
            gaps.append(
                "SWIR1 data are unavailable for at least one image; built-up spectral proxy is omitted."
            )
        gaps.extend(image_errors)
        gaps.extend(document_errors)

        chosen_area_label = (
            area_label.strip()
            or (
                "Uploaded GeoJSON boundary"
                if area_choice == "Uploaded boundary"
                else (
                    "Image footprint (sample area)"
                    if area_choice == "Image footprint (sample area)"
                    else "Full image footprint (no boundary)"
                )
            )
        )
        report = render_report(analyzed_images, comparison, passages, gaps, chosen_area_label)
        st.session_state["review_result"] = {
            "images": analyzed_images,
            "comparison": comparison,
            "passages": passages,
            "gaps": gaps,
            "image_errors": image_errors,
            "document_errors": document_errors,
            "report": report,
            "area_label": chosen_area_label,
        }

result = st.session_state.get("review_result")
if result:
    st.divider()
    st.header("Review evidence")
    for error in result["image_errors"]:
        st.error(error)
    for error in result["document_errors"]:
        st.warning(error)

    image_columns = st.columns(len(result["images"])) if result["images"] else []
    for column, image in zip(image_columns, result["images"]):
        with column:
            st.subheader(image["name"])
            st.caption(
                f"Acquisition date: {image['date'] or 'not supplied'} · "
                f"analysis grid: {image['analysis_pixel_scale']} · CRS: {image['crs']}"
            )
            metric_columns = st.columns(2)
            metric_columns[0].metric("Mean heuristic score", f"{image['average_confidence']:.1%}")
            metric_columns[1].metric("Low-score pixels", f"{image['uncertainty_fraction']:.1%}")
            st.caption("Scores are not calibrated probabilities; low-score pixels are shown as uncertainty.")
            st.dataframe(
                [
                    {
                        "Spectral class": label,
                        "Estimated area": format_area(area),
                        "Share of valid sample": f"{image['class_shares_pct'][label]:.1f}%",
                        "Mean heuristic class score": f"{image['mean_class_scores'][label]:.1%}",
                    }
                    for label, area in image["class_areas_m2"].items()
                ],
                hide_index=True,
                width="stretch",
            )

    comparison = result["comparison"]
    if comparison.get("available"):
        st.subheader("Potential temporal changes")
        metric_columns = st.columns(3)
        metric_columns[0].metric("Potential changed area", format_area(comparison["changed_area_m2"]))
        metric_columns[1].metric("Sampled area changed", f"{comparison['changed_fraction_pct']:.1f}%")
        metric_columns[2].metric("Change confidence proxy", f"{comparison['confidence']:.1%}")
        st.caption(
            f"{comparison['earlier']['date']} → {comparison['later']['date']} · "
            f"{comparison['changed_pixel_count']:,} sampled class flips retained above the "
            f"{0.55:.0%} threshold. Red map pixels mark potential spectral class changes; "
            "not verified construction."
        )
    else:
        st.info(f"Change detection unavailable: {comparison.get('reason', 'No valid image pair.')}")

    if result["images"]:
        map_image = result["images"][0]
        if comparison.get("available"):
            map_image = comparison["later"]
        map_modes = ["Input imagery", "Land-cover classes"]
        if comparison.get("available"):
            map_modes.append("Potential change overlay")
        map_mode = st.radio("Map layer", map_modes, horizontal=True)
        bounds = map_image["bounds"]
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        map_object = folium.Map(location=center, zoom_start=14, tiles=None, control_scale=True)
        if map_mode == "Input imagery":
            ImageOverlay(
                map_image["rgb_png"],
                bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                opacity=0.95,
                name="RGB image preview",
            ).add_to(map_object)
        elif map_mode == "Land-cover classes":
            ImageOverlay(
                map_image["rgb_png"],
                bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                opacity=0.9,
                name="RGB image preview",
            ).add_to(map_object)
            ImageOverlay(
                map_image["classification_png"],
                bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                opacity=0.9,
                name="Heuristic classes",
            ).add_to(map_object)
            st.caption(
                "Class colors: "
                + " · ".join(
                    f"{label} RGB {color}"
                    for label, color in CLASS_COLORS.items()
                    if label in map_image["class_labels"]
                )
            )
        else:
            ImageOverlay(
                comparison["earlier"]["rgb_png"],
                bounds=[
                    [bounds[1], bounds[0]],
                    [bounds[3], bounds[2]],
                ],
                opacity=0.9,
                name=f"Earlier image · {comparison['earlier']['date']}",
            ).add_to(map_object)
            ImageOverlay(
                comparison["change_png"],
                bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                opacity=0.9,
                name="Potential spectral class changes",
            ).add_to(map_object)
            folium.LayerControl().add_to(map_object)
            st.caption("Red pixels show retained class changes; dates are shown above the map.")
        if map_image.get("boundary_geojson"):
            folium.GeoJson(
                map_image["boundary_geojson"],
                name="Area-of-interest boundary",
                style_function=lambda _: {"color": "#f8c537", "weight": 2, "fillOpacity": 0.02},
            ).add_to(map_object)
        map_object.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        st_folium(map_object, use_container_width=True, height=500, key=f"review_map_{map_mode}")

    st.subheader("Planning-document evidence")
    if result["passages"]:
        st.caption("Verbatim excerpts from supplied documents. Retrieval does not determine legal applicability.")
        for passage in result["passages"]:
            with st.expander(f"{passage.document} · {passage.location}"):
                st.write(f"> {passage.text}")
    else:
        st.info("No relevant searchable passages were found in the supplied documents.")

    st.subheader("Data gaps and limitations")
    if result["gaps"]:
        for gap in result["gaps"]:
            st.warning(gap)
    else:
        st.success("All expected inputs were supplied. Human verification is still required.")
    st.caption(
        "All processing runs locally in this app process. The map uses uploaded imagery without external map tiles."
    )

    with st.expander("Review report (Markdown)", expanded=True):
        st.markdown(result["report"])
    st.download_button(
        "Download review report",
        data=result["report"],
        file_name="satellite-land-use-review.md",
        mime="text/markdown",
    )
