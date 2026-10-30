"""GeoTIFF inspection, spectral-proxy classification, and paired comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import BytesIO
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.io import MemoryFile
from shapely.geometry import mapping, shape
from shapely.ops import transform as transform_geometry

MAX_UPLOAD_BYTES = 150 * 1024 * 1024
MAX_ANALYSIS_SIDE = 512
MAX_SOURCE_PIXELS = 100_000_000
CONFIDENCE_THRESHOLD = 0.55

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "Water": (40, 125, 235),
    "Vegetation": (53, 161, 91),
    "Built-up spectral proxy": (226, 123, 55),
    "Other surface": (186, 168, 117),
}

BAND_ALIASES = {
    "red": ("red", "b04", "band 4"),
    "green": ("green", "b03", "band 3"),
    "blue": ("blue", "b02", "band 2"),
    "nir": ("nir", "near infrared", "near-infrared", "b08", "b8", "band 8"),
    "swir1": ("swir1", "swir 1", "shortwave infrared", "b11", "b12"),
}


class ImageryValidationError(ValueError):
    """An uploaded image cannot be safely interpreted as georeferenced imagery."""


@dataclass(frozen=True)
class RasterInfo:
    width: int
    height: int
    band_count: int
    crs: str
    descriptions: tuple[str | None, ...]
    bounds: tuple[float, float, float, float]
    transform: Any
    nodata: float | None
    acquisition_date: date | None


def _read_acquisition_date(tags: dict[str, str]) -> date | None:
    for key in ("ACQUISITION_DATE", "DATE_ACQUIRED", "TIFFTAG_DATETIME"):
        raw = tags.get(key)
        if not raw:
            continue
        candidate = raw.strip()[:10]
        for pattern in ("%Y-%m-%d", "%Y:%m:%d"):
            try:
                return date.fromisoformat(candidate) if pattern == "%Y-%m-%d" else date(
                    *map(int, candidate.split(":"))
                )
            except (TypeError, ValueError):
                pass
    return None


def inspect_geotiff(file_bytes: bytes) -> RasterInfo:
    """Validate a GeoTIFF header and expose metadata without loading its pixels."""
    if not file_bytes:
        raise ImageryValidationError("The uploaded image is empty.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise ImageryValidationError("Each image must be 150 MB or smaller.")
    try:
        with MemoryFile(file_bytes) as memory_file, memory_file.open() as dataset:
            if dataset.driver != "GTiff":
                raise ImageryValidationError("Only georeferenced GeoTIFF (.tif/.tiff) images are supported.")
            if dataset.crs is None:
                raise ImageryValidationError("This image has no coordinate reference system (CRS).")
            if dataset.transform is None or dataset.transform.is_identity:
                raise ImageryValidationError("This image has no usable georeferencing transform.")
            if dataset.width < 1 or dataset.height < 1 or dataset.count < 1:
                raise ImageryValidationError("The image has no readable pixels or bands.")
            if dataset.width * dataset.height > MAX_SOURCE_PIXELS:
                raise ImageryValidationError("Images larger than 100 million pixels are not supported.")
            return RasterInfo(
                width=dataset.width,
                height=dataset.height,
                band_count=dataset.count,
                crs=dataset.crs.to_string(),
                descriptions=tuple(dataset.descriptions),
                bounds=(dataset.bounds.left, dataset.bounds.bottom, dataset.bounds.right, dataset.bounds.top),
                transform=dataset.transform,
                nodata=dataset.nodata,
                acquisition_date=_read_acquisition_date(dataset.tags()),
            )
    except ImageryValidationError:
        raise
    except (rasterio.errors.RasterioError, OSError, ValueError) as error:
        raise ImageryValidationError(
            "The upload could not be opened as a GeoTIFF. Check the file and try again."
        ) from error


def suggested_band_roles(descriptions: tuple[str | None, ...]) -> dict[str, int | None]:
    """Suggest roles only when a band description explicitly matches a known alias."""
    suggestions: dict[str, int | None] = {role: None for role in BAND_ALIASES}
    for index, description in enumerate(descriptions, start=1):
        if not description:
            continue
        normalized = description.strip().lower()
        for role, aliases in BAND_ALIASES.items():
            if suggestions[role] is None and any(alias in normalized for alias in aliases):
                suggestions[role] = index
    return suggestions


def parse_boundary_geojson(file_bytes: bytes) -> dict[str, Any]:
    """Read an RFC 7946 boundary and reject ambiguous or invalid geometries."""
    import json

    try:
        document = json.loads(file_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ImageryValidationError("The boundary is not valid UTF-8 GeoJSON.") from error

    try:
        if document.get("type") == "Feature":
            geometries = [shape(document["geometry"])]
        elif document.get("type") == "FeatureCollection":
            geometries = [shape(feature["geometry"]) for feature in document["features"]]
        elif document.get("type") in ("Polygon", "MultiPolygon"):
            geometries = [shape(document)]
        else:
            raise ImageryValidationError("Use a GeoJSON Polygon or MultiPolygon boundary.")
        if not geometries:
            raise ImageryValidationError("The GeoJSON contains no boundary geometry.")
        from shapely.ops import unary_union

        boundary = unary_union(geometries)
        if boundary.geom_type not in ("Polygon", "MultiPolygon") or boundary.is_empty:
            raise ImageryValidationError("The boundary must contain a Polygon or MultiPolygon.")
        if not boundary.is_valid or boundary.area <= 0:
            raise ImageryValidationError("The GeoJSON boundary is invalid or has zero area.")
        return mapping(boundary)
    except (KeyError, TypeError, IndexError, ValueError) as error:
        if isinstance(error, ImageryValidationError):
            raise
        raise ImageryValidationError("The GeoJSON has a malformed or invalid boundary.") from error


def _scaled_shape(width: int, height: int) -> tuple[int, int]:
    scale = min(1.0, MAX_ANALYSIS_SIDE / max(width, height))
    return max(1, round(height * scale)), max(1, round(width * scale))


def _boundary_in_raster_crs(
    boundary_geojson: dict[str, Any] | None, raster_crs: CRS
) -> dict[str, Any] | None:
    if boundary_geojson is None:
        return None
    boundary = shape(boundary_geojson)
    transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    projected = transform_geometry(transformer.transform, boundary)
    if projected.is_empty:
        raise ImageryValidationError("The supplied boundary does not overlap the image.")
    return mapping(projected)


def _area_per_pixel_m2(crs: CRS, transform: Any) -> float:
    if crs.is_geographic:
        raise ValueError("Geographic pixel areas must be calculated from geodesic polygon areas.")
    unit_factor = crs.axis_info[0].unit_conversion_factor if crs.axis_info else None
    if not unit_factor or not np.isfinite(unit_factor):
        raise ValueError("The image CRS has no recognized linear units.")
    return abs(transform.a * transform.e - transform.b * transform.d) * unit_factor**2


def _geographic_mask_area_m2(mask: np.ndarray, transform: Any, crs: CRS) -> float:
    geod = crs.get_geod()
    total = 0.0
    center_column = mask.shape[1] / 2
    for row in np.flatnonzero(np.any(mask, axis=1)):
        pixel_polygon = [
            transform * (center_column, row),
            transform * (center_column + 1, row),
            transform * (center_column + 1, row + 1),
            transform * (center_column, row + 1),
        ]
        longitude, latitude = zip(*pixel_polygon)
        pixel_area = abs(geod.polygon_area_perimeter(longitude, latitude)[0])
        total += int(np.count_nonzero(mask[row])) * pixel_area
    return total


def _mask_area_m2(mask: np.ndarray, crs: CRS, transform: Any) -> float:
    if not np.any(mask):
        return 0.0
    if crs.is_geographic:
        return _geographic_mask_area_m2(mask, transform, crs)
    return float(np.count_nonzero(mask) * _area_per_pixel_m2(crs, transform))


def _image_rgb(bands: dict[str, np.ndarray]) -> np.ndarray:
    rgb = np.stack((bands["red"], bands["green"], bands["blue"]), axis=-1).astype("float32")
    output = np.zeros_like(rgb, dtype="uint8")
    for channel in range(3):
        values = rgb[:, :, channel]
        valid = values[np.isfinite(values)]
        if valid.size == 0:
            continue
        low, high = np.percentile(valid, (2, 98))
        if high <= low:
            high = low + 1
        scaled = np.clip((values - low) * (255.0 / (high - low)), 0, 255)
        output[:, :, channel] = np.where(np.isfinite(scaled), scaled, 0).astype("uint8")
    return output


def _png_data_url(pixels: np.ndarray) -> str:
    image = Image.fromarray(pixels)
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    import base64

    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _composite_png(background: np.ndarray, overlay: np.ndarray) -> str:
    alpha = overlay[:, :, 3:4].astype("float32") / 255
    blended = np.clip(
        background.astype("float32") * (1 - alpha)
        + overlay[:, :, :3].astype("float32") * alpha,
        0,
        255,
    ).astype("uint8")
    return _png_data_url(blended)


def analyze_geotiff(
    file_bytes: bytes,
    band_map: dict[str, int | None],
    acquisition_date: date | None,
    reflectance_scale: float = 10000,
    boundary_geojson: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify a small grid sampled from a georeferenced GeoTIFF."""
    info = inspect_geotiff(file_bytes)
    required = ("red", "green", "blue", "nir")
    missing = [role for role in required if band_map.get(role) is None]
    if missing:
        raise ImageryValidationError(
            "Assign a band for each required role: " + ", ".join(missing).upper() + "."
        )
    selected = [index for index in band_map.values() if index is not None]
    if any(not isinstance(index, int) or index < 1 or index > info.band_count for index in selected):
        raise ImageryValidationError("A selected band number is outside this image's band range.")
    if len(selected) != len(set(selected)):
        raise ImageryValidationError("Assign each spectral role to a different band.")
    if reflectance_scale <= 0 or not np.isfinite(reflectance_scale):
        raise ImageryValidationError("The reflectance divisor must be a positive number.")

    target_height, target_width = _scaled_shape(info.width, info.height)
    crs = CRS.from_user_input(info.crs)
    projected_boundary = _boundary_in_raster_crs(boundary_geojson, crs)

    try:
        with MemoryFile(file_bytes) as memory_file, memory_file.open() as dataset:
            transform = dataset.transform @ dataset.transform.scale(
                dataset.width / target_width, dataset.height / target_height
            )
            arrays: dict[str, np.ndarray] = {}
            valid = np.ones((target_height, target_width), dtype=bool)
            for role, index in band_map.items():
                if index is None:
                    continue
                band = dataset.read(
                    index,
                    out_shape=(target_height, target_width),
                    resampling=Resampling.average,
                    masked=True,
                )
                arrays[role] = (
                    np.asarray(band.astype(np.float32).filled(np.nan), dtype=np.float32)
                    / reflectance_scale
                )
                valid &= ~np.ma.getmaskarray(band) & np.isfinite(arrays[role])
            if projected_boundary is not None:
                inside = geometry_mask(
                    [projected_boundary],
                    out_shape=valid.shape,
                    transform=transform,
                    invert=True,
                )
                valid &= inside
            if not np.any(valid):
                raise ImageryValidationError(
                    "No valid image pixels fall inside the supplied boundary. Check its location and CRS."
                )
    except ImageryValidationError:
        raise
    except (rasterio.errors.RasterioError, OSError, ValueError) as error:
        raise ImageryValidationError("One or more selected image bands could not be read.") from error

    red, green, nir = arrays["red"], arrays["green"], arrays["nir"]
    ndvi = np.divide(nir - red, nir + red, out=np.zeros_like(nir), where=(nir + red) != 0)
    ndwi = np.divide(green - nir, green + nir, out=np.zeros_like(nir), where=(green + nir) != 0)
    scores = [2.8 * ndwi, 2.8 * ndvi]
    labels = ["Water", "Vegetation"]
    if "swir1" in arrays:
        swir = arrays["swir1"]
        ndbi = np.divide(swir - nir, swir + nir, out=np.zeros_like(nir), where=(swir + nir) != 0)
        scores.extend((2.3 * ndbi, np.zeros_like(ndvi)))
        labels.extend(("Built-up spectral proxy", "Other surface"))
    else:
        scores.append(np.zeros_like(ndvi))
        labels.append("Other surface")

    logits = np.stack(scores, axis=-1)
    logits -= np.max(logits, axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    predicted = np.argmax(probabilities, axis=-1)
    pixel_confidence = np.max(probabilities, axis=-1)
    predicted[~valid] = -1

    class_areas = {
        label: _mask_area_m2((predicted == index) & valid, crs, transform)
        for index, label in enumerate(labels)
    }
    valid_area = sum(class_areas.values())
    class_shares = {
        label: (area / valid_area * 100 if valid_area else 0.0)
        for label, area in class_areas.items()
    }
    mean_class_scores = {
        label: float(np.mean(probabilities[valid, index]))
        for index, label in enumerate(labels)
    }
    valid_confidence = pixel_confidence[valid]
    uncertainty_fraction = float(np.mean(valid_confidence < CONFIDENCE_THRESHOLD))
    average_confidence = float(np.mean(valid_confidence))

    rgb = _image_rgb(arrays)
    rgba = np.zeros((*valid.shape, 4), dtype="uint8")
    for index, label in enumerate(labels):
        rgba[predicted == index, :3] = CLASS_COLORS[label]
        rgba[predicted == index, 3] = 125
    class_rgba = rgba
    full_bounds = rasterio.transform.array_bounds(target_height, target_width, transform)
    geographic_bounds = rasterio.warp.transform_bounds(
        crs, "EPSG:4326", *full_bounds, densify_pts=21
    )
    if not all(np.isfinite(geographic_bounds)):
        raise ImageryValidationError("The image extent cannot be displayed in geographic coordinates.")

    return {
        "name": "",
        "date": acquisition_date,
        "band_map": {role: index for role, index in band_map.items() if index is not None},
        "has_swir1": "swir1" in arrays,
        "reflectance_scale": float(reflectance_scale),
        "width": info.width,
        "height": info.height,
        "sample_width": target_width,
        "sample_height": target_height,
        "crs": info.crs,
        "transform": transform,
        "crs_object": crs,
        "bounds": tuple(geographic_bounds),
        "valid_mask": valid,
        "predicted": predicted,
        "probabilities": probabilities,
        "pixel_confidence": pixel_confidence,
        "class_labels": labels,
        "class_areas_m2": class_areas,
        "class_shares_pct": class_shares,
        "mean_class_scores": mean_class_scores,
        "valid_area_m2": float(valid_area),
        "average_confidence": average_confidence,
        "uncertainty_fraction": uncertainty_fraction,
        "rgb_pixels": rgb,
        "rgb_png": _png_data_url(rgb),
        "classification_png": _png_data_url(class_rgba),
        "classification_map_png": _composite_png(rgb, class_rgba),
        "boundary_geojson": boundary_geojson,
        "analysis_pixel_scale": f"{target_width} x {target_height}",
    }


def compare_images(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Compare two exactly co-registered images; otherwise return an explicit limitation."""
    if not first.get("date") or not second.get("date"):
        return {"available": False, "reason": "Both images need an acquisition date."}
    if first["date"] == second["date"]:
        return {"available": False, "reason": "The two images must have different acquisition dates."}
    earlier, later = sorted((first, second), key=lambda image: image["date"])
    if earlier["crs"] != later["crs"]:
        return {"available": False, "reason": "The images use different coordinate reference systems."}
    if (earlier["width"], earlier["height"]) != (later["width"], later["height"]):
        return {"available": False, "reason": "The images have different pixel dimensions and are not confirmed aligned."}
    if not np.allclose(
        tuple(earlier["transform"]), tuple(later["transform"]), rtol=0, atol=1e-8
    ):
        return {"available": False, "reason": "The images have different geotransforms and are not confirmed co-registered."}
    if earlier["band_map"].keys() != later["band_map"].keys():
        return {"available": False, "reason": "The images do not have matching assigned spectral band roles."}
    if earlier["reflectance_scale"] != later["reflectance_scale"]:
        return {"available": False, "reason": "The images use different reflectance divisors."}
    if earlier["class_labels"] != later["class_labels"]:
        return {"available": False, "reason": "The images do not support the same classification classes."}

    valid = earlier["valid_mask"] & later["valid_mask"]
    if not np.any(valid):
        return {"available": False, "reason": "The images have no overlapping valid pixels."}
    flips = (
        (earlier["predicted"] != later["predicted"])
        & (earlier["predicted"] >= 0)
        & (later["predicted"] >= 0)
    )
    qualified = (
        flips
        & (earlier["pixel_confidence"] >= CONFIDENCE_THRESHOLD)
        & (later["pixel_confidence"] >= CONFIDENCE_THRESHOLD)
    )
    low_confidence_flips = int(np.count_nonzero(flips & ~qualified))
    crs = earlier["crs_object"]
    transform = earlier["transform"]
    area_m2 = _mask_area_m2(qualified, crs, transform)
    qualified_count = int(np.count_nonzero(qualified))
    transition_counts: dict[str, int] = {}
    for old_index, old_label in enumerate(earlier["class_labels"]):
        for new_index, new_label in enumerate(later["class_labels"]):
            count = int(
                np.count_nonzero(
                    qualified
                    & (earlier["predicted"] == old_index)
                    & (later["predicted"] == new_index)
                )
            )
            if count:
                transition_counts[f"{old_label} → {new_label}"] = count
    confidence = (
        float(
            np.mean(
                np.minimum(
                    earlier["pixel_confidence"][qualified],
                    later["pixel_confidence"][qualified],
                )
            )
        )
        if qualified_count
        else 0.0
    )
    change_rgba = np.zeros((*qualified.shape, 4), dtype="uint8")
    change_rgba[qualified] = (231, 58, 76, 195)
    change_map_png = _composite_png(earlier["rgb_pixels"], change_rgba)
    return {
        "available": True,
        "earlier": earlier,
        "later": later,
        "changed_area_m2": area_m2,
        "changed_area_ha": area_m2 / 10_000,
        "changed_fraction_pct": (
            float(np.count_nonzero(qualified) / np.count_nonzero(valid) * 100)
            if np.any(valid)
            else 0.0
        ),
        "confidence": confidence,
        "changed_pixel_count": qualified_count,
        "low_confidence_flip_count": low_confidence_flips,
        "transition_counts": transition_counts,
        "change_png": _png_data_url(change_rgba),
        "change_map_png": change_map_png,
        "boundary_geojson": earlier["boundary_geojson"],
        "bounds": earlier["bounds"],
    }
