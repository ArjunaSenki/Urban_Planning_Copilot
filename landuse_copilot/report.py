"""Traceable human-review report rendering."""

from __future__ import annotations

from datetime import date
from typing import Any


def format_area(area_m2: float) -> str:
    if area_m2 >= 1_000_000:
        return f"{area_m2 / 1_000_000:,.2f} km²"
    if area_m2 >= 10_000:
        return f"{area_m2 / 10_000:,.2f} ha"
    return f"{area_m2:,.0f} m²"


def _source_date(image: dict[str, Any]) -> str:
    value: date | None = image.get("date")
    return value.isoformat() if value else "Not supplied"


def render_report(
    images: list[dict[str, Any]],
    comparison: dict[str, Any],
    passages: list[Any],
    data_gaps: list[str],
    area_label: str,
) -> str:
    """Render a concise, exportable report without adding unsupported conclusions."""
    lines = [
        "# Satellite land-use review",
        "",
        f"- **Area:** {area_label}",
        "- **Purpose:** Preliminary evidence for municipal human review; not a legal determination.",
        "",
        "## Image evidence",
    ]
    for image in images:
        lines.extend(
            [
                f"- **{image['name']}** — acquisition date: {_source_date(image)}; "
                f"CRS: `{image['crs']}`; source dimensions: {image['width']} × {image['height']} pixels.",
                f"  Classification proxy: average top heuristic class score {image['average_confidence']:.1%}; "
                f"{image['uncertainty_fraction']:.1%} of sampled pixels fall below the "
                f"{0.55:.0%} confidence threshold. These are heuristic scores, not calibrated probabilities.",
            ]
        )
        lines.extend(
            [
                "",
                f"![{image['name']} sampled RGB and land-cover classification overlay]("
                f"{image['classification_map_png']})",
                "",
            ]
        )
        lines.extend(
            f"  - {label}: {format_area(area)} "
            f"({image['class_shares_pct'][label]:.1f}% of valid sampled area; "
            f"mean heuristic class score {image['mean_class_scores'][label]:.1%})"
            for label, area in image["class_areas_m2"].items()
        )
    lines.extend(["", "## Temporal comparison"])
    if comparison.get("available"):
        earlier, later = comparison["earlier"], comparison["later"]
        lines.extend(
            [
                f"- **Dates compared:** {_source_date(earlier)} to {_source_date(later)}.",
                f"- **Potential changed area:** {format_area(comparison['changed_area_m2'])} "
                f"({comparison['changed_fraction_pct']:.1f}% of overlapping valid sampled area).",
                f"- **Comparison confidence:** {comparison['confidence']:.1%} average minimum per-pixel "
                "classification score on retained changes; heuristic, not calibrated.",
                "- Changes are spectral class differences in pixels above the confidence threshold; "
                "they are potential changes, not verified construction or land-use violations.",
            ]
        )
        lines.extend(
            [
                "",
                f"![Potential spectral class-change overlay, {_source_date(earlier)} to "
                f"{_source_date(later)}]({comparison['change_map_png']})",
                "",
            ]
        )
        if comparison["transition_counts"]:
            lines.append("- **Observed class transitions:**")
            lines.extend(
                f"  - {transition}: {count:,} sampled pixels"
                for transition, count in sorted(comparison["transition_counts"].items())
            )
        if comparison["low_confidence_flip_count"]:
            lines.append(
                f"- **Unresolved lower-confidence class flips:** "
                f"{comparison['low_confidence_flip_count']:,} sampled pixels (excluded from changed-area estimate)."
            )
    else:
        lines.append(
            f"- **Unavailable:** {comparison.get('reason', 'A valid paired-image comparison was not supplied.')}"
        )
    lines.extend(["", "## Planning-document evidence"])
    if passages:
        lines.append(
            "Retrieved text is quoted verbatim for human review; retrieval does not establish that a rule applies."
        )
        for passage in passages:
            lines.extend(
                [
                    f"- **{passage.document}, {passage.location}** — {passage.text}",
                ]
            )
    else:
        lines.append("No relevant, searchable passages were retrieved from supplied planning documents.")
    lines.extend(["", "## Data gaps and limitations"])
    if data_gaps:
        lines.extend(f"- {gap}" for gap in data_gaps)
    else:
        lines.append("- No additional input gaps detected; all results still require human verification.")
    lines.extend(
        [
            "- The land-cover output is an interpretable spectral-index heuristic, not a trained or validated "
            "land-cover model. Displayed class scores are not calibrated probabilities. No standalone EuroSAT "
            "baseline is used to classify these uploaded images.",
            "- Images are sampled to at most 512 pixels on their longest side. Areas are approximate and reflect "
            "the sampled image resolution, boundary rasterization, CRS accuracy, and the selected reflectance divisor.",
            "- Image alignment is checked from CRS, dimensions, and affine geotransform only; this does not prove "
            "sensor-level co-registration, equivalent acquisition conditions, or radiometric calibration.",
            "- Zoning passages are retrieval evidence only. No finding of unlawful activity or legal conclusion is made.",
        ]
    )
    return "\n".join(lines)
