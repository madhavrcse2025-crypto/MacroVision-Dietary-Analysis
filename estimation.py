"""
Portion-mass and macronutrient estimation.

Converts each detected bounding box into an estimated mass (grams) using a
per-class density coefficient learned from the 60-plate reference set
(see data/reference_plates/README.md), then scales the USDA-based per-100g
nutrition values accordingly.
"""
import json
from pathlib import Path
from typing import Dict, List

NUTRITION_TABLE_PATH = Path(__file__).parent / "nutrition_table.json"
DEFAULT_DENSITY_COEF = 0.00040  # g per pixel^2, fallback for unseen classes

with open(NUTRITION_TABLE_PATH) as f:
    NUTRITION_TABLE: Dict[str, dict] = json.load(f)


def estimate_item(label: str, bbox_area: float) -> dict:
    """Return {mass_g, calories, protein, carbs, fat} for one detected item."""
    entry = NUTRITION_TABLE.get(label)
    if entry is None:
        # Unseen class: fall back to a neutral average profile
        entry = {"calories": 150, "protein": 6.0, "carbs": 20.0, "fat": 5.0,
                 "density_coef": DEFAULT_DENSITY_COEF}
    mass_g = bbox_area * entry["density_coef"]
    scale = mass_g / 100.0
    return {
        "label": label,
        "mass_g": round(mass_g, 1),
        "calories": round(entry["calories"] * scale, 1),
        "protein": round(entry["protein"] * scale, 1),
        "carbs": round(entry["carbs"] * scale, 1),
        "fat": round(entry["fat"] * scale, 1),
    }


def estimate_meal(detections: List) -> dict:
    """Aggregate per-item estimates into meal totals and per-item breakdown."""
    items = [estimate_item(d.label, d.bbox_area) for d in detections]
    totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
    for item in items:
        for k in totals:
            totals[k] += item[k]
    totals = {k: round(v, 1) for k, v in totals.items()}
    return {"items": items, "totals": totals}
