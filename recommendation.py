"""
Rule-based recommendation engine.

Compares a meal's estimated macro totals against the user's per-meal target
(daily goal / meals_per_day) and flags each macro as surplus, on-target, or
deficit. Thresholds (110% / 80%) were chosen empirically to absorb the
estimation error characterised in Chapter 6 of the report rather than flag on
every small deviation from 100%.
"""
from typing import Dict

SURPLUS_RATIO = 1.10
DEFICIT_RATIO = 0.80

DEFAULT_DAILY_GOALS = {"calories": 2000, "protein": 50, "carbs": 275, "fat": 70}

SUGGESTIONS = {
    ("protein", "deficit"): "Add a protein source such as chicken, eggs, or legumes.",
    ("carbs", "surplus"): "Consider a smaller carbohydrate portion next time.",
    ("fat", "surplus"): "Watch for hidden fats (dressings, fried items) in this meal.",
    ("calories", "surplus"): "This meal is calorie-dense relative to your per-meal target.",
    ("calories", "deficit"): "This meal is lighter than your per-meal target.",
}


def recommend(meal_totals: Dict[str, float], daily_goals: Dict[str, float] = None,
              meals_per_day: int = 3) -> dict:
    daily_goals = daily_goals or DEFAULT_DAILY_GOALS
    per_meal_target = {k: v / meals_per_day for k, v in daily_goals.items()}

    status, notes = {}, []
    for macro, total in meal_totals.items():
        target = per_meal_target.get(macro)
        if not target:
            continue
        ratio = total / target
        if ratio > SURPLUS_RATIO:
            status[macro] = "surplus"
        elif ratio < DEFICIT_RATIO:
            status[macro] = "deficit"
        else:
            status[macro] = "on-target"
        if (macro, status[macro]) in SUGGESTIONS:
            notes.append(SUGGESTIONS[(macro, status[macro])])

    return {"status": status, "notes": notes, "per_meal_target": per_meal_target}
