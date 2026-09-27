# MacroVision - AI-Driven Dietary Analysis and Wellness Guide

Estimate the calorie and macronutrient content of a meal from a single photograph.
Built as a Project-Based Learning (PBL) submission for Machine Learning (CS3505),
Department of Computer Science and Engineering, Chennai Institute of Technology.

**Team:** Madhav R (Systems & Integration Lead) - Sabari Saatvik KG (Team Lead)
**Mentor:** S. Raja Priya, M.E

## What it does

1. **Detect** - a YOLOv8 model fine-tuned on a Food-101 subset (101 classes) localises
   and classifies every food item in one photo (mAP@0.5 = 88.5%, ~28 ms/frame on a Tesla T4).
2. **Estimate** - each detected box's pixel area is converted to an estimated portion
   mass, then looked up against a USDA-based nutrition table for calories, protein, carbs,
   and fat. Validated against 60 manually weighed reference plates (calorie MAE = 34.2 kcal,
   R-squared = 0.89).
3. **Recommend** - a rule-based engine compares the meal's totals against the user's
   per-meal macro targets and flags surplus (>110%) or deficit (<80%) categories.
4. **Serve** - a FastAPI backend exposes a single `/predict` endpoint; a lightweight
   web dashboard renders the bounding boxes, macro breakdown, and recommendation status.

See the full PBL report (`docs/MacroVision_PBL_Report.docx`) for the literature review,
architecture diagrams, training procedure, and evaluation results.

## Repository layout

```
MacroVision-Dietary-Analysis/
|- backend/
|  |- main.py              FastAPI app (/predict endpoint)
|  |- detection.py         YOLOv8 inference wrapper
|  |- estimation.py        Area-to-mass regression + nutrition look-up
|  |- recommendation.py    Rule-based surplus/deficit engine
|  `- nutrition_table.json USDA-based per-100g nutrition table (excerpt)
|- training/
|  |- prepare_dataset.py   Converts Food-101 labels to YOLO bounding-box format
|  |- train_yolov8.py      Fine-tuning script (augmentation, AdamW, cosine LR)
|  `- augmentation.yaml    Flip / rotation / HSV-jitter augmentation config
|- frontend/
|  `- index.html           Minimal capture -> result -> dashboard demo (vanilla JS)
|- data/reference_plates/
|  `- README.md            How the 60-plate calibration/validation set was collected
|- requirements.txt
`- LICENSE
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train (optional -- a fine-tuned checkpoint can be dropped into backend/ instead)
python training/prepare_dataset.py --food101-root /path/to/food-101
python training/train_yolov8.py --data food101.yaml --epochs 50

# Serve
uvicorn backend.main:app --reload --port 8000

# Open frontend/index.html in a browser (defaults to http://localhost:8000)
```

## Status

Detection, estimation, recommendation, and the backend service are complete and evaluated
(Chapter 6 of the report). The React/Tailwind production frontend and on-device mobile
build are tracked under Future Scope.

## License

MIT - see `LICENSE`.
