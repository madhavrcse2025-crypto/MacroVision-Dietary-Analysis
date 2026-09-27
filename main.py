"""
FastAPI backend for MacroVision.

Single endpoint: POST /predict, accepting a meal photo and an optional user
goal profile, returning detections, estimated macros, and a recommendation.
"""
import json

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.detection import FoodDetector
from backend.estimation import estimate_meal
from backend.recommendation import DEFAULT_DAILY_GOALS, recommend

app = FastAPI(title="MacroVision API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = FoodDetector()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), goals: str = Form(default=None)):
    raw = await file.read()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

    detections = detector.infer(image)
    meal = estimate_meal(detections)

    user_goals = json.loads(goals) if goals else DEFAULT_DAILY_GOALS
    rec = recommend(meal["totals"], daily_goals=user_goals)

    return JSONResponse({
        "detections": [
            {"label": d.label, "confidence": round(d.confidence, 3), "bbox": d.bbox}
            for d in detections
        ],
        "macros": meal,
        "recommendation": rec,
    })
