from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from backend.api.service import get_dashboard_payload, predict_text


class PredictionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_id: str = Field(alias="modelId")
    profile: str = "sample"
    text: str


app = FastAPI(title="Spam Detector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
    allow_origins=["*"],
)


@app.get("/api/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/dashboard")
def dashboard(profile: str = "sample") -> dict:
    try:
        return get_dashboard_payload(profile)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict:
    try:
        if not request.text.strip():
            raise ValueError("Prediction text cannot be empty.")
        return predict_text(
            profile=request.profile,
            model_id=request.model_id,
            text=request.text,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
