"""
api.py
FastAPI service that scores one loan applicant with the current
Production champion model.

Run from the PROJECT ROOT:
    python -m uvicorn --app-dir src api:app --reload
Interactive docs:  http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from serving import (
    DECISION_BANDS, MODEL_NAME, allowed_extra_keys, decide,
    features_from_applicant, load_champion, predict_proba,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.champion = load_champion()
    yield


app = FastAPI(title="sMLOps Credit Risk API", version="1.0", lifespan=lifespan)


class Applicant(BaseModel):
    """Input validation (Pydantic): anything outside these rules is rejected
    with HTTP 422 before it reaches the model."""

    model_config = ConfigDict(json_schema_extra={"example": {
        "amt_income_total": 150000, "amt_credit": 500000, "amt_annuity": 25000,
        "amt_goods_price": 450000, "cnt_children": 1, "age_years": 35,
        "years_employed": 5, "ext_source_1": 0.5, "ext_source_2": 0.6,
        "ext_source_3": 0.55, "name_contract_type": "Cash loans",
        "code_gender": "F", "flag_own_car": "N", "flag_own_realty": "Y",
        "name_income_type": "Working",
        "name_education_type": "Secondary / secondary special",
        "name_family_status": "Married", "name_housing_type": "House / apartment",
    }})

    amt_income_total: float = Field(..., gt=0, le=5e8)
    amt_credit: float = Field(..., gt=0, le=1e8)
    amt_annuity: float | None = Field(None, gt=0, le=5e6)
    amt_goods_price: float | None = Field(None, gt=0, le=1e8)
    cnt_children: int = Field(0, ge=0, le=30)
    age_years: float = Field(..., ge=18, le=100)
    years_employed: float | None = Field(
        None, ge=0, le=70, description="Leave empty if not employed (pensioner / unemployed)"
    )
    ext_source_1: float | None = Field(None, ge=0, le=1)
    ext_source_2: float | None = Field(None, ge=0, le=1)
    ext_source_3: float | None = Field(None, ge=0, le=1)

    name_contract_type: Literal["Cash loans", "Revolving loans"]
    code_gender: Literal["F", "M", "XNA"]
    flag_own_car: Literal["Y", "N"]
    flag_own_realty: Literal["Y", "N"]
    name_income_type: Literal[
        "Working", "State servant", "Commercial associate", "Pensioner",
        "Unemployed", "Student", "Businessman", "Maternity leave",
    ]
    name_education_type: Literal[
        "Secondary / secondary special", "Higher education",
        "Incomplete higher", "Lower secondary", "Academic degree",
    ]
    name_family_status: Literal[
        "Married", "Single / not married", "Civil marriage",
        "Separated", "Widow", "Unknown",
    ]
    name_housing_type: Literal[
        "House / apartment", "With parents", "Municipal apartment",
        "Rented apartment", "Office apartment", "Co-op apartment",
    ]

    # Optional: any other raw application_train.csv column by its dataset
    # name (e.g. "OCCUPATION_TYPE": "Laborers"). Anything not supplied is
    # treated as missing.
    extra_features: dict[str, float | str] = Field(default_factory=dict)


class Prediction(BaseModel):
    default_probability: float
    decision: Literal["approve", "manual_review", "reject"]
    model_version: str


def _model_info(champ: dict) -> dict:
    return {
        "model_name": MODEL_NAME,
        "version": champ["version"],
        "source": champ["source"],
        "n_features": len(champ["columns"]),
        "decision_bands": DECISION_BANDS,
    }


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_loaded": hasattr(request.app.state, "champion")}


@app.get("/model-info")
def model_info(request: Request):
    return _model_info(request.app.state.champion)


@app.post("/reload")
def reload_model(request: Request):
    """Pick up a newly promoted champion without restarting the server."""
    request.app.state.champion = load_champion()
    return _model_info(request.app.state.champion)


@app.post("/predict", response_model=Prediction)
def predict(applicant: Applicant, request: Request):
    champ = request.app.state.champion
    extras = applicant.extra_features

    bad = sorted(set(extras) - allowed_extra_keys(champ["columns"]))
    if bad:
        raise HTTPException(422, f"Unknown or reserved extra_features: {bad}")

    fields = applicant.model_dump(exclude={"extra_features"})
    try:
        x = features_from_applicant(fields, extras, champ["columns"])
    except (ValueError, TypeError) as e:
        raise HTTPException(422, f"Could not process features: {e}")

    p = float(predict_proba(champ["model"], x)[0])
    return {
        "default_probability": round(p, 4),
        "decision": decide(p),
        "model_version": champ["version"],
    }