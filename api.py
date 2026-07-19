# app/api.py
import sys
import os
from fastapi import FastAPI, HTTPException # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from pydantic import BaseModel
from typing import Optional
import uvicorn # type: ignore

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Customer_Risk_Escalation.pipeline.predict_pipeline import PredictPipeline
from Customer_Risk_Escalation.exceptions.exception import CustomException
from Customer_Risk_Escalation.logger.logging import logging


#  Pydantic input model 
class TicketInput(BaseModel):
    customer_segment      : str
    product_area          : str
    issue_type            : str
    priority              : str
    status                : str
    sla_plan              : str
    initial_message       : str
    agent_first_reply     : str
    resolution_summary    : Optional[str] = None
    resolution_time_hours : Optional[float] = None
    reopened              : int = 0
    customer_sentiment    : str = "neutral"
    csat_score            : int = 3
    has_attachment        : int = 0
    platform              : str
    region                : Optional[str] = None
    created_at            : str


#  Pydantic response model 
class PredictionResponse(BaseModel):
    risk_score  : float
    risk_level  : str
    escalated   : bool
    top_reasons : list
    model_used  : str


#FastAPI app 
app = FastAPI(
    title       = "Customer Risk & Escalation Engine",
    description = "Predicts escalation risk for support tickets",
    version     = "1.0.0"
)

#CORS — allows Streamlit to call this API 
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"]
)

#  Load pipeline once at startup 
pipeline = None

@app.on_event("startup")
async def load_pipeline():
    global pipeline
    try:
        logging.info("Loading PredictPipeline at startup...")
        pipeline = PredictPipeline()
        logging.info("PredictPipeline loaded ✅")
    except Exception as e:
        logging.error(f"Failed to load pipeline: {e}")


#  Routes 
@app.get("/health")
def health_check():
    return {
        "status"         : "running",
        "pipeline_loaded": pipeline is not None,
        "model"          : "LateFusion_LightGBM"
    }


@app.get("/model-info")
def model_info():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    return {
        "model_name"    : "Late Fusion LightGBM",
        "tabular_features": 12,
        "embedding_dim" : 768,
        "total_features": 780,
        "base_model"    : "distilbert-base-uncased"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(ticket: TicketInput):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    try:
        raw_ticket = ticket.dict()
        result     = pipeline.predict(raw_ticket)

        return PredictionResponse(
            risk_score  = result.risk_score,
            risk_level  = result.risk_level,
            escalated   = result.escalated,
            top_reasons = result.top_reasons,
            model_used  = result.model_used
        )
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)