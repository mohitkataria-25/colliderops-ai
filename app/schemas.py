from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    DER_mass_MMC: float = Field(..., example=138.4)
    DER_mass_transverse_met_lep: float = Field(..., example=51.6)
    DER_mass_vis: float = Field(..., example=97.8)
    PRI_tau_pt: float = Field(..., example=32.6)
    PRI_lep_pt: float = Field(..., example=44.1)

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    model_name: str
    model_version: str
    risk_level: str
    needs_review: bool


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]

class HealthResponse(BaseModel):
    status: str

class AskRequest(BaseModel):
    question: str

class SourceDocument(BaseModel):
    file_name: str
    source:str
    document_type:str

class AskResponse(BaseModel):
    question: str
    answer: str
    sources : list[SourceDocument]

class BatchPredictionSummary(BaseModel):
    total_events: int
    signal_count: int
    background_count: int
    high_confidence_count: int
    low_confidence_count: int
    review_required_count: int
    average_probability: float | None

class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    summary: BatchPredictionSummary






