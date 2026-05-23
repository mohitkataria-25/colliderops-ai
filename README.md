# ColliderOpsAI

ColliderOpsAI is an AI/ML and MLOps project inspired by CERN/LHC-style collider event analysis.

The current version is a working FastAPI-based ML inference service, MLOps workflow, agentic research assistant, and early real CERN/Open Data ingestion pipeline. It includes a local PySpark ETL pipeline that writes processed and curated datasets, trains and evaluates baseline models from curated data, tracks training/evaluation runs in MLflow, serves single-event predictions, JSON batch predictions, CSV file-based batch predictions, exports prediction results as CSV, applies confidence-based triage, generates batch-level summary statistics, includes a LangChain-powered RAG assistant over project documentation, provides a tested LangGraph agent workflow for routing research and prediction requests, and now includes a real CERN ROOT-file ingestion path that extracts readable event-level features into processed and curated CSV outputs.

The long-term goal is to evolve this into a practical AI-assisted collider analysis workbench with real CERN/Open Data ingestion, batch file uploads, confidence-based triage, model monitoring, LangChain-powered research assistance, LangGraph orchestration, and AWS/PySpark data engineering layers.

---

## Current MVP Status

Completed:

- Baseline ML model training
- Model evaluation
- Local PySpark ETL pipeline
- Processed Parquet output
- Curated ML-ready dataset output
- Training pipeline updated to consume curated data
- Evaluation pipeline updated to consume curated data
- Timestamped evaluation metrics output
- MLflow training run tracking
- MLflow evaluation run tracking
- Model metadata file for active model/version tracking
- Metadata-driven model loading
- Saved model artifacts
- FastAPI inference API
- Single-event prediction endpoint
- Batch prediction endpoint
- Batch CSV upload prediction endpoint
- Batch CSV prediction export endpoint
- LangChain RAG assistant endpoint
- Confidence triage with `risk_level` and `needs_review`
- Batch prediction summary statistics
- Prediction-layer tests
- API endpoint tests
- Dockerfile
- Requirements file
- RAG documentation assets
- Chroma vector index
- RAG pipeline tests
- LangGraph agent state, tools, router, nodes, and graph
- Agent routing tests
- Agentic workflow smoke tests
- CERN Open Data record metadata registration
- CERN ROOT file-location enrichment through `cernopendata-client`
- Dataset registry at `data/dataset_registry.json`
- ROOT file inspection with `uproot`
- CMS Events tree branch probing
- Readable event-level feature extraction from real CERN ROOT files
- Real CERN processed CSV output
- Real CERN curated ML-ready CSV output
- Real CERN feature-engineering validation mode

Current test status:

```bash
All current tests passing
```

---

## Project Architecture

```text
Raw / sample collider data
        ↓
Local PySpark ETL
        ↓
Processed Parquet dataset
        ↓
Curated ML-ready dataset
        ↓
Feature engineering
        ↓
Baseline model training from curated data
        ↓
Model evaluation from curated data
        ↓
Saved model artifacts + model metadata
        ↓
MLflow tracking for training/evaluation
        ↓
FastAPI inference service
        ↓
/predict, /batch-predict, /batch-predict-file, and /batch-predict-file-export endpoints
        ↓
Confidence triage + batch summary
        ↓
LangChain RAG assistant through /ask
        ↓
LangGraph agent routing across RAG, metadata, prediction, batch, export, and ETL status tools
        ↓
Tests + Docker-ready API
```

Planned architecture:

```text
CERN / raw data sources
        ↓
AWS-style data lake: raw → processed → curated
        ↓
PySpark / AWS Glue ETL
        ↓
ML training + model registry
        ↓
FastAPI inference API
        ↓
Batch analysis + confidence triage
        ↓
LangChain RAG assistant
        ↓
LangGraph workflow orchestration
        ↓
Researcher-facing UI + real CERN/Open Data ingestion
```

---

## Repository Structure

```text
colliderops-ai/
│
├── app/
│   ├── main.py              # FastAPI app and API routes
│   ├── model_loader.py      # Model artifact loading and metadata
│   ├── predict.py           # Prediction, batch prediction, file prediction, triage
│   └── schemas.py           # Pydantic request/response schemas
│
├── agent/
│   ├── graph.py             # LangGraph workflow definition
│   ├── nodes.py             # Agent nodes that call tools and update state
│   ├── router.py            # Intent routing logic
│   ├── state.py             # AgentState TypedDict
│   ├── tools.py             # Agent tool wrappers around RAG, prediction, metadata, ETL status
│   └── test_agent.py        # Agent workflow smoke tests
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── curated/
│   └── sample/
│
├── evaluation_metrics/
│   └── model_comparison_<timestamp>.csv
│
├── etl/
│   ├── download_cern_data.py
│   ├── glue_etl_job.py
│   ├── upload_to_s3.py
│   └── validate_schema.py
│
├── docs/
│   ├── data_dictionary.md
│   ├── evaluation_card.md
│   ├── model_card.md
│   └── research_workbench_overview.md
│
├── rag/
│   ├── build_vector_index.py
│   ├── config.py
│   ├── ingest_docs.py
│   ├── models.py
│   ├── pipeline.py
│   ├── prompts.py
│   └── retriever.py
│
├── models/
│   ├── logistic_regression_baseline.joblib
│   ├── random_forest_baseline.joblib
│   └── model_metadata.json
│
├── training/
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── logs.py              # MLflow helper utilities
│   └── model_registry.py
│
├── tests/
│   ├── test_api.py
│   ├── test_prediction.py
│   └── test_rag.py
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Current API Endpoints

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

---

### Single Prediction

```http
POST /predict
```

Example request:

```json
{
  "DER_mass_MMC": 138.4,
  "DER_mass_transverse_met_lep": 51.6,
  "DER_mass_vis": 97.8,
  "PRI_tau_pt": 32.6,
  "PRI_lep_pt": 44.1
}
```

Example response:

```json
{
  "prediction": "signal",
  "probability": 0.91,
  "model_name": "random_forest_baseline",
  "model_version": "v1",
  "risk_level": "high_confidence",
  "needs_review": false
}
```

---

### Batch Prediction

```http
POST /batch-predict
```

Example request:

```json
[
  {
    "DER_mass_MMC": 138.4,
    "DER_mass_transverse_met_lep": 51.6,
    "DER_mass_vis": 97.8,
    "PRI_tau_pt": 32.6,
    "PRI_lep_pt": 44.1
  },
  {
    "DER_mass_MMC": 80.1,
    "DER_mass_transverse_met_lep": 88.4,
    "DER_mass_vis": 58.3,
    "PRI_tau_pt": 18.4,
    "PRI_lep_pt": 24.1
  }
]
```

Example response:

```json
{
  "predictions": [
    {
      "prediction": "signal",
      "probability": 0.91,
      "model_name": "random_forest_baseline",
      "model_version": "v1",
      "risk_level": "high_confidence",
      "needs_review": false
    },
    {
      "prediction": "background",
      "probability": 0.62,
      "model_name": "random_forest_baseline",
      "model_version": "v1",
      "risk_level": "low_confidence",
      "needs_review": true
    }
  ],
  "summary": {
    "total_events": 2,
    "signal_count": 1,
    "background_count": 1,
    "high_confidence_count": 1,
    "low_confidence_count": 1,
    "review_required_count": 1,
    "average_probability": 0.765
  }
}
```

---

### Batch Prediction from CSV File

```http
POST /batch-predict-file
```

This endpoint allows researchers or analysts to upload a CSV file instead of manually sending JSON payloads.

The uploaded CSV must contain these feature columns:

```text
DER_mass_MMC
DER_mass_transverse_met_lep
DER_mass_vis
PRI_tau_pt
PRI_lep_pt
```

Optional column:

```text
event_id
```

If `event_id` is present, it is preserved in the row-level prediction output for traceability.

Example workflow:

```text
upload CSV
→ validate required feature columns
→ convert rows into prediction payloads
→ run batch prediction
→ return predictions + batch summary
```

---

### Export Batch Predictions as CSV

```http
POST /batch-predict-file-export
```

This endpoint allows researchers or analysts to upload a CSV file and receive a downloadable CSV containing prediction results.

The uploaded CSV follows the same schema as `/batch-predict-file`.

Output CSV columns:

```text
event_id
prediction
probability
risk_level
needs_review
model_name
model_version
```

Example workflow:

```text
upload CSV
→ validate required feature columns
→ run batch prediction
→ preserve event_id if available
→ return downloadable colliderops_predictions.csv
```

---

### Research Assistant

```http
POST /ask
```

Example request:

```json
{
  "question": "What is ColliderOpsAI and what problem does it solve?"
}
```

Example response:

```json
{
  "question": "What is ColliderOpsAI and what problem does it solve?",
  "answer": "ColliderOpsAI is an AI-assisted research workbench for collider-style datasets...",
  "sources": [
    {
      "file_name": "research_workbench_overview.md",
      "source": "docs/research_workbench_overview.md",
      "document_type": "md"
    }
  ]
}
```

The `/ask` endpoint uses the LangChain RAG pipeline to retrieve relevant project documentation and generate grounded answers.

---

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Data Download

```bash
python -m etl.download_cern_data
```

This currently searches CERN Open Data metadata and creates a fallback collider-style dataset for local MVP development.

The fallback dataset is written to:

```text
data/raw/collider_events.csv
```

---

## Run Local ETL

```bash
python -m etl.glue_etl_job
```

The local ETL pipeline reads raw collider-style data and writes processed and curated outputs.

Input:

```text
data/raw/collider_events.csv
```

Processed output:

```text
data/processed/collider_events/
```

Curated training output:

```text
data/curated/training_dataset/
```

Current ETL steps:

- Read raw CSV or JSON data
- Flatten nested JSON records when needed
- Validate required schema
- Cast numerical feature columns
- Normalize labels
- Drop missing required values
- Deduplicate records by `event_id`
- Write processed Parquet output
- Write curated ML-ready training output

---

## Run Real CERN/Open Data Ingestion

ColliderOpsAI now includes an early real CERN/Open Data ingestion path.

### 1. Search CERN Open Data records

```bash
python -m etl.search_cern_records
```

This searches the CERN Open Data Portal, enriches candidate records with file locations through `cernopendata-client`, and identifies processable file formats such as ROOT.

### 2. Register a CERN record

Example using CMS Higgs Monte Carlo record `7901`:

```bash
python -c "from etl.cern_client import register_cern_record; summary = register_cern_record(record_id='7901'); print(summary['dataset_name']); print(summary['file_count']); print(summary['registry_path'])"
```

This writes source metadata and ROOT file URLs to:

```text
data/dataset_registry.json
```

### 3. Inspect and probe ROOT files

```bash
python -m etl.adapters.root_adapter
```

This uses `uproot` to:

- Open a real CERN/CMS ROOT file from URL
- Inspect top-level ROOT keys
- Detect the `Events;1` tree
- List branch previews
- Probe which branches are readable
- Extract a small readable event-level feature table

### 4. Run the real CERN ETL job

```bash
python -m etl.real_cern_etl_job
```

Current real CERN ETL output:

```text
data/processed/real_cern_events/events.csv
data/processed/real_cern_events/run_metadata.json
data/curated/real_cern_training_dataset/training_dataset.csv
```

Current extracted real CERN feature columns:

```text
gen_event_present
gen_event_weight_count
gen_event_signal_process_id
gen_event_qscale
gen_particles_present
gen_particle_count
ak5_genjets_present
label
```

Current limitation: The project now includes real CERN/Open Data ROOT ingestion, but the default real CERN output is signal-only and not yet suitable for meaningful binary classifier training.. The feature-engineering layer can validate the dataset schema and numeric features, but binary classifier training is intentionally blocked until a background CERN/Open Data sample is added.

## Train Models

```bash
python -m training.train
```

Current baseline models:

Training now consumes the curated dataset generated by the local ETL pipeline.

Training runs are tracked in MLflow under the `ColliderOpsAI` experiment. Model artifacts are logged to MLflow and also saved locally under `models/`.

- Logistic Regression
- Random Forest

Saved under:

```text
models/
```

---

## Evaluate Models

```bash
python -m training.evaluate
```

Current evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1 score

Evaluation now consumes the curated dataset generated by the local ETL pipeline.

Evaluation outputs are saved under:

```text
evaluation_metrics/model_comparison_<timestamp>.csv
```

Evaluation runs are also tracked in MLflow, including model metrics and the generated evaluation metrics CSV as an artifact.

---

## Run MLflow UI

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

Current MLflow tracking includes:

- Training run parameters
- Training dataset path
- Feature columns
- Train/test row counts
- Saved model artifacts
- Evaluation run parameters
- Accuracy, precision, recall, and F1 metrics
- Evaluation metrics CSV artifact

MLflow runs are stored locally under:

```text
mlruns/
```

---

## Run API Locally

```bash
python -m uvicorn app.main:app --reload
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

---

## Run Tests

```bash
python -m pytest -v
```

Current passing tests:

- Health endpoint
- Single prediction endpoint
- Batch prediction endpoint
- Batch CSV upload endpoint
- Batch CSV prediction export endpoint
- RAG prompt formatting
- RAG retrieval flow
- RAG answer generation with mocked LLM
- RAG pipeline output structure
- Confidence triage fields
- Batch summary output
- API validation error
- Single prediction function
- Batch prediction function
- LangGraph model metadata route
- LangGraph ETL status route
- LangGraph single prediction route
- LangGraph batch prediction route
- LangGraph mocked RAG route

---

## Run with Docker

Build image:

```bash
docker build -t colliderops-ai .
```

Run container:

```bash
docker run -p 8000:8000 colliderops-ai
```

Open:

```text
http://localhost:8000/docs
```

---

## Run the RAG Pipeline

Build the local Chroma vector index:

```bash
python -m rag.build_vector_index
```

Run the RAG pipeline directly:

```bash
python -m rag.pipeline
```

The RAG pipeline currently ingests:

```text
README.md
docs/research_workbench_overview.md
docs/model_card.md
docs/evaluation_card.md
docs/data_dictionary.md
```

Local LLM support is currently configured through Ollama using:

```text
llama3.1:8b
```

Make sure Ollama is running before using `/ask` or `rag.pipeline`.

---

## Run the LangGraph Agent

The LangGraph agent routes user requests across the project’s RAG, model metadata, prediction, batch prediction, export, and ETL status tools.

Example smoke test:

```bash
python -c "from agent.graph import run_agent; result = run_agent({'user_query': 'What model is currently active?', 'errors': []}); print(result['intent']); print(result['final_response'])"
```

Expected intent:

```text
model_metadata
```

Current supported agent intents:

```text
rag_question
model_metadata
single_prediction
batch_prediction
batch_file_prediction
batch_export
etl_status
unknown
```

Agent workflow:

```text
User request
        ↓
AgentState
        ↓
Intent router
        ↓
Conditional LangGraph routing
        ↓
Selected tool node
        ↓
Final response
```

The current router is deterministic and keyword/state based. This keeps v1 simple, testable, and cheap. A future version can add hybrid or LLM-based intent classification with confidence scores.

---

## Why This Project Exists

ColliderOpsAI is being built as a practical AI/MLOps workbench, not just a model demo.

It is designed to demonstrate:

- Python engineering
- ML model training and evaluation
- FastAPI model serving
- Batch inference
- Confidence-based event triage
- Research-run summary statistics
- CSV-based researcher workflows
- Downloadable prediction exports
- LangChain-based research Q&A
- Chroma-based local vector search
- Testing and Dockerization
- Data lake thinking
- Local PySpark ETL
- Raw / processed / curated data pipeline
- Curated-data-driven training and evaluation
- PySpark / AWS Glue ETL patterns
- LangChain-based RAG
- LangGraph-based orchestration
- Agentic AI workflow design
- Tool-based routing across prediction, RAG, metadata, export, and ETL status
- MLflow-based experiment tracking
- Metadata-driven model serving

---

## Roadmap

### Phase 1 — FastAPI ML Inference MVP

Completed:

- Train baseline models
- Evaluate models
- Serve predictions through FastAPI
- Add single and batch prediction endpoints
- Add tests
- Add Dockerfile

### Phase 2 — Practical Workbench Features

Planned:

Completed in this phase:

- Confidence thresholding
- `risk_level` classification
- `needs_review` flag
- Batch prediction summary statistics
- Batch CSV upload endpoint
- Prediction result export as downloadable CSV

### Phase 3 — MLOps Layer

Completed:

- MLflow training run tracking
- MLflow evaluation run tracking
- Model artifacts logged to MLflow
- Evaluation metrics logged to MLflow
- Evaluation CSV artifact logging
- `models/model_metadata.json`
- Metadata-driven model loading
- Model card and evaluation card documentation

Planned:

- Automated model metadata refresh from training/evaluation runs
- Model registry promotion workflow
- Model metadata API endpoint
- Dataset versioning and lineage

### Phase 4 — Data Engineering Layer

Completed:

- Local PySpark ETL pipeline
- Raw / processed / curated data lake-style zones
- Processed Parquet output
- Curated ML-ready dataset output
- Schema validation
- Data cleaning, type casting, label normalization, and deduplication
- Training updated to consume curated data
- Evaluation updated to consume curated data

Planned:

- AWS Glue-compatible runtime adaptation
- S3 integration
- Glue Data Catalog / crawler integration
- Dataset versioning and lineage
- Additional CERN/Open Data ingestion adapters

### Phase 5 — LangChain RAG Assistant

Completed:

- Project documentation assets created
- Markdown document ingestion
- Recursive text chunking
- Local Chroma vector index
- Retriever module
- Prompt builder
- LLM factory module
- RAG pipeline orchestration
- FastAPI `/ask` endpoint
- RAG tests

Planned:

- Add external CERN/Open Data documentation
- Add answer evaluation / judge model flow
- Add richer source citation handling
- Add RAG response logging

### Phase 6 — LangGraph Agentic Workflow

Completed:

- `AgentState` definition
- Tool wrappers for RAG, model metadata, single prediction, batch prediction, CSV file prediction, CSV export, and ETL status
- Deterministic intent router
- LangGraph nodes
- Conditional graph routing
- Compiled `run_agent()` workflow
- Agent smoke tests

Planned:

- Hybrid / LLM-based intent classification
- Agent API endpoint
- Agent-facing report generation
- Safer controlled execution for long-running ETL/training workflows
- Researcher-facing UI integration

---

## Current Limitations

- Current dataset is a small fallback dataset, not yet a large real CERN dataset.
- Model performance metrics are not meaningful yet because the dataset is small.
- Local PySpark ETL is working, but AWS Glue runtime integration and S3-based data lake deployment are not yet complete.
- LangChain RAG is implemented over local project documentation, but external CERN/Open Data sources are not yet integrated.
- LangGraph orchestration is implemented for v1, but it currently uses deterministic routing rather than LLM-based intent classification.
- Model explainability is not yet implemented.

---

## Near-Term Next Steps

1. Find/register a suitable CERN/Open Data background sample
2. Extend `etl.real_cern_etl_job` to combine signal + background configurations
3. Train/evaluate a real CERN `dataset_mode="real_cern"` classifier once two classes are available
4. Add tests for CERN client, ROOT adapter helpers, and real CERN ETL row generation
5. Add model explainability with feature importance / SHAP
6. Add an agent API endpoint for LangGraph workflows
7. Add external CERN/Open Data documentation to RAG
8. Add AWS Glue/S3 adaptation for the local and real CERN ETL pipelines
9. Clean up Pydantic and LangChain deprecation warnings
