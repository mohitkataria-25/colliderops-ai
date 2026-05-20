# ColliderOpsAI Research Workbench Overview

## Purpose

ColliderOpsAI is designed as an AI-assisted research workbench for scientists, research analysts, and ML engineers working with collider-style experimental datasets.

The goal is not to replace specialized physics analysis frameworks. Instead, ColliderOpsAI provides a practical workflow for ingesting structured event data, running ML-based event classification, triaging uncertain predictions, and helping users understand model outputs through documentation, reports, and eventually a RAG-powered research assistant.

## Target Users

ColliderOpsAI is intended for:

- Scientific researchers working with experimental or simulation-style event datasets
- Physics students and research assistants learning ML workflows for scientific data
- ML engineers building classification and inference workflows for structured scientific data
- Data engineers supporting research teams with repeatable ETL and data validation pipelines
- Analysts who need to inspect prediction outputs and identify uncertain events for review

## Core User Problems

Scientific and experimental datasets can be difficult to work with because they often involve:

- Large volumes of event-level records
- Raw files that require cleaning and schema validation
- Multiple transformation stages before data is ML-ready
- Models that produce predictions without enough operational context
- Uncertain or borderline predictions that require expert review
- Scattered documentation across code, reports, notebooks, and research references
- Difficulty reproducing model and dataset versions across experiment runs

ColliderOpsAI is being built to address these problems through a research-oriented workflow.

## Current Capabilities

The current MVP includes:

- Local data ingestion using a collider-style starter dataset
- Feature engineering for model-ready tabular data
- Baseline model training using Logistic Regression and Random Forest
- Model evaluation using standard classification metrics
- Saved model artifacts using `joblib`
- FastAPI-based model serving
- Single-event prediction through `POST /predict`
- Batch prediction through `POST /batch-predict`
- Confidence-based triage using `risk_level` and `needs_review`
- Batch summary statistics for quick research-run review
- Automated tests for prediction logic and API endpoints
- Docker-ready API deployment setup

## Current Workflow

```text
Raw collider-style data
        ↓
Feature engineering
        ↓
Train baseline ML models
        ↓
Evaluate model performance
        ↓
Save model artifacts
        ↓
FastAPI inference service
        ↓
Single and batch predictions
        ↓
Confidence triage and batch summary
```

## Prediction Workflow

A user can submit a single collider-style event with numerical features. The API returns:

- Predicted class: `signal` or `background`
- Prediction probability
- Model name
- Model version
- Risk level
- Review flag

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

## Batch Prediction Workflow

A user can submit multiple events in one request. The system returns row-level predictions and a batch-level summary.

The summary includes:

- Total events processed
- Number of signal predictions
- Number of background predictions
- Number of high-confidence predictions
- Number of low-confidence predictions
- Number of events requiring review
- Average prediction probability

This makes the tool more useful for research workflows because users can quickly understand the overall result of a batch run instead of inspecting every event manually.

## Confidence Triage

ColliderOpsAI uses prediction probability to classify output confidence.

The current default confidence threshold is `0.75`.

The triage logic is:

```text
probability >= 0.75 → high_confidence, needs_review = false
probability < 0.75  → low_confidence, needs_review = true
probability missing → unknown, needs_review = true
```

This mirrors real operational ML workflows where high-confidence predictions can be processed automatically, while low-confidence predictions are routed for human or expert review.

## Practical Research Use Cases

ColliderOpsAI can support the following research-oriented workflows:

### 1. Event Classification

Researchers can classify collider-style events as signal or background using a deployed model rather than running predictions manually in notebooks.

### 2. Uncertainty Review

Researchers can identify low-confidence predictions and prioritize them for manual review.

### 3. Batch Run Summary

Researchers can submit a batch of events and receive a compact summary of the run, including signal/background distribution and review counts.

### 4. Reproducible ML Workflow

ML engineers can track how data moves from ingestion to feature engineering to training to model serving.

### 5. Future Research Assistant

A LangChain-powered RAG assistant will help users ask questions about the dataset, features, model card, evaluation results, and pipeline behavior.

Example questions:

- What does `DER_mass_MMC` represent?
- Why was this event marked as low confidence?
- What are the limitations of the current model?
- How was the model trained?
- What changed between model versions?
- Which events should be reviewed first?

## Planned Product Direction

ColliderOpsAI will evolve into a practical research workbench with the following capabilities:

1. CSV file upload for batch prediction
2. Exportable prediction results
3. Research-run summary reports
4. Local PySpark ETL for raw, processed, and curated data layers
5. MLflow tracking for experiments and model versions
6. Model cards and evaluation reports
7. LangChain RAG assistant over project documentation and research references
8. LangGraph workflow orchestration for prediction, explanation, and report generation
9. AWS Glue and cloud data lake support
10. Research dashboard for event triage and model run review

## Future LangChain RAG Role

LangChain will be used to build a retrieval-augmented assistant over:

- Project README
- Data dictionary
- Model card
- Evaluation report
- Batch prediction reports
- Research workbench documentation
- Future CERN/Open Data references

The assistant should provide grounded answers with references to internal project documentation and selected external research sources.

## Future LangGraph Role

LangGraph will be used to coordinate multi-step research workflows.

Example workflow:

```text
User request
    ↓
Classify intent
    ↓
Route to prediction, batch analysis, RAG retrieval, or report generation
    ↓
Execute tool
    ↓
Validate output
    ↓
Return final response
```

Potential agent tools:

- `predict_single_event_tool`
- `batch_predict_tool`
- `summarize_batch_results_tool`
- `retrieve_docs_tool`
- `generate_research_report_tool`
- `validate_dataset_tool`

## What ColliderOpsAI Is Not

ColliderOpsAI is not claiming to discover new particles or replace production-grade CERN analysis frameworks.

It is a practical AI/MLOps research workflow inspired by collider-event classification. The project demonstrates how scientific event data can be processed, classified, triaged, explained, and operationalized using modern AI engineering patterns.

## Current Status

Completed:

- ML training and evaluation MVP
- FastAPI inference service
- Single prediction endpoint
- Batch prediction endpoint
- Confidence triage
- Batch summary statistics
- Automated tests
- Docker setup

Next planned features:

- Batch CSV upload endpoint
- Prediction result export
- Data dictionary
- Model card
- Evaluation report
- LangChain RAG setup
- LangGraph research assistant workflow
