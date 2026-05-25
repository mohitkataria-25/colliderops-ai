

# ColliderOpsAI Orchestration Plan

## Purpose

This document defines the target orchestration plan for ColliderOpsAI.

ColliderOpsAI currently supports local execution for:

- Raw CERN/Open Data ROOT ingestion through `real_cern`.
- Curated HIGGS benchmark ingestion through `curated_higgs`.
- Model training and evaluation across multiple dataset modes.
- MLflow experiment tracking.
- Evaluation reports, model comparison, confusion matrices, classification reports, and feature importance outputs.

The next architecture milestone is to convert these local workflows into a cloud-ready orchestration pattern using AWS Step Functions, AWS Glue, AWS Lambda, S3, and containerized training/evaluation jobs.

This plan is a blueprint first. It is designed to make the local project production-shaped without requiring every AWS component to be deployed immediately.

---

## Orchestration Goals

The orchestration layer should support:

- Repeatable end-to-end pipeline execution.
- Dataset-mode-specific workflows.
- Validation before expensive ETL/training jobs.
- Clear failure handling.
- Metadata capture at each stage.
- Model and evaluation artifact registration.
- Human-readable status and report outputs.
- Future scheduling through EventBridge, Airflow, or manual triggers.

---

## Supported Dataset Modes

Current dataset modes:

```text
sample_collider
real_cern
curated_higgs
```

### `sample_collider`

Purpose:

- Lightweight sample dataset used for local development and fast test runs.

Expected use:

- Unit tests.
- API smoke tests.
- Local development.

### `real_cern`

Purpose:

- Raw CERN/Open Data pipeline.
- Demonstrates ROOT file discovery, caching, branch probing, lightweight feature extraction, and lineage tracking.

Expected orchestration behavior:

- Validate dataset registry.
- Download/cache ROOT files or verify cached files.
- Run controlled extraction step.
- Write processed and curated outputs.
- Train/evaluate baseline models.

### `curated_higgs`

Purpose:

- ML-ready HIGGS benchmark pipeline.
- Demonstrates tabular ML modeling, model comparison, and explainability reporting.

Expected orchestration behavior:

- Download/cache HIGGS source file.
- Validate schema and label balance.
- Write curated dataset.
- Train/evaluate baseline and nonlinear models.
- Generate evaluation summary and feature importance artifacts.

---

## Local Pipeline Commands

### Curated HIGGS pipeline

```bash
python -m etl.download_curated_higgs_dataset --total-rows 10000
python -m training.train --dataset-mode curated_higgs
python -m training.evaluate --dataset-mode curated_higgs
```

### Real CERN pipeline

```bash
python -m etl.real_cern_etl_job
python -m training.train --dataset-mode real_cern
python -m training.evaluate --dataset-mode real_cern
```

### Smoke test pipeline

```bash
python -m training.train --dataset-mode sample_collider
python -m training.evaluate --dataset-mode sample_collider
```

---

## Target AWS Orchestration Pattern

The AWS orchestration should be implemented as a Step Functions state machine.

High-level flow:

```text
Start
  -> Validate Pipeline Request
  -> Validate Dataset Metadata
  -> Run Dataset ETL
  -> Validate Curated Dataset
  -> Run Training Job
  -> Run Evaluation Job
  -> Check Evaluation Gates
  -> Register Model Metadata
  -> Publish Evaluation Summary
  -> Notify Completion
End
```

---

## Step Functions State Design

### 1. Validate Pipeline Request

Service:

```text
AWS Lambda
```

Purpose:

- Validate requested `dataset_mode`.
- Validate run configuration.
- Validate required S3 prefixes.
- Validate whether the run is `dev`, `staging`, or `prod`.

Input example:

```json
{
  "dataset_mode": "curated_higgs",
  "run_type": "manual",
  "total_rows": 10000,
  "train_models": true,
  "evaluate_models": true
}
```

Output example:

```json
{
  "dataset_mode": "curated_higgs",
  "run_id": "curated_higgs-2026-05-25-0830",
  "validated": true
}
```

Failure cases:

- Unsupported dataset mode.
- Missing required config.
- Invalid row-count request.

---

### 2. Validate Dataset Metadata

Service:

```text
AWS Lambda
```

Purpose:

- Check whether source files exist in S3 or are downloadable.
- Validate dataset registry for `real_cern`.
- Validate HIGGS source path for `curated_higgs`.
- Check whether previous cached source data can be reused.

Example checks:

```text
real_cern:
  - dataset_registry.json exists
  - selected record IDs exist
  - source ROOT files are known or cached

curated_higgs:
  - HIGGS.csv.gz exists in raw zone or source URL is configured
  - requested row count is even
  - expected feature count is 28
```

Failure cases:

- Missing source file.
- Missing registry metadata.
- Invalid dataset-mode configuration.

---

### 3. Run Dataset ETL

Service options:

```text
AWS Glue for tabular ETL
ECS/Fargate task for ROOT extraction or Python-native ingestion
```

Recommended implementation:

```text
curated_higgs -> AWS Glue / PySpark
real_cern     -> ECS/Fargate extraction task first, then Glue on standardized outputs
```

Reason:

- HIGGS is already tabular and fits Glue well.
- CERN ROOT AODSIM files may contain complex CMS EDM objects that are not ideal for direct Glue parsing.

Outputs:

```text
s3://colliderops-ai/processed/<dataset_mode>/
s3://colliderops-ai/curated/<dataset_mode>/training_dataset.parquet
s3://colliderops-ai/processed/<dataset_mode>/run_metadata.json
```

Failure cases:

- Download failure.
- Schema mismatch.
- Empty output.
- Single-class dataset.
- Missing required features.

---

### 4. Validate Curated Dataset

Service:

```text
AWS Lambda or lightweight ECS task
```

Purpose:

- Confirm curated dataset exists.
- Confirm required columns are present.
- Confirm feature columns are numeric.
- Confirm no leakage-prone columns are included.
- Confirm at least two labels exist.
- Confirm row count and label counts meet expectations.

Validation output example:

```json
{
  "dataset_mode": "curated_higgs",
  "row_count": 10000,
  "feature_count": 28,
  "label_counts": {
    "background": 5000,
    "signal": 5000
  },
  "schema_valid": true,
  "features_numeric": true,
  "two_class_training_ready": true
}
```

Failure cases:

- Curated dataset missing.
- Missing feature columns.
- Null/non-numeric features.
- Only one label class present.
- Leakage-prone columns present.

---

### 5. Run Training Job

Service options:

```text
ECS/Fargate task
AWS Batch
SageMaker Training Job
```

Recommended portfolio implementation:

```text
ECS/Fargate task running python -m training.train --dataset-mode <dataset_mode>
```

Training responsibilities:

- Load curated dataset.
- Select dataset-mode-specific features.
- Train baseline models.
- Save model artifacts.
- Log training metadata to MLflow.

Current models:

```text
Logistic Regression
Random Forest
HistGradientBoostingClassifier
```

Outputs:

```text
s3://colliderops-ai/models/<dataset_mode>/
MLflow run metadata
training run logs
```

Failure cases:

- Dataset load failure.
- Training failure.
- Model artifact not created.
- MLflow logging failure.

---

### 6. Run Evaluation Job

Service options:

```text
ECS/Fargate task
AWS Batch
SageMaker Processing Job
```

Recommended portfolio implementation:

```text
ECS/Fargate task running python -m training.evaluate --dataset-mode <dataset_mode>
```

Evaluation responsibilities:

- Load saved models.
- Recreate deterministic test split.
- Generate model comparison metrics.
- Generate ROC-AUC.
- Generate confusion matrices.
- Generate classification reports.
- Generate built-in and permutation feature importance artifacts.
- Generate `evaluation_summary.md`.
- Log artifacts to MLflow.

Outputs:

```text
s3://colliderops-ai/evaluation/<dataset_mode>/model_comparison.csv
s3://colliderops-ai/evaluation/<dataset_mode>/evaluation_summary.md
s3://colliderops-ai/evaluation/<dataset_mode>/confusion_matrices/
s3://colliderops-ai/evaluation/<dataset_mode>/classification_reports/
s3://colliderops-ai/evaluation/<dataset_mode>/feature_importance/
```

Failure cases:

- Model artifact missing.
- Evaluation dataset missing.
- Metrics generation failure.
- Evaluation report write failure.

---

### 7. Check Evaluation Gates

Service:

```text
AWS Lambda
```

Purpose:

- Decide whether model performance is good enough to register or deploy.

Example gates:

```text
curated_higgs:
  - ROC-AUC >= 0.75 for at least one model
  - F1 >= 0.68 for at least one model

real_cern:
  - F1 >= 0.70 for baseline acceptance
  - No leakage-prone columns included
```

Output example:

```json
{
  "passed": true,
  "best_model_by_roc_auc": "hist_gradient_boosting_curated_higgs",
  "best_roc_auc": 0.786
}
```

Failure cases:

- Metrics file missing.
- No model passes threshold.
- Metric regression from previous accepted run.

---

### 8. Register Model Metadata

Service:

```text
AWS Lambda
```

Storage options:

```text
S3 JSON registry
DynamoDB model registry table
MLflow model registry
```

Minimum registry fields:

```json
{
  "dataset_mode": "curated_higgs",
  "model_name": "hist_gradient_boosting_curated_higgs",
  "model_artifact_path": "s3://colliderops-ai/models/curated_higgs/hist_gradient_boosting.joblib",
  "metrics_path": "s3://colliderops-ai/evaluation/curated_higgs/model_comparison.csv",
  "evaluation_summary_path": "s3://colliderops-ai/evaluation/curated_higgs/evaluation_summary.md",
  "roc_auc_score": 0.786,
  "f1_score": 0.700,
  "registered_at": "2026-05-25T08:35:48Z",
  "status": "candidate"
}
```

Failure cases:

- Missing model artifact.
- Missing metrics artifact.
- Registry write failure.

---

### 9. Publish Evaluation Summary

Service:

```text
AWS Lambda
```

Purpose:

- Copy or publish `evaluation_summary.md` to a stable location.
- Optionally update RAG source documents.
- Optionally trigger vector index refresh.

Outputs:

```text
s3://colliderops-ai/evaluation/latest/<dataset_mode>/evaluation_summary.md
s3://colliderops-ai/rag/source_docs/evaluation_summary_<dataset_mode>.md
```

Failure cases:

- Summary file missing.
- RAG document update failure.

---

### 10. Notify Completion

Service:

```text
AWS Lambda + SNS / Slack webhook / email
```

Notification should include:

- Dataset mode.
- Run ID.
- Status.
- Best model.
- Best ROC-AUC.
- Best F1.
- Links to evaluation summary and MLflow run.

Example message:

```text
ColliderOpsAI pipeline completed successfully.
Dataset mode: curated_higgs
Best ROC-AUC: hist_gradient_boosting_curated_higgs — 0.786
Best F1: hist_gradient_boosting_curated_higgs — 0.700
Evaluation summary: s3://colliderops-ai/evaluation/latest/curated_higgs/evaluation_summary.md
```

---

## Failure Handling

Each orchestration step should return structured status.

Recommended failure object:

```json
{
  "status": "failed",
  "stage": "ValidateCuratedDataset",
  "error_type": "SchemaValidationError",
  "message": "Missing required feature columns: ['m_bb']",
  "run_id": "curated_higgs-2026-05-25-0830"
}
```

Failure handling rules:

- Validation failures should stop the pipeline immediately.
- ETL failures should not overwrite previous successful curated outputs.
- Training failures should preserve logs and input metadata.
- Evaluation gate failures should register the run as `rejected`, not deployable.
- Notification should run for both success and failure paths.

---

## Scheduling Options

### Manual trigger

Useful for development and portfolio demos.

```text
Developer manually starts Step Functions execution with dataset_mode input.
```

### EventBridge scheduled trigger

Useful for periodic retraining experiments.

```text
Run curated_higgs weekly or monthly.
```

### S3 event trigger

Useful when new dataset files arrive.

```text
New file in raw zone -> trigger metadata validation -> start pipeline.
```

### Airflow alternative

Airflow can be used instead of Step Functions when:

- DAG-level visibility is preferred.
- Multi-cloud or hybrid orchestration is needed.
- The team already uses Airflow.
- Complex dependency management is required.

Recommended Airflow DAG tasks:

```text
validate_pipeline_request
validate_dataset_metadata
run_glue_etl
validate_curated_dataset
run_training
run_evaluation
check_evaluation_gates
register_model_metadata
publish_evaluation_summary
notify_completion
```

---

## Step Functions vs Airflow

### Step Functions is better when:

- The workflow is AWS-native.
- Glue, Lambda, ECS, and SageMaker need direct orchestration.
- Managed retries and execution history are enough.
- Infrastructure simplicity matters.

### Airflow is better when:

- Workflows are complex and cross-platform.
- There are many DAGs and dependency chains.
- Team already has Airflow operational knowledge.
- Scheduling and backfill are major requirements.

For ColliderOpsAI portfolio v1, Step Functions is the cleaner AWS-native choice.

---

## Near-Term Implementation Order

Recommended implementation order:

```text
1. Add this orchestration plan.
2. Add Step Functions ASL skeleton.
3. Add Lambda validator skeleton.
4. Add Glue ETL skeleton for curated_higgs.
5. Add config layer for local/S3 paths.
6. Add ECS task command documentation for training/evaluation.
7. Add README section showing local-to-cloud architecture path.
```

---

## Portfolio Framing

This orchestration layer shows that ColliderOpsAI is not just a notebook or script collection.

It demonstrates:

- Production-style pipeline decomposition.
- ETL and ML job separation.
- Dataset validation before training.
- Evaluation gates before model registration.
- MLflow-backed tracking and artifact logging.
- AWS-native orchestration thinking.
- Clear migration path from local prototype to cloud workflow.

Best framing:

```text
I designed ColliderOpsAI as a local-first AI/MLOps research workbench with a clear AWS orchestration path. The pipeline separates ingestion, validation, ETL, training, evaluation, model registration, and serving. This makes the system easier to productionize using S3, Glue, Lambda, Step Functions, ECS/Fargate, and MLflow.
```