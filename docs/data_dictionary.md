

# ColliderOpsAI Data Dictionary

## Purpose

This data dictionary documents the current dataset schema used by ColliderOpsAI for collider-style event classification.

It is intended to help researchers, ML engineers, and future RAG/agentic assistant workflows understand what each field means, how the fields are used, and what validation expectations apply before model training or inference.

The current dataset is a small collider-inspired fallback dataset used to validate the end-to-end pipeline. As larger real CERN/Open Data sources are integrated, this dictionary should be updated with domain-accurate definitions, units, provenance, and references.

---

## Current Dataset

Current raw dataset path:

```text
data/raw/collider_events.csv
```

Current dataset type:

```text
structured tabular collider-style event data
```

Current target task:

```text
binary classification: signal vs background
```

---

## Required Columns

ColliderOpsAI currently expects the following columns in the raw or curated training dataset:

```text
event_id
label
DER_mass_MMC
DER_mass_transverse_met_lep
DER_mass_vis
PRI_tau_pt
PRI_lep_pt
```

The required schema is also defined in:

```text
etl/validate_schema.py
```

---

## Column Definitions

| Column | Type | Required | Used for Training | Description |
|---|---:|---:|---:|---|
| `event_id` | string | Yes | No | Unique identifier for each event record. Used for traceability, deduplication, batch output review, and future report generation. |
| `label` | string | Yes | Target only | Ground-truth class label used during training and evaluation. Allowed values are `signal` and `background`. Not required for inference-only payloads. |
| `DER_mass_MMC` | float | Yes | Yes | Collider-inspired derived mass feature. In real Higgs-style datasets, this type of feature can represent reconstructed mass information. In the current MVP, it is used as a numerical predictor. |
| `DER_mass_transverse_met_lep` | float | Yes | Yes | Collider-inspired transverse mass feature involving missing transverse energy and lepton information. Used as a numerical predictor. |
| `DER_mass_vis` | float | Yes | Yes | Collider-inspired visible mass feature derived from visible event components. Used as a numerical predictor. |
| `PRI_tau_pt` | float | Yes | Yes | Collider-inspired primary tau transverse momentum feature. Used as a numerical predictor. |
| `PRI_lep_pt` | float | Yes | Yes | Collider-inspired primary lepton transverse momentum feature. Used as a numerical predictor. |

---

## Target Label

The target column is:

```text
label
```

Allowed values:

```text
signal
background
```

During model training and evaluation, labels are mapped as:

```text
background → 0
signal     → 1
```

This mapping is used internally by the training pipeline. API responses map numeric predictions back to human-readable labels.

---

## Feature Columns

The current model uses these predictor columns:

```text
DER_mass_MMC
DER_mass_transverse_met_lep
DER_mass_vis
PRI_tau_pt
PRI_lep_pt
```

The following columns are excluded from model training:

```text
event_id
label
```

Reason:

- `event_id` is an identifier, not a predictive feature.
- `label` is the target variable and should never be used as an input predictor.

---

## Training Dataset Schema

Training data should contain:

```text
event_id
label
DER_mass_MMC
DER_mass_transverse_met_lep
DER_mass_vis
PRI_tau_pt
PRI_lep_pt
```

Training requires `label` because the model needs ground-truth classes.

---

## Inference Dataset Schema

Inference data only needs the model feature columns:

```text
DER_mass_MMC
DER_mass_transverse_met_lep
DER_mass_vis
PRI_tau_pt
PRI_lep_pt
```

`event_id` is optional for inference today, but it will become useful for batch CSV upload, result export, and report generation.

`label` should not be required for inference because prediction requests are meant for unknown events.

---

## API Input Schema

The FastAPI `/predict` endpoint expects one event with the model feature columns.

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

The `/batch-predict` endpoint expects a list of the same event objects.

API schemas are defined in:

```text
app/schemas.py
```

---

## API Output Fields

Prediction responses include:

| Field | Type | Description |
|---|---:|---|
| `prediction` | string | Predicted class: `signal` or `background`. |
| `probability` | float | Highest class probability returned by the model. |
| `model_name` | string | Name of the model artifact used for prediction. |
| `model_version` | string | Version label for the served model. |
| `risk_level` | string | Confidence category based on probability threshold. |
| `needs_review` | boolean | Whether the event should be reviewed based on confidence. |

---

## Confidence Triage Fields

ColliderOpsAI adds operational triage fields to every prediction.

Current threshold:

```text
0.75
```

Rules:

```text
probability >= 0.75 → risk_level = high_confidence, needs_review = false
probability < 0.75  → risk_level = low_confidence, needs_review = true
probability missing → risk_level = unknown, needs_review = true
```

These fields make the tool more useful for research workflows where users need to quickly identify uncertain or borderline events.

---

## Batch Summary Fields

Batch predictions return row-level predictions plus a summary object.

Summary fields:

| Field | Type | Description |
|---|---:|---|
| `total_events` | integer | Total number of events processed in the batch. |
| `signal_count` | integer | Number of events predicted as `signal`. |
| `background_count` | integer | Number of events predicted as `background`. |
| `high_confidence_count` | integer | Number of predictions above or equal to the confidence threshold. |
| `low_confidence_count` | integer | Number of predictions below the confidence threshold. |
| `review_required_count` | integer | Number of predictions where `needs_review` is true. |
| `average_probability` | float or null | Average prediction probability across the batch. |

---

## Data Quality Expectations

For the current MVP, a valid training dataset should meet the following rules:

- All required columns must be present.
- Feature columns should be numeric or convertible to numeric values.
- `label` should only contain `signal` or `background` during training/evaluation.
- `event_id` should uniquely identify each event.
- Rows with missing required values should be removed or flagged before training.
- Duplicate `event_id` records should be removed or resolved during ETL.

Schema validation helpers are stored in:

```text
etl/validate_schema.py
```

---

## Raw, Processed, and Curated Data Layers

ColliderOpsAI is moving toward a data lake-style structure.

Planned layers:

```text
data/raw/        → source or generated input data
data/processed/  → cleaned and validated data
data/curated/    → ML-ready training and inference data
```

The long-term goal is to support:

```text
raw collider data
→ local PySpark / AWS Glue transformation
→ processed Parquet
→ curated ML-ready dataset
→ model training and inference
```

---

## Current Limitations

Current limitations of the dataset/schema:

- The current data is a small fallback dataset, not a real large CERN/LHC dataset.
- Feature definitions are simplified and project-facing.
- No real external CERN record has been finalized as the canonical source yet.
- No full feature provenance is available yet.
- No dataset versioning has been implemented yet.
- No data drift monitoring exists yet.
- No feature range validation exists yet.

---

## Future Data Dictionary Improvements

Planned improvements:

1. Add domain-accurate feature definitions from selected CERN/Open Data sources.
2. Add units for each numerical feature.
3. Add accepted ranges and validation rules for each feature.
4. Add dataset version information.
5. Add lineage from raw source to processed and curated datasets.
6. Add references to external CERN/Open Data documentation.
7. Add fields for batch upload schemas.
8. Add drift monitoring fields and summary statistics.
9. Add feature provenance and transformation notes.

---

## RAG Assistant Use

This data dictionary will be part of the future LangChain RAG knowledge base.

The assistant should be able to answer questions such as:

- What columns are required for prediction?
- Which columns are used by the model?
- Why is `event_id` excluded from training?
- What does `label` represent?
- What does `needs_review` mean?
- What fields are returned in a batch summary?
- What data quality checks are expected before training?
- What is the difference between training schema and inference schema?

---

## Current Status

Completed:

- Current required columns documented
- Target label documented
- Training schema documented
- Inference schema documented
- Prediction input schema documented
- Prediction output fields documented
- Confidence triage fields documented
- Batch summary fields documented

Next planned data work:

- Add larger real dataset
- Add curated dataset output
- Add batch CSV upload schema
- Add feature ranges and units
- Add dataset versioning and lineage