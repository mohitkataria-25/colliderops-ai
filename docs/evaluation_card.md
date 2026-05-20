# ColliderOpsAI Evaluation Card

## Purpose

This evaluation card documents the current evaluation approach for ColliderOpsAI models.

The goal is to make model performance transparent, reproducible, and easy to interpret for researchers, ML engineers, and future RAG/agentic assistant workflows.

The current evaluation should be treated as an MVP pipeline validation, not as scientifically meaningful model performance.

---

## Current Evaluation Status

Current status:

```text
MVP evaluation pipeline is working.
```

The current evaluation verifies that the project can:

- Load prepared collider-style data
- Split features and labels
- Load saved model artifacts
- Generate predictions
- Calculate classification metrics
- Compare baseline model performance

---

## Models Evaluated

Current evaluated models:

```text
logistic_regression_baseline.joblib
random_forest_baseline.joblib
```

The FastAPI inference service currently uses:

```text
random_forest_baseline.joblib
```

---

## Evaluation Dataset

Current dataset:

```text
data/raw/collider_events.csv
```

Important note:

The current dataset is a small fallback collider-style dataset used to validate the end-to-end ML workflow. It is not yet a large real CERN/LHC dataset.

Because of the small dataset size, evaluation metrics may appear perfect or near-perfect. These scores should not be interpreted as meaningful scientific performance.

---

## Target Variable

The target label is:

```text
label
```

Allowed classes:

```text
signal
background
```

During training and evaluation, labels are mapped as:

```text
background → 0
signal     → 1
```

---

## Input Features Used for Evaluation

Current features:

```text
DER_mass_MMC
DER_mass_transverse_met_lep
DER_mass_vis
PRI_tau_pt
PRI_lep_pt
```

Identifier columns such as `event_id` are excluded from model training and evaluation.

---

## Evaluation Metrics

Current metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall percentage of correct predictions |
| Precision | Measures how many predicted positive/signal events are actually positive |
| Recall | Measures how many actual positive/signal events were correctly identified |
| F1 Score | Harmonic mean of precision and recall |

These metrics are appropriate for the MVP, but future versions should include additional scientific and operational metrics.

---

## Current Evaluation Flow

Evaluation is handled by:

```text
training/evaluate.py
```

Current flow:

```text
load test data
→ load saved models
→ generate predictions
→ calculate metrics
→ compare model performance
→ optionally save metrics output
```

---

## Known Evaluation Limitation

The current evaluation is limited because:

- Dataset is very small
- Dataset is fallback/synthetic-style, not real collider-scale data
- Train/test split is not yet robust
- No cross-validation yet
- No external validation dataset yet
- No experiment tracking through MLflow yet
- No statistical uncertainty analysis yet
- No physics-domain benchmark comparison yet

For this reason, current metrics should be interpreted as:

```text
pipeline validation metrics
```

not:

```text
scientific model performance metrics
```

---

## Confidence Triage Evaluation

ColliderOpsAI uses model probability to assign confidence triage fields:

```text
risk_level
needs_review
```

Default threshold:

```text
0.75
```

Rules:

```text
probability >= 0.75 → high_confidence, needs_review = false
probability < 0.75  → low_confidence, needs_review = true
probability missing → unknown, needs_review = true
```

Future evaluation should measure:

- Number of low-confidence predictions
- Percentage of predictions requiring review
- Whether low-confidence predictions correspond to higher error rates
- Whether threshold tuning improves review efficiency

---

## Batch Evaluation and Research-Run Summary

Batch prediction outputs include a summary with:

```text
total_events
signal_count
background_count
high_confidence_count
low_confidence_count
review_required_count
average_probability
```

This summary is useful for research-run inspection, but it is not a replacement for formal model evaluation.

Future batch evaluation should include:

- Per-batch class distribution
- Per-batch confidence distribution
- Drift against historical batch distributions
- Low-confidence event clustering
- Exportable evaluation reports

---

## Recommended Next Evaluation Improvements

Short-term improvements:

1. Add a larger collider-style dataset
2. Add stratified cross-validation
3. Save evaluation outputs to a versioned report file
4. Add confusion matrix output
5. Add ROC-AUC where appropriate
6. Track evaluation results with MLflow
7. Compare Logistic Regression, Random Forest, XGBoost, and LightGBM

Medium-term improvements:

1. Add SHAP-based feature importance
2. Add model calibration analysis
3. Add threshold tuning for `needs_review`
4. Add dataset versioning
5. Add model version comparison reports
6. Add drift monitoring on batch prediction inputs

Long-term improvements:

1. Evaluate on real CERN/Open Data-style datasets
2. Add domain expert validation notes
3. Add physics-aware metrics where appropriate
4. Compare against accepted baseline methods
5. Add reproducible experiment reports for each model version

---

## Evaluation Report for RAG Assistant

This evaluation card will be used as part of the future LangChain RAG knowledge base.

The assistant should be able to answer questions such as:

- What models have been evaluated?
- What metrics are currently used?
- Why are current scores not scientifically meaningful?
- What does `needs_review` mean?
- What should be improved in the next evaluation phase?
- How should batch prediction summaries be interpreted?

---

## Current Evaluation Interpretation

The correct interpretation of the current evaluation is:

> The model evaluation pipeline works end to end, but the current metrics are only useful for validating the software workflow. They should not be used to claim strong scientific performance until the project uses a larger, validated collider dataset.

---

## Current Status

Completed:

- Evaluation module exists
- Baseline models can be evaluated
- Standard classification metrics are calculated
- API-level tests are passing
- Prediction confidence fields are implemented
- Batch summary statistics are implemented

Next planned evaluation work:

- Add larger dataset
- Add confusion matrix
- Add ROC-AUC
- Add MLflow tracking
- Add model comparison table
- Add versioned evaluation reports
- Add SHAP/feature importance
