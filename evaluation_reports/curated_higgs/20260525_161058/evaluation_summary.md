# Evaluation Summary: curated_higgs

Timestamp: `20260525_161058`

## Model comparison

| model_name                           |   recall_score |   precision_score |   accuracy_score |   f1_score |   roc_auc_score |
|:-------------------------------------|---------------:|------------------:|-----------------:|-----------:|----------------:|
| logistic_regression_curated_higgs    |          0.649 |          0.620459 |            0.626 |   0.634409 |        0.678715 |
| random_forest_curated_higgs          |          0.68  |          0.714286 |            0.704 |   0.696721 |        0.781093 |
| hist_gradient_boosting_curated_higgs |          0.698 |          0.702213 |            0.701 |   0.7001   |        0.786047 |

## Best models

- Best accuracy: `random_forest_curated_higgs` (0.7040)
- Best F1: `hist_gradient_boosting_curated_higgs` (0.7001)
- Best ROC-AUC: `hist_gradient_boosting_curated_higgs` (0.7860)

## Top permutation-importance features

### logistic_regression_curated_higgs

- `m_wwbb`: mean=0.091074, std=0.009436
- `m_bb`: mean=0.065396, std=0.008720
- `m_wbb`: mean=0.033881, std=0.007294
- `jet_1_pt`: mean=0.018423, std=0.006896
- `missing_energy_magnitude`: mean=0.014681, std=0.005317

### random_forest_curated_higgs

- `m_bb`: mean=0.079751, std=0.012377
- `m_wwbb`: mean=0.072830, std=0.004717
- `m_wbb`: mean=0.040191, std=0.005190
- `m_jlv`: mean=0.023903, std=0.004834
- `m_jjj`: mean=0.016916, std=0.006610

### hist_gradient_boosting_curated_higgs

- `m_bb`: mean=0.091789, std=0.010539
- `m_wwbb`: mean=0.070723, std=0.006892
- `m_wbb`: mean=0.044149, std=0.006034
- `m_jlv`: mean=0.027632, std=0.004729
- `m_jjj`: mean=0.017851, std=0.005005

## Interpretation notes

- Accuracy, precision, recall, and F1 evaluate the model at its default classification threshold.
- ROC-AUC evaluates how well the model ranks signal-like events above background-like events across thresholds.
- Permutation importance estimates how much model performance drops when each feature is shuffled.
- Features with high mean importance and relatively low standard deviation are stronger, more stable drivers of model performance.
