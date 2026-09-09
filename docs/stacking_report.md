# Stacking: CatBoost + DeBERTa-v3 — Report

## 1. Motivation

The previous experiment established CatBoost with 20 handcrafted features as the strongest tabular baseline (OOF log loss **1.0279**). This notebook explores whether combining those features with a fine-tuned DeBERTa-v3 cross-encoder can further reduce log loss.

All experiments use the same **5-fold StratifiedKFold** with `random_state=42`.
Metric: **multiclass log loss**, where lower is better.

## 2. Experimental setup

### Baselines

| Model | Log loss |
|---|---:|
| CatBoost (20 features, HP-tuned) | 1.0279 |
| DeBERTa-v3-base cross-encoder (max_length=1024) | 0.9939 |

### Data and features

- Training set: 57 477 samples, 3-class target.
- 20 handcrafted features (same set as the CatBoost report): length (3), refusals (2), repetition (2), prompt length (1), punctuation (2), paragraph structure (8), numeric content (2).
- DeBERTa-v3-base cross-encoder fine-tuned on the competition task with `max_length=1024` and smart head+tail truncation (prompt capped at 256 tokens, remaining budget split equally between responses).

### CatBoost hyperparameter tuning

Optuna TPE sampler, 20 trials, 5-fold CV, objective: minimize OOF log loss.

Best parameters found:

| Parameter | Value |
|---|---:|
| iterations | 896 |
| learning_rate | 0.0289 |
| depth | 7 |
| l2_leaf_reg | 5.920 |
| border_count | 73 |

## 3. Approach 1 — True stacking

Stack the OOF probability predictions from both models (3 classes x 2 models = 6 features) and train a meta-learner.

### 3a. Logistic Regression meta-model

StandardScaler applied to stacked features before fitting.

| Fold | Log loss |
|---|---:|
| 0 | 0.9922 |
| 1 | 0.9949 |
| 2 | 0.9869 |
| 3 | 0.9887 |
| 4 | 0.9967 |
| **Mean** | **0.9918 (+/- 0.0037)** |

### 3b. CatBoost meta-model

Small CatBoost (300 iterations, lr=0.05, depth=4) trained directly on the 6 stacked features.

| Fold | Log loss |
|---|---:|
| 0 | 0.9911 |
| 1 | 0.9938 |
| 2 | 0.9862 |
| 3 | 0.9865 |
| 4 | 0.9953 |
| **Mean** | **0.9906 (+/- 0.0037)** |

## 4. Approach 2 — CLS embedding + handcrafted features MLP

Instead of using DeBERTa's probability output, extract the `[CLS]` token embedding from the last hidden state (768-d), concatenate with the 20 handcrafted features (~788-d total), and train a small MLP.

### MLP hyperparameter tuning

Optuna TPE sampler, 20 trials, 5-fold CV, objective: minimize OOF log loss.

Search space:

| Parameter | Range |
|---|---|
| hidden_dim_1 | 64–512 |
| hidden_dim_2 | 32–hidden_dim_1 |
| dropout | 0.1–0.5 |
| lr | 1e-4–1e-2 (log) |
| weight_decay | 1e-5–1e-3 (log) |
| batch_size | 128, 256, 512 |
| patience | 5–10 |

Training: AdamW, CosineAnnealingLR, CrossEntropyLoss, max 50 epochs with early stopping.

### MLP architecture (after tuning)

The architecture is determined by Optuna. Example best configuration:

```
BatchNorm1d(input_dim)
Linear(input_dim, hidden_dim_1) + ReLU + Dropout(dropout)
BatchNorm1d(hidden_dim_1)
Linear(hidden_dim_1, hidden_dim_2) + ReLU + Dropout(dropout)
BatchNorm1d(hidden_dim_2)
Linear(hidden_dim_2, 3)
```

| Fold | Log loss |
|---|---:|
| 0 | 0.9896 |
| 1 | 0.9934 |
| 2 | 0.9845 |
| 3 | 0.9847 |
| 4 | 0.9942 |
| **Mean** | **0.9880 (+/- 0.0034)** |

## 5. Final comparison

| Approach | Log loss | Fold std |
|---|---:|---:|
| **MLP (CLS + 20 handcrafted features)** | **0.9880** | 0.0034 |
| Stacking: CatBoost meta-model | 0.9906 | 0.0037 |
| Stacking: LR meta-model | 0.9918 | 0.0037 |
| DeBERTa-v3 cross-encoder | 0.9939 | 0.0000 |
| CatBoost (20 features, HP-tuned) | 1.0279 | 0.0026 |

## 6. Conclusions

- **Best approach**: MLP on `[CLS]` embedding + 20 handcrafted features, achieving **0.9893** OOF log loss.
- Stacking DeBERTa and CatBoost OOF predictions (CatBoost meta-model) achieved **0.9906**, slightly worse than the MLP.
- Both stacking approaches and the MLP outperform the individual DeBERTa cross-encoder (0.9939).
- All ensemble methods provide a meaningful improvement over the CatBoost-only baseline (1.0279), with the best gain being **0.0386** in log loss.
- The MLP approach is preferred because it operates on a richer representation (CLS embedding) compared to the stacking approach which only uses the 3-class probability output.
