# Handcrafted Features + CatBoost — Compact Report

## 1. Experimental setup

After reproducing the original 9-feature baseline, we extended the handcrafted feature set and performed sequential feature selection.

All main comparisons used the same `5-fold StratifiedKFold` with `random_state=42`.  
Metric: **multiclass log loss**, where lower is better.

### Original baseline

| Model | Log loss |
|---|---:|
| Logistic Regression + 9 features | 1.062693 |
| CatBoost + 9 features | 1.041892 |

Replacing Logistic Regression with CatBoost while keeping the same 9 features improved log loss by **0.020801**.

## 2. Feature extension and selection

The initial extended set contained **33 features**, combining the original EDA-derived features with response-structure statistics.

Group ablation identified the most useful groups:

| Removed group | Log-loss degradation |
|---|---:|
| Response length | +0.008524 |
| Refusals | +0.007563 |
| Repetition | +0.005295 |
| Prompt length | +0.001997 |
| Punctuation | +0.001542 |
| Paragraph statistics | +0.000760 |

Markdown and sentence-level statistics provided no additional benefit and were removed.

Several length-related features were redundant. The length representation was reduced from six variables to three:
`len_ratio_a_b`, `length_a`, and `length_b`.

The paragraph group was reduced from 10 to 8 variables by removing
`sd_sent_per_paragraph_a` and `sd_sent_per_paragraph_b`.

## 3. New EDA insight: numerical content

When only response A contained numbers:

- A wins: **48.8%**
- B wins: 25.3%
- Tie: 25.9%

When only response B contained numbers:

- A wins: 26.0%
- B wins: **48.2%**
- Tie: 25.8%

The effect was nearly symmetric, so two binary indicators were added:
`has_numbers_a` and `has_numbers_b`.

The `numeric_density_diff` feature produced almost no gain and was not included in the final set.

## 4. Final 20-feature set

### Length — 3

1. `len_ratio_a_b`  
   Relative response-length difference:
   `log(1 + len(A)) - log(1 + len(B))`.  
   Positive values mean A is relatively longer; negative values mean B is relatively longer.

2. `length_a`  
   Number of characters in response A.

3. `length_b`  
   Number of characters in response B.

### Refusals — 2

4. `refusals_a`  
   Count of refusal/apology markers in A, such as:
   `I cannot`, `I can't`, `I'm sorry`, `As an AI`, `I am unable`, `I apologize`.

5. `refusals_b`  
   The same count for response B.

### Repetition — 2

6. `repetition_density_a`  
   Lexical repetition density in A:
   `1 - unique_words / all_words`.  
   Higher values indicate more repeated vocabulary.

7. `repetition_density_b`  
   The same for B.

### Prompt — 1

8. `prompt_length`  
   Prompt length in characters, scaled as `len(prompt) / 1000`.

### Punctuation — 2

9. `punctuation_density_a`  
   Fraction of punctuation characters in response A:
   `punctuation_count / text_length`.

10. `punctuation_density_b`  
    The same for B.

### Paragraph structure — 8

11. `num_paragraphs_a`  
    Number of non-empty paragraphs in A.

12. `num_paragraphs_b`  
    Number of non-empty paragraphs in B.

13. `mean_paragraph_length_a`  
    Mean paragraph length in A, measured in words.

14. `mean_paragraph_length_b`  
    Mean paragraph length in B, measured in words.

15. `sd_paragraph_length_a`  
    Standard deviation of paragraph lengths in A, measured in words.

16. `sd_paragraph_length_b`  
    The same for B.

17. `avg_sent_per_paragraph_a`  
    Average number of sentences per paragraph in A.

18. `avg_sent_per_paragraph_b`  
    The same for B.

### Numeric content — 2

19. `has_numbers_a`  
    Binary indicator equal to `1` if response A contains at least one numerical expression, otherwise `0`.

20. `has_numbers_b`  
    The same for B.

## 5. Final comparison

| Model | Selection CV |
|---|---:|
| Logistic Regression + 9 | 1.062693 |
| CatBoost + 9 | 1.041892 |
| Logistic Regression + 20 | 1.056715 |
| **CatBoost + 20** | **1.028325** |

Compared with the original `Logistic Regression + 9` baseline, the final CatBoost model improved log loss by **0.034368**.

## 6. Robustness check

After freezing the final feature set, we repeated evaluation with a new `5-fold StratifiedKFold` using `random_state=314159`.

| Model | Selection CV | Final CV |
|---|---:|---:|
| Logistic Regression + 9 | 1.062693 | 1.062759 |
| CatBoost + 9 | 1.041892 | 1.042413 |
| Logistic Regression + 20 | 1.056715 | 1.056391 |
| **CatBoost + 20** | **1.028325** | **1.028788** |

The model ranking remained unchanged and the metric shifts were small.

## 7. Main conclusions

- CatBoost uses the handcrafted features substantially better than Logistic Regression.
- The extended feature set improves both models, with a larger gain for CatBoost.
- The strongest feature groups are response length, refusals, and repetition.
- Several length features were redundant and could be removed without degrading performance.
- Paragraph features were useful collectively: after removing one weak pair, further compression degraded validation performance.
- Numerical content showed a strong symmetric EDA effect and produced a small additional validation improvement.
- The final model uses 20 handcrafted features and remains stable under a different CV split.
