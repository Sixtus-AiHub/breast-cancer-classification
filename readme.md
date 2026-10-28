# Breast Cancer Classification: Data Understanding and Exploratory Analysis

## 1. Project objective

The project investigates how supervised machine learning can distinguish benign from malignant breast-cancer samples using numerical measurements of cell nuclei.

Learning objectives:
- Reliable data auditing and exploratory analysis.
- Statistical comparison of benign and malignant groups.
- Understanding feature correlation and redundancy.
- Preventing data leakage.
- Designing a defensible model-evaluation workflow.
- Comparing predictive performance and interpretation methods in later phases.

This is an educational machine-learning project, not a clinically validated diagnostic system.

## 2. Dataset and setup

The dataset was loaded using `sklearn.datasets.load_breast_cancer`. The working notebook objects are:
- `X`: DataFrame containing the 30 numerical predictors.
- `y`: the original scikit-learn target.
- `df`: predictors plus a human-readable `diagnosis` column.

The dataset contains **569 observations** and **30 numerical features**.

For this project, the intended target encoding is:
- `malignant = 1`
- `benign = 0`

This is the reverse of the original scikit-learn target encoding, where 0 represents malignant and 1 represents benign. Explicit encoding makes recall/sensitivity for the positive class refer to malignant cases.

## 3. Data audit

| Audit item | Result | Interpretation |
|---|---:|---|
| Observations | 569 | Small-to-moderate tabular dataset |
| Numerical predictors | 30 | All numerical measurements |
| Missing values | 0 | No imputation currently required |
| Exact duplicate rows | 0 | No exact duplicate rows found |
| Numeric feature types | `float64` | Numeric predictors |

### Class distribution

| Diagnosis | Count | Percentage |
|---|---:|---:|
| Benign | 357 | 62.74% |
| Malignant | 212 | 37.26% |
| **Total** | **569** | **100%** |

The classes are moderately imbalanced. An all-benign classifier would achieve approximately 62.7% accuracy but would fail to identify malignant cases. Accuracy alone is insufficient for model evaluation.

## 4. Numerical summaries and feature distributions

The `X.describe().T` summary showed that feature scales vary substantially. Selected values:

| Feature | Mean | Standard deviation | Median | Maximum |
|---|---:|---:|---:|---:|
| `worst area` | 880.583 | 569.357 | 686.5 | 4254 |
| `mean area` | 654.889 | 351.914 | 551.1 | 2501 |
| `area error` | 40.337 | 45.491 | 24.53 | 542.2 |
| `worst perimeter` | 107.261 | 33.603 | 97.66 | 251.2 |
| `mean perimeter` | 91.969 | 24.299 | 86.24 | 188.5 |
| `worst texture` | 25.677 | 6.146 | 25.41 | 49.54 |
| `worst radius` | 16.269 | 4.833 | 14.97 | 36.04 |
| `mean texture` | 19.290 | 4.301 | 18.84 | 39.28 |
| `mean radius` | 14.127 | 3.524 | 13.37 | 28.11 |
| `perimeter error` | 2.866 | 2.022 | 2.287 | 21.98 |

Standardization may be useful for scale-sensitive algorithms such as regularized Logistic Regression and SVM. Tree-based models such as Random Forest generally do not require standardization.

### Histograms and boxplots

Histograms and boxplots were reviewed for six features with the largest standard deviations: `worst area`, `mean area`, `area error`, `worst perimeter`, `mean perimeter`, and `worst texture`.

Main observations:
1. `worst area`, `mean area`, and `area error` are strongly right-skewed, with long upper tails.
2. `worst perimeter` and `mean perimeter` also show right-skewed distributions and high-value observations.
3. `worst texture` appears more balanced than the area features, though not perfectly symmetric.
4. Boxplots flag potential outliers. These are points beyond the boxplot whiskers, not proof of erroneous data.
5. Extreme values may represent genuine variation. No outlier removal or transformation has been performed.

**Decision:** Retain all observations for now. Investigate suspected data errors rather than removing points solely because they are flagged as outliers. Establish baseline performance before considering transformations.

## 5. Feature distributions by diagnosis

Boxplots compared benign and malignant observations for `mean area`, `worst area`, `mean perimeter`, `worst perimeter`, `area error`, and `worst texture`.

For all six, the malignant group has a higher sample median than the benign group. Area and perimeter measurements show particularly clear differences in central values and spread. `worst texture` also has a higher malignant median, although the distributions overlap.

These plots are exploratory. Median separation does not mean every malignant observation has a larger value than every benign observation. Visual differences alone do not establish statistical significance or predictive performance.

## 6. Inferential analysis: Mann–Whitney U tests

Two-sided Mann–Whitney U tests compared feature distributions between malignant and benign cases. P-values were adjusted using the Benjamini–Hochberg false discovery rate procedure: first across the six initially selected features, then across all 30 predictors.

Hypotheses:
- **Null hypothesis:** the feature distributions are the same in benign and malignant groups.
- **Alternative hypothesis:** the feature distributions differ between groups.

The Mann–Whitney U test is rank-based. It tests for distributional differences; it is not automatically a test of median differences unless additional assumptions are met.

### Selected results

| Feature | Malignant median | Benign median | Median difference (malignant − benign) |
|---|---:|---:|---:|
| `worst perimeter` | 138.000 | 86.920 | 51.080 |
| `worst radius` | 20.590 | 13.350 | 7.240 |
| `worst area` | 1303.000 | 547.400 | 755.600 |
| `worst concave points` | 0.18200 | 0.07431 | 0.10769 |
| `mean concave points` | 0.08628 | 0.02344 | 0.06284 |
| `mean perimeter` | 114.200 | 78.180 | 36.020 |
| `mean area` | 932.000 | 458.400 | 473.600 |
| `mean concavity` | 0.15135 | 0.03709 | 0.11426 |
| `mean radius` | 17.325 | 12.200 | 5.125 |
| `area error` | 58.455 | 19.630 | 38.825 |
| `worst concavity` | 0.40490 | 0.14120 | 0.26370 |
| `perimeter error` | 3.6795 | 1.8510 | 1.8285 |
| `radius error` | 0.5472 | 0.2575 | 0.2897 |
| `mean compactness` | 0.13235 | 0.07529 | 0.05706 |
| `worst compactness` | 0.35635 | 0.16980 | 0.18655 |

### Multiple-testing result

Using an adjusted significance threshold of 0.05:
- **27 of 30 features** had adjusted p-values below 0.05.
- **3 features** did not meet this threshold:
  - `smoothness error`: adjusted p-value 0.228891
  - `mean fractal dimension`: adjusted p-value 0.555709
  - `texture error`: adjusted p-value 0.643693

Many p-values appeared as `0.000000` because of display rounding; this does not mean they are mathematically zero. Scientific notation should be used to inspect very small values.

The results provide strong evidence that many feature distributions differ between diagnosis groups. These results do not show that the features cause malignancy, that statistically significant features are automatically the best predictors, or that a model will perform well on unseen data. Raw median differences cannot be compared directly across features with different units and scales.

The tests were performed on the full dataset for exploratory inference. Their results must not be used to preselect features before cross-validation for predictive modeling. Any predictive feature selection must be fitted within training folds to prevent information leakage.

## 7. Correlation analysis

A Pearson correlation matrix was computed for all 30 numerical predictors. It showed clusters of strongly positively correlated measurements, particularly among related size and shape features.

### Strongest observed feature pairs

| Feature 1 | Feature 2 | Pearson correlation |
|---|---|---:|
| `mean radius` | `mean perimeter` | 0.998 |
| `worst radius` | `worst perimeter` | 0.994 |
| `mean radius` | `mean area` | 0.987 |
| `mean perimeter` | `mean area` | 0.987 |
| `worst radius` | `worst area` | 0.984 |
| `worst perimeter` | `worst area` | 0.978 |
| `radius error` | `perimeter error` | 0.973 |
| `mean perimeter` | `worst perimeter` | 0.970 |
| `mean radius` | `worst radius` | 0.970 |
| `mean perimeter` | `worst radius` | 0.969 |
| `mean radius` | `worst perimeter` | 0.965 |
| `mean area` | `worst radius` | 0.963 |
| `mean area` | `worst area` | 0.959 |
| `mean area` | `worst perimeter` | 0.959 |
| `radius error` | `area error` | 0.952 |

### Interpretation

1. Radius, perimeter, and area measurements are highly correlated, plausibly because they describe related geometric aspects of nuclei.
2. Mean and worst measurements of related quantities are also strongly correlated.
3. Compactness, concavity, and concave-point measurements form another positively correlated group.
4. Strong predictor correlation can create multicollinearity and make individual coefficients harder to interpret in regression models.
5. Correlation does not establish causation, does not determine target-predictive value, and does not automatically justify removing features.

**Decision:** Keep all 30 features for the initial baseline. Investigate feature selection later as a modeling experiment, with selection performed inside cross-validation. Logistic Regression and Random Forest will provide useful contrasts because they respond differently to correlated predictors.

### Preprocessing and candidate models

- StandardScaler was included inside the Logistic Regression and SVC pipelines.
- Random Forest and XGBoost were used without feature scaling.
- All 30 features were retained in the selected final workflow.
- The final estimator was `LogisticRegression(C=1, max_iter=5000, random_state=42)` inside a `StandardScaler` pipeline.

## Model comparison

Nested five-fold cross-validation on the training split produced the following ROC-AUC summaries:

| Model | Mean outer-fold ROC-AUC | Standard deviation |
|---|---:|---:|
| SVC (RBF) | 0.9954 | 0.0064 |
| Logistic Regression | 0.9953 | 0.0046 |
| XGBoost | 0.9939 | 0.0046 |
| Random Forest | 0.9883 | 0.0072 |

SVC and Logistic Regression were effectively tied by mean ROC-AUC. Logistic Regression was selected for the final workflow because its ranking performance was comparable and its behavior is more straightforward to interpret. The fold standard deviations are descriptive variability, not confidence intervals or formal significance tests.

## Final model and held-out test results

**Selected workflow:** `StandardScaler` + Logistic Regression, all 30 features, decision threshold **0.40**. Malignant is the positive class.

### Test confusion matrix

Rows are actual classes; columns are predicted classes.

| Actual class | Predicted benign | Predicted malignant |
|---|---:|---:|
| Benign | 71 (true negative) | 1 (false positive) |
| Malignant | 2 (false negative) | 40 (true positive) |

### Test metrics

| Metric | Result |
|---|---:|
| Accuracy | 0.9737 (97.37%) |
| Malignant recall / sensitivity | 0.9524 (95.24%) |
| Precision | 0.9756 (97.56%) |
| Specificity | 0.9861 (98.61%) |
| F1 score | 0.9639 |
| ROC-AUC | 0.9960 |
| Average precision | 0.9943 |

On this test split, the model identified 40 of 42 malignant samples and missed 2. It correctly classified 71 of 72 benign samples and produced 1 false positive. The observed malignant recall met the project's development target of at least 95%; this is a project criterion, not a clinical recommendation or guarantee. Because the test set contains only 42 malignant samples, one additional missed case would change recall by about 2.38 percentage points.

The held-out test set was used for the final evaluation only. **Do not use these test results to tune the model or threshold.**

## Reproducibility

The notebook was run from top to bottom successfully in the current environment. The confirmed project Python executable was:

```text
.venv\Scripts\python.exe
```

The environment has XGBoost **3.4.1**, and `requirements.txt` was updated to include:

```text
xgboost==3.4.1
```

The current `requirements.txt` is a full `pip freeze`-style list of installed packages. The existing environment's `pip check` reported no broken requirements. A clean-environment installation test has **not** been performed, so successful reproduction in a fresh environment has not yet been confirmed.

To install the listed dependencies in an activated project environment, run:

```bash
python -m pip install -r requirements.txt
```

The dataset is loaded through scikit-learn; a separate dataset download is not required for the documented data-loading step.

## Limitations

- This is a small, educational dataset; the results do not establish performance on other populations, institutions, or real-world clinical settings.
- Threshold selection was performed on training-only out-of-fold predictions, but model and threshold selection can still overfit development data.
- Cross-validation fold standard deviations are descriptive and should not be interpreted as confidence intervals or formal tests of model superiority.
- Statistical tests and correlations describe associations in this dataset; they do not establish causation.
- The selected 0.40 threshold is an educational operating choice, not a clinically validated decision threshold.

## Skills demonstrated

- Data auditing and exploratory data analysis
- Statistical testing and multiple-comparison adjustment
- Correlation analysis and multicollinearity awareness
- Scikit-learn pipelines and preprocessing
- Stratified cross-validation and nested model evaluation
- Hyperparameter tuning
- Leakage-aware out-of-fold threshold selection
- Feature-selection comparison with RFE
- Confusion-matrix analysis and classification metrics
- Reproducibility and documentation

## Post-evaluation error analysis

After the one-time held-out test evaluation, the three misclassified observations were inspected descriptively. This analysis documents the observed errors; it was not used to change the fitted model, the decision threshold, or the reported test metrics.

### Misclassified observations

| Test index label | Actual diagnosis | Predicted diagnosis | Predicted malignant probability | Error type |
|---:|---|---|---:|---|
| 455 | Benign | Malignant | 0.509155 (50.92%) | False positive |
| 73 | Malignant | Benign | 0.061429 (6.14%) | False negative |
| 190 | Malignant | Benign | 0.314532 (31.45%) | False negative |

The values 455, 73, and 190 are DataFrame index labels, not necessarily original dataset row numbers. Case 455's probability was above the fixed 0.40 threshold. Case 190's probability was below the threshold but closer to it than Case 73's. Case 73 received a particularly low malignant probability despite being malignant in the dataset.

### Selected feature measurements

| Test index label | Worst radius | Worst area | Worst concave points | Mean concave points |
|---:|---:|---:|---:|---:|
| 455 (false positive) | 15.05 | 705.6 | 0.07763 | 0.03264 |
| 73 (false negative) | 16.57 | 812.4 | 0.13830 | 0.05069 |
| 190 (false negative) | 15.74 | 762.4 | 0.17720 | 0.06618 |

For context, the medians of these features within the held-out test set were:

| Feature | Benign median | Malignant median |
|---|---:|---:|
| Worst radius | 13.335 | 20.075 |
| Worst area | 538.2 | 1248.5 |
| Worst concave points | 0.07597 | 0.17565 |
| Mean concave points | 0.023815 | 0.08057 |

All four selected measurements for Case 455 were above the benign-group medians, which may help describe why this benign case received a relatively high malignant probability. For Case 73, all four measurements were below the malignant-group medians. For Case 190, three were below the malignant medians and worst concave points was slightly above its malignant median. These are descriptive comparisons only: group medians are not decision boundaries, and four features cannot fully explain a Logistic Regression prediction based on all 30 features.

### What this analysis does—and does not—show

- The errors had different probability patterns: one false positive just above the threshold and two false negatives below it.
- The two false negatives were not identical in their selected feature measurements or predicted probabilities.
- This small, post-evaluation inspection does not establish the cause of any error, prove that the probabilities are calibrated, or establish clinical significance.
- The observations and findings were not used to tune the model or threshold. The held-out test metrics above remain the final reported results.

## Final project status

The core modeling and evaluation work is complete: exploratory analysis, statistical tests, baseline comparison, hyperparameter tuning, nested cross-validation, RFE comparison, training-only out-of-fold threshold selection, final fitting, one-time held-out test evaluation, notebook execution check, and descriptive post-evaluation error analysis. Remaining publication work is repository housekeeping: place this README as `README.md` in the project repository, ensure the updated `requirements.txt` is included, and commit the final files. A clean-environment installation test has not been completed and is not claimed here.
