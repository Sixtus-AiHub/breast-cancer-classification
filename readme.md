# Breast Cancer Classification: Data Understanding and Exploratory Analysis

## Project status

**Current phase:** Data audit and exploratory data analysis (EDA) completed; preprocessing and evaluation design planned but not yet executed.

This README documents the work completed so far on the Breast Cancer Wisconsin (Diagnostic) dataset. It records observed findings, the statistical analyses performed, and decisions for the next stage. No predictive model has been trained in this phase.

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

## 8. Preprocessing plan (planned, not yet executed)

| Issue | Current decision | Rationale |
|---|---|---|
| Missing values | No imputation initially | Audit found no missing values |
| Duplicate rows | No removal | Audit found no exact duplicate rows |
| Target encoding | Malignant = 1; benign = 0 | Makes malignant recall/sensitivity easy to interpret |
| Scaling | Standardize for regularized Logistic Regression | Predictors have very different scales |
| Random Forest scaling | Not required by default | Tree-based splits generally are not sensitive to feature scaling |
| Outliers | Retain initially | Outliers may be genuine observations |
| Skewness | No automatic transformation | Establish baseline before testing transformations |
| Feature selection | None for initial baseline | Avoid leakage and establish a comparison point |

Any learned preprocessing, transformation, or feature selection used in prediction should be placed inside a scikit-learn `Pipeline` and fitted within the appropriate training fold.

## 9. Evaluation design (planned, not yet executed)

Proposed workflow:
1. Encode the target as malignant = 1 and benign = 0.
2. Split once using a stratified 80/20 train/test split and a fixed random seed.
3. Keep the test set untouched during model selection and tuning.
4. Use stratified five-fold cross-validation on the training set.
5. Place scaling and any learned feature selection inside the model pipeline.
6. Compare a regularized Logistic Regression baseline with a Random Forest baseline.
7. Choose the model and classification threshold using training data and a predeclared decision procedure.
8. Evaluate the finalized approach on the test set once.

With 569 observations, the expected split is approximately 455 training observations and 114 test observations; exact class counts depend on the stratified split.

### Metrics to report

- Recall/sensitivity for malignant cases.
- Specificity for benign cases.
- Precision for malignant predictions.
- F1-score.
- ROC-AUC.
- Average precision / precision–recall analysis.
- Confusion matrix.
- Accuracy and balanced accuracy as supplementary metrics.

Accuracy alone will not be the main performance measure. Threshold selection will consider the trade-off between false negatives and false positives. This is an educational project, not a clinically validated diagnostic tool.

## 10. Current status and next steps

### Completed
- Dataset loaded and organized into `X`, `y`, and `df`.
- Missing values and exact duplicate rows audited.
- Class distribution examined.
- Numerical summaries reviewed.
- Histograms and boxplots examined for six high-variability features.
- Feature distributions compared by diagnosis.
- Mann–Whitney U tests run for all 30 features, with Benjamini–Hochberg adjustment.
- Pearson correlation matrix and strongest feature pairs examined.
- Initial preprocessing and evaluation plan documented.

### Not yet completed
- Train/test split.
- Model pipelines.
- Cross-validation comparison.
- Hyperparameter tuning.
- Threshold selection.
- Predictive feature selection within cross-validation.
- Final test evaluation.
- Model interpretation and uncertainty analysis.
- Reproducibility checks and final project reporting.

### Immediate next notebook task

Create the stratified train/test split with `malignant = 1` and `benign = 0`, inspect class counts and proportions, and then build the initial model pipelines without exposing the test set to training or model-selection decisions.

## 11. Reproducibility notes

- Dataset source in the current notebook: `sklearn.datasets.load_breast_cancer`.
- A fixed random seed of `42` is proposed for the train/test split.
- The dataset objects and variable names described here reflect the current notebook workflow.
- Record package versions, random seeds, and final evaluation choices in the project repository before publishing.
- The statistical tests and correlation analysis documented here were exploratory analyses on the full dataset; they are not a substitute for leakage-safe predictive evaluation.

---
*This README documents the work completed up to the end of EDA and the planned evaluation strategy. Update the status and results as the project progresses.*
