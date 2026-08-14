# Heart Disease Threshold Explorer

An interactive Streamlit application for exploring diagnostic classification thresholds in an AIC-selected stepwise logistic regression model for binary heart disease prediction.

> **Educational use only.** The project is not a clinical diagnostic tool and must not be used to make real patient-care decisions.

## Overview

This project explores how clinical and biomedical measurements can be used to predict the binary outcome of heart disease using the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease).

The model was trained on a randomly selected 80% of the cleaned dataset. The remaining 20% was held out as an independent test set for model evaluation and threshold analysis.

## Selected Model

The final model was selected using bidirectional AIC stepwise selection.

```text
disease ~ sex + cp + restecg + exang + oldpeak + slope + ca + thal
```

The stepwise logistic regression model is:

```text
logit[P(Heart Disease = Yes)] = η

η = -5.866
    + 0.915(sex1)
    + 1.579(cp2)
    + 0.070(cp3)
    + 2.392(cp4)
    - 0.098(restecg1)
    + 0.906(restecg2)
    + 0.703(exang1)
    + 0.634(oldpeak)
    + 1.427(slope2)
    + 0.247(slope3)
    + 2.381(ca1)
    + 3.007(ca2)
    + 2.305(ca3)
    - 0.076(thal6)
    + 1.565(thal7)

P(Heart Disease = Yes) = 1 / [1 + exp(-η)]
```

All estimated beta coefficients are rounded to three decimal places.

Reference categories:

- `sex = 0`
- `cp = 1`
- `restecg = 0`
- `exang = 0`
- `slope = 1`
- `ca = 0`
- `thal = 3`

## Performance

| Metric | Result |
|---|---:|
| Full model AIC | 188.104 |
| Stepwise model AIC | 181.918 |
| Full model test-set AUC | 0.912 |
| Stepwise model test-set AUC | 0.904 |
| Full model repeated CV AUC | 0.892 |
| Stepwise model repeated CV AUC | 0.898 |
| Youden-optimal threshold | 0.490 |
| Sensitivity at Youden threshold | 81.5% |
| Specificity at Youden threshold | 96.9% |
| Maximum-utility threshold | 0.383 |
| Sensitivity at maximum-utility threshold | 85.2% |
| Specificity at maximum-utility threshold | 87.5% |
| Maximum total utility | 302 |

The Youden-optimal threshold was calculated in R with:

```r
pROC::coords(
  roc_step,
  x = "best",
  best.method = "youden"
)
```

The default clinical utility assumptions were:

| Outcome | Utility |
|---|---:|
| True Positive | +10 |
| True Negative | +5 |
| False Positive | -2 |
| False Negative | -15 |

## Application Features

The Streamlit application allows users to:

- Explore model-predictor distributions using the full cleaned dataset.
- View bar charts for categorical predictors and a boxplot for `oldpeak`.
- Select a predicted-probability threshold for binary classification.
- View threshold-dependent sensitivity and specificity on the held-out test set.
- Set the utility values of TP, FP, TN, and FN outcomes.
- Calculate total utility for the selected threshold.
- View TP, FP, TN, and FN counts at the selected threshold.
- Compare the selected threshold with the Youden-optimal threshold.
- Compare the selected threshold with the utility-maximizing threshold.

## Structure

```text
heart_disease_threshold_explorer/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── test_predictions.csv
└── heart_for_explorer.csv
```

## Run Locally

From the root folder of the parent GitHub repository, enter the project folder and run:

```bash
cd heart_disease_threshold_explorer
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

On Windows PowerShell, activate the virtual environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open the displayed URL in a browser if the application does not open automatically.

## Data Files

The application uses two CSV files stored in the same folder as `app.py`:

| File | Purpose |
|---|---|
| `test_predictions.csv` | Held-out test-set observed labels and predicted probabilities. It is used for threshold evaluation, sensitivity, specificity, confusion-matrix counts, and clinical utility calculations. |
| `heart_for_explorer.csv` | The full cleaned dataset. It is used exclusively for Predictor Explorer bar charts and boxplots. |

## Data Source

- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)

This project uses the Cleveland heart disease data.

## Analysis

The complete R analysis is in a different directory of the parent repository:

It includes data cleaning, exploratory analysis, logistic-regression fitting, AIC stepwise model selection, ROC/AUC comparison, repeated cross-validation, and threshold analyses.

## Author

Joanna Zou  
[GitHub Profile](https://github.com/joannaazou)
