# Heart Disease Prediction and Diagnostic Threshold Analysis

## Overview

This project investigates the predictive capability of **13 clinical and biochemical variables** for identifying whether an individual has heart disease using the **UCI Heart Disease dataset**.

The main objective is to build a binary classification model using **logistic regression**, evaluate its predictive performance on a held-out prediction set, compare alternative model specifications, and explore how different probability thresholds affect diagnostic performance.

The project combines statistical modeling with decision-oriented threshold analysis, including:

- Data cleaning
- Exploratory data analysis
- Multivariable logistic regression
- VIF analysis
- AIC stepwise regression
- 5-fold cross-validation
- ROC and AUC analysis
- Threshold sensitivity-specifity analysis
- Confusion matrix analysis
- Utility-based threshold optimization
- Interactive visualization with Python and Streamlit

---

## Central Question of the Project

> **How well can 13 clinical and biochemical variables predict whether an individual has heart disease, and how should the predicted probability be converted into a binary diagnostic decision?**

The project focuses on two related questions:

1. How accurately can the clinical variables distinguish individuals with and without heart disease?
2. How should a predicted probability be converted into a final binary classification when false positives and false negatives may have different consequences?

---

## Social Meaning
Although heart disease can be directly checked-using the coronary angiography, this invasive procedure involves threading a catheter into the coronary arteries, injecting contrast dye, and visualizing blockages in real time via X-ray, which is very expansive and can pose harm to human body. As a result, predictions using such logistic regression model can serve as a reference to reduce unnecessary angiographies.

## Dataset

The analysis is based on the **UCI Heart Disease dataset**(https://archive.ics.uci.edu/dataset/45/heart+disease).

The 13 predictors used in the project are:

- `age`
- `sex`
- `cp`
- `trestbps`
- `chol`
- `fbs`
- `restecg`
- `thalach`
- `exang`
- `oldpeak`
- `slope`
- `ca`
- `thal`

The binary outcome is:

```text
disease
├── No
└── Yes
```

Categorical clinical variables were treated as factors in the logistic regression model where appropriate.

---

## Analytical Workflow

```text
UCI Heart Disease Dataset
          ↓
    Data Cleaning
          ↓
Exploratory Data Analysis
          ↓
      Train/Test Split
          ↓
   ┌───────────────┐
   │               │
   ↓               ↓
Full Logistic    AIC Stepwise
Regression       Regression
   │               │
   └───────┬───────┘
           ↓
     Model Comparison
(AIC, CV-5fold, ROC&AUC on test set)
           ↓
    Final Stepwise Model
           ↓
     Test-Set Prediction
           ↓
     Threshold Analysis
           ↓
 ┌─────────┼──────────────┐
 ↓         ↓              ↓
Youden   Sensitivity    Utility
         Constraint
 └─────────┼──────────────┘
           ↓
 Clinical / Decision Analysis
           ↓
Interactive Python + Streamlit App
```

---

## 1. Data Cleaning and Exploratory Analysis

The analysis began with data cleaning and exploratory analysis.

The dataset was examined for:

- Missing observations
- Variable types
- Factor levels
- Outcome distribution
- Distributions of clinical variables
- Relationships between predictors and heart disease status

Missing observations were removed before model fitting, and categorical variables were explicitly treated as factors where appropriate.

Exploratory analysis was then used to understand the data and the relationships between the clinical variables and the disease outcome. **Plots from R are included in the file plot of this repository.**

**After data cleaning and exploratory analysis, the dataset is partitioned into a training set(80%) and a prediction set(20%). Models are built based on the training set.**

---

## 2. Full Logistic Regression Model

A multivariable logistic regression model was first fitted using all 13 predictors.

The model estimates the probability of heart disease as:

**ln(P(Disease = Yes) / (1 - P(Disease = Yes)))
= β₀ + β₁X₁ + ... + β₁₃X₁₃**

---

## 3. Multicollinearity Assessment

Variance Inflation Factor (VIF) was used to investigate potential multicollinearity among the predictors in the full model.

The VIF analysis did **not identify obvious severe multicollinearity**.

Therefore, the reduction in predictors produced by the subsequent stepwise procedure was not interpreted simply as a consequence of severe multicollinearity.

---

## 4. AIC Stepwise Regression

An **AIC-based stepwise logistic regression** was performed starting from the full model. 

The stepwise procedure selected **8 of the original 13 predictors**:

```text
sex
cp
restecg
exang
oldpeak
slope
ca
thal
```

The final stepwise model was:

```text
disease ~ sex + cp + restecg + exang +
          oldpeak + slope + ca + thal
```

Some variables showed stronger parameter significance in the stepwise model than in the full model.

---

## 5. Full Model vs. Stepwise Model

The two models were compared using several criteria rather than relying only on individual coefficient significance.

### AIC

**AIC<sub>step</sub> = 181.9**

**AIC<sub>full</sub> = 188.1**

The stepwise model had a lower AIC than the full model, indicating a slightly more favorable trade-off between model fit and complexity according to AIC.

### Test-Set AUC and ROC

Both models were evaluated on the held-out prediction set using ROC curves and AUC.

**AUC<sub>step</sub> = 0.904**

**AUC<sub>full</sub> = 0.912**

The stepwise model had a slightly lower test-set AUC than the full model, but the difference was small and the ROC curves showed substantial overlap.

### 5-Fold Cross-Validation

**Mean-ROC<sub>cv_step</sub> = 0.8979434**

**Mean-ROC<sub>cv_full</sub> = 0.8918531**

Five-fold cross-validation was performed on the training data.

The stepwise model achieved a slightly higher mean cross-validated AUC than the full model.

### Model Selection

Overall, the predictive performance of the two models was very similar.

Although the stepwise model had a slightly lower AUC on the prediction set, it:

- Used fewer predictors
- Had a lower AIC
- Had a slightly higher 5-fold cross-validated mean AUC
- Produced a highly similar ROC curve

Therefore, the **8-variable stepwise model was selected as the final model** because it provided a more parsimonious model while maintaining broadly comparable predictive performance.

---

## 6. Diagnostic Threshold Analysis

After selecting the final stepwise model, predicted probabilities were generated for the prediction set.

A probability threshold is required to convert the continuous predicted probability into a binary classification:

$$
P(Disease=Yes) \geq threshold
\Rightarrow Yes
$$

Different thresholds produce different combinations of:

- True Positives (TP)
- False Positives (FP)
- True Negatives (TN)
- False Negatives (FN)

Therefore, three threshold-selection strategies were explored.

### 6.1 Youden's Index

The first approach used **Youden's Index**:

$$
J = Sensitivity + Specificity - 1
$$

The threshold maximizing Youden's Index was selected.

This approach seeks a balance between sensitivity and specificity.

### 6.2 Sensitivity-Constrained Threshold

Because sensitivity can be particularly important in disease screening, a second threshold-selection method imposed a minimum sensitivity requirement.

The analysis searched for a threshold satisfying:

$$
Sensitivity \geq 0.85
$$

and selected the threshold with the highest specificity among the eligible thresholds.

This approach prioritizes reducing false negatives while retaining as much specificity as possible.

**The threshold value selected here is 0.382, with a sensitivity of 85.19% and a specificity of 87.50%.**

### 6.3 Utility-Based Threshold

The third approach incorporated the relative benefits and losses associated with TP, TN, FP, and FN outcomes.

A total utility was defined as:

$$
Utility =
U_{TP}TP+
U_{TN}TN+
U_{FP}FP+
U_{FN}FN
$$

The threshold producing the highest total utility was selected.

This framework allows the preferred threshold to depend on the assumed consequences of different classification outcomes. This utility-based threshold selection may have substantial potential in **real-world healthcare decision-making**, as it allows classification thresholds to reflect the relative clinical and economic consequences of false positives and false negatives.

**Here I define:**

**U<sub>TP</sub> = 10**

**U<sub>FP</sub> = -2**

**U<sub>TN</sub> = 5**

**U<sub>FN</sub> = -15**

**The threshold value selected using these defined values is 0.383, with a sensitivity of 85.19% and a specificity of 87.50%.**

---

## 7. Sensitivity-Specificity Analysis

Sensitivity and specificity were evaluated across a range of classification thresholds.

A **threshold sensitivity analysis graph** was created to visualize how sensitivity and specificity change as the threshold changes *(see the plot file)*.

The three selected thresholds were also marked on the graph:

- Youden threshold
- Sensitivity-constrained threshold
- Social utility threshold

This allows the different threshold-selection strategies to be compared visually.

---

## 8. Clinical Decision Perspective

Sensitivity is particularly important in this context because a false negative represents an individual who has heart disease but is classified as not having the disease.

If the potential loss associated with a false negative is high, a lower classification threshold may be preferred in order to identify more individuals with disease, even if this results in more false positives.

However, this project is not intended to establish actual clinical or societal costs **due to the lack of related utility data**. The utility analysis is instead used to demonstrate how different assumptions about TP, TN, FP, and FN can influence threshold selection.

---

## 9. Interactive Python + Streamlit Application: Heart Disease Threshold Explorer

To extend the statistical analysis into an interactive decision-making framework, a small application was developed using **Python and Streamlit**.

The application uses predictions generated by the final stepwise logistic regression model.

Users can interactively explore the model and threshold-selection process.

The application allows users to:

- Visualize relationships between selected model variables and disease status
- Explore variables using bar plots or box plots
- Adjust the classification threshold using an interactive slider
- Manually enter the assumed benefit or loss associated with:
  - True Positive (TP)
  - False Positive (FP)
  - True Negative (TN)
  - False Negative (FN)
- Observe how changing the threshold affects sensitivity and specificity
- Observe how changing the threshold affects total utility
- Explore how different TP/FP/TN/FN utility assumptions change the optimal threshold

The application is designed as a simplified simulation of **clinical decision-making under different cost-benefit assumptions**.

The underlying logistic regression model remains fixed while the threshold changes. Changing the threshold changes the classification rule rather than refitting the model.

---

## 10. Limitations

### Small Sample Size

The UCI Heart Disease dataset has a relatively small sample size.

This may partly explain why the full and stepwise models showed relatively similar predictive performance.

### Limited Training and Prediction Sets

Because the original dataset is relatively small, both the training and prediction sets contain limited numbers of observations.

Therefore, the estimated predictive performance may be uncertain, and the generalizability of the model requires further validation using larger and more diverse datasets.

### Prediction Does Not Imply Causation

This project focuses on **prediction rather than causal inference**.

The associations identified by logistic regression should therefore not be interpreted as causal relationships.

The project did not deeply investigate:

- Causal relationships
- Mediator effects
- Confounding structures
- Omitted variable bias

These represent important directions for further learning and improvement.

### Utility Assumptions

The utility-based threshold analysis depends on assumed values for the benefits and losses of TP, TN, FP, and FN.

These values are used for methodological exploration and should not be interpreted as validated clinical or economic estimates.

A real-world clinical implementation would require evidence-based utility estimates, clinical expertise, patient preferences, and potentially health-economic data.

---

## 11. Future Directions

Future improvements could include:

- External validation using a larger and independent dataset
- Bootstrap confidence intervals
- Repeated cross-validation
- More rigorous clinical utility estimation
- Health-economic analysis
- Investigation of mediator effects
- Investigation of omitted variable bias
- Causal inference
- Comparison with alternative machine-learning models
- Validation on independent clinical populations

In particular, further investigation into **causal relationships, mediator effects, and omitted variable bias** could help distinguish variables that are useful for prediction from variables that have meaningful causal relationships with heart disease.

---

## 12. Key Takeaways

This project demonstrates a complete workflow from clinical data to statistical prediction and decision-oriented threshold analysis:

```text
Clinical Data
      ↓
Data Cleaning
      ↓
Exploratory Analysis
      ↓
Full Logistic Regression
      ↓
VIF Assessment
      ↓
AIC Stepwise Selection
      ↓
Model Comparison
      ↓
Final Stepwise Model
      ↓
Prediction
      ↓
Threshold Analysis
      ↓
Youden / Sensitivity Constraint / Utility
      ↓
Clinical Decision Analysis
      ↓
Interactive Python Application
```

The main model-selection result was that the **8-variable stepwise model achieved predictive performance broadly comparable to the 13-variable full model while using fewer predictors**.

The threshold analysis further demonstrates that there is no single universally optimal classification threshold. The preferred threshold depends on the relative importance assigned to sensitivity, specificity, and the consequences of different classification errors.

Overall, this project explores how **biostatistical modeling can move beyond prediction toward transparent and decision-oriented clinical analysis**.

---

## 13. Notes from the Author

While conducting this project, I am a high school graduate student feeling lost about what to do in the future. Yet with a deep interest in exploring math and statistical formulas, together with a passion of life science related stuff, I thought it's a great idea to first learn some statistical theory and use some online public dataset to "play around" without the help of a third institution. Throughout the completion of this project, I systematically studied logistic regression on my own, including the underlying statistical principles of maximum likelihood estimation (MLE), the Hessian matrix, Newton–Raphson iteration, and the consistency and asymptotic normality of MLE logistic regression estimators under large-sample conditions.

I used AI only as a coding tool when writing and debugging code. I acknowledge that the modeling strategy and the logic behind the statistical analysis—including model selection and the comparison of statistical measures—were developed by me.

Through this project, I gained a deeper understanding that statistical tools can do more than capture patterns through mathematical models; they are also closely connected to decision-making, much like the role of economic considerations in determining costs, benefits, and trade-offs.

I hope to further explore methods for optimizing statistical models and the mathematical theories underlying them, while also strengthening my understanding of the domain-specific knowledge behind the data—for example, economic costs and the biological and biochemical meaning of clinical variables. I believe that combining stronger mathematical foundations with domain knowledge will allow me to better understand, evaluate, and optimize statistical and machine-learning models.

## Author

Joanna Zou  
[GitHub Profile](https://github.com/joannaazou)
