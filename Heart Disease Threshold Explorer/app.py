import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Heart Disease Threshold Explorer",
    page_icon="❤️",
    layout="wide"
)

DEFAULT_TP_UTILITY = 10
DEFAULT_FP_UTILITY = -2
DEFAULT_TN_UTILITY = 5
DEFAULT_FN_UTILITY = -15

YOUDEN_THRESHOLD = 0.4897795
YOUDEN_SENSITIVITY = 0.8148148
YOUDEN_SPECIFICITY = 0.96875

MODEL_PREDICTORS = {
    "Sex": "sex",
    "Chest Pain Type": "cp",
    "Resting ECG Result": "restecg",
    "Exercise-Induced Angina": "exang",
    "ST Depression (Oldpeak)": "oldpeak",
    "ST-Segment Slope": "slope",
    "Number of Major Vessels": "ca",
    "Thalassemia Test Result": "thal",
}

PREDICTOR_AXIS_LABELS = {
    "sex": "Sex (0 = Female, 1 = Male)",
    "cp": "Chest Pain Type",
    "restecg": "Resting ECG Result",
    "exang": "Exercise-Induced Angina (0 = No, 1 = Yes)",
    "oldpeak": "ST Depression Induced by Exercise",
    "slope": "ST-Segment Slope",
    "ca": "Number of Major Vessels",
    "thal": "Thalassemia Test Result",
}

CATEGORY_LABELS = {
    "sex": {
        0: "Female",
        1: "Male",
    },
    "cp": {
        1: "Typical Angina",
        2: "Atypical Angina",
        3: "Non-Anginal Pain",
        4: "Asymptomatic",
    },
    "restecg": {
        0: "Normal",
        1: "ST-T Wave Abnormality",
        2: "Left Ventricular Hypertrophy",
    },
    "exang": {
        0: "No",
        1: "Yes",
    },
    "slope": {
        1: "Upsloping",
        2: "Flat",
        3: "Downsloping",
    },
    "ca": {
        0: "0 Vessels",
        1: "1 Vessel",
        2: "2 Vessels",
        3: "3 Vessels",
    },
    "thal": {
        3: "Normal",
        6: "Fixed Defect",
        7: "Reversible Defect",
    },
}


@st.cache_data
def load_predictions():
    df = pd.read_csv("test_predictions.csv")

    required_columns = {"actual", "probability"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "test_predictions.csv must contain the columns "
            "'actual' and 'probability'."
        )

    df = df.copy()
    df["actual"] = df["actual"].astype(int)
    df["probability"] = df["probability"].astype(float)

    if not df["actual"].isin([0, 1]).all():
        raise ValueError(
            "The 'actual' column must contain only 0 "
            "(No Heart Disease) and 1 (Heart Disease)."
        )

    if not df["probability"].between(0, 1).all():
        raise ValueError(
            "The 'probability' column must contain values between 0 and 1."
        )

    return df

@st.cache_data
def load_explorer_data():
    explorer_df = pd.read_csv("heart_for_explorer.csv")

    if "actual" not in explorer_df.columns:
        raise ValueError(
            "heart_for_explorer.csv must contain an 'actual' column."
        )

    explorer_df = explorer_df.copy()
    explorer_df["actual"] = explorer_df["actual"].astype(int)

    if not explorer_df["actual"].isin([0, 1]).all():
        raise ValueError(
            "The 'actual' column in heart_for_explorer.csv must contain "
            "only 0 (No Heart Disease) and 1 (Heart Disease)."
        )

    return explorer_df

def confusion_counts(df, threshold):
    predicted_positive = df["probability"] >= threshold
    actual_positive = df["actual"] == 1

    tp = int((predicted_positive & actual_positive).sum())
    fp = int((predicted_positive & ~actual_positive).sum())
    tn = int((~predicted_positive & ~actual_positive).sum())
    fn = int((~predicted_positive & actual_positive).sum())

    return tp, fp, tn, fn


def evaluate_threshold(df, threshold, tp_value, fp_value, tn_value, fn_value):
    tp, fp, tn, fn = confusion_counts(df, threshold)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    youden_index = sensitivity + specificity - 1

    total_utility = (
        tp * tp_value
        + fp * fp_value
        + tn * tn_value
        + fn * fn_value
    )

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Youden Index": youden_index,
        "Total Utility": total_utility,
    }


def create_threshold_table(df, tp_value, fp_value, tn_value, fn_value):
    thresholds = np.sort(df["probability"].unique())
    rows = []

    for threshold in thresholds:
        metrics = evaluate_threshold(
            df=df,
            threshold=float(threshold),
            tp_value=tp_value,
            fp_value=fp_value,
            tn_value=tn_value,
            fn_value=fn_value,
        )

        rows.append({
            "Threshold": float(threshold),
            **metrics,
        })

    return pd.DataFrame(rows)


def get_best_thresholds(df, tp_value, fp_value, tn_value, fn_value):
    threshold_table = create_threshold_table(
        df=df,
        tp_value=tp_value,
        fp_value=fp_value,
        tn_value=tn_value,
        fn_value=fn_value,
    )

    best_youden = threshold_table.loc[
        threshold_table["Youden Index"].idxmax()
    ]

    best_utility = threshold_table.loc[
        threshold_table["Total Utility"].idxmax()
    ]

    return best_youden, best_utility


def show_predictor_chart(df, predictor_label, predictor_column):
    if predictor_column not in df.columns:
        st.error(
            f"The column '{predictor_column}' is missing from "
            "test_predictions.csv. Please export the full test-set data "
            "from R, including the predictor columns."
        )
        return

    axis_label = PREDICTOR_AXIS_LABELS.get(
        predictor_column,
        predictor_label
    )

    chart_df = df.copy()

    chart_df["Heart Disease Status"] = chart_df["actual"].map({
        0: "No Heart Disease",
        1: "Heart Disease",
    })

    if predictor_column in CATEGORY_LABELS:
        chart_df["Display Category"] = chart_df[predictor_column].map(
            CATEGORY_LABELS[predictor_column]
        ).fillna(chart_df[predictor_column].astype(str))
    else:
        chart_df["Display Category"] = chart_df[predictor_column].astype(str)

    colors = {
        "No Heart Disease": "#4C78A8",
        "Heart Disease": "#E45756",
    }

    if predictor_column == "oldpeak":
        fig = px.box(
            chart_df,
            x="Heart Disease Status",
            y=predictor_column,
            color="Heart Disease Status",
            points="all",
            title=f"{predictor_label} by Heart Disease Status",
            labels={
                predictor_column: axis_label,
                "Heart Disease Status": "Observed Heart Disease Status",
            },
            color_discrete_map=colors,
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="Observed Heart Disease Status",
            yaxis_title=axis_label,
        )

    else:
        count_df = (
            chart_df
            .groupby(["Heart Disease Status", "Display Category"])
            .size()
            .reset_index(name="Patient Count")
        )

        fig = px.bar(
            count_df,
            x="Heart Disease Status",
            y="Patient Count",
            color="Display Category",
            barmode="group",
            title=f"{predictor_label} Distribution by Heart Disease Status",
            labels={
                "Heart Disease Status": "Observed Heart Disease Status",
                "Display Category": axis_label,
                "Patient Count": "Number of Patients",
            },
            color_discrete_sequence=px.colors.qualitative.Set2,
        )

        fig.update_layout(
            xaxis_title="Observed Heart Disease Status",
            yaxis_title="Number of Patients",
            legend_title=axis_label,
        )

    st.plotly_chart(fig, use_container_width=True)


df = load_predictions()
explorer_df = load_explorer_data()

st.title("Heart Disease Threshold Explorer")

st.markdown(
    """
This application explores how clinical and biomedical measurements can be used
to predict the binary outcome of heart disease. It presents test-set predictions
from an AIC-selected stepwise logistic regression model trained on a randomly
selected 80% of the UCI Heart Disease dataset. The remaining 20% of observations
were used as an independent test set.

Users can select a classification threshold to simulate a diagnostic decision.
At the selected threshold, the application reports sensitivity, specificity,
confusion-matrix counts, and total clinical utility. Users can also define the
relative utility or cost of true-positive, false-positive, true-negative, and
false-negative diagnostic outcomes.

Developer: Joanna Zou | zouyijunjoanna@outlook.com
"""
)

st.markdown(
    "Dataset source: "
    "[UCI Heart Disease Dataset]"
    "(https://archive.ics.uci.edu/dataset/45/heart+disease)"
)

st.warning(
    "Educational and analytical use only. This application is not a clinical "
    "diagnostic tool and must not be used for real patient-care decisions."
)

st.divider()

left_panel, right_panel = st.columns([1.65, 1])

with left_panel:
    st.subheader("Model")

    st.markdown(
        """
**Model:** AIC-Selected Stepwise Logistic Regression  
**Training/Test Split:** Random 80% training set / 20% test set  
**Repeated Cross-Validation AUC:** 0.899  
**Test-Set AUC:** 0.904  
**Model AIC:** 181.918
"""
    )

    with st.expander("Model Equation and Estimated Coefficients", expanded=True):
        st.markdown(
            "The stepwise logistic regression model estimates the probability "
            "of heart disease as follows:"
        )

        st.latex(
            r"""
\begin{aligned}
\eta ={}& -5.866 + 0.915(\mathrm{sex1}) \\
&+ 1.579(\mathrm{cp2}) + 0.070(\mathrm{cp3}) \\
&+ 2.392(\mathrm{cp4}) - 0.098(\mathrm{restecg1}) \\
&+ 0.906(\mathrm{restecg2}) + 0.703(\mathrm{exang1}) \\
&+ 0.634(\mathrm{oldpeak}) + 1.427(\mathrm{slope2}) \\
&+ 0.247(\mathrm{slope3}) + 2.381(\mathrm{ca1}) \\
&+ 3.007(\mathrm{ca2}) + 2.305(\mathrm{ca3}) \\
&- 0.076(\mathrm{thal6}) + 1.565(\mathrm{thal7})
\end{aligned}
"""
        )

        st.latex(
            r"""
P(\mathrm{Heart\ Disease}=\mathrm{Yes})
=
\frac{1}{1+\exp(-\eta)}
"""
        )

        st.caption(
            "All estimated beta coefficients are rounded to three decimal places."
        )

        st.caption(
            "Reference categories: sex = 0, cp = 1, restecg = 0, "
            "exang = 0, slope = 1, ca = 0, and thal = 3."
        )

        st.info(
            "For categorical predictors, each coefficient represents the "
            "change in log-odds relative to its reference category, holding "
            "all other model predictors constant."
        )

with right_panel:
    st.subheader("Predictor Explorer")

    selected_label = st.selectbox(
        "Select a model predictor",
        options=list(MODEL_PREDICTORS.keys()),
    )

    selected_column = MODEL_PREDICTORS[selected_label]

    st.caption(
        "Based on the full cleaned dataset. Select a predictor to view its "
        "observed distribution by heart disease status."
    )

    show_predictor_chart(
        df=explorer_df,
        predictor_label=selected_label,
        predictor_column=selected_column,
    )

st.divider()

st.subheader("Classification Threshold")

threshold = st.slider(
    "Select a probability threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.38,
    step=0.01,
    help=(
        "An observation is classified as heart-disease positive when its "
        "predicted probability is greater than or equal to this threshold."
    ),
)

st.caption(
    f"At a threshold of {threshold:.2f}, observations with predicted "
    f"probabilities greater than or equal to {threshold:.2f} are classified "
    f"as heart-disease positive."
)

st.divider()

st.subheader("Clinical Utility Weights")

st.write(
    "Specify the relative utility or cost assigned to each diagnostic outcome."
)

utility_col1, utility_col2, utility_col3, utility_col4 = st.columns(4)

with utility_col1:
    utility_tp = st.number_input(
        "True Positive (TP)",
        value=DEFAULT_TP_UTILITY,
        step=1,
    )

with utility_col2:
    utility_fp = st.number_input(
        "False Positive (FP)",
        value=DEFAULT_FP_UTILITY,
        step=1,
    )

with utility_col3:
    utility_tn = st.number_input(
        "True Negative (TN)",
        value=DEFAULT_TN_UTILITY,
        step=1,
    )

with utility_col4:
    utility_fn = st.number_input(
        "False Negative (FN)",
        value=DEFAULT_FN_UTILITY,
        step=1,
    )

st.caption(
    "Total utility = (TP × TP utility) + (FP × FP utility) + "
    "(TN × TN utility) + (FN × FN utility)"
)

result = evaluate_threshold(
    df=df,
    threshold=threshold,
    tp_value=utility_tp,
    fp_value=utility_fp,
    tn_value=utility_tn,
    fn_value=utility_fn,
)

best_youden, best_utility = get_best_thresholds(
    df=df,
    tp_value=utility_tp,
    fp_value=utility_fp,
    tn_value=utility_tn,
    fn_value=utility_fn,
)

st.divider()

st.subheader("Optimal Thresholds")

best_col1, best_col2 = st.columns(2)

with best_col1:
    st.metric(
        "Best Threshold by Youden's J",
        f"{YOUDEN_THRESHOLD:.3f}",
        help=(
            "Youden's J = Sensitivity + Specificity − 1. "
            "This threshold was calculated in R using pROC::coords() "
            "with best.method = 'youden'."
        ),
    )

    st.caption(
        f"Sensitivity: {YOUDEN_SENSITIVITY:.1%} | "
        f"Specificity: {YOUDEN_SPECIFICITY:.1%} | "
        f"Youden's J: "
        f"{YOUDEN_SENSITIVITY + YOUDEN_SPECIFICITY - 1:.3f}"
    )

with best_col2:
    st.metric(
        "Best Threshold by Maximum Utility",
        f"{best_utility['Threshold']:.3f}",
        help=(
            "This threshold maximizes total utility using the utility weights "
            "currently specified above."
        ),
    )

    st.caption(
        f"Maximum total utility: {best_utility['Total Utility']:.0f} | "
        f"Sensitivity: {best_utility['Sensitivity']:.1%} | "
        f"Specificity: {best_utility['Specificity']:.1%}"
    )

st.divider()

st.subheader("Performance at the Selected Threshold")

metric_col1, metric_col2, metric_col3 = st.columns(3)

metric_col1.metric(
    "Sensitivity",
    f"{result['Sensitivity']:.1%}",
    help="Sensitivity = TP / (TP + FN)",
)

metric_col2.metric(
    "Specificity",
    f"{result['Specificity']:.1%}",
    help="Specificity = TN / (TN + FP)",
)

metric_col3.metric(
    "Total Utility",
    f"{result['Total Utility']:.0f}",
)

st.subheader("Confusion Matrix Counts")

count_col1, count_col2, count_col3, count_col4 = st.columns(4)

count_col1.metric("True Positive", result["TP"])
count_col2.metric("False Positive", result["FP"])
count_col3.metric("True Negative", result["TN"])
count_col4.metric("False Negative", result["FN"])

st.caption(
    f"Youden's J at the selected threshold: "
    f"{result['Youden Index']:.3f}"
)
