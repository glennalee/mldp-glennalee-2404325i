# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib

# =============================
# Page Config
# =============================
st.set_page_config(
    page_title="Loan Risk Checker (LDA)",
    page_icon="💳",
    layout="centered"
)

# =============================
# Constants
# =============================
THRESHOLD = 0.40
MODEL_PATH = "loan_approval_model.pkl"

# MUST match training encoded feature order
FEATURE_ORDER = [
    'loan_amnt', 'loan_int_rate', 'loan_percent_income',
    'cb_person_cred_hist_length', 'credit_score',
    'person_gender_male',
    'person_education_Bachelor', 'person_education_Doctorate',
    'person_education_High School', 'person_education_Master',
    'person_home_ownership_OTHER', 'person_home_ownership_OWN',
    'person_home_ownership_RENT',
    'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT',
    'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE',
    'previous_loan_defaults_on_file_Yes'
]

# One-hot mappings
EDU_MAP = {
    "High School": "person_education_High School",
    "Bachelor": "person_education_Bachelor",
    "Master": "person_education_Master",
    "Doctorate": "person_education_Doctorate"
}

HOME_MAP = {
    "RENT": "person_home_ownership_RENT",
    "OWN": "person_home_ownership_OWN",
    "OTHER": "person_home_ownership_OTHER"
}

INTENT_MAP = {
    "EDUCATION": "loan_intent_EDUCATION",
    "HOMEIMPROVEMENT": "loan_intent_HOMEIMPROVEMENT",
    "MEDICAL": "loan_intent_MEDICAL",
    "PERSONAL": "loan_intent_PERSONAL",
    "VENTURE": "loan_intent_VENTURE"
}

# =============================
# Load Pipeline Model
# =============================
@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(MODEL_PATH)

# =============================
# Helper: Build Encoded Row
# =============================
def build_encoded_row(
    loan_amnt,
    loan_int_rate,
    loan_percent_income,
    credit_hist_len,
    credit_score,
    gender,
    education,
    home,
    intent,
    prev_default
):
    row = {col: 0 for col in FEATURE_ORDER}

    # Numeric features
    row["loan_amnt"] = loan_amnt
    row["loan_int_rate"] = loan_int_rate
    row["loan_percent_income"] = loan_percent_income
    row["cb_person_cred_hist_length"] = credit_hist_len
    row["credit_score"] = credit_score

    # Categorical (one-hot)
    row["person_gender_male"] = 1 if gender == "male" else 0

    if education in EDU_MAP:
        row[EDU_MAP[education]] = 1

    if home in HOME_MAP:
        row[HOME_MAP[home]] = 1

    if intent in INTENT_MAP:
        row[INTENT_MAP[intent]] = 1

    row["previous_loan_defaults_on_file_Yes"] = 1 if prev_default == "Yes" else 0

    return pd.DataFrame([[row[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)

# =============================
# UI
# =============================
st.title("💳 Loan Risk Checker (LDA)")
st.caption(
    "Final model: **LDA pipeline (scaled + encoded)**  \n"
    f"Decision threshold = **{THRESHOLD:.2f}** to prioritise recall for risky borrowers."
)

with st.form("loan_form"):
    st.subheader("Loan & Credit Inputs")

    c1, c2 = st.columns(2)

    with c1:
        loan_amnt = st.number_input("Loan amount", 500.0, 35000.0, 8000.0, step=500.0)
        loan_int_rate = st.number_input("Interest rate (%)", 5.0, 20.0, 12.0, step=0.1)
        loan_percent_income = st.number_input("Loan percent of income", 0.0, 0.66, 0.20, step=0.01)

    with c2:
        credit_hist_len = st.number_input("Credit history length (years)", 0.0, 30.0, 6.0, step=1.0)
        credit_score = st.number_input("Credit score", 390, 850, 650, step=1)

    st.subheader("Categorical Inputs")

    c3, c4 = st.columns(2)

    with c3:
        gender = st.selectbox("Gender", ["female", "male"])
        education = st.selectbox("Education", ["High School", "Bachelor", "Master", "Doctorate"])
        home = st.selectbox("Home ownership", ["RENT", "OWN", "OTHER"])

    with c4:
        intent = st.selectbox("Loan intent", ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"])
        prev_default = st.selectbox("Previous loan defaults on file", ["No", "Yes"])

    submitted = st.form_submit_button("Predict risk")

# =============================
# Prediction
# =============================
if submitted:
    X_new = build_encoded_row(
        loan_amnt, loan_int_rate, loan_percent_income,
        credit_hist_len, credit_score,
        gender, education, home, intent, prev_default
    )

    # PIPELINE handles scaling internally
    prob_default = float(model.predict_proba(X_new)[0][1])
    prediction = int(prob_default >= THRESHOLD)

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠️ **HIGH RISK (Likely Default)**\n\n"
            f"Probability of default: **{prob_default:.2f}**"
        )
    else:
        st.success(
            f"✅ **LOW RISK (Likely Non-Default)**\n\n"
            f"Probability of default: **{prob_default:.2f}**"
        )

    with st.expander("Show encoded features sent to model (debug)"):
        st.dataframe(X_new)
