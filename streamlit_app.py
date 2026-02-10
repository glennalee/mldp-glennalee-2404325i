# streamlit_app.py
# Loan Repayment Risk Assessment Tool
# LDA-based early-stage screening application for loan applicants and bank staff

import streamlit as st
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Loan Repayment Risk Assessment",
    page_icon="💳",
    layout="centered"
)

# =========================================================
# MODEL & FEATURE CONFIGURATION
# =========================================================
# Path to the trained LDA model
MODEL_PATH = "loan_approval_model - Copy.pkl"

# Probability threshold above which an applicant is flagged as high-risk
# Set conservatively to minimize false negatives (missed defaulters)
THRESHOLD = 0.40

# All features expected by the model in the correct order
FEATURE_ORDER = [
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length',
    'credit_score',
    'person_gender_male',
    'person_education_Bachelor',
    'person_education_Doctorate',
    'person_education_High School',
    'person_education_Master',
    'person_home_ownership_OTHER',
    'person_home_ownership_OWN',
    'person_home_ownership_RENT',
    'loan_intent_EDUCATION',
    'loan_intent_HOMEIMPROVEMENT',
    'loan_intent_MEDICAL',
    'loan_intent_PERSONAL',
    'loan_intent_VENTURE',
    'previous_loan_defaults_on_file_Yes'
]

# Maps user-facing categorical values to one-hot encoded feature names
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

# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource
def load_model(path):
    """Load pre-trained LDA model from disk."""
    return joblib.load(path)

model = load_model(MODEL_PATH)

# =========================================================
# DATA ENCODING FUNCTION
# =========================================================
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
    """
    Convert user input into model-ready feature vector.
    
    Handles one-hot encoding for categorical variables and creates
    a DataFrame with features in the correct order expected by the model.
    """
    # Initialize all features to 0
    row = {col: 0 for col in FEATURE_ORDER}

    # Set numeric features
    row["loan_amnt"] = loan_amnt
    row["loan_int_rate"] = loan_int_rate
    row["loan_percent_income"] = loan_percent_income
    row["cb_person_cred_hist_length"] = credit_hist_len
    row["credit_score"] = credit_score

    # Encode gender (1 for male, 0 for female)
    row["person_gender_male"] = 1 if gender == "male" else 0

    # One-hot encode education level
    if education in EDU_MAP:
        row[EDU_MAP[education]] = 1

    # One-hot encode home ownership
    if home in HOME_MAP:
        row[HOME_MAP[home]] = 1

    # One-hot encode loan purpose
    if intent in INTENT_MAP:
        row[INTENT_MAP[intent]] = 1

    # Encode previous defaults (1 for Yes, 0 for No)
    row["previous_loan_defaults_on_file_Yes"] = 1 if prev_default == "Yes" else 0

    return pd.DataFrame([[row[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)

# =========================================================
# PAGE HEADER & INSTRUCTIONS
# =========================================================
st.title("Loan Repayment Risk Assessment")
st.caption(
    "Early-stage decision support tool designed to minimise missed high-risk applicants."
)

# =========================================================
# SIDEBAR CONFIGURATION
# =========================================================
st.sidebar.header("Screening Mode")

# Allow user to select between applicant and staff perspectives
perspective = st.sidebar.radio(
    "User perspective",
    ["Applicant (Self-Check)", "Bank Staff (Screening View)"]
)

# Display context-appropriate guidance
if perspective == "Applicant (Self-Check)":
    st.info(
        "Estimate your likelihood of passing early loan eligibility screening. "
        "This self-check is advisory and does not represent a final decision."
    )
else:
    st.info(
        "Early-stage risk screening view for bank staff. "
        "The model is intentionally conservative to minimise missed high-risk applicants."
    )

st.sidebar.markdown("---")
st.sidebar.caption(f"Model: LDA")
st.sidebar.caption(f"Decision threshold: {THRESHOLD:.0%}")

# =========================================================
# INPUT FORM
# =========================================================
st.markdown("### Applicant Financial Profile")

with st.form("loan_form"):

    # Financial metrics (left column)
    c1, c2 = st.columns(2)

    with c1:
        loan_amnt = st.number_input(
            "Loan amount",
            min_value=500.0,
            max_value=35000.0,
            value=8000.0,
            step=500.0
        )

        loan_int_rate = st.number_input(
            "Interest rate (%)",
            min_value=5.0,
            max_value=20.0,
            value=12.0,
            step=0.1
        )

        loan_percent_income = st.number_input(
            "Loan as % of income",
            min_value=0.0,
            max_value=0.66,
            value=0.20,
            step=0.01
        )

    # Credit profile (right column)
    with c2:
        credit_hist_len = st.number_input(
            "Credit history length (years)",
            min_value=0.0,
            max_value=30.0,
            value=6.0,
            step=1.0
        )

        credit_score = st.number_input(
            "Credit score",
            min_value=390,
            max_value=850,
            value=650,
            step=1
        )

    # Demographic and loan details
    st.markdown("### Applicant Background")

    c3, c4 = st.columns(2)

    with c3:
        gender = st.selectbox("Gender", ["female", "male"])
        education = st.selectbox(
            "Education level",
            ["High School", "Bachelor", "Master", "Doctorate"]
        )
        home = st.selectbox(
            "Home ownership",
            ["MORTGAGE", "RENT", "OWN", "OTHER"]
        )

    with c4:
        intent = st.selectbox(
            "Loan purpose",
            ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"]
        )
        prev_default = st.selectbox(
            "Previous loan defaults on record",
            ["No", "Yes"]
        )

    submitted = st.form_submit_button("Assess Risk")

# =========================================================
# RISK PREDICTION & RESULTS DISPLAY
# =========================================================
if submitted:
    # Encode user input for model prediction
    X_new = build_encoded_row(
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
    )

    # Get probability of default from model
    prob_default = float(model.predict_proba(X_new)[0][1])
    high_risk = prob_default >= THRESHOLD

    st.markdown("---")
    st.markdown("### Risk Assessment Result")

    # Display probability metric
    st.metric(
        label="Estimated Probability of Repayment Failure",
        value=f"{prob_default:.0%}",
        delta="High risk" if high_risk else "Low risk"
    )

    # Visual risk indicator
    st.progress(min(prob_default, 1.0))
    st.caption(f"Screening threshold: {THRESHOLD:.0%}")

    # Display perspective-specific recommendation
    if perspective == "Applicant (Self-Check)":
        if high_risk:
            st.warning(
                f"""
Your profile is **at or above the bank's early screening threshold**.

This does not guarantee rejection, but additional review is likely.

**Ways to improve eligibility:**
- Reduce loan amount
- Improve credit score
- Increase income stability
"""
            )
        else:
            st.success(
                """
Your profile is **below the early screening threshold**.

You are more likely to pass initial eligibility checks,
subject to standard verification.
"""
            )
    else:
        # Bank staff view with actionable recommendations
        if high_risk:
            st.error(
                f"""
HIGH RISK — SCREENING FLAG

Estimated default probability: **{prob_default:.0%}**

Model bias: prioritises avoiding missed defaulters.
Recommended action: enhanced review or rejection.
"""
            )
        else:
            st.success(
                f"""
LOW RISK — ELIGIBLE FOR APPROVAL

Estimated default probability: **{prob_default:.0%}**

Recommended action: proceed with standard verification.
"""
            )

    # Expandable sections for transparency and audit trail
    with st.expander("How to interpret this score"):
        st.markdown("""
- The score represents the model's estimated probability of repayment failure  
- Scores above the threshold are flagged conservatively  
- This tool supports **early-stage screening**, not final approval  
- Human judgment is required for final decisions
""")

    with st.expander("Model inputs (encoded)"):
        st.caption("Shown for transparency and audit purposes.")
        st.dataframe(X_new)
