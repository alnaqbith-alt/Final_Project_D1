import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import date

# ---------------------------------------------------------
# Dubai Real Estate Investment Prediction - Decision Tree App
# ---------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "dubai_property_model.pkl"
COLUMNS_PATH = APP_DIR / "model_columns.pkl"

st.set_page_config(
    page_title="Dubai Real Estate Investment Prediction",
    page_icon="🏙️",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Load the trained preprocessing pipeline with the selected model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "dubai_property_model.pkl was not found. "
            "Run the training script first and keep the "
            "generated model file in the same folder as app.py."
        )

    if not COLUMNS_PATH.exists():
        raise FileNotFoundError(
            "model_columns.pkl was not found. Keep it in the same folder as app.py."
        )

    return joblib.load(MODEL_PATH), joblib.load(COLUMNS_PATH)


st.title("Dubai Real Estate Investment Prediction")
st.write(
    "Enter the property and transaction details below. The trained Decision Tree "
    "model will predict whether the property is a Premium Value investment (>= 1.36M AED)."
)

try:
    model, feature_columns = load_model()
except Exception as error:
    st.error(str(error))
    st.stop()

st.info(
    "The prediction is an estimate based on the information entered. "
    "Use it as supporting information when making business decisions."
)

with st.form("property_prediction_form"):
    st.subheader("Property and Transaction Details")
    col1, col2 = st.columns(2)

    with col1:
        property_area_sqm = st.number_input("Property Area (Sq.m)", min_value=1.0, value=100.0, step=10.0)
        rooms_bucket = st.number_input("Rooms Count (0 = Studio)", min_value=0, value=2, step=1)
        transaction_date = st.date_input("Transaction Date", value=date.today())
        registration_type = st.selectbox("Registration Type", ["Ready", "Off-Plan"])
        property_type = st.selectbox("Property Type", ["Unit", "Villa", "Land", "Building"])
        property_sub_type = st.text_input("Property Sub Type", value="Flat")

    with col2:
        property_usage = st.selectbox("Property Usage", ["Residential", "Commercial", "Other"])
        area_name = st.text_input("Area Name", value="Marsa Dubai")
        master_project = st.text_input("Master Project", value="Dubai Marina")
        procedure_name = st.text_input("Procedure Name", value="Sell")
        has_parking = st.selectbox("Has Parking", ["True", "False"])
        is_pre_registration = st.selectbox("Is Pre Registration", ["False", "True"])

    submitted = st.form_submit_button("Predict Investment Tier", use_container_width=True)

if submitted:
    input_values = {
        "Registration_Type": registration_type.strip() or np.nan,
        "Area_Name": area_name.strip() or np.nan,
        "Master_Project": master_project.strip() or np.nan,
        "Property_Type": property_type.strip() or np.nan,
        "Property_Sub_Type": property_sub_type.strip() or np.nan,
        "Property_Usage": property_usage.strip() or np.nan,
        "Rooms_Bucket": rooms_bucket,
        "Has_Parking": has_parking,
        "Property_Area_Sqm": property_area_sqm,
        "Procedure_Name": procedure_name.strip() or np.nan,
        "Is_Pre_Registration": is_pre_registration,
        "Trans_Year": transaction_date.year,
        "Trans_Month": transaction_date.month,
        "Trans_Quarter": (transaction_date.month - 1) // 3 + 1,
    }

    input_data = pd.DataFrame([input_values]).reindex(columns=feature_columns)

    # Match the training script's conversion of categorical values to strings.
    for column in input_data.select_dtypes(include=["object"]).columns:
        input_data[column] = input_data[column].apply(
            lambda value: str(value) if pd.notna(value) else np.nan
        )

    try:
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        classes = list(model.classes_)
        premium_index = classes.index(1)
        premium_probability = float(probabilities[premium_index])

        st.subheader("Prediction Result")
        if prediction == 1:
            st.success("Predicted Result: Premium Value Property")
        else:
            st.warning("Predicted Result: Standard Value Property")

        st.metric("Premium Property Probability", f"{premium_probability:.2%}")
        st.progress(float(np.clip(premium_probability, 0.0, 1.0)))
        st.caption(
            "Class 1 means Premium Value Property (>= 1.36M AED). Class 0 means Standard Value Property. "
            "The probability is the model's estimated likelihood of a premium valuation."
        )
    except Exception as error:
        st.error(
            "The prediction could not be completed. Check the entered values "
            f"and make sure the model file was generated by the updated script. Details: {error}"
        )

st.divider()
st.caption("Dubai Real Estate Investment Prediction | Decision Tree Model")