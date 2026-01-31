import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="sp1505/Predictive-Maintenace-Model", filename="best_predictive_maintenace_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Prediction Maintenance App")
st.write("""
This application predicts the likelihood of a machine failing based on its operational parameters.
Please enter the sensor and configuration data below to get a prediction.
""")

# User input
rpm = st.number_input("Engine rpm (K)")
lub_oil_pressure = st.number_input("Lub oil pressure")
fuel_pressure = st.number_input("Fuel pressure")
coolant_pressure = st.number_input("Coolant pressure")
lub_oil_temp = st.number_input("Lub oil temp")
coolant_temp = st.number_input("Coolant temp")


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Engine rpm': rpm,
    'Lub oil pressure': lub_oil_pressure,
    'Fuel pressure': fuel_pressure,
    'Coolant pressure': coolant_pressure,
    'lub oil temp': lub_oil_temp,
    'Coolant temp': coolant_temp
}])

if st.button("Predict Failure"):
    prediction = model.predict(input_data)[0]
    result = "Engine Failure" if prediction == 1 else "No Failure"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
