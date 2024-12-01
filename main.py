import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("heart_disease_model.pkl")

# Define app title and description
st.title("Heart Disease Prediction App")
st.write("""
This application predicts the likelihood of heart disease based on user-provided input data.  
Fill in the required details below and click **Predict** to see the result.
""")

# Define the expected feature names based on training
expected_features = [
    "days_to_birth", "height", "weight", "tobacco_smoking_history",
    "frequency_of_alcohol_consumption", 
    "primary_pathology_age_at_initial_pathologic_diagnosis",
    "person_neoplasm_cancer_status_WITH TUMOR"  # Only this dummy variable is retained
]

# Collect user input for each feature
days_to_birth = st.number_input("Days to Birth (negative for past days):", min_value=-50000, max_value=0, step=1)
height = st.number_input("Height (cm):", min_value=50, max_value=250, step=1)
weight = st.number_input("Weight (kg):", min_value=10, max_value=200, step=1)
tobacco_smoking_history = st.selectbox("Tobacco Smoking History (1 for Yes, 0 for No):", [0, 1])
frequency_of_alcohol_consumption = st.number_input("Frequency of Alcohol Consumption (times per week):", min_value=0, max_value=50, step=1)
primary_pathology_age = st.number_input("Age at Initial Pathologic Diagnosis:", min_value=0, max_value=120, step=1)
cancer_status = st.selectbox("Cancer Status:", ["WITH TUMOR", "NO TUMOR"])

# Prepare the input data
input_data = {
    "days_to_birth": days_to_birth,
    "height": height,
    "weight": weight,
    "tobacco_smoking_history": tobacco_smoking_history,
    "frequency_of_alcohol_consumption": frequency_of_alcohol_consumption,
    "primary_pathology_age_at_initial_pathologic_diagnosis": primary_pathology_age,
    "person_neoplasm_cancer_status_WITH TUMOR": 1 if cancer_status == "WITH TUMOR" else 0
}

# Convert input to a DataFrame
input_df = pd.DataFrame([input_data])

# Make predictions when the button is clicked
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.success("The model predicts a high likelihood of heart disease.")
    else:
        st.success("The model predicts a low likelihood of heart disease.")
