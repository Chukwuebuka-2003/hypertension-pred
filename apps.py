import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load your trained model
model = joblib.load('logistic_model.sav')

# Function to preprocess input data
def preprocess_data(input_data):
    # Assuming all necessary preprocessing steps here
    # Label encoding for categorical variables
    label_encoder = LabelEncoder()
    for column in input_data.columns:
        if input_data[column].dtype == 'object':
            input_data[column] = label_encoder.fit_transform(input_data[column])
    # Standardize numerical features
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    return input_data_scaled

# Streamlit app layout
st.set_page_config(page_title="Hypertension Prediction Tool", layout="wide")
st.title('Hypertension Prediction App')

# Input fields for user data
st.sidebar.title('User Input')
age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=30, help='Enter the age of the patient.')
heart_disease = st.sidebar.selectbox('Heart Disease', ['No', 'Yes'], help='Does the patient have a history of heart disease?')
ever_married = st.sidebar.selectbox('Ever Married?', ['No', 'Yes'], help='Marital status of the patient.')
work_type = st.sidebar.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'], help='Type of occupation of the patient.')
residence_type = st.sidebar.selectbox('Residence Type', ['Rural', 'Urban'], help='Residential living area of the patient.')
avg_glucose_level = st.sidebar.number_input('Average Glucose Level', min_value=0.0, max_value=300.0, value=100.0, help='Average glucose level in the blood.')
bmi = st.sidebar.number_input('Body Mass Index (BMI)', min_value=10.0, max_value=100.0, value=22.0, help='Body mass index of the patient.')
smoking_status = st.sidebar.selectbox('Smoking Status', ['never smoked', 'formerly smoked', 'smokes', 'unknown'], help='Smoking behavior of the patient.')

input_dict = {
    'age': [age],
    'heart_disease': [heart_disease],
    'ever_married': [ever_married],
    'work_type': [work_type],
    'Residence_type': [residence_type],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'smoking_status': [smoking_status]
}
input_df = pd.DataFrame(input_dict)

# Button to make prediction
if st.sidebar.button('Predict Hypertension'):
    input_df_processed = preprocess_data(input_df)  # Process the data
    prediction = model.predict(input_df_processed)
    result_text = 'The patient is likely to have hypertension.' if prediction[0] == 1 else 'The patient is unlikely to have hypertension.'
    st.success(result_text)

# Styling
st.markdown("""
<style>
.sidebar .widget-title {
    font-size: 16px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)
