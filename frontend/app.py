import streamlit as st
import pickle
import pandas as pd

# load the trained model 
model = pickle.load(open("result.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
cols = pickle.load(open("cols.pkl", "rb"))

st.title("Heart Disease Prediction App")

st.markdown("""
This app predicts whether a person has heart disease or not based on various health parameters.
""")

st.markdown("""
So enter the following details:
""")

age = st.slider("Age", 10, 100, 30)
sex = st.selectbox("Sex", ["Male","Female"])
chest_pain_type = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.number_input("Resting Blood Pressure (in mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (in mg/dl)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["0", "1"])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
oldpeak = st.number_input("Oldpeak (ST depression induced by exercise)", 0.0, 10.0, 1.0)
st_slope = st.selectbox("Slope of the peak exercise ST segment", ["Up", "Flat", "Down"])

if st.button("Predict"):
    input_data = {
        "Age": age,
        "Sex_" + sex: 1,
        "ChestPainType_"+chest_pain_type: 1,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG_"+resting_ecg: 1,
        "MaxHR": max_hr,
        "ExerciseAngina_"+exercise_angina: 1,
        "Oldpeak": oldpeak,
        "ST_Slope_"+st_slope: 1
    }
    input_dataframe = pd.DataFrame([input_data])
    for col in cols:
        # print(col)
        if col not in input_dataframe.columns:
            input_dataframe[col] = 0
        
    input_df = input_dataframe[cols]
    # scale input if using  knn or other distance based model
    # scaled_input = scaler.transform(input_df)
    prediction = model.predict(input_df)[0]


    if prediction == 1:
        st.error("The person is likely to have heart disease.")
    else:
        st.success("The person is unlikely to have heart disease.")

