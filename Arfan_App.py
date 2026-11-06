import streamlit as st
import pandas as pd
import joblib

# Set page title
st.title("🎓 Student Performance Prediction App")
st.write("Enter student lifestyle habits to predict whether they will pass or fail.")

# Load the saved model pipeline
@st.cache_resource
def load_model():
    return joblib.load('student_performance_model.pkl')
model = load_model()

# User Inputs in Sidebar
st.sidebar.header("Student Habit Inputs")

study_hours = st.sidebar.slider("Study Hours / Day", 0.0, 10.0, 5.0, 0.1)
social_media = st.sidebar.slider("Social Media Hours / Day", 0.0, 10.0, 2.0, 0.1)
netflix = st.sidebar.slider("Netflix Hours / Day", 0.0, 10.0, 1.5, 0.1)
attendance = st.sidebar.slider("Attendance Percentage", 0.0, 100.0, 85.0, 1.0)
sleep = st.sidebar.slider("Sleep Hours / Day", 0.0, 12.0, 7.0, 0.5)
exercise = st.sidebar.slider("Exercise Days / Week", 0, 7, 3)
mental_health = st.sidebar.slider("Mental Health Rating (1-10)", 1, 10, 7)
age = st.sidebar.number_input("Age", 17, 30, 20)

diet = st.sidebar.selectbox("Diet Quality", ["Poor", "Fair", "Good"])
parent_edu = st.sidebar.selectbox("Parental Education Level", ["High School", "Bachelor", "Master", "PhD"])
internet = st.sidebar.selectbox("Internet Quality", ["Poor", "Average", "Good"])
extracurricular = st.sidebar.selectbox("Extracurricular Participation", ["No", "Yes"])
gender = st.sidebar.selectbox("Gender", ["Female", "Male", "Other"])
part_time = st.sidebar.selectbox("Part-Time Job", ["No", "Yes"])

# Combine inputs into DataFrame matching original features
input_data = pd.DataFrame([{
    'age': age,
    'gender': gender,
    'study_hours_per_day': study_hours,
    'social_media_hours': social_media,
    'netflix_hours': netflix,
    'part_time_job': part_time,
    'attendance_percentage': attendance,
    'sleep_hours': sleep,
    'diet_quality': diet,
    'exercise_frequency': exercise,
    'parental_education_level': parent_edu,
    'internet_quality': internet,
    'mental_health_rating': mental_health,
    'extracurricular_participation': extracurricular
}])

# Predict button
if st.button("Predict Result"):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.subheader("Prediction Outcome")
    if prediction == 1:
        st.success(f"🎉 Result: **PASS** (Confidence: {probabilities[1]*100:.1f}%)")
    else:
        st.error(f"⚠️ Result: **FAIL** (Confidence: {probabilities[0]*100:.1f}%)")
