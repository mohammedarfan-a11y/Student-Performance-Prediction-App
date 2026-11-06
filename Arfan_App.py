import streamlit as st
import pandas as pd
import joblib

# Set page configuration with custom logo title
st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")

# Display Logo image in the sidebar top corner
st.sidebar.image("https://i.postimg.cc/vDFNncsc/Arfan-2.jpg", use_container_width=True)

# Main Title & Header
st.title("🎓 Student Performance Prediction App")
st.write("Enter student lifestyle habits to predict whether they will pass or fail.")

# Load the saved model pipeline directly from root directory
@st.cache_resource
def load_model():
    return joblib.load('student_performance_model.pkl')

model = load_model()

st.divider()
st.header("📋 Student Habit Inputs")

# Create two columns for clean input organization
col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input("Study Hours / Day", min_value=0.0, max_value=24.0, value=5.0, step=0.1)
    social_media = st.number_input("Social Media Hours / Day", min_value=0.0, max_value=24.0, value=2.0, step=0.1)
    netflix = st.number_input("Netflix Hours / Day", min_value=0.0, max_value=24.0, value=1.5, step=0.1)
    attendance = st.number_input("Attendance Percentage (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
    sleep = st.number_input("Sleep Hours / Day", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
    exercise = st.number_input("Exercise Days / Week", min_value=0, max_value=7, value=3, step=1)
    mental_health = st.number_input("Mental Health Rating (1-10)", min_value=1, max_value=10, value=7, step=1)

with col2:
    age = st.number_input("Age", min_value=15, max_value=60, value=20, step=1)
    diet = st.selectbox("Diet Quality", ["Poor", "Fair", "Good"])
    parent_edu = st.selectbox("Parental Education Level", ["High School", "Bachelor", "Master", "PhD"])
    internet = st.selectbox("Internet Quality", ["Poor", "Average", "Good"])
    extracurricular = st.selectbox("Extracurricular Participation", ["No", "Yes"])
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    part_time = st.selectbox("Part-Time Job", ["No", "Yes"])

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

st.divider()

# Predict button and outcome below inputs
if st.button("Predict Result", type="primary", use_container_width=True):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.subheader("📊 Prediction Outcome")
    if prediction == 1:
        st.success(f"🎉 Result: **PASS** (Confidence: {probabilities[1]*100:.1f}%)")
    else:
        st.error(f"⚠️ Result: **FAIL** (Confidence: {probabilities[0]*100:.1f}%)")
