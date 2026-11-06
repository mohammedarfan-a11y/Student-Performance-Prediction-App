import streamlit as st
import pandas as pd
import joblib

# Set page configuration with custom logo title
st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓")

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

# User Inputs in Sidebar (Changed to manual number inputs)
st.sidebar.header("Student Habit Inputs")

study_hours = st.sidebar.number_input("Study Hours / Day", min_value=0.0, max_value=24.0, value=5.0, step=0.1)
social_media = st.sidebar.number_input("Social Media Hours / Day", min_value=0.0, max_value=24.0, value=2.0, step=0.1)
netflix = st.sidebar.number_input("Netflix Hours / Day", min_value=0.0, max_value=24.0, value=1.5, step=0.1)
attendance = st.sidebar.number_input("Attendance Percentage (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
sleep = st.sidebar.number_input("Sleep Hours / Day", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
exercise = st.sidebar.number_input("Exercise Days / Week", min_value=0, max_value=7, value=3, step=1)
mental_health = st.sidebar.number_input("Mental Health Rating (1-10)", min_value=1, max_value=10, value=7, step=1)
age = st.sidebar.number_input("Age", min_value=15, max_value=60, value=20, step=1)

diet = st.sidebar.selectbox("Diet Quality", ["Poor", "Fair", "Good"])
parent_edu = st.sidebar.selectbox("Parental Education Level", ["High School", "Bachelor", "Master", "PhD"])
internet = st.sidebar.selectbox("Internet Quality", ["Poor", "Average", "Good"])
extracurricular = st.sidebar.selectbox("Extracurricular Participation", ["No", "Yes"])
gender = st.sidebar.selectbox("Gender", ["Female", "Male", "Other"])
part_time = st.sidebar.selectbox("Part-Time Job", ["No", "Yes"])

# Combine inputs into DataFrame matching feature columns
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
if st.button("Predict Result", type="primary"):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.subheader("Prediction Outcome")
    if prediction == 1:
        st.success(f"🎉 Result: **PASS** (Confidence: {probabilities[1]*100:.1f}%)")
    else:
        st.error(f"⚠️ Result: **FAIL** (Confidence: {probabilities[0]*100:.1f}%)")
