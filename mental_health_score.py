import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
st.set_page_config(page_title="Mental Health Score Estimator", layout="centered")
st.title("🧠 Mental Health Score Estimator")
st.markdown("""
Estimate your mental well-being score (0–100) based on your daily habits. This tool is for awareness only, not a professional diagnosis.
""")
np.random.seed(42)
n_samples = 1000
data = {
    'Sleep_Hours': np.random.uniform(4, 12, n_samples),
    'Screen_Time': np.random.uniform(0, 12, n_samples),
    'Physical_Activity': np.random.uniform(0, 5, n_samples),
    'Social_Interaction': np.random.uniform(0, 5, n_samples)
}
df = pd.DataFrame(data)
df['Mental_Wellbeing_Score'] = (
    10 * df['Sleep_Hours'] - 
    5 * df['Screen_Time'] + 
    8 * df['Physical_Activity'] + 
    7 * df['Social_Interaction'] + 
    np.random.normal(0, 5, n_samples)
).clip(0, 100) 
X = df[['Sleep_Hours', 'Screen_Time', 'Physical_Activity', 'Social_Interaction']]
y = df['Mental_Wellbeing_Score']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LinearRegression()
model.fit(X_scaled, y)
st.subheader("Enter Your Daily Habits")
with st.form("input_form"):
    sleep_hours = st.slider("Sleep Hours (per night)", 0.0, 12.0, 7.0, 0.5)
    screen_time = st.slider("Screen Time (hours per day)", 0.0, 24.0, 4.0, 0.5)
    physical_activity = st.slider("Physical Activity (hours per day)", 0.0, 5.0, 1.0, 0.1)
    social_interaction = st.slider("Social Interaction (hours per day)", 0.0, 5.0, 1.0, 0.1)
    submit_button = st.form_submit_button("Estimate Score")
if submit_button:
    input_data = np.array([[sleep_hours, screen_time, physical_activity, social_interaction]])
    input_scaled = scaler.transform(input_data)
    predicted_score = model.predict(input_scaled)[0]
    predicted_score = np.clip(predicted_score, 0, 100) 
    st.subheader("Your Estimated Mental Well-being Score")
    st.metric("Score", f"{predicted_score:.1f}/100")
    if predicted_score >= 80:
        st.success("High well-being! Your habits are likely supporting a positive mental state.")
    elif predicted_score >= 60:
        st.info("Moderate well-being. Try reducing screen time or increasing activity.")
    elif predicted_score >= 40:
        st.warning("Low well-being. Consider more sleep or social interaction.")
    else:
        st.error("Very low well-being. Please consult a professional for support.")
    
    st.markdown("*Note: This is an estimate based on a simple model. For accurate assessment, consult a mental health professional.*")
st.markdown("---")
st.markdown("Built with Streamlit | For health awareness only")