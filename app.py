import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("fake_job_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Streamlit App
st.title("Fake Job Postings Classifier")

user_input = st.text_area("Enter the job description:")

if st.button("Predict"):
    if user_input.strip():
        # Transform input
        user_tfidf = vectorizer.transform([user_input])
        # Prediction
        prediction = model.predict(user_tfidf)[0]
        # Output
        if prediction == 1:
            st.error("⚠️ This job posting is likely FAKE.")
        else:
            st.success("✅ This job posting seems REAL.")
    else:
        st.warning("Please enter a job description before predicting.")


