import streamlit as st
import pickle

# Load model and vectorizer
with open("fake_job_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.set_page_config(page_title="Fake Job Detector", page_icon="🕵️")

# Title & Instructions
st.title("🕵️ Fake Job Detector")
st.write("Paste a job description below and click **Predict** to see if it's real or fake.")

# Text area for user input
user_input = st.text_area("Job description:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter a job description.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        if prediction == 1:
            st.error("🚨 Likely FAKE job posting.")
        else:
            st.success("✅ Likely REAL job posting.")

st.write("---")
st.caption("Model trained on Kaggle Fake Job Postings Dataset")
