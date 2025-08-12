import streamlit as st
import joblib

# Load model + vectorizer once
@st.cache_resource
def load_model():
    model = joblib.load("fake_job_model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    return model, vectorizer

model, vectorizer = load_model()

st.title("Fake Job Postings Classifier")

user_input = st.text_area("Enter the job description:")

if st.button("Predict"):
    if user_input.strip():
        X_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(X_tfidf)[0]
        proba = model.predict_proba(X_tfidf)[0][prediction]

        if prediction == 1:
            st.error(f"⚠️ Likely FAKE ({proba:.2%} confidence)")
        else:
            st.success(f"✅ Likely REAL ({proba:.2%} confidence)")
    else:
        st.warning("Please enter a job description.")




