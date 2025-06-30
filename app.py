import streamlit as st
import joblib

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("model/phishing_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer

st.set_page_config(page_title="Phishing Detector", layout="centered")
st.title("AI-Powered Phishing Email Detector")
st.markdown("Paste the email content below to check if it's phishing:")

# Load model
model, vectorizer = load_model()

# User input
email_input = st.text_area("📩 Email Content:", height=200)

if st.button("Check for Phishing"):
    if not email_input.strip():
        st.warning("Please paste an email to analyze.")
    else:
        # Clean + vectorize (no need for clean_text; vectorizer handles it)
        email_vector = vectorizer.transform([email_input])
        prediction = model.predict(email_vector)[0]
        proba = model.predict_proba(email_vector)[0][prediction]

        if prediction == 1:
            st.error(f"Phishing Detected\nConfidence: `{proba:.2%}`")
        else:
            st.success(f"Legitimate Email\nConfidence: `{proba:.2%}`")
