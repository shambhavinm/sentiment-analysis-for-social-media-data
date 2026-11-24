import streamlit as st
import joblib

# Load model
model = joblib.load("models/sentiment_model.pkl")

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Social Media Sentiment Analyzer")

st.write(
    "Analyze social media comments and identify user sentiment."
)

text = st.text_area(
    "Enter Tweet / Social Media Comment"
)

if st.button("Analyze Sentiment"):

    prediction = model.predict([text])[0]

    if prediction == 1:
        st.success("✅ Positive Sentiment")

    elif prediction == -1:
        st.error("❌ Negative Sentiment")

    else:
        st.warning("⚠️ Neutral Sentiment")