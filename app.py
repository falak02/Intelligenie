import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Intelligent NLP-Based Text Assistant",
    page_icon="🤖"
)

st.title("Intelligent NLP-Based Text Assistant")
st.write(
    "Analyze text sentiment using a pretrained Hugging Face Transformer model."
)

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_model()

text = st.text_area(
    "Enter a sentence or paragraph:",
    height=150
)

if st.button("Analyze Sentiment"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        result = classifier(text)[0]

        label = result["label"]
        confidence = result["score"] * 100

        st.success(f"Sentiment: {label}")
        st.info(f"Confidence: {confidence:.2f}%")