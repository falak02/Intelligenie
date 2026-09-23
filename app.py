import streamlit as st
from transformers import pipeline
import re
from collections import Counter


# ==========================================
# PAGE SETUP
# ==========================================

st.set_page_config(
    page_title="Intelligent NLP Assistant",
    page_icon="🤖",
    layout="centered"
)


# ==========================================
# CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e1b4b, #312e81);
}

.block-container {
    max-width: 850px;
    padding-top: 45px;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: white;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 17px;
    margin-bottom: 30px;
}

.result-box {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 18px;
    margin-top: 25px;
    border: 1px solid rgba(255,255,255,0.15);
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    font-size: 17px;
    font-weight: bold;
    padding: 12px;
}

.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 40px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="title">🤖 Intelligent NLP Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze and summarize text using NLP techniques'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# SENTIMENT ANALYSIS MODEL
# ==========================================

@st.cache_resource
def load_sentiment_model():

    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )


# ==========================================
# FAST CPU SUMMARIZATION
# ==========================================

def fast_summarize(text, num_sentences=3):

    # Split text into sentences
    sentences = re.split(
        r'(?<=[.!?])\s+',
        text.strip()
    )

    # If text is already short
    if len(sentences) <= num_sentences:
        return text

    # Extract words
    words = re.findall(
        r'\b[a-zA-Z]+\b',
        text.lower()
    )

    # Common words that are less useful
    stop_words = {
        "the", "is", "a", "an", "and", "or",
        "of", "to", "in", "on", "for", "with",
        "that", "this", "it", "as", "are",
        "was", "were", "be", "by", "from",
        "at", "which", "has", "have", "had",
        "but", "not", "they", "their", "its"
    }

    # Count important words
    word_frequency = Counter(
        word
        for word in words
        if word not in stop_words
    )

    # Calculate score for each sentence
    sentence_scores = []

    for sentence in sentences:

        sentence_words = re.findall(
            r'\b[a-zA-Z]+\b',
            sentence.lower()
        )

        score = sum(
            word_frequency[word]
            for word in sentence_words
        )

        sentence_scores.append(
            (score, sentence)
        )

    # Select highest-scoring sentences
    best_sentences = sorted(
        sentence_scores,
        reverse=True
    )[:num_sentences]

    # Put them back in original order
    best_sentences = sorted(
        best_sentences,
        key=lambda x: sentences.index(x[1])
    )

    # Create final summary
    summary = " ".join(
        sentence
        for score, sentence in best_sentences
    )

    return summary


# ==========================================
# FEATURE SELECTION
# ==========================================

feature = st.selectbox(
    "Choose a feature",
    [
        "😊 Sentiment Analysis",
        "📝 Text Summarization"
    ]
)


# ==========================================
# TEXT INPUT
# ==========================================

text = st.text_area(
    "Enter your text",
    placeholder="Write a sentence or paragraph here...",
    height=180
)


# ==========================================
# SENTIMENT ANALYSIS
# ==========================================

if feature == "😊 Sentiment Analysis":

    if st.button("✨ Analyze Sentiment"):

        if not text.strip():

            st.warning(
                "Please enter some text."
            )

        else:

            with st.spinner(
                "Analyzing your text..."
            ):

                classifier = load_sentiment_model()

                result = classifier(text)[0]

            label = result["label"]

            confidence = result["score"] * 100

            # Choose emoji
            if label == "POSITIVE":
                emoji = "😊"
            else:
                emoji = "😞"

            # Result box
            st.markdown(
                '<div class="result-box">',
                unsafe_allow_html=True
            )

            st.subheader(
                "Analysis Result"
            )

            st.markdown(
                f"### {emoji} {label}"
            )

            st.progress(
                int(confidence)
            )

            st.write(
                f"**Confidence:** "
                f"{confidence:.2f}%"
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


# ==========================================
# TEXT SUMMARIZATION
# ==========================================

else:

    if st.button("📝 Summarize Text"):

        if not text.strip():

            st.warning(
                "Please enter some text."
            )

        elif len(text.split()) < 30:

            st.warning(
                "Please enter at least 30 words "
                "for summarization."
            )

        else:

            with st.spinner(
                "Creating summary..."
            ):

                # Fast CPU summarization
                summary = fast_summarize(
                    text,
                    num_sentences=3
                )

            # Result box
            st.markdown(
                '<div class="result-box">',
                unsafe_allow_html=True
            )

            st.subheader(
                "📝 Summary"
            )

            st.write(
                summary
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


# ==========================================
# FOOTER
# ==========================================

st.markdown(
    '<div class="footer">'
    'Powered by Hugging Face Transformers • '
    'DistilBERT • NLP'
    '</div>',
    unsafe_allow_html=True
)