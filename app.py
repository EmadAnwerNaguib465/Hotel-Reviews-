import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Hotel Review Sentiment Analysis",
    page_icon="🏨",
    layout="centered"
)

# ---------------------------
# Header Image
# ---------------------------
# st.image("hotel.jpg", use_container_width=True)

# ---------------------------
# Title
# ---------------------------
st.title("🏨 Hotel Review Sentiment Analyzer")
st.write("Enter a hotel review and the AI model will predict the sentiment.")

# ---------------------------
# Constants
# ---------------------------
MAX_LEN = 100

# ---------------------------
# Load Model & Tokenizer
# ---------------------------
@st.cache_resource
def load_resources():
    model = load_model("best_lstm_model.keras", compile=False)
    tokenizer = pickle.load(open("tokenizer_best_lstm.pkl", "rb"))
    return model, tokenizer

model, tokenizer = load_resources()

# ---------------------------
# User Input
# ---------------------------
review = st.text_area(
    "Write your hotel review here:",
    placeholder="Example: The hotel was very clean and the staff were friendly..."
)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:

        # Convert text → sequence
        seq = tokenizer.texts_to_sequences([review])

        # Padding
        padded = pad_sequences(seq, maxlen=MAX_LEN)

        # Prediction
        prediction = model.predict(padded)

        sentiment_class = np.argmax(prediction)

        sentiment_map = {
            0: "Negative 😞",
            1: "Neutral 😐",
            2: "Positive 😊"
        }

        confidence = np.max(prediction)

        st.success(f"Sentiment: **{sentiment_map[sentiment_class]}**")
        st.info(f"Confidence: **{confidence*100:.2f}%**")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("AI Sentiment Analysis for Hotel Reviews | Built with Streamlit")
