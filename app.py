import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# Load tokenizer
@st.cache_data
def load_tokenizer():
    with open("tokenizer.json", "r") as f:
        data = f.read()
        tokenizer = tokenizer_from_json(data)
    return tokenizer

# Load model
@st.cache_resource
def load_siamese_model():
    model = load_model("final_model.h5")
    return model

tokenizer = load_tokenizer()
model = load_siamese_model()

MAX_LEN = 200  # Adjust based on your training

def preprocess(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_LEN, padding='post')
    return padded

def predict_similarity(text1, text2):
    seq1 = preprocess(text1)
    seq2 = preprocess(text2)
    score = model.predict([seq1, seq2])[0][0]
    return float(score)

# Streamlit UI
st.set_page_config(page_title="Plagiarism Detection", layout="centered")
st.title("📄 Plagiarism Detection App")

text1 = st.text_area("Enter Text 1")
text2 = st.text_area("Enter Text 2")

if st.button("Check Plagiarism"):
    if not text1.strip() or not text2.strip():
        st.warning("Please enter both texts.")
    else:
        with st.spinner("Checking..."):
            score = predict_similarity(text1, text2)
            st.success(f"Plagiarism Score: {score:.2f}")
            if score > 0.5:
                st.info("⚠️ High similarity detected!")
            else:
                st.info("✅ Texts are mostly different.")
