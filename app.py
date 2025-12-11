import streamlit as st
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("final_model.h5")
    return model

model = load_model()

# -----------------------------
# Load Tokenizer
# -----------------------------
@st.cache_resource
def load_tokenizer():
    with open("tokenizer.json", "r") as f:
        data = f.read()
    tokenizer = tokenizer_from_json(data)
    return tokenizer

tokenizer = load_tokenizer()

# -----------------------------
# Prediction Function
# -----------------------------
def predict(text1, text2):
    try:
        combined = text1 + " " + text2
        seq = tokenizer.texts_to_sequences([combined])
        seq = pad_sequences(seq, maxlen=200)

        pred = model.predict(seq)[0][0]
        pred = float(pred)

        if pred > 0.5:
            return "⚠️ **Plagiarism Detected**", pred
        else:
            return "✅ **No Plagiarism Detected**", pred

    except Exception as e:
        return f"Error: {str(e)}", None

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Plagiarism Detection", layout="centered")

st.ti
