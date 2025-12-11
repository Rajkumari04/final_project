import streamlit as st
import numpy as np
import pandas as pd
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# --- Load Tokenizer ---
@st.cache_resource
def load_tokenizer():
    with open("tokenizer.json", "r", encoding="utf-8") as f:
        data = f.read()
    tokenizer = tokenizer_from_json(data)
    return tokenizer

# --- Load Model ---
@st.cache_resource
def load_siamese_model():
    model = load_model("final_model.keras")
    return model

tokenizer = load_tokenizer()
model = load_siamese_model()

MAX_LEN = 200  # Use the same max length you trained on

# --- Streamlit UI ---
st.title("📄 Plagiarism Detection App")
st.write("Enter two texts below to check for plagiarism. The model will return a plagiarism score between 0 and 1.")

text1 = st.text_area("Enter Text 1")
text2 = st.text_area("Enter Text 2")

if st.button("Check Plagiarism"):
    if text1.strip() == "" or text2.strip() == "":
        st.warning("Please enter both texts.")
    else:
        # Tokenize and pad
        seq1 = tokenizer.texts_to_sequences([text1])
        seq2 = tokenizer.texts_to_sequences([text2])
        pad1 = pad_sequences(seq1, maxlen=MAX_LEN, padding='post', truncating='post')
        pad2 = pad_sequences(seq2, maxlen=MAX_LEN, padding='post', truncating='post')

        # Predict
        score = model.predict([pad1, pad2])[0][0]
        st.success(f"Plagiarism Score: {score:.4f}")
