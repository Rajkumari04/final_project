import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# --- Load Tokenizer ---
@st.cache_data
def load_tokenizer():
    with open("tokenizer.json", "r") as f:
        data = f.read()
        tokenizer = tokenizer_from_json(data)
    return tokenizer

# --- Load Model ---
@st.cache_resource
def load_siamese_model():
    model = load_model("final_model.keras")  # changed from h5 to keras
    return model

tokenizer = load_tokenizer()
model = load_siamese_model()

st.title("Siamese Text Similarity App")
st.write("Enter two texts to check similarity.")

text1 = st.text_area("Text 1")
text2 = st.text_area("Text 2")

max_len = 100  # adjust based on your model

def preprocess(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len, padding='post')
    return padded

if st.button("Check Similarity"):
    if text1.strip() == "" or text2.strip() == "":
        st.warning("Please enter both texts.")
    else:
        seq1 = preprocess(text1)
        seq2 = preprocess(text2)
        pred = model.predict([seq1, seq2])[0][0]
        st.success(f"Similarity Score: {pred:.4f}")
