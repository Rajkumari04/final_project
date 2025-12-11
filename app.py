import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# ---- CONFIG ----
MODEL_PATH = "final_model.keras"
TOKENIZER_PATH = "tokenizer.json"
MAX_LEN = 100  # Adjust based on your model

# ---- LOAD TOKENIZER ----
@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "r") as f:
        data = f.read()
        tokenizer = tokenizer_from_json(data)
    return tokenizer

# ---- LOAD MODEL ----
@st.cache_resource
def load_siamese_model():
    model = load_model(MODEL_PATH)
    return model

# ---- PREDICTION FUNCTION ----
def preprocess_text(text, tokenizer, max_len=MAX_LEN):
    seq = tokenizer.texts_to_sequences([text])
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = pad_sequences(seq, maxlen=max_len, padding="post")
    return seq

def predict_pair(text1, text2, tokenizer, model, max_len=MAX_LEN):
    seq1 = preprocess_text(text1, tokenizer, max_len)
    seq2 = preprocess_text(text2, tokenizer, max_len)
    score = float(model.predict([seq1, seq2])[0][0])
    result = "Likely PLAGIARIZED" if score > 0.5 else "Likely NOT plagiarized"
    return score, result

# ---- STREAMLIT INTERFACE ----
st.title("Plagiarism Checker")
st.write("Compare two texts to check if they are likely plagiarized.")

text1 = st.text_area("Text A")
text2 = st.text_area("Text B")

if st.button("Check Plagiarism"):
    if not text1.strip() or not text2.strip():
        st.warning("Please enter both texts!")
    else:
        try:
            tokenizer = load_tokenizer()
            model = load_siamese_model()
            score, result = predict_pair(text1, text2, tokenizer, model)
            
            st.write(f"**Prediction score:** {score:.8f}")
            st.success(result)
        except Exception as e:
            st.error(f"An error occurred: {e}")
