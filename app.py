import streamlit as st
from keras.models import load_model
from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
import json
import numpy as np

# -------------------------
# Load Tokenizer
# -------------------------
@st.cache_data(show_spinner=True)
def load_tokenizer():
    with open("tokenizer.json", "r") as f:
        data = json.load(f)
    tokenizer = tokenizer_from_json(data)
    return tokenizer

# -------------------------
# Load Model
# -------------------------
@st.cache_resource(show_spinner=True)
def load_siamese_model():
    model = load_model("final_model.keras")
    return model

# -------------------------
# Prediction Function
# -------------------------
def predict_pair(text1, text2, tokenizer, model, max_len=100):
    seq1 = tokenizer.texts_to_sequences([text1])
    seq2 = tokenizer.texts_to_sequences([text2])

    seq1 = pad_sequences(seq1, maxlen=max_len)
    seq2 = pad_sequences(seq2, maxlen=max_len)

    score = model.predict([seq1, seq2])[0][0]
    
    if score > 0.5:
        result = "⚠️ Likely PLAGIARIZED"
    else:
        result = "✅ Likely NOT Plagiarized"
    
    return score, result

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Plagiarism Checker", layout="centered")
st.title("📄 Plagiarism Detection with Score")

text1 = st.text_area("Enter first text", height=150)
text2 = st.text_area("Enter second text", height=150)

if st.button("Check Plagiarism"):
    if not text1.strip() or not text2.strip():
        st.warning("Please enter both texts!")
    else:
        try:
            tokenizer = load_tokenizer()
            model = load_siamese_model()
            score, result = predict_pai_
