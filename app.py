import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json
import json

# ------------------------------
# Load tokenizer
# ------------------------------
@st.cache_data
def load_tokenizer():
    with open("tokenizer.json", "r", encoding="utf-8") as f:
        data = f.read()  # Read as string
    tokenizer = tokenizer_from_json(data)
    return tokenizer

# ------------------------------
# Load model
# ------------------------------
@st.cache_resource
def load_siamese_model():
    model = load_model("final_model.keras")
    return model

# ------------------------------
# Preprocess text
# ------------------------------
def preprocess_text(text, tokenizer, max_len=200):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    return padded

# ------------------------------
# Predict plagiarism score
# ------------------------------
def get_plagiarism_score(text1, text2, model, tokenizer):
    seq1 = preprocess_text(text1, tokenizer)
    seq2 = preprocess_text(text2, tokenizer)
    score = model.predict([seq1, seq2])[0][0]  # Siamese model expects two inputs
    return score

# ------------------------------
# Streamlit app UI
# ------------------------------
st.title("📄 Plagiarism Detection App")
st.write("Enter two texts below to check for plagiarism. The model will return a score between 0 and 1.")

text1 = st.text_area("Enter Text 1")
text2 = st.text_area("Enter Text 2")

model = load_siamese_model()
tokenizer = load_tokenizer()

if st.button("Check Plagiarism"):
    if not text1.strip() or not text2.strip():
        st.warning("⚠️ Please enter both texts!")
    else:
        try:
            score = get_plagiarism_score(text1, text2, model, tokenizer)
            st.success(f"Plagiarism Score: {score:.4f}")
            if score > 0.5:
                st.info("⚠️ The texts are likely plagiarized.")
            else:
                st.info("✅ The texts are likely not plagiarized.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
