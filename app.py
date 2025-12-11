import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# Constants
MAX_LEN = 50  # Must match your model's training input length

# Load tokenizer
@st.cache_data
def load_tokenizer():
    with open("tokenizer.json", "r") as f:
        data = f.read()
        tokenizer = tokenizer_from_json(data)
    return tokenizer

# Load model
@st.cache_resource
def load_model_file():
    model = load_model("final_model.keras")
    return model

tokenizer = load_tokenizer()
model = load_model_file()

# Function to preprocess and predict
def predict_pair(text1, text2, tokenizer, max_len):
    seq1 = tokenizer.texts_to_sequences([text1])
    seq2 = tokenizer.texts_to_sequences([text2])
    
    seq1 = pad_sequences(seq1, maxlen=max_len, padding="post")
    seq2 = pad_sequences(seq2, maxlen=max_len, padding="post")
    
    score = model.predict([seq1, seq2])[0][0]
    result = "Likely PLAGIARIZED" if score > 0.5 else "Likely NOT plagiarized"
    
    return score, result

# Streamlit UI
st.title("Plagiarism Detection")

text1 = st.text_area("Enter Source Text:")
text2 = st.text_area("Enter Text to Compare:")

if st.button("Check"):
    if text1.strip() == "" or text2.strip() == "":
        st.warning("Please enter both texts to compare.")
    else:
        try:
            score, result = predict_pair(text1, text2, tokenizer, MAX_LEN)
            st.write(f"Prediction score: {score:.8f}")
            st.write(result)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
