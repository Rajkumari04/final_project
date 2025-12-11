import streamlit as st
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------------
# Load Model
# -------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("final_model.keras")
    return model

model = load_model()

# -------------------------------------------------------
# Load Tokenizer
# -------------------------------------------------------
@st.cache_resource
def load_tokenizer():
    with open("tokenizer.json", "r") as f:
        data = json.load(f)
    tokenizer = tokenizer_from_json(data)
    return tokenizer

tokenizer = load_tokenizer()

# -------------------------------------------------------
# Preprocess
# -------------------------------------------------------
def preprocess(text):
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=200)
    return seq

# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------
st.set_page_config(page_title="Plagiarism Detection", layout="centered")

st.title("📄 Plagiarism Detection App")
st.write("Enter two texts to check for semantic similarity (plagiarism score).")

text1 = st.text_area("Enter Text 1")
text2 = st.text_area("Enter Text 2")

# -------------------------------------------------------
# Prediction Handling
# -------------------------------------------------------
if st.button("Check Plagiarism"):

    if text1.strip() == "" or text2.strip() == "":
        st.warning("⚠️ Please enter both texts.")
    else:
        try:
            seq1 = preprocess(text1)
            seq2 = preprocess(text2)

            prediction = model.predict([seq1, seq2])[0][0]
            score = float(prediction)

            st.subheader("🔍 Result:")
            st.write(f"**Plagiarism Score: `{score:.4f}`**")

            if score > 0.5:
                st.error("⚠️ Plagiarism Detected")
            else:
                st.success("✅ No Plagiarism Detected")

        except Exception as e:
            st.error(f"Error: {str(e)}")
