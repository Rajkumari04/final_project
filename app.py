import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

# -----------------------------
# Load Siamese Model
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("final_model.keras")
    return model

model = load_model()

# -----------------------------
# Load Tokenizer
# -----------------------------
@st.cache_resource
def load_tokenizer():
    try:
        with open("tokenizer.json", "r", encoding="utf-8") as f:
            data = f.read()  # raw JSON string
        tokenizer = tokenizer_from_json(data)
        return tokenizer
    except Exception as e:
        st.error(f"Tokenizer load error: {e}")
        st.stop()

tokenizer = load_tokenizer()

# -----------------------------
# Prediction Function
# -----------------------------
def predict_plagiarism(text1, text2):
    if text1.strip() == "" or text2.strip() == "":
        return "⚠️ Please enter both texts", None

    # Convert texts to sequences
    seq1 = tokenizer.texts_to_sequences([text1])
    seq2 = tokenizer.texts_to_sequences([text2])

    # Pad sequences
    maxlen = 200
    seq1 = pad_sequences(seq1, maxlen=maxlen)
    seq2

                st.success("✅ No Plagiarism Detected")

        except Exception as e:
            st.error(f"Error: {str(e)}")
