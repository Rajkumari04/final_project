import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# ----------------------------
# Load tokenizer
# ----------------------------
with open("tokenizer.json", "r") as f:
    tokenizer_json = f.read()  # read as string
tokenizer = tokenizer_from_json(tokenizer_json)

# ----------------------------
# Load trained model for inference
# ----------------------------
model = load_model("final_model.keras", compile=False)

# ----------------------------
# Settings
# ----------------------------
MAX_LEN = 50

# ----------------------------
# Prediction function
# ----------------------------
def predict_pair(source_text, plag_text):
    # Convert to sequences
    src_seq = tokenizer.texts_to_sequences([source_text])
    plg_seq = tokenizer.texts_to_sequences([plag_text])

    # Pad sequences
    src_pad = pad_sequences(src_seq, maxlen=MAX_LEN)
    plg_pad = pad_sequences(plg_seq, maxlen=MAX_LEN)

    # Predict
    pred = model.predict([src_pad, plg_pad], verbose=0)[0][0]

    # Interpret
    if pred > 0.5:
        result = "Likely PLAGIARIZED"
    else:
        result = "Likely NOT plagiarized"

    return pred, result

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Plagiarism Detection App")
st.write("Check if a text is plagiarized against another text.")

source_text = st.text_area("Source Text", height=150)
plag_text = st.text_area("Text to Check", height=150)

if st.button("Check Plagiarism"):
    if not source_text.strip() or not plag_text.strip():
        st.error("Please enter both texts.")
    else:
        score, result = predict_pair(source_text, plag_text)
        st.success(f"Prediction score: {score:.6f} — {result}")
