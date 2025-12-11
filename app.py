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
    tokenizer_data = json.load(f)
tokenizer = tokenizer_from_json(tokenizer_data)

# ----------------------------
# Load model
# ----------------------------
model = load_model("final_model.keras")

# ----------------------------
# Constants
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
    pred = model.predict([src_pad, plg_pad])[0][0]
    score = np.float32(pred)
    result = "Likely PLAGIARIZED" if pred > 0.5 else "Likely NOT plagiarized"
    return score, result

# ----------------------------
# Streamlit interface
# ----------------------------
st.title("Plagiarism Detection")
st.write("Enter two texts to check if the second text is likely plagiarized from the first.")

source_text = st.text_area("Source Text")
plag_text = st.text_area("Text to Check")

if st.button("Check Plagiarism"):
    if source_text.strip() == "" or plag_text.strip() == "":
        st.error("Please enter both texts.")
    else:
        score, result = predict_pair(source_text, plag_text)
        st.write(f"**Prediction score:** {score:.4f}")
        st.write(f"**Result:** {result}")
