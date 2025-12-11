import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

# -----------------------------
# Load Model (cached for efficiency)
# -----------------------------
@st.cache_resource
def load_model(path="final_model.keras"):
    """
    Load the Keras model and cache it to avoid reloading on every interaction.
    """
    model = tf.keras.models.load_model(path)
    return model

model = load_model()

# -----------------------------
# Load Tokenizer (cached)
# -----------------------------
@st.cache_resource
def load_tokenizer(path="tokenizer.json"):
    """
    Load the tokenizer from JSON and cache it.
    """
    with open(path, "r") as f:
        data = f.read()
    tokenizer = tokenizer_from_json(data)
    return tokenizer

tokenizer = load_tokenizer()

# -----------------------------
# Prediction Function
# -----------------------------
def predict_plagiarism(text1, text2, max_len=200):
    """
    Predict plagiarism between two texts.
    Returns a label and the plagiarism score.
    """
    try:
        combined_text = text1 + " " + text2
        seq = tokenizer.texts_to_sequences([combined_text])
        seq_padded = pad_sequences(seq, maxlen=max_len)

        score = float(model.predict(seq_padded, verbose=0)[0][0])

        if score > 0.5:
            return "⚠️ **Plagiarism Detected**", score
        else:
            return "✅ **No Plagiarism Detected**", score

    except Exception as e:
        return f"Error: {str(e)}", None

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Plagiarism Detection", layout="centered")
st.title("📄 Plagiarism Detection App")

st.markdown(
    "Enter two texts below to check for plagiarism. "
    "The model will return a plagiarism score between 0 and 1."
)

text1 = st.text_area("Enter Text 1")
text2 = st.text_area("Enter Text 2")

if st.button("Check Plagiarism"):
    if not text1.strip() or not text2.strip():
        st.warning("Please enter both texts!")
    else:
        label, score = predict_plagiarism(text1, text2)
        st.write(label)
        if score is not None:
            st.info(f"Plagiarism Score: {score:.2f}")
