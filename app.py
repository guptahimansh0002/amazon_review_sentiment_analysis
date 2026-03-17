import os
import streamlit as st
import numpy as np
import pickle
import re
from nltk.corpus import stopwords

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

stop_words = set(stopwords.words("english"))
os.chdir(r'D:\AMAZON_REVIEW_SENTIMENT')
# Settings 
MAX_LEN = 100

# Load TF-IDF vectorizer and sentiment model (Loading V1 Model)
@st.cache_resource
def load_v1_models():
    with open('final_models/gridtfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('final_models/sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return tfidf, model

# Text cleaning function for v1
def clean_text_V1(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


# Load V2A Models (Word2Vec + LSTM)
@st.cache_resource
def load_v2a_models():
    try:
        model = load_model('models_v2a_word2vec/lstm_model.h5')
        with open('models_v2a_word2vec/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f"V2A model not found: {e}")
        st.info("Run v2a_word2vec_lstm.ipynb first")
        return None, None
    

##Define a function for cleaning text
def clean_text_V2a(text):
  text = str(text).lower()
  text = re.sub(r'<.*?>', '', text)     # remove HTML tags
  text = re.sub(r'[^a-zA-Z\s]', '', text)   # keep only letters
  text = re.sub(r'\s+', ' ', text).strip()   # extra spaces   
  
  return text

# Load V2B Models (GloVe + LSTM)
@st.cache_resource
def load_v2b_models():
    try:
        model = load_model('models_v2b_glove/lstm_glove_model.h5')
        with open('models_v2b_glove/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f"V2B model not found: {e}")
        st.info("Run v2b_glove_lstm.ipynb first")
        return None, None

def clean_text_V2b(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_v2b(text, model, tokenizer):
    clean  = clean_text_V2b(text)
    seq    = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    prob   = model.predict(padded, verbose=0)[0][0]
    label  = 'POSITIVE ✅' if prob > 0.5 else 'NEGATIVE ❌'
    conf   = prob if prob > 0.5 else 1 - prob
    return label, round(float(conf), 3)


# Predict V1
def predict_v1(text, tfidf, model):
    clean   = clean_text_V1(text)
    vector  = tfidf.transform([clean])
    pred    = model.predict(vector)[0]
    prob    = model.predict_proba(vector)[0]
    label   = 'POSITIVE ✅' if pred == 1 else 'NEGATIVE ❌'
    conf    = max(prob)
    return label, round(conf, 3)

# Predict V2A 
def predict_v2a(text, model, tokenizer):
    clean   = clean_text_V2a(text)
    seq     = tokenizer.texts_to_sequences([clean])
    padded  = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    prob    = model.predict(padded, verbose=0)[0][0]
    label   = 'POSITIVE ✅' if prob > 0.5 else 'NEGATIVE ❌'
    conf    = prob if prob > 0.5 else 1 - prob
    return label, round(float(conf), 3)

# ── Streamlit UI ──────────────────────────────────────────
st.title("🛍️ Amazon Review Sentiment Analysis")
st.markdown("Compare Classical ML vs Deep Learning")

# Version selector
version = st.selectbox(
    "Choose Model Version:",
    [
        "V1 — TF-IDF + Logistic Regression",
        "V2A — Word2Vec + LSTM",
        "V2B — GloVe + LSTM (Fine-Tuned)"
    ]
)

# Review input
review = st.text_area(
    "Enter Amazon Review:",
    placeholder="Type your review here..."
)

# Predict button
if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        if version == "V1 — TF-IDF + Logistic Regression":
            tfidf, model = load_v1_models()
            label, conf  = predict_v1(review, tfidf, model)
            st.markdown(f"### {label}")
            st.markdown(f"**Confidence:** {conf*100:.1f}%")
            st.info("Model: TF-IDF + Logistic Regression (V1)")

        elif version == "V2A — Word2Vec + LSTM":
            model, tokenizer = load_v2a_models()
            if model is None:
                st.error("V2A model not loaded")
            else:
                label, conf = predict_v2a(review, model, tokenizer)
                st.markdown(f"### {label}")
                st.markdown(f"**Confidence:** {conf*100:.1f}%")
                st.info("Model: Word2Vec + LSTM (V2A)")

        elif version == "V2B — GloVe + LSTM (Fine-Tuned)":
            model, tokenizer = load_v2b_models()
            if model is None:
                st.error("V2B model not loaded")
            else:
                label, conf = predict_v2b(review, model, tokenizer)
                st.markdown(f"### {label}")
                st.markdown(f"**Confidence:** {conf*100:.1f}%")
                st.info("Model: GloVe + LSTM — Fine-Tuned (V2B)")
           

# Footer
st.write("---")
st.caption("Built by Himanshu Gupta | Amazon Review Sentiment Analysis Project")