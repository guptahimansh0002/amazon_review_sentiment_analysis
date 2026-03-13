import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

# Load models
import os

# Load TF-IDF vectorizer
with open('final_models/gridtfidf_vectorizer.pkl', 'rb') as f:
    gridtfidf = pickle.load(f)  

# Load trained model
with open('final_models/sentiment_model.pkl', 'rb') as f:
    best_model = pickle.load(f)  

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ─ Streamlit UI ─
st.set_page_config(page_title="Amazon Review Sentiment", page_icon="🛒")

st.title("🛒 Amazon Review Sentiment Analyzer")
st.write("Enter any Amazon product review to check if it's Positive or Negative")

# Text input box
review = st.text_area("Paste your Amazon review here:", height=150,
                       placeholder="e.g. This product is amazing, works perfectly!")

# Predict button
if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review first!")
    else:
        # Clean and predict
        cleaned = clean_text(review)
        vector = gridtfidf.transform([cleaned])
        prediction = best_model.predict(vector)[0]
        probability = best_model.predict_proba(vector)[0]

        # Show result
        if prediction == 1:
            st.success(f"✅ POSITIVE Review  —  {round(probability[1]*100, 1)}% confident")
        else:
            st.error(f"❌ NEGATIVE Review  —  {round(probability[0]*100, 1)}% confident")

        # Show probability bar
        st.write("---")
        st.write("**Confidence Breakdown:**")
        col1, col2 = st.columns(2)
        col1.metric("Positive", f"{round(probability[1]*100, 1)}%")
        col2.metric("Negative", f"{round(probability[0]*100, 1)}%")

# Footer
st.write("---")
st.caption("Built by Himanshu Gupta | Amazon Review Sentiment Analysis Project")