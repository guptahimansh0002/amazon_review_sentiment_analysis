
# 🛒 Amazon Review Sentiment Analysis

Classifying Amazon product reviews as **Positive** or **Negative** using NLP and Machine Learning.

> Built to compare Classical ML vs Deep Learning approaches on real-world imbalanced data.

---

## 🧠 Problem Statement

Given a product review written by a customer, classify the sentiment of the review as:
- Positive
- Negative

---

## 🚀 Run Locally
```bash
git clone https://github.com/guptahimansh0002/amazon-sentiment-analysis.git
cd AMAZON_REVIEW_SENTIMENT
pip install -r requirements.txt
streamlit run app.py
```


## 📊 Model Performance

### Before Fixes (Baseline)
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Negative | 0.00 ❌ | 0.00 ❌ | 0.00 ❌ |
| Positive | 0.95 | 1.00 | 0.97 |
| Accuracy | | | 94.78% |

> Model was completely blind to negative reviews — always predicted Positive.

---

### After SMOTE + GridSearchCV
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Negative | 0.33 | 0.45 | 0.38 ✅ |
| Positive | 0.97 | 0.95 | 0.96 |
| Accuracy | | | 92.41% |

> Accuracy dropped slightly but model is now honest —
> actually detecting negative reviews instead of ignoring them.

---

## ⚠️ Where Model Still Fails & Why

**Review 3 — Predicted Positive ❌**
```
"Not bad at all! Initially I was worried but this 
 product exceeded my expectations completely."
```
**Why TF-IDF fails:**
TF-IDF sees individual words — *"bad"*, *"worried"* get high negative
scores but *"exceeded"*, *"expectations"* get positive scores.
No understanding of *"not bad"* = good or that the overall
sentence turns positive by the end.

---

**Review 4 — Predicted Positive ❌**
```
"The packaging was nice and delivery was fast but the 
 product itself is absolutely useless and stopped working 
 within a week."
```
**Why TF-IDF fails:**
Starts with positive words — *"nice"*, *"fast"* — which
dominate the TF-IDF score. The critical negative part
*"useless"*, *"stopped working"* comes later but TF-IDF
treats all words equally regardless of position or context.

---

## 🔮 Why LSTM is the Next Step

| What TF-IDF Cannot Do | How LSTM Fixes It |
|----------------------|-------------------|
| *"not bad"* → reads "bad" as negative | Reads full sequence → understands "not" flips meaning |
| Ignores word position | Remembers word order left to right |
| Mixed reviews confuse it | Understands sentiment shifts mid-sentence |
| Sarcasm detection impossible | Context memory catches tone change |

> V2 will use LSTM with Word Embeddings trained on the same
> Amazon dataset — directly fixing the failures shown above.
```

---

## 🧠 Why This Works
```
Tells a clear story:
Broken → Fixed → Still has limits → Here is why → Here is next step

Feels human because:
→ Shows honest failures not just good results
→ Uses real review examples
→ Explains reasoning not just metrics
→ Natural transition to V2

