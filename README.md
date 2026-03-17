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

---

## 📁 Project Structure
```
AMAZON_REVIEW_SENTIMENT/
├── notebooks/
│   ├── sentiment_analysis_tfidf_logistic_regression_model.ipynb  ← V1: TF-IDF + LogReg
│   ├── sentiment_analysis_word2vec_LSTM_model.ipynb              ← V2A: Word2Vec + LSTM
│   └── sentiment_analysis_GLoVE_LSTM_model.ipynb                 ← V2B: GloVe + LSTM
├── data/
│   └── raw/amazon_review_dataset.csv          ← same dataset all versions
├── final_models/                               ← V1 saved models
├── models_v2a_word2vec/                        ← V2A saved models
├── models_v2b_glove/                           ← V2B saved models
├── results_v2a_word2vec/                       ← V2A confusion matrix + curves
├── results_v2b_glove/                          ← V2B confusion matrix + curves
├── glove/                                      ← GloVe vectors (not in repo — see setup)
├── app.py                                      ← Streamlit app (V1 + V2A + V2B)
└── requirements.txt
```

---

## ⚙️ GloVe Setup

GloVe file is NOT included in this repo (331 MB — too large for GitHub).
```
1. Download → https://nlp.stanford.edu/projects/glove/
2. Download → glove.6B.zip
3. Extract  → glove.6B.100d.txt
4. Place at → AMAZON_REVIEW_SENTIMENT/glove/glove.6B.100d.txt
```

---

## 📊 Model Performance

### V1 — Before Fixes (Baseline)
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Negative | 0.00 ❌ | 0.00 ❌ | 0.00 ❌ |
| Positive | 0.95 | 1.00 | 0.97 |
| Accuracy | | | 94.78% |

> Model was completely blind to negative reviews — always predicted Positive.

---

### V1 — After SMOTE + GridSearchCV
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Negative | 0.33 | 0.45 | 0.38 |
| Positive | 0.97 | 0.95 | 0.96 |
| Accuracy | | | 92.41% |

> Accuracy dropped slightly but model is now honest —
> actually detecting negative reviews instead of ignoring them.

---

### V2A — Word2Vec + LSTM
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Negative | 0.39 | 0.47 | 0.42 |
| Positive | 0.96 | 0.94 | 0.95 |
| Accuracy | | | 91.00% |

> LSTM reads word sequences with memory — understands context better than TF-IDF.
> class_weight replaces SMOTE for handling imbalance in Keras.

---

### V2B — GloVe + LSTM (Frozen)
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Negative | 0.18 | 0.53 | 0.27 |
| Positive | 0.96 | 0.81 | 0.88 |
| Accuracy | | | 79.15% |

> Frozen GloVe vectors (trainable=False) could not adapt to informal Amazon review language.
> 18.9% of vocab missing from GloVe — informal words like "dont", "isnt", "paperwhite".

---

### V2B — GloVe + LSTM (Fine-Tuned) ✅ Final
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Negative | 0.35 | 0.73 | 0.48 |
| Positive | 0.98 | 0.90 | 0.94 |
| Accuracy | | | 88.63% |

> Fine-tuning (trainable=True) allows GloVe vectors to adapt to Amazon review language.
> Neg Recall jumped from 0.53 → 0.73 — model now catches 73% of negative reviews.
> Tricky reviews improved from 3/8 correct → 5/8 correct after fine-tuning.

---

## 🔄 Full Model Comparison

| | V1 | V2A | V2B Frozen | V2B Fine-Tuned |
|---|---|---|---|---|
| Accuracy | 92.41% | 91.00% | 79.15% | 88.63% |
| Neg Precision | 0.33 | 0.39 | 0.18 | 0.35 |
| Neg Recall | 0.45 | 0.47 | 0.53 | **0.73** ✅ |
| Neg F1 | 0.38 | 0.42 | 0.27 | **0.48** ✅ |
| Macro F1 | 0.67 | 0.69 | 0.57 | **0.71** ✅ |
| Word order | ❌ | ✅ | ✅ | ✅ |
| Negation | ❌ | partial | partial | partial |
| Sarcasm | ❌ | ❌ | ❌ | ❌ |

> **Fine-Tuned GloVe beats all previous models on Neg Recall and Macro F1.**
> Best model so far for catching negative reviews.

---

## ⚠️ What None of These Models Can Solve

| Review | Expected | All Models |
|--------|----------|------------|
| "Oh great... does not work at all" | NEGATIVE | POSITIVE ❌ |
| "Not bad at all" | POSITIVE | NEGATIVE ❌ |
| "Would not recommend this to anyone" | NEGATIVE | fails ❌ |

> Root cause — static embeddings give ONE fixed vector per word
> regardless of surrounding context.
> Sarcasm, double negation, indirect negation need contextual understanding.

---

## ⚠️ Where V1 Still Fails & Why

**Review — Predicted Positive ❌**
```
"Not bad at all! Initially I was worried but this 
 product exceeded my expectations completely."
```
**Why TF-IDF fails:**
TF-IDF sees individual words — *"bad"*, *"worried"* get high negative
scores but *"exceeded"*, *"expectations"* get positive scores.
No understanding of *"not bad"* = good.

---

**Review — Predicted Positive ❌**
```
"The packaging was nice and delivery was fast but the 
 product itself is absolutely useless and stopped working 
 within a week."
```
**Why TF-IDF fails:**
Starts with positive words — *"nice"*, *"fast"* — which
dominate the TF-IDF score. Critical negative part
*"useless"*, *"stopped working"* comes later but TF-IDF
treats all words equally regardless of position.

---

## 🔮 Next Step — V3 BERT

> All three models (V1, V2A, V2B) fail on the same reviews —
> sarcasm, double negation, indirect negation.
>
> **Root cause:** Static embeddings assign ONE fixed vector per word.
> "not" always means the same thing regardless of context.
>
> **BERT** uses contextual embeddings — the vector of each word
> changes based on ALL surrounding words simultaneously.
> Pre-trained on 3.3 Billion words.
> Fine-tuned on our data = domain adaptation.
> Expected to solve what V1, V2A, V2B all failed on.

---

## 🧠 Why This Works
```
Tells a clear story:
Broken → Fixed → Word meaning added → Better embeddings
→ Fine-tuning → Still has limits → BERT next
```