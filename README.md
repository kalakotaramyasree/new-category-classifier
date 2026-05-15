# 📰 BBC News Text Classifier — Streamlit App

A Naive Bayes text classification app built from your ML notebook.

## Files
```
bbc_classifier/
├── app.py                  ← Streamlit app
├── tfidf_vectorizer.pkl    ← Trained TF-IDF vectorizer
├── mnb_model.pkl           ← Trained Naive Bayes model
├── requirements.txt        ← Python dependencies
└── README.md
```

---

## 🚀 Run Locally (Step-by-Step)

### Step 1 — Install Python (if not installed)
Download Python 3.10+ from https://python.org

### Step 2 — Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the app
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

---

## ☁️ Deploy to Streamlit Cloud (Free)

### Step 1 — Push to GitHub
1. Create a new GitHub repo (e.g. `bbc-news-classifier`)
2. Upload all files in this folder to the repo

### Step 2 — Go to Streamlit Cloud
Visit https://share.streamlit.io and sign in with GitHub

### Step 3 — Deploy
- Click **"New app"**
- Select your GitHub repo
- Set **Main file path** → `app.py`
- Click **"Deploy!"**

Your app will be live at:
`https://<your-username>-bbc-news-classifier.streamlit.app`

---

## 🔁 Retrain with Full BBC Dataset
To retrain with the real BBC dataset, run this in a notebook or script:

```python
import pandas as pd, pickle, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

url = "https://raw.githubusercontent.com/ronaldleonardo/Data_Transformer_Datasets/refs/heads/main/bbc_text_cls.csv"
df = pd.read_csv(url)

def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return " ".join(text.split())

df["cleaned"] = df["text"].apply(clean)
X_train, X_test, y_train, y_test = train_test_split(df["cleaned"], df["labels"], test_size=0.2, random_state=42)

tfidf = TfidfVectorizer()
mnb = MultinomialNB(alpha=1.0)
mnb.fit(tfidf.fit_transform(X_train), y_train)

pickle.dump(tfidf, open("tfidf_vectorizer.pkl","wb"))
pickle.dump(mnb,   open("mnb_model.pkl","wb"))
print("Done! Replace pkl files and redeploy.")
```
