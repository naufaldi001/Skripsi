import joblib
import os
from preprocess import clean_text

# Dapatkan directory dari file predict.py saat ini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model dengan path lengkap
model_path = os.path.join(BASE_DIR, "naive_bayes_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return pred
