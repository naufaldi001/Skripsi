import joblib
from preprocess import clean_text

model = joblib.load("/Sentiment App/naive_bayes_model.pkl")
vectorizer = joblib.load("/Sentiment App/tfidf_vectorizer.pkl")

def predict_sentiment(text):
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    return model.predict(vec)[0]

