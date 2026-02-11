import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ===== LOAD DATA =====
df = pd.read_csv("data/clean_ulasan_product_dataset.csv")

X = df["clean_text"].astype(str)
y = df["label"]

# ===== SPLIT DATA =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ===== TF-IDF =====
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ===== MODEL =====
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ===== EVALUATION =====
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, digits=4))

print("Confusion Matrix (POS, NET, NEG):")
print(confusion_matrix(y_test, y_pred, labels=["POS", "NET", "NEG"]))

# ===== SAVE MODEL =====
joblib.dump(model, "naive_bayes_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel & vectorizer saved.")
