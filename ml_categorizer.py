"""
AI categorizer: predicts an expense category from its text description.

Pipeline = TfidfVectorizer (text -> numeric vector) + MultinomialNB (classifier).
We start with seed training data, then retrain on real user data as it grows,
so accuracy improves the more the app is used.
"""

import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

MODEL_PATH = os.path.join(os.path.dirname(__file__), "categorizer.joblib")

# Seed training data: (description, category)
SEED_DATA = [
    ("starbucks coffee", "Food"),
    ("mcdonalds burger", "Food"),
    ("zomato order", "Food"),
    ("swiggy dinner", "Food"),
    ("grocery store", "Food"),
    ("restaurant dinner", "Food"),
    ("uber ride", "Transport"),
    ("ola cab", "Transport"),
    ("petrol pump fuel", "Transport"),
    ("metro card recharge", "Transport"),
    ("flight ticket booking", "Transport"),
    ("amazon purchase", "Shopping"),
    ("flipkart order", "Shopping"),
    ("clothing store", "Shopping"),
    ("shoes purchase", "Shopping"),
    ("electricity bill", "Bills"),
    ("water bill payment", "Bills"),
    ("mobile recharge", "Bills"),
    ("internet bill", "Bills"),
    ("rent payment", "Bills"),
    ("netflix subscription", "Entertainment"),
    ("movie tickets", "Entertainment"),
    ("spotify subscription", "Entertainment"),
    ("gaming purchase", "Entertainment"),
    ("gym membership", "Other"),
    ("doctor visit", "Other"),
    ("medicine purchase", "Other"),
    ("books purchase", "Other"),
]

CATEGORIES = ["Food", "Transport", "Shopping", "Bills", "Entertainment", "Other"]


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", MultinomialNB()),
    ])


def train_and_save(extra_data=None):
    """Train on seed data + any extra (description, category) pairs, then save."""
    data = list(SEED_DATA) + (extra_data or [])
    descriptions = [d for d, _ in data]
    categories = [c for _, c in data]

    pipeline = build_pipeline()
    pipeline.fit(descriptions, categories)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline


def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return train_and_save()


def predict_category(description: str) -> str:
    model = load_model()
    return model.predict([description])[0]


def predict_with_confidence(description: str):
    model = load_model()
    proba = model.predict_proba([description])[0]
    classes = model.classes_
    best_idx = proba.argmax()
    return classes[best_idx], round(float(proba[best_idx]) * 100, 1)


if __name__ == "__main__":
    train_and_save()
    print("Model trained and saved.")
    tests = ["uber to airport", "dominos pizza", "electricity bill payment", "netflix monthly"]
    for t in tests:
        cat, conf = predict_with_confidence(t)
        print(f"{t!r:35s} -> {cat} ({conf}%)")
