#!/usr/bin/env python3
"""
Train a tuned intent classifier for the paddy disease chatbot.

Usage:
    python train_intent_classifier.py

Input:
    - intent_training_data.csv  (columns: text,label)

Output:
    - intent_classifier.joblib  (best pipeline: vectorizer + model)
"""

from pathlib import Path
import csv

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "intent_training_synth.csv"
MODEL_PATH = BASE_DIR / "intent_training_synth.joblib"


def load_data():
    texts, labels = [], []

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")

    with DATA_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if text and label:
                texts.append(text)
                labels.append(label)

    return texts, labels


def main():
    texts, labels = load_data()
    print(f"Loaded {len(texts)} training examples.")

    # Hold-out set to sanity-check the final best model
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    # Base pipeline (we’ll swap vectorizer + model via GridSearch)
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression()),
        ]
    )

    # Parameter grid:
    # - Try different n-grams (more context)
    # - Try both LogisticRegression and LinearSVC
    # - Try a few C values
    param_grid = [
        {
            "tfidf__lowercase": [True],
            "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
            "tfidf__min_df": [1, 2],
            "clf": [LogisticRegression(max_iter=300, multi_class="auto")],
            "clf__C": [0.5, 1.0, 2.0],
            "clf__class_weight": [None, "balanced"],
        },
        {
            "tfidf__lowercase": [True],
            "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
            "tfidf__min_df": [1, 2],
            "clf": [LinearSVC()],
            "clf__C": [0.5, 1.0, 2.0],
            # LinearSVC doesn't support class_weight="balanced" in the same way for all configs,
            # so we leave it default here.
        },
    ]

    # Grid search with 5-fold CV
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring="f1_weighted",  # good for imbalanced classes
    )

    print("Starting grid search...")
    grid.fit(X_train, y_train)
    print("Grid search done.")

    print("\nBest CV score (weighted F1):", grid.best_score_)
    print("Best params:\n", grid.best_params_)

    best_model = grid.best_estimator_

    # Evaluate on held-out test set
    y_pred = best_model.predict(X_test)
    print("\nValidation report on held-out test set:")
    print(classification_report(y_test, y_pred))

    # Save the entire pipeline (vectorizer + classifier)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nSaved best intent classifier to {MODEL_PATH}")


if __name__ == "__main__":
    main()
