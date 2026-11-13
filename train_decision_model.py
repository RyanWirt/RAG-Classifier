"""Decision-model training demo: labeled CSV -> embeddings -> classifier -> report.

Run: python train_decision_model.py [data/labeled_examples.csv]
CPU only: embeddings come from the configured endpoint; the model is logistic regression.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from app.config import Settings
from app.retrieval import create_embeddings

SAMPLES = [
    "My card was billed again after I cancelled.",
    "The app crashes whenever I open reports.",
    "Do you have an API reference?",
    "I can't sign in, it says my account is locked.",
    "Please let me export everything to Excel.",
    "The weather is lovely today.",
]


def main(path: str = "data/labeled_examples.csv") -> int:
    frame = pd.read_csv(path).dropna(subset=["text", "label"])
    print(frame.head(), "\n")
    print(frame["label"].value_counts().to_string(), "\n")

    embeddings = create_embeddings(Settings.from_env())
    vectors = np.array(embeddings.embed_documents(frame["text"].tolist()))
    print(f"Embedded {vectors.shape[0]} texts into {vectors.shape[1]} dimensions.\n")

    x_train, x_test, y_train, y_test = train_test_split(
        vectors, frame["label"], test_size=0.3, stratify=frame["label"], random_state=0
    )
    model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
    print(f"Held-out test set ({len(y_test)} rows):")
    print(classification_report(y_test, model.predict(x_test), zero_division=0))

    probabilities = model.predict_proba(np.array(embeddings.embed_documents(SAMPLES)))
    results = pd.DataFrame(
        {
            "text": SAMPLES,
            "predicted": model.classes_[probabilities.argmax(axis=1)],
            "probability": probabilities.max(axis=1).round(2),
        }
    )
    print("New inputs:")
    print(results.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:2]))
