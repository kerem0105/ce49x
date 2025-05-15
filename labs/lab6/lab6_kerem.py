#!/usr/bin/env python3
"""
Lab 6: Document Classification Pipeline with Advanced Tasks
Author: Kerem
Date: 2025-05-15
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from scipy.sparse import hstack

from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    StratifiedKFold
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def clean_text(text: str) -> str:
    """
    Lowercase, remove common abbreviations (co, rfi, qi, si),
    strip digits and non-word characters.
    """
    text = text.lower()
    text = re.sub(r'\b(co|rfi|qi|si)\b', '', text)
    text = re.sub(r'[\d\W]+', ' ', text)
    return text.strip()


def classify_new_documents(path: str,
                           dv: DictVectorizer,
                           tfidf_vec: TfidfVectorizer,
                           clf) -> None:
    """
    Loads a JSON file with the same schema and predicts document types.
    """
    new_df = pd.read_json(path)
    # Metadata features
    meta = new_df[['project_phase', 'author_role']].fillna('Missing')
    X_meta_new = dv.transform(meta.to_dict(orient='records'))
    # Text features
    new_df['clean_content'] = new_df['content'].apply(clean_text)
    X_tfidf_new = tfidf_vec.transform(new_df['clean_content'])
    # Combined features
    X_comb_new = hstack([X_meta_new, X_tfidf_new])
    preds = clf.predict(X_comb_new)
    for i, p in enumerate(preds):
        print(f"Document #{i+1}: Predicted = {p}")


def main():
    # —————————————
    # 1. Load dataset
    # —————————————
    df = pd.read_json('construction_documents.json')
    df['date'] = pd.to_datetime(df['date'])

    # —————————————
    # 2. Metadata preprocessing
    # —————————————
    df[['project_phase', 'author_role']] = df[['project_phase', 'author_role']].fillna('Missing')
    dv = DictVectorizer(sparse=False)
    X_meta = dv.fit_transform(df[['project_phase', 'author_role']].to_dict(orient='records'))

    # —————————————
    # 3. Text cleaning & vectorization
    # —————————————
    df['clean_content'] = df['content'].apply(clean_text)
    count_vec = CountVectorizer()
    tfidf_vec = TfidfVectorizer()
    X_count = count_vec.fit_transform(df['clean_content'])
    X_tfidf = tfidf_vec.fit_transform(df['clean_content'])

    # —————————————
    # 4. Train/Test split & basic NB evaluation
    # —————————————
    y = df['document_type'].values
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(idx, test_size=0.2,
                                           random_state=42, stratify=y)

    X_meta_tr, X_meta_te = X_meta[train_idx], X_meta[test_idx]
    X_count_tr, X_count_te = X_count[train_idx], X_count[test_idx]
    X_tfidf_tr, X_tfidf_te = X_tfidf[train_idx], X_tfidf[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    X_comb_tr = hstack([X_meta_tr, X_tfidf_tr])
    X_comb_te = hstack([X_meta_te, X_tfidf_te])

    model_features = {
        'Metadata Only':     (X_meta_tr,  X_meta_te),
        'Count Vector':      (X_count_tr, X_count_te),
        'TF-IDF Vector':     (X_tfidf_tr, X_tfidf_te),
        'Combined Features': (X_comb_tr,  X_comb_te),
    }

    print("\n=== Basic NB Evaluation on Held-Out Test ===")
    best_clf = None
    for name, (X_tr, X_te) in model_features.items():
        clf = MultinomialNB()
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        acc = accuracy_score(y_te, preds)
        print(f"{name:17s} — Accuracy: {acc:.2%}")

        if name == 'Combined Features':
            best_clf = clf
            # Confusion matrix
            cm = confusion_matrix(y_te, preds, labels=clf.classes_)
            plt.figure(figsize=(5,4))
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.title('Confusion Matrix — Combined Features')
            plt.xticks(range(len(clf.classes_)), clf.classes_, rotation=45)
            plt.yticks(range(len(clf.classes_)), clf.classes_)
            plt.colorbar()
            plt.tight_layout()
            plt.show()

    # —————————————
    # 5. Advanced Task: Cross-validation (5-fold)
    # —————————————
    print("\n=== 5-Fold Cross-Validation (Naive Bayes) ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, X_all in [
        ('Metadata Only', X_meta),
        ('Count Vector',  X_count),
        ('TF-IDF Vector', X_tfidf),
        ('Combined',      hstack([X_meta, X_tfidf]))
    ]:
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        scores = cross_validate(MultinomialNB(), X_all, y,
                                cv=cv, scoring=scoring,
                                return_train_score=False)
        print(f"\n{name}:")
        for metric in scoring:
            mean = np.mean(scores[f'test_{metric}'])
            std  = np.std(scores[f'test_{metric}'])
            print(f"  {metric:16s}: {mean:.2%} ± {std:.2%}")

    # —————————————
    # 6. Advanced Task: Compare with other classifiers on Combined features
    # —————————————
    print("\n=== Classifier Comparison (5-fold CV on Combined) ===")
    classifier_configs = {
        'Multinomial NB':       MultinomialNB(),
        'Logistic Regression':  LogisticRegression(max_iter=1000),
        'Random Forest':        RandomForestClassifier(n_estimators=100, random_state=42),
    }
    combined_X = hstack([X_meta, X_tfidf])
    for clf_name, clf_obj in classifier_configs.items():
        scores = cross_validate(clf_obj, combined_X, y,
                                cv=cv,
                                scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])
        acc = np.mean(scores['test_accuracy'])
        f1  = np.mean(scores['test_f1_macro'])
        print(f"{clf_name:20s} — Accuracy: {acc:.2%}, F1-macro: {f1:.2%}")

    # —————————————
    # 7. Advanced Task: Temporal Pattern Analysis
    # —————————————
    df_ts = df.set_index('date')
    monthly = df_ts.groupby([pd.Grouper(freq='M'), 'document_type']) \
                   .size().unstack(fill_value=0)
    monthly.plot(figsize=(10,5))
    plt.title('Monthly Document Type Counts')
    plt.xlabel('Month')
    plt.ylabel('Number of Documents')
    plt.tight_layout()
    plt.show()

    # —————————————
    # 8. Advanced Task: Simple CLI Interface for New Docs
    # —————————————
    print("\n=== Enter Interactive Classification Mode ===")
    print("Type a JSON file path to classify, or 'exit' to quit.")
    while True:
        path = input(">> ").strip()
        if path.lower() in ('exit', 'quit'):
            break
        try:
            classify_new_documents(path, dv, tfidf_vec, best_clf)
        except Exception as e:
            print("Error:", e)

    print("Goodbye!")


if __name__ == '__main__':
    main()
