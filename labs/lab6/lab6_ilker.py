import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Load your JSON dataset
df = pd.read_json('construction_documents.json')

# 2. Extract the 'content' field and the labels
texts = df['content'].fillna('')      # fill missing content with empty string
labels = df['document_type']

# 3. Vectorize all documents using TF-IDF
tfidf = TfidfVectorizer(
    stop_words='english',              # remove English stop words
    ngram_range=(1, 2),                # use unigrams and bigrams
    max_features=1000                  # keep only top 1000 features
)
X = tfidf.fit_transform(texts)        # fit on all texts
feature_names = np.array(tfidf.get_feature_names_out())

# 4. Train a small Naive Bayes model to get feature log-probs
X_train, X_test, y_train, y_test = train_test_split(
    X, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Collect the union of top-10 terms per class
top_terms = set()
for i, cls in enumerate(model.classes_):
    idxs = np.argsort(model.feature_log_prob_[i])[-10:]
    top_terms |= set(feature_names[idxs])
top_terms = sorted(top_terms)
top_idxs = [np.where(feature_names == t)[0][0] for t in top_terms]

# 6. Build a correlation matrix between each class and each term
#    using Pearson correlation of (binary class indicator) vs TF-IDF value
corr_df = pd.DataFrame(index=model.classes_, columns=top_terms, dtype=float)

for cls in model.classes_:
    # binary indicator vector: 1 if doc is of this class, else 0
    y_bin = (labels == cls).astype(int).values
    for term in top_terms:
        j = top_idxs[top_terms.index(term)]
        term_vec = X[:, j].toarray().ravel()   # TF-IDF scores for that term
        if term_vec.std() == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(y_bin, term_vec)[0, 1]
        corr_df.loc[cls, term] = (corr + 1) / 2  # scale from [-1,1] to [0,1]

# 7. Print the correlation DataFrame
print("\nCorrelation (scaled 0–1) between document types and top terms:\n")
print(corr_df)

# 8. Plot as heatmap
plt.figure(figsize=(12, 6))
plt.imshow(corr_df.values.astype(float), vmin=0, vmax=1, cmap='viridis')
plt.title('Correlation (scaled 0–1) between Document Types and Top Terms')
plt.colorbar(label='Scaled Correlation')
plt.xticks(range(len(top_terms)), top_terms, rotation=90)
plt.yticks(range(len(model.classes_)), model.classes_)
plt.tight_layout()
plt.show()