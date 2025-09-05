import numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score

import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load embeddings
data = np.load("models/face_embeds.npz")
X, y = data["X"], data["y"]

X = np.asarray(X)
y = np.asarray(y)

# encode first
le0 = LabelEncoder()
y0 = le0.fit_transform(y)

# Keep only classes with >= MIN_PER_CLASS
MIN_PER_CLASS = 2
cnt = Counter(y0)
keep_mask = np.array([cnt[c] >= MIN_PER_CLASS for c in y0])
X = X[keep_mask]
y0 = y0[keep_mask]

# Re-encode to 0..K-1 after filtering
le = LabelEncoder()
y_enc = le.fit_transform(y0)

# Stratify and split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)
X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=42)

# Scale X
scaler = StandardScaler()
X_tr  = scaler.fit_transform(X_tr)
X_val = scaler.transform(X_val)
X_te  = scaler.transform(X_te)

# Use Logisic Regression for classification tasks
clf = LogisticRegression(
    max_iter=100, n_jobs=-1, solver='saga', verbose=1,  # <= print progress
    multi_class='multinomial'
)
# This model improved very very slow (almost no progress after 100 iterations)

clf.fit(X_tr, y_tr)
joblib.dump(clf, "models/emb_classifier.pkl")

n_classes   = len(clf.classes_)
labels_full = clf.classes_                  # full list of class ids the model was trained on

# Top-1
val_top1 = accuracy_score(y_val, clf.predict(X_val))

# Top-5
if hasattr(clf, "predict_proba"):
    proba_val = clf.predict_proba(X_val)    # shape (N, n_classes)
    val_top5  = top_k_accuracy_score(
        y_val, proba_val, k=min(5, n_classes), labels=labels_full
    )
else:
    scores = clf.decision_function(X_val)
    val_top5 = top_k_accuracy_score(
        y_val, scores, k=min(5, n_classes), labels=labels_full
    )

print(f"VAL  | top1={val_top1:.4f} | top5={val_top5:.4f}")

# Do the same for TEST:
test_top1 = accuracy_score(y_te, clf.predict(X_te))
if hasattr(clf, "predict_proba"):
    proba_te = clf.predict_proba(X_te)
    test_top5 = top_k_accuracy_score(y_te, proba_te, k=min(5, n_classes), labels=labels_full)
else:
    scores_te = clf.decision_function(X_te)
    test_top5 = top_k_accuracy_score(y_te, scores_te, k=min(5, n_classes), labels=labels_full)

print(f"TEST | top1={test_top1:.4f} | top5={test_top5:.4f}")

# Save artifacts
joblib.dump(scaler, "models/emb_scaler.pkl")
joblib.dump(clf,    "models/emb_classifier.pkl")
print("Saved: models/emb_scaler.pkl, models/emb_classifier.pkl")
