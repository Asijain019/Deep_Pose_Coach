import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# FIXED: Proper data loading
print("🔥 Loading pose features...")
data = np.load("data/pose_features.npz", allow_pickle=True)
X = data["X"]
y = data["y"]

print(f"📊 Loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"🏷️  Classes: {len(np.unique(y))}")
print(f"Classes: {np.unique(y)[:10]}...")

# FIXED: Robust train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

# OPTIMIZED SVM PIPELINE
pipeline = Pipeline([
    ("scaler", StandardScaler()),  # Critical for SVM
    ("svm", SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        probability=True,      # For confidence scores
        class_weight="balanced",  # Handle class imbalance
        random_state=42
    ))
])

# Train model
print("🚀 Training SVM...")
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print("\n📈 VALIDATION RESULTS:")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
print("\nDetailed Report:")
print(classification_report(y_val, y_pred))

# Cross-validation for robustness
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for train_idx, val_idx in cv.split(X, y):
    X_cv_train, X_cv_val = X[train_idx], X[val_idx]
    y_cv_train, y_cv_val = y[train_idx], y[val_idx]
    pipeline.fit(X_cv_train, y_cv_train)
    cv_scores.append(accuracy_score(y_cv_val, pipeline.predict(X_cv_val)))

print(f"\n🔍 5-Fold CV: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# FIXED: Ensure models directory
import os
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(pipeline, "models/svm_pose_classifier.pkl")
print("\n✅ Model saved: models/svm_pose_classifier.pkl")
