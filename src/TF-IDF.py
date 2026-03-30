import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ================== PATHS ==================
DATA_PATH = r"D:\SocialEngineeringAIPoweredAttackPredictor\data\processed\cleaned_dataset.csv"
MODEL_PATH = r"D:\SocialEngineeringAIPoweredAttackPredictor\models"

os.makedirs(MODEL_PATH, exist_ok=True)

# ================== LOAD DATA ==================
df = pd.read_csv(DATA_PATH)

X = df["cleaned_text"]
y = df["label"]

print("Dataset size:", df.shape)
print("Label distribution:\n", y.value_counts())

# ================== TRAIN-TEST SPLIT ==================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================== TF-IDF VECTORIZATION ==================
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Save vectorizer
with open(os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf, f)

# ================== MODELS ==================
models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "svm_model": LinearSVC(),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

results = []

for name, model in models.items():
    print(f"\n🚀 Training {name}...")
    
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save model
    with open(os.path.join(MODEL_PATH, f"{name}.pkl"), "wb") as f:
        pickle.dump(model, f)

    results.append({
        "model": name,
        "accuracy": acc
    })

# ================== SAVE COMPARISON ==================
results_df = pd.DataFrame(results)
results_df.to_csv(
    r"D:\SocialEngineeringAIPoweredAttackPredictor\data\processed\model_comparison.csv",
    index=False
)

print("\n✅ STEP 6A COMPLETED SUCCESSFULLY")
print(results_df)
