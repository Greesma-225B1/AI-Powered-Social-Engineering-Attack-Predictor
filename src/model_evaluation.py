import os
import pickle
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ================= PATHS =================
DATA_PATH = r"D:\SocialEngineeringAIPoweredAttackPredictor\data\processed\cleaned_dataset.csv"

MODEL_DIR = r"D:\SocialEngineeringAIPoweredAttackPredictor\models"
DL_MODEL_DIR = r"D:\SocialEngineeringAIPoweredAttackPredictor\models\models_dl"
BERT_MODEL_DIR = r"D:\SocialEngineeringAIPoweredAttackPredictor\bert_model"

OUTPUT_PATH = r"D:\SocialEngineeringAIPoweredAttackPredictor\data\processed\model_comparison_final.csv"

# ================= LOAD DATA =================
df = pd.read_csv(DATA_PATH)

X = df["cleaned_text"].tolist()

# 🔑 FORCE LABELS TO NUMERIC (THIS IS CRITICAL)
label_map = {
    "legit": 0,
    "ham": 0,
    "phishing": 1,
    "spam": 1,
    "smishing": 1
}

y = df["label"].str.lower().map(label_map).astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

results = []

# ================= METRIC HELPER =================
def evaluate_model(name, y_true, y_pred):
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0)
    })

# ================= TF-IDF + ML MODELS =================
print("🔹 Evaluating TF-IDF + ML models...")

vectorizer = pickle.load(open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb"))
X_test_tfidf = vectorizer.transform(X_test)

for model_name in ["logistic_regression", "svm_model", "random_forest"]:
    model = pickle.load(open(os.path.join(MODEL_DIR, f"{model_name}.pkl"), "rb"))
    preds = model.predict(X_test_tfidf)

    # 🔥 ABSOLUTE FIX: convert string predictions → numeric
    if preds.dtype == object:
        preds = np.array([
            0 if p in ["legit", "ham"] else 1
            for p in preds
        ])

    evaluate_model(model_name.upper(), y_test, preds)

# ================= LSTM & GRU =================
print("🔹 Evaluating LSTM & GRU...")

tokenizer = pickle.load(open(os.path.join(DL_MODEL_DIR, "tokenizer.pkl"), "rb"))

X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_pad = pad_sequences(X_test_seq, maxlen=200)

lstm_model = load_model(os.path.join(DL_MODEL_DIR, "lstm_model.h5"))
gru_model = load_model(os.path.join(DL_MODEL_DIR, "gru_model.h5"))

lstm_preds = (lstm_model.predict(X_test_pad) > 0.5).astype(int).ravel()
gru_preds = (gru_model.predict(X_test_pad) > 0.5).astype(int).ravel()

evaluate_model("LSTM", y_test, lstm_preds)
evaluate_model("GRU", y_test, gru_preds)

# ================= BERT =================
print("🔹 Evaluating BERT...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_bert = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
bert_model.to(device)
bert_model.eval()

bert_preds = []

with torch.no_grad():
    for text in X_test:
        inputs = tokenizer_bert(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = bert_model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        bert_preds.append(pred)

evaluate_model("BERT", y_test, bert_preds)

# ================= SAVE RESULTS =================
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)

print("\n✅ STEP 7 COMPLETED SUCCESSFULLY")
print(results_df)
print(f"\n📁 Results saved at: {OUTPUT_PATH}")
