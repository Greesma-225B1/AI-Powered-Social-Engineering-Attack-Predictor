import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

# ================= PATHS =================
DATA_PATH = r"D:\SocialEngineeringAIPoweredAttackPredictor\data\processed\cleaned_dataset.csv"
MODEL_SAVE_PATH = r"D:\SocialEngineeringAIPoweredAttackPredictor\bert_model"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# ================= LOAD DATA =================
df = pd.read_csv(DATA_PATH)

# ---------------- CLEAN LABELS ----------------
df["label"] = df["label"].str.strip().str.lower()

# Map all possible variants
label_map = {
    "legit": 0,
    "ham": 0,
    "phishing": 1,
    "spam": 1,
    "smishing": 1
}

df["label_id"] = df["label"].map(label_map)

# DROP rows with invalid labels
df = df.dropna(subset=["label_id"])

df["label_id"] = df["label_id"].astype(int)

print("Clean dataset shape:", df.shape)
print("Label distribution:\n", df["label_id"].value_counts())

# ================= TRAIN TEST SPLIT =================
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["cleaned_text"].tolist(),
    df["label_id"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label_id"]
)

# ================= TOKENIZER =================
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_dataset = Dataset.from_dict({
    "text": train_texts,
    "label": train_labels
})

test_dataset = Dataset.from_dict({
    "text": test_texts,
    "label": test_labels
})

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# ================= MODEL =================
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# ================= TRAINING ARGS =================

training_args = TrainingArguments(
    output_dir="./bert_output",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./bert_logs",
    report_to="none"
)


# ================= TRAINER =================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

# ================= TRAIN =================
#trainer.train()
trainer.train(resume_from_checkpoint=True)

# ================= SAVE =================
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

print("\n✅ STEP 6B COMPLETED SUCCESSFULLY")
print(f"BERT model saved at: {MODEL_SAVE_PATH}")
