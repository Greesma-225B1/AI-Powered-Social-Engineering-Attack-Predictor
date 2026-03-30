import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords

# Download stopwords once
nltk.download("stopwords")

RAW_INPUT = r"D:\SocialEngineeringAIPoweredAttackPredictor\data\processed\combined_dataset.csv"
OUTPUT_PATH = r"D:\SocialEngineeringAIPoweredAttackPredictor\data\processed\cleaned_dataset.csv"

stop_words = set(stopwords.words("english"))

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove emails
    text = re.sub(r"\S+@\S+", "", text)

    # Remove phone numbers
    text = re.sub(r"\b\d{10,}\b", "", text)

    # Remove punctuation and numbers
    text = re.sub(r"[^a-z\s]", "", text)

    # Remove stopwords
    text = " ".join(word for word in text.split() if word not in stop_words)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ================== LOAD DATA ==================
df = pd.read_csv(RAW_INPUT)

# Safety check
print("Original dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# ================== CLEAN TEXT ==================
df["cleaned_text"] = df["text"].apply(clean_text)

# Remove empty rows after cleaning
df = df[df["cleaned_text"].str.len() > 0]

# ================== SAVE ==================
df.to_csv(OUTPUT_PATH, index=False)

print("\n✅ STEP 5 COMPLETED SUCCESSFULLY")
print("Cleaned dataset shape:", df.shape)
print("\nLabel distribution:")
print(df["label"].value_counts())
