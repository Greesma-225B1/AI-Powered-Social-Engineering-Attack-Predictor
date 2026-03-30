import os
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

# ================= PATHS =================
DATA_PATH = r"D:\SocialEngineeringAIPoweredAttackPredictor\data\processed\cleaned_dataset.csv"
MODEL_DL_PATH = r"D:\SocialEngineeringAIPoweredAttackPredictor\models\models_dl"

os.makedirs(MODEL_DL_PATH, exist_ok=True)

# ================= LOAD DATA =================
df = pd.read_csv(DATA_PATH)

df["label"] = df["label"].str.strip().str.lower()

label_map = {
    "legit": 0,
    "ham": 0,
    "phishing": 1,
    "spam": 1,
    "smishing": 1
}

df["label_id"] = df["label"].map(label_map)
df = df.dropna(subset=["label_id"])
df["label_id"] = df["label_id"].astype(int)

texts = df["cleaned_text"].values
labels = df["label_id"].values

print("Dataset size:", len(texts))
print("Label distribution:", np.bincount(labels))

# ================= TOKENIZATION =================
MAX_WORDS = 10000
MAX_LEN = 150

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN)

# Save tokenizer
with open(os.path.join(MODEL_DL_PATH, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# ================= LSTM MODEL =================
lstm_model = Sequential([
    Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

lstm_model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

print("\n🚀 Training LSTM model...")
lstm_model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=3,
    batch_size=64
)

lstm_model.save(os.path.join(MODEL_DL_PATH, "lstm_model.h5"))

# ================= GRU MODEL =================
gru_model = Sequential([
    Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
    GRU(64, return_sequences=False),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

gru_model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

print("\n🚀 Training GRU model...")
gru_model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=3,
    batch_size=64
)

gru_model.save(os.path.join(MODEL_DL_PATH, "gru_model.h5"))

print("\n✅ STEP 6C COMPLETED SUCCESSFULLY")
print("Models saved in:", MODEL_DL_PATH)
