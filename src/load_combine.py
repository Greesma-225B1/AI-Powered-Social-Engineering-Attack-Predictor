import pandas as pd
import os

RAW_PATH = r"D:\SocialEngineeringAIPoweredAttackPredictor\data\raw"
PROCESSED_PATH = r"D:\SocialEngineeringAIPoweredAttackPredictor\data\processed"

os.makedirs(PROCESSED_PATH, exist_ok=True)

# ================== 1. SMS Spam Collection ==================
sms_spam = pd.read_csv(
    os.path.join(RAW_PATH, "SpamCollection"),
    sep="\t",
    header=None,
    names=["label", "text"]
)

sms_spam["label"] = sms_spam["label"].map({
    "ham": "legit",
    "spam": "phishing"
})

sms_spam = sms_spam[["text", "label"]]

# ================== 2. SMS Phishing Dataset ==================
sms_phishing = pd.read_csv(
    os.path.join(RAW_PATH, "Phishing.csv")
)

sms_phishing = sms_phishing.rename(columns={
    "TEXT": "text",
    "LABEL": "label"
})

sms_phishing["label"] = sms_phishing["label"].str.lower()
sms_phishing = sms_phishing[["text", "label"]]

# ================== 3. Email Spam Dataset ==================
email_data = pd.read_csv(
    os.path.join(RAW_PATH, "EmailsLegitAndSpam.csv")
)

email_data = email_data.rename(columns={
    "mensaje": "text",
    "tipo": "label"
})

email_data["label"] = email_data["label"].str.lower()
email_data = email_data[["text", "label"]]

# ================== COMBINE ALL DATASETS ==================
combined_df = pd.concat(
    [sms_spam, sms_phishing, email_data],
    ignore_index=True
)

combined_df.dropna(inplace=True)

# ================== SAVE ==================
combined_df.to_csv(
    os.path.join(PROCESSED_PATH, "combined_dataset.csv"),
    index=False
)

print("✅ STEP 4 COMPLETED SUCCESSFULLY")
print("Total samples:", combined_df.shape[0])
print("\nLabel distribution:")
print(combined_df["label"].value_counts())
