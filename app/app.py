import streamlit as st
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Social Engineering Attack Predictor",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Social Engineering Attack Predictor")

# ================= MODEL PATH =================
MODEL_PATH = r"D:\SocialEngineeringAIPoweredAttackPredictor\bert_model"

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ================= PREPROCESS FUNCTION =================
def preprocess_by_mode(text, mode):
    text = text.lower().strip()

    if mode == "SMS":
        text = re.sub(r"http\S+", " link ", text)
        text = re.sub(r"\d+", " number ", text)

    elif mode == "Email":
        text = re.sub(r"from:.*", "", text)
        text = re.sub(r"subject:.*", "", text)
        text = re.sub(r"http\S+", " link ", text)

    elif mode == "Chat":
        text = re.sub(r"@[A-Za-z0-9_]+", " user ", text)
        text = re.sub(r"http\S+", " link ", text)

    return text


# ================= GMAIL SECTION =================
st.divider()
st.subheader("📧 Check New Gmail Messages")

if st.button("🔄 Fetch Unread Emails"):

    from gmail_reader import fetch_unread_emails
    emails = fetch_unread_emails()

    if not emails:
        st.info("No new unread emails found.")
    else:
        for mail in emails:

            processed_text = preprocess_by_mode(mail, "Email")

            inputs = tokenizer(
                processed_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(probs, dim=1).item()
                confidence = probs[0][prediction].item()

            label = "🚨 PHISHING / ATTACK" if prediction == 1 else "✅ LEGIT"

            st.write("---")
            st.write(f"📨 Email Result: {label}")
            st.write(f"Confidence: {confidence * 100:.2f}%")

            if prediction == 1 and confidence > 0.90:
                st.error("⚠️ HIGH RISK: Strong social engineering indicators detected.")

                from alert_service import send_email_alert
                send_email_alert(mail, confidence * 100)

            elif prediction == 1:
                st.warning("⚠️ MEDIUM RISK: Suspicious message.")
            else:
                st.success("✅ LOW RISK: Message appears safe.")

st.markdown("---")
st.caption("Final Year Major Project | AI-Powered Social Engineering Attack Predictor")