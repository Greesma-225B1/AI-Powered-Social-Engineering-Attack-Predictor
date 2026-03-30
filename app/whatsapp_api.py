from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)
CORS(app)  # 🔥 allows Chrome extension to connect

# ==============================
# LOAD BERT MODEL
# ==============================
MODEL_PATH = "bert_model"   # adjust if needed

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("✅ BERT Model Loaded")

# ==============================
# ANALYZE API
# ==============================
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"label": "Safe"})

    text_lower = text.lower()

    # ==============================
    # 🔥 RULE-BASED OVERRIDE (FOR DEMO)
    # ==============================
    if any(word in text_lower for word in ["otp", "urgent", "password", "bank", "verify", "account"]):
        print("⚠️ Rule-based Attack detected")
        return jsonify({"label": "Attack"})

    # ==============================
    # 🤖 BERT MODEL PREDICTION
    # ==============================
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

    # ==============================
    # LABEL MAPPING
    # ==============================
    if predicted_class == 0:
        label = "Safe"
    elif predicted_class == 1:
        label = "Suspicious"
    else:
        label = "Attack"

    print(f"🤖 BERT Prediction: {label}")

    return jsonify({"label": label})


# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    app.run(port=5000, debug=True)