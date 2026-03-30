import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
from app.gmail_labeler import (
    authenticate_gmail,
    create_label_if_not_exists,
    apply_label,
    get_unread_messages,
    get_email_body
)

# Authenticate Gmail
service = authenticate_gmail()

# Load BERT model
MODEL_PATH = "bert_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Create labels
safe_id = create_label_if_not_exists(service, "🟢 AI-Safe")
suspicious_id = create_label_if_not_exists(service, "🟡 AI-Suspicious")
attack_id = create_label_if_not_exists(service, "🔴 AI-Attack")

print("Monitoring Gmail...")

while True:
    messages = get_unread_messages(service)

    if messages:
        for msg in messages:
            message_id = msg['id']
            email_text = get_email_body(service, message_id)

            print("------ NEW EMAIL ------")
            print("EMAIL TEXT:", email_text[:200])

            # Tokenize
            inputs = tokenizer(
                email_text,
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

            # Adjust this mapping based on your training labels
            if predicted_class == 0:
                apply_label(service, message_id, safe_id)
                print("Marked as SAFE")

            elif predicted_class == 1:
                apply_label(service, message_id, suspicious_id)
                print("Marked as SUSPICIOUS")

            else:
                apply_label(service, message_id, attack_id)
                print("Marked as ATTACK")

    time.sleep(30)  # check every 30 seconds