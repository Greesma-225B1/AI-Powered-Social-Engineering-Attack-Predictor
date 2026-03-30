from app.gmail_labeler import (
    authenticate_gmail,
    create_label_if_not_exists,
    apply_label,
    get_unread_messages,
    get_email_body
)

service = authenticate_gmail()

safe_id = create_label_if_not_exists(service, "AI-Safe")
suspicious_id = create_label_if_not_exists(service, "AI-Suspicious")
attack_id = create_label_if_not_exists(service, "AI-Attack")

messages = get_unread_messages(service)

if not messages:
    print("No unread emails found.")
else:
    for msg in messages:
        message_id = msg['id']
        email_text = get_email_body(service, message_id)

        # TEMP MODEL LOGIC (Replace with your real BERT later)
        if any(word in email_text.lower() for word in ["urgent", "otp", "password"]):
            apply_label(service, message_id, attack_id)
            print("Marked as ATTACK")
        else:
            apply_label(service, message_id, safe_id)
            print("Marked as SAFE")