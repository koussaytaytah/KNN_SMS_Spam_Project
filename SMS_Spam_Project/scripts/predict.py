import joblib
import os

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "../models/knn_spam_model.joblib"

# 👇 PUT YOUR SMS HERE TO TEST
SMS_TEXT = "Congratulations! You won a free ticket. Reply YES to claim."

# ===============================
# LOAD MODEL
# ===============================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

model, vectorizer = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully")

# ===============================
# VECTORIZE SMS
# ===============================
sms_vector = vectorizer.transform([SMS_TEXT])

# ===============================
# PREDICT
# ===============================
prediction = model.predict(sms_vector)[0]

label = "Spam" if prediction==1 else "Ham"
print(f"\n📩 SMS Prediction: {label}")
