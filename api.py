# api.py
import os
import re
import joblib
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # --- IMPORT THIS ---
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="PhishDetect API")

# --- ADD THIS CORS MIDDLEWARE SECTION ---
# This allows your Chrome extension (running on mail.google.com)
# to make requests to this local server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://mail.google.com"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
# -----------------------------------------

MODEL_DIR = os.environ.get("MODEL_DIR", "models")

# Global vars
tfidf_model = None
transformer = None
tokenizer = None
use_transformer = False

# Load TF-IDF model
if os.path.exists(os.path.join(MODEL_DIR, "tfidf_lr.joblib")):
    try:
        tfidf_model = joblib.load(os.path.join(MODEL_DIR, "tfidf_lr.joblib"))
        print("✅ Loaded TF-IDF model.")
    except Exception as e:
        print(f"⚠️ TF-IDF load fail: {e}")

# Load transformer model
if os.path.isdir(os.path.join(MODEL_DIR, "transformer_model")):
    try:
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "transformer_model"))
        transformer = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(MODEL_DIR, "transformer_model")
        )
        transformer.eval()
        use_transformer = True
        print("✅ Loaded transformer model.")
    except Exception as e:
        print(f"⚠️ Transformer load fail: {e}")

class InputEmail(BaseModel):
    text: str

# Heuristics & evidence
def inspect_links(text: str):
    return re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', text)

def sender_personalization(text: str):
    m = re.search(r"Dear\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", text)
    return bool(m)

def urgency_words(text: str):
    words = ["verify now", "immediately", "urgent", "account will be locked", "action required", "limited time"]
    found = [w for w in words if w in text.lower()]
    return found

def categorize_phish(text: str):
    text_lower = text.lower()
    if any(w in text_lower for w in ["invoice", "payment", "charge", "billing"]):
        return "Payment/Financial phishing"
    elif any(w in text_lower for w in ["account suspended", "verify your account", "login attempt"]):
        return "Account takeover attempt"
    elif any(w in text_lower for w in ["reward", "congratulations", "claim now", "prize"]):
        return "Fake offer/reward"
    else:
        return "General phishing"

@app.post("/predict")
def predict(inp: InputEmail):
    text = inp.text or ""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    links = inspect_links(text)
    personalized = sender_personalization(text)
    urgency = urgency_words(text)
    category = categorize_phish(text)

    # Model inference
    if use_transformer:
        inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=256)
        with torch.no_grad():
            logits = transformer(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            phish_prob = float(probs[1])
    elif tfidf_model:
        proba = tfidf_model.predict_proba([text])[0]
        phish_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
    else:
        raise HTTPException(status_code=500, detail="No model available")

    label = int(phish_prob >= 0.5)

    evidence = {
        "links_found": links,
        "personalized_greeting": personalized,
        "urgency_words": urgency
    }

    return {
        "prob": phish_prob,
        "label": label,
        "category": category,
        "evidence": evidence
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(use_transformer or tfidf_model)}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)