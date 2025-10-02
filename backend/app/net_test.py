import requests
import os
from dotenv import load_dotenv
load_dotenv()  


HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
MODEL = "tabularisai/multilingual-sentiment-analysis"

def map_to_binary_label(preds: list[dict]) -> str:
    """Map 5-class prediction to binary ('positive'/'negative') with neutral fallback."""
    if not preds:
        return "unknown"

    # Sort predictions by score descending
    preds = sorted(preds, key=lambda x: x["score"], reverse=True)

    # If top is neutral, use the next best one
    top = preds[0]
    if top["label"].lower() == "neutral":
        for alt in preds[1:]:
            if alt["label"].lower() != "neutral":
                top = alt
                break

    label = top["label"].lower()
    if "positive" in label:
        return "positive"
    elif "negative" in label:
        return "negative"
    else:
        return "unknown"

def classify_text(text: str) -> str:
    """Run sentiment classification and map to binary label."""
    payload = {"inputs": text.strip()}
    try:
        resp = requests.post(
            f"https://api-inference.huggingface.co/models/{MODEL}",
            headers=HEADERS,
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        # HF returns a list of lists
        if isinstance(data, list) and isinstance(data[0], list):
            preds = data[0]
        elif isinstance(data, list):
            preds = data
        else:
            return "unknown"

        return map_to_binary_label(preds)

    except Exception as e:
        print(f"[ERROR] HF API failed: {e}")
        return "unknown"

# 🔍 TESTING
texts = [
    "Eli Lilly is great",
    "Eli Lilly is terrible",
    "Eli Lilly is okay"
]

for t in texts:
    label = classify_text(t)
    print(f"{t!r} → {label}")
