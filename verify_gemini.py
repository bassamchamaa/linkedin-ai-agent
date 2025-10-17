import os, requests, json

API_KEY = os.getenv("GEMINI_API_KEY")
assert API_KEY, "Set GEMINI_API_KEY in env"

def list_models():
    r = requests.get("https://generativelanguage.googleapis.com/v1/models",
                     headers={"x-goog-api-key": API_KEY}, timeout=20)
    r.raise_for_status()
    names = [m["name"] for m in r.json().get("models", [])]
    print("Models visible:", names[:10])
    return names

def gen(model):
    url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={API_KEY}"
    body = {
        "contents":[{"role":"user","parts":[{"text":"Give one concrete, non-hype tip for B2B partnerships (2 sentences)."}]}],
        "generationConfig":{"maxOutputTokens": 1200, "temperature": 0.6}
    }
    r = requests.post(url, json=body, timeout=30, headers={"Content-Type": "application/json"})
    print("Status:", r.status_code)
    print("Body preview:", r.text[:500])
    r.raise_for_status()
    data = r.json()
    cand = data.get("candidates", [{}])[0]
    content = cand.get("content", {})
    parts = content.get("parts", [])
    text = "\n".join([p.get("text","") for p in parts if isinstance(p, dict)])
    print("\n=== TEXT ===\n", text or "(no text)", "\n============\n")
    print("finishReason:", cand.get("finishReason"))

if __name__ == "__main__":
    names = list_models()
    for m in ["gemini-2.5-flash", "gemini-1.5-flash-8b"]:
        if f"models/{m}" in names or m in names:
            print("\n--- Trying", m, "---")
            gen(m)
