# CatSpy — Local Phishing & URL Detector (Prototype)

This repository contains a small, local prototype for phishing and URL detection. It pairs a local transformer-based classifier (DistilBERT via Hugging Face) with lightweight heuristic checks to produce a human-friendly JSON result for a simple web UI. It now also bundles a **completely local conversational large language model** endpoint so you can run a chatbot without calling external services once the weights are downloaded.

Contents
- backend: FastAPI server, phishing detector, local chat model, utils, and a small frontend bundle (served at `/index.html`).
- frontend: single-page HTML demo located at `backend/frontend/index.html`.

This README explains how to run the project locally, test the API, sample test inputs, and the quality/disclaimer statement.

---

## Quick start (Windows PowerShell)

1. Open PowerShell and change to the backend folder, then create the virtual environment and install dependencies (script included):

```powershell
cd backend
.\start.ps1 -CreateVenv
```

2. Start the server (if not already started by the script):

```powershell
.\start.ps1
# or (development)
# python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

3. Open the demo UI in your browser. The backend serves the demo page at the root:

- http://127.0.0.1:8000/

4. Try the local chatbot endpoint with curl (the first request may take a few seconds while the model downloads and warms up):

```powershell
curl -X POST "http://127.0.0.1:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
           "messages": [
             {"role": "system", "content": "你是一名乐于助人的网络安全助手。"},
             {"role": "user", "content": "你好，给我讲讲如何识别钓鱼邮件？"}
           ]
         }'
```

Alternatively, serve the frontend directory with a simple static server (optional):

```powershell
cd backend/frontend
python -m http.server 8080
# then open http://127.0.0.1:8080/index.html
```

---

## API (important endpoints)

- GET `/health` — health check
	- Response: `{"status":"ok"}`

- POST `/predict` — single-text prediction
	- Request JSON: `{ "text": "..." }`
	- Response JSON (fields):
		- result: "SAFE" | "SUSPICIOUS" | "PHISHING"
		- score: integer 0..100 (combined model probability + heuristic adjustments)
		- summary: short human-readable summary
		- heuristics: object with small detectors (domain_risk, urgency_language, creds_request, shortener_obfuscation, brand_spoof)
		- cues: list of discovered cues (type, text, details)
		- model: object with model name, predicted label, and raw scores

- POST `/train` — synchronous, small-batch training (for debugging only)
        - Request JSON: `{ "texts": ["..."], "labels": [0,1,...] }`
        - Response JSON: `{ "loss": float }`

- POST `/chat` — local conversational large language model
        - Request JSON: `{ "messages": [{"role":"system","content":"..."}, ...] }`
        - Response JSON: `{ "response": "assistant reply", "prompt_tokens": int, "generated_tokens": int, "model": "..." }`
        - Notes: Provide the full conversation history in chronological order. The first call will download the
          model (~350 MB) and can take a moment; subsequent calls are fully offline.

---

## Example curl tests

Health:
```bash
curl -X GET "http://127.0.0.1:8000/health"
```

Predict (safe):
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
	-H "Content-Type: application/json" \
	-d '{"text":"Hello friend, just checking in about our meeting next week."}'
```

Predict (phishing):
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
	-H "Content-Type: application/json" \
	-d '{"text":"URGENT: Verify your account now at http://bit.ly/abc123 and enter your password."}'
```

Train (small batch):
```bash
curl -X POST "http://127.0.0.1:8000/train" \
	-H "Content-Type: application/json" \
	-d '{"texts":["You won a prize! Click here","Hello friend"], "labels":[1,0]}'
```

You can import these curl commands into Postman (Import -> Raw Text) for interactive testing.

---

## Sample test texts (copy into the demo UI or use via curl)

SAFE example:
```
Hi Anna, just a reminder about our meeting next Tuesday at 10am.
```

PHISHING example:
```
URGENT: Your bank account will be suspended. Click http://bit.ly/verify123 and enter your password immediately.
```

Shortener / obfuscation example:
```
Action required: Click http://t.co/AbCd12 to confirm your account.
```

URL only:
```
http://192.168.0.1/login
```

---

## Implementation notes

- Model: the backend instantiates a local Hugging Face `DistilBertForSequenceClassification` model by default (`distilbert-base-uncased`).
- Chatbot: the `/chat` endpoint loads `microsoft/DialoGPT-medium` locally for lightweight dialogue. You can swap to another Hugging Face causal LM by editing `ChatConfig` in `backend/chat_model.py`.
- PEFT/LoRA: the code attempts to apply a PEFT/LoRA adapter. If your peft version or configuration requires `target_modules` the adapter may not be applied; the server catches this and continues running without PEFT (a warning is printed).
- Heuristics: lightweight rule-based detectors are in `backend/utils.py`. The `map_model_to_frontend` function combines the model's phishing probability with heuristic-derived weights to produce the frontend `score` and `result`.

Performance: DistilBERT is relatively small and runs on CPU, but initial model load and the first prediction may be slow. For production use, consider GPU or an optimized inference stack.

---

## Quality assurance / Warranty statement

This project is provided as a prototype demonstration. The following points summarize expected usage, limitations, and liability:

1. Functionality
	 - The system produces classification outputs using a local transformer model and additional heuristic checks. It is intended for prototyping and demonstration.

2. Scope of warranty
	 - The deliverable is validated as a working prototype on a developer machine. It is **not** a production-ready security appliance. It is not guaranteed to detect all phishing variants or to have acceptable false positive/negative rates for a production environment.

3. Testing and acceptance
	 - You should perform acceptance testing with representative datasets. Recommended criteria include acceptable false positive rate and recall for phishing samples in your environment.

4. Liability and disclaimer
	 - We disclaim any liability for damages or loss arising from use of this prototype. It should not be used as the sole control for critical security workflows.

5. Suggested steps to productionize
	 - Fine-tune the model on larger labeled datasets.
	 - Add model versioning, A/B testing, monitoring, and an async training pipeline.
	 - Add external signals (DNS/WHOIS, URL reputation, email header checks) to improve accuracy.

---

## Next steps I can implement for you (optional)

- Serve the entire frontend via FastAPI's static files (currently `/index.html` is served). I can register a `StaticFiles` route so assets are delivered under `/static/`.
- Add a `/model_info` endpoint that returns model name, parameter count, device (cpu/cuda), and whether PEFT was applied.
- Create a Postman Collection JSON for easy import.
- Add example files (`backend/test_samples.txt`, `backend/train_samples.jsonl`).

---

## LoRA (PEFT) fine-tuning guide

This repository includes a training example `backend/train_lora.py` that demonstrates fine-tuning DistilBERT with LoRA (PEFT). The script expects a JSONL file with one JSON object per line: `{ "text": "...", "label": 0 }`.

Basic usage (from the `backend` directory):

```powershell
python train_lora.py --train_file ../data/train_samples.jsonl --output_dir ./lora_out --epochs 3 --batch_size 8
```

Notes:
- The script will attempt to auto-detect LoRA `target_modules` for the model. You can override with `--target_modules q_lin,v_lin`.
- LoRA greatly reduces the number of trainable parameters and is suitable for quick adaptations on small datasets.
- For larger-scale production fine-tuning use a GPU, proper validation split, checkpointing, and consider using accelerate.


Tell me which of the above you want and I'll implement it and validate the change.

# CatSpy
CatSpy AI Security Detection Platform
