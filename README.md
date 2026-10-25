# ScamLens: AI Scam Investigator

ScamLens helps users and cybersecurity analysts assess suspicious messages and screenshots with explainable, multimodal scam detection powered by Google Gemini.

## Features

- Analyze pasted messages or uploaded screenshots.
- Return a 0–100 risk score with LOW, MEDIUM, HIGH, or CRITICAL severity.
- Highlight evidence and manipulation tactics such as urgency, impersonation, credential harvesting, and payment traps.
- Retry transient service failures and optionally use a fallback Gemini model.
- Report service outages and invalid API keys without presenting fabricated safety assessments.
- Recommend practical next steps, such as verifying through an official app or blocking a sender.
- Process uploaded content in memory; do not store message text or screenshots on disk.
- Never fetch or browse links contained in analyzed messages.

ScamLens provides probabilistic decision support, not legal or forensic conclusions. Never enter real passwords, recovery keys, or secret PINs into an AI system.

## Requirements

- Python 3.11 or later
- A Google Gemini API key

## Setup

From the project directory, create and activate a virtual environment in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Copy `.env.example` to `.env`, then set your API key and model preferences:

```powershell
Copy-Item .env.example .env
```

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
GEMINI_FALLBACK_MODEL=gemini-3.5-flash
GEMINI_MAX_RETRIES=3
```

Install dependencies and run the tests:

```powershell
pip install -r requirements.txt
python test_app.py
```

Launch the Streamlit application:

```powershell
streamlit run app.py
```

The app is available at `http://localhost:8501`. If PowerShell blocks virtual-environment activation, run commands with ` .\.venv\Scripts\python.exe` instead.

## Demo Scenarios

The sidebar includes synthetic examples for an urgent OTP scam, a fake delivery-fee message, and a routine appointment reminder. Do not test with real credentials or sensitive personal information.
# ScamLens-Threat