"""
ScamLens Specialist Text Classifier
Supports Hugging Face Transformers (e.g. Horizon-Labs/phishing-scam-detector-base) when installed and enabled,
with an intelligent local NLP heuristic fallback when transformers/torch is unavailable.
Never crashes application on model loading or missing dependency.
"""

import os
import re
import logging
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

logger = logging.getLogger("scamlens.ml_classifier")

class ClassifierResult(BaseModel):
    label: str = Field(description="Predicted class label (e.g., PHISHING, SPAM, BENIGN)")
    score: float = Field(ge=0.0, le=1.0, description="Model raw confidence score (not a calibrated probability)")
    is_phishing: bool = Field(description="Normalized boolean flag indicating elevated scam/phishing risk")
    engine: str = Field(description="Identifier of classification engine used (transformers or local_heuristic)")
    model_name: str = Field(description="Name of model or heuristic profile")
    is_calibrated_probability: bool = Field(default=False, description="Always False. Raw classification score is not statistically calibrated.")
    explanation: str = Field(description="Summary of classifier rationale")

# High-risk trigger phrases for local heuristic classifier
URGENCY_PATTERNS = re.compile(r'(?i)\b(10 minutes|immediately|urgent|within \d+ hours|act now|suspended|locked|blocked|freeze|expir(?:e|ing)|action required)\b')
CREDENTIAL_PATTERNS = re.compile(r'(?i)\b(otp|one-time password|passcode|pin|security code|verify your identity|bank account|cvv|credentials|ssn|social security)\b')
PAYMENT_PATTERNS = re.compile(r'(?i)\b(fee|dispatch fee|delivery fee|unpaid|pay \$\d+|pay ₹\d+|settle payment|bitcoin|crypto|gift card)\b')
IMPERSONATION_PATTERNS = re.compile(r'(?i)\b(chase|usps|wells fargo|bank of america|paypal|netflix|fedex|irs|apple support|amazon security)\b')
BENIGN_INDICATORS = re.compile(r'(?i)\b(appointment|dental|checkup|arrive 10 minutes early|opt out|reply stop|scheduled for tomorrow|meeting reminder)\b')

def local_heuristic_classify(text: str) -> ClassifierResult:
    """
    High-speed, dependency-free local NLP classifier used when Hugging Face / Torch is unavailable.
    Evaluates multi-attribute social engineering features.
    """
    clean_text = text or ""
    
    urgency_hits = len(URGENCY_PATTERNS.findall(clean_text))
    credential_hits = len(CREDENTIAL_PATTERNS.findall(clean_text))
    payment_hits = len(PAYMENT_PATTERNS.findall(clean_text))
    impersonation_hits = len(IMPERSONATION_PATTERNS.findall(clean_text))
    benign_hits = len(BENIGN_INDICATORS.findall(clean_text))

    # Calculate heuristic score
    risk_points = 0
    rationales = []

    if credential_hits > 0:
        risk_points += 45
        rationales.append("Harvesting sensitive credentials or passcodes")
    if urgency_hits > 0:
        risk_points += 25
        rationales.append("Artificial urgency and pressure triggers")
    if payment_hits > 0:
        risk_points += 20
        rationales.append("Unsolicited fee or payment demand")
    if impersonation_hits > 0:
        risk_points += 15
        rationales.append("Brand or financial institution mention")
    if benign_hits > 0 and credential_hits == 0:
        risk_points -= 40
        rationales.append("Routine calendar/appointment confirmation language")

    raw_score = max(0.05, min(0.99, risk_points / 100.0))
    is_phishing = raw_score >= 0.50
    label = "PHISHING_SCAM" if is_phishing else "BENIGN_NOTIFICATION"

    explanation = "; ".join(rationales) if rationales else "Neutral pattern without prominent phishing markers"

    return ClassifierResult(
        label=label,
        score=round(raw_score, 3),
        is_phishing=is_phishing,
        engine="local_pattern_heuristic",
        model_name="ScamLens-NLP-Heuristic-v2",
        explanation=explanation
    )

class SpecialistClassifier:
    """
    Manages specialist phishing classification with Transformers support and graceful fallback.
    """
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("SCAM_CLASSIFIER_MODEL", "Horizon-Labs/phishing-scam-detector-base")
        self.pipeline = None
        self.is_loaded = False
        self._attempted_load = False

    def _lazy_load_pipeline(self):
        if self._attempted_load:
            return
        self._attempted_load = True

        enable_hf = os.getenv("ENABLE_HF_CLASSIFIER", "false").lower() in ("true", "1", "yes")
        if not enable_hf:
            logger.info("HF Classifier disabled via ENABLE_HF_CLASSIFIER=false. Using local heuristic classifier.")
            return

        try:
            from transformers import pipeline
            logger.info("Attempting to load specialist classifier: %s", self.model_name)
            self.pipeline = pipeline("text-classification", model=self.model_name, truncation=True, max_length=512)
            self.is_loaded = True
            logger.info("Specialist classifier loaded successfully.")
        except Exception as e:
            logger.warning("Could not load Hugging Face classifier '%s': %s. Falling back to local pattern heuristic.", self.model_name, str(e))
            self.pipeline = None
            self.is_loaded = False

    def classify(self, text: str) -> ClassifierResult:
        if not text or not text.strip():
            return ClassifierResult(
                label="EMPTY_INPUT",
                score=0.0,
                is_phishing=False,
                engine="none",
                model_name="none",
                explanation="No text provided for classification"
            )

        self._lazy_load_pipeline()

        if self.is_loaded and self.pipeline is not None:
            try:
                raw_out = self.pipeline(text[:512])
                if raw_out and isinstance(raw_out, list) and len(raw_out) > 0:
                    top_pred = raw_out[0]
                    raw_label = str(top_pred.get("label", "")).upper()
                    raw_score = float(top_pred.get("score", 0.5))
                    is_phishing = any(k in raw_label for k in ["PHISH", "SCAM", "MALICIOUS", "LABEL_1", "SPAM"])
                    return ClassifierResult(
                        label=raw_label,
                        score=round(raw_score, 3),
                        is_phishing=is_phishing,
                        engine="transformers_pipeline",
                        model_name=self.model_name,
                        explanation=f"Deep learning classification from {self.model_name} (Class: {raw_label}, Score: {raw_score:.3f})"
                    )
            except Exception as e:
                logger.warning("HF pipeline inference failed: %s. Using local pattern fallback.", str(e))

        return local_heuristic_classify(text)
