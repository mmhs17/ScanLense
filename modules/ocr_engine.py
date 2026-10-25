"""
ScamLens Screenshot and OCR Analysis Engine
Extracts visible text, URLs, sender names, payment demands, and urgency signals from screenshots.
Preserves raw OCR output as distinct evidence from AI interpretations.
"""

import re
import logging
from typing import List, Optional, Dict, Any
from PIL import Image, ImageStat
from pydantic import BaseModel, Field
from modules.url_threat_intel import extract_urls

logger = logging.getLogger("scamlens.ocr")

class OCRExtractedEvidence(BaseModel):
    raw_text: str = Field(description="Raw text extracted by optical character recognition")
    extracted_urls: List[str] = Field(default_factory=list, description="URLs or domains extracted from visible text")
    sender_indicators: List[str] = Field(default_factory=list, description="Identified sender names, headers, or shortcodes")
    payment_indicators: List[str] = Field(default_factory=list, description="Detected financial demands, fee amounts, or currency symbols")
    urgency_indicators: List[str] = Field(default_factory=list, description="Detected urgency words, countdowns, or threats")
    is_readable: bool = True
    ocr_engine: str = Field(description="OCR provider or method used (pytesseract, easyocr, or fallback_vision_heuristics)")
    notes: str = Field(description="Observations regarding image clarity, resolution, or legibility")

# Regex for key evidence in OCR text
SENDER_REGEX = re.compile(r'(?i)(?:from|sender|caller|id):\s*([a-zA-Z0-9_\-\.\+@\s]{3,30})')
PAYMENT_REGEX = re.compile(r'(?i)(?:pay|\$|₹|€|£|usd|inr|fee|charge|bill|amount|balance)\s*[:\$₹€£]?\s*(\d+(?:\.\d{2})?)')
URGENCY_REGEX = re.compile(r'(?i)\b(urgent|immediately|within \d+ (?:mins|minutes|hours)|suspend|blocked|locked|expire|action required|final notice)\b')

def extract_evidence_from_ocr_text(raw_text: str, engine_name: str, notes: str) -> OCRExtractedEvidence:
    """Parses raw text to identify structured cybersecurity signals."""
    clean_text = raw_text.strip()
    urls = extract_urls(clean_text)
    
    senders = [m.strip() for m in SENDER_REGEX.findall(clean_text) if len(m.strip()) > 2]
    
    # Also look for known institution names at start of message
    for brand in ["Chase", "USPS", "Wells Fargo", "Bank of America", "PayPal", "FedEx", "DHL", "Apple", "Amazon", "Netflix"]:
        if re.search(rf'(?i)\b{brand}\b', clean_text) and brand not in senders:
            senders.append(f"{brand} (Mentioned)")

    payments = []
    for match in PAYMENT_REGEX.finditer(clean_text):
        snippet = clean_text[max(0, match.start() - 5):min(len(clean_text), match.end() + 15)].strip()
        if snippet not in payments:
            payments.append(snippet)

    urgencies = list(set(URGENCY_REGEX.findall(clean_text)))

    return OCRExtractedEvidence(
        raw_text=clean_text,
        extracted_urls=urls,
        sender_indicators=senders,
        payment_indicators=payments,
        urgency_indicators=urgencies,
        is_readable=len(clean_text) > 10,
        ocr_engine=engine_name,
        notes=notes
    )

def perform_screenshot_ocr(image: Image.Image) -> OCRExtractedEvidence:
    """
    Performs OCR and signal extraction on screenshot image.
    Uses pytesseract if installed and configured, otherwise provides transparent image quality metrics.
    """
    if image is None:
        return OCRExtractedEvidence(
            raw_text="",
            is_readable=False,
            ocr_engine="none",
            notes="No image provided."
        )

    width, height = image.size
    is_low_res = width < 300 or height < 150

    # Test for blank/uniform image
    stat = ImageStat.Stat(image)
    variance = sum(stat.var) / len(stat.var) if stat.var else 0
    is_blank = variance < 10.0

    notes_list = [f"Resolution: {width}x{height}px"]
    if is_low_res:
        notes_list.append("Low resolution detected; fine text may have reduced legibility.")
    if is_blank:
        notes_list.append("Image appears nearly uniform or blank.")

    # Try pytesseract if available
    try:
        import pytesseract
        raw_text = pytesseract.image_to_string(image)
        if raw_text and len(raw_text.strip()) > 5:
            notes_list.append("Successfully extracted visible text via Tesseract OCR engine.")
            return extract_evidence_from_ocr_text(raw_text, "Tesseract-OCR", "; ".join(notes_list))
    except Exception as e:
        logger.debug("Local Tesseract OCR unavailable: %s", str(e))

    # Transparent fallback when local Tesseract binary is not installed on system PATH
    notes_list.append("Local Tesseract binary not detected; raw text extraction deferred to Gemini Multimodal Vision API.")
    return OCRExtractedEvidence(
        raw_text="",
        extracted_urls=[],
        sender_indicators=[],
        payment_indicators=[],
        urgency_indicators=[],
        is_readable=not is_blank,
        ocr_engine="Gemini-Multimodal-Vision-Delegate",
        notes="; ".join(notes_list)
    )
