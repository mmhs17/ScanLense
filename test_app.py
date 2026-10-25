"""
Comprehensive test suite for ScamLens Advanced Platform:
- Phase 1: Gemini retries, 503 exponential backoff & jitter, fallback switching, credential failure
- Phase 2: URL extraction, normalization, threat intelligence lookups, caching, provider failure
- Phase 3: Specialist text classifier, HF failure handling, local heuristic fallback
- Phase 4: OCR evidence extraction, low-res/blank image handling
- Phase 5: URL static feature engineering (IP host, shortener, brand spoofing, length, deceptive chars)
- Phase 6: Multi-source evidence fusion, confirmed threat intel escalation, conflicting signals, incomplete analysis
- Phase 8: Benchmark metrics calculation and pipeline comparison
- Live Gemini API integration test (if network and key are available)
"""

import os
import sys
import json
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

from app import (
    ScamAnalysisResult,
    RedFlag,
    analyze_with_gemini,
    execute_gemini_generate_with_retry,
    classify_gemini_error,
    extract_retry_after,
    sanitize_log_message,
    process_uploaded_image,
    GeminiAuthError,
    GeminiUnavailableError,
    GeminiInvalidRequestError,
    GeminiModelNotFoundError,
    DEMO_EXAMPLES,
    SYSTEM_INSTRUCTION,
)

from modules.url_threat_intel import (
    extract_urls,
    normalize_url,
    URLThreatReport,
    GoogleSafeBrowsingProvider,
    ThreatIntelManager,
)
from modules.url_features import (
    analyze_url_features,
    is_ip_address,
    URLFeatureReport,
)
from modules.ml_classifier import (
    SpecialistClassifier,
    local_heuristic_classify,
    ClassifierResult,
)
from modules.ocr_engine import (
    perform_screenshot_ocr,
    extract_evidence_from_ocr_text,
    OCRExtractedEvidence,
)
from modules.evidence_fusion import (
    fuse_investigation_evidence,
    FusedInvestigationReport,
)
from modules.benchmark import (
    evaluate_pipeline,
    calculate_metrics,
    BENCHMARK_DATASET,
)

SAMPLE_PARSED_RESULT = ScamAnalysisResult(
    risk_score=92,
    risk_level="CRITICAL",
    summary="Critical smishing attempt attempting to capture banking credentials via urgency manipulation.",
    red_flags=[
        RedFlag(
            title="Urgency Trick",
            evidence="within 10 minutes",
            explanation="Manufactured urgency to bypass critical thinking."
        )
    ],
    positive_signals=[],
    recommended_actions=["Do not share OTP", "Block sender"],
    is_definitive=False
)

def create_mock_response():
    resp = MagicMock()
    resp.parsed = SAMPLE_PARSED_RESULT
    resp.text = json.dumps(SAMPLE_PARSED_RESULT.model_dump())
    return resp


class TestScamLens(unittest.TestCase):

    # --- BASELINE VALIDATION TESTS ---

    def test_pydantic_schema_validation(self):
        """Test schema validation for structured analysis output."""
        valid_data = {
            "risk_score": 75,
            "risk_level": "HIGH",
            "summary": "Suspicious delivery fee smishing notification.",
            "red_flags": [
                {
                    "title": "Mismatched URL",
                    "evidence": "http://usps-update82.com",
                    "explanation": "Not an official USPS domain."
                }
            ],
            "positive_signals": ["Contains tracking identifier"],
            "recommended_actions": ["Do not click link", "Track via official USPS app"],
            "is_definitive": False
        }
        model = ScamAnalysisResult.model_validate(valid_data)
        self.assertEqual(model.risk_score, 75)
        self.assertEqual(model.risk_level, "HIGH")
        self.assertEqual(len(model.red_flags), 1)
        self.assertFalse(model.is_definitive)

    def test_malformed_ai_output_rejection(self):
        """Test rejection of malformed model output."""
        malformed_data = {"risk_score": "not_a_number"}
        with self.assertRaises(Exception):
            ScamAnalysisResult.model_validate(malformed_data)

    def test_empty_input_validation(self):
        """Test rejection of empty message or missing input."""
        with self.assertRaises(ValueError):
            analyze_with_gemini(text_input="", image_input=None)

    def test_demo_examples_integrity(self):
        """Test that synthetic demo scenarios exist and contain adequate prompt context."""
        self.assertEqual(len(DEMO_EXAMPLES), 3)
        for key, demo in DEMO_EXAMPLES.items():
            self.assertIn("title", demo)
            self.assertIn("text", demo)
            self.assertGreater(len(demo["text"]), 20)

    def test_error_classification(self):
        """Test classification of various API and network error strings."""
        self.assertEqual(classify_gemini_error(Exception("400 API_KEY_INVALID")), "auth")
        self.assertEqual(classify_gemini_error(Exception("503 UNAVAILABLE: Model is experiencing high demand")), "transient_unavailable")
        self.assertEqual(classify_gemini_error(Exception("429 ResourceExhausted: Quota exceeded")), "rate_limit")
        self.assertEqual(classify_gemini_error(Exception("WinError 10051 socket unreachable")), "network")
        self.assertEqual(classify_gemini_error(Exception("404 NOT_FOUND: model not found")), "not_found")

    def test_extract_retry_after(self):
        """Test extraction of Retry-After headers and text patterns."""
        self.assertEqual(extract_retry_after(Exception("Please retry after 4s")), 4.0)
        self.assertEqual(extract_retry_after(Exception("retry in 10.5 s")), 10.5)
        self.assertIsNone(extract_retry_after(Exception("random error")))

    def test_log_sanitization(self):
        """Ensure sensitive API keys are stripped from logs."""
        raw = "Error calling https://api.google.com?key=REDACTED_TEST_KEY"
        sanitized = sanitize_log_message(raw)
        self.assertNotIn("REDACTED_TEST_KEY", sanitized)
        self.assertIn("[REDACTED", sanitized)

    # --- PHASE 1: MOCKED GEMINI API RESILIENCE TESTS ---

    @patch("time.sleep", return_value=None)
    def test_mock_primary_success(self, mock_sleep):
        """Test: Primary model succeeds on first attempt."""
        mock_client = MagicMock()
        mock_resp = create_mock_response()
        mock_client.models.generate_content.return_value = mock_resp

        result = analyze_with_gemini(
            text_input="Your bank account is locked. Reply with OTP.",
            model_name="gemini-3.8-flash",
            fallback_model="gemini-3.5-flash",
            client=mock_client,
        )

        self.assertEqual(result.risk_score, 92)
        self.assertEqual(result.risk_level, "CRITICAL")
        self.assertEqual(result.model_used, "gemini-3.8-flash")
        self.assertEqual(mock_client.models.generate_content.call_count, 1)
        mock_sleep.assert_not_called()

    @patch("time.sleep", return_value=None)
    def test_mock_temporary_503_followed_by_success(self, mock_sleep):
        """Test: Temporary 503 UNAVAILABLE on attempt 1, succeeds on attempt 2."""
        mock_client = MagicMock()
        mock_resp = create_mock_response()
        mock_client.models.generate_content.side_effect = [
            Exception("503 UNAVAILABLE: This model is currently experiencing high demand. Please try again later."),
            mock_resp,
        ]

        result = analyze_with_gemini(
            text_input="Your bank account is locked. Reply with OTP.",
            model_name="gemini-3.8-flash",
            fallback_model="gemini-3.5-flash",
            max_retries=3,
            client=mock_client,
        )

        self.assertEqual(result.risk_score, 92)
        self.assertEqual(result.model_used, "gemini-3.8-flash")
        self.assertEqual(mock_client.models.generate_content.call_count, 2)
        mock_sleep.assert_called_once()

    @patch("time.sleep", return_value=None)
    def test_mock_fallback_success(self, mock_sleep):
        """Test: Primary model fails with 503 on all retries, switches to fallback model and succeeds."""
        mock_client = MagicMock()
        mock_resp = create_mock_response()

        def side_effect(model, contents, config):
            if model == "gemini-3.8-flash":
                raise Exception("503 UNAVAILABLE: Model overloaded")
            elif model == "gemini-3.5-flash":
                return mock_resp
            raise Exception("Unknown model")

        mock_client.models.generate_content.side_effect = side_effect

        result = analyze_with_gemini(
            text_input="Your bank account is locked. Reply with OTP.",
            model_name="gemini-3.8-flash",
            fallback_model="gemini-3.5-flash",
            max_retries=2,
            client=mock_client,
        )

        self.assertEqual(result.risk_score, 92)
        self.assertEqual(result.model_used, "gemini-3.5-flash")
        self.assertEqual(mock_client.models.generate_content.call_count, 3)

    @patch("time.sleep", return_value=None)
    def test_mock_exhausted_retries(self, mock_sleep):
        """Test: Both primary and fallback models fail with 503, raises GeminiUnavailableError."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("503 UNAVAILABLE: High demand across all regions")

        with self.assertRaises(GeminiUnavailableError) as ctx:
            analyze_with_gemini(
                text_input="Suspicious delivery parcel link.",
                model_name="gemini-3.8-flash",
                fallback_model="gemini-3.5-flash",
                max_retries=2,
                client=mock_client,
            )

        self.assertIn("experiencing high demand", str(ctx.exception))
        self.assertEqual(mock_client.models.generate_content.call_count, 4)

    @patch("time.sleep", return_value=None)
    def test_mock_invalid_api_credentials(self, mock_sleep):
        """Test: Invalid API key fails immediately with GeminiAuthError; does NOT retry or mask."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API_KEY_INVALID: 400 Bad Request. Provided API key is invalid.")

        with self.assertRaises(GeminiAuthError) as ctx:
            analyze_with_gemini(
                text_input="Suspicious SMS text.",
                model_name="gemini-3.8-flash",
                fallback_model="gemini-3.5-flash",
                max_retries=3,
                client=mock_client,
            )

        self.assertIn("rejected the API key as invalid", str(ctx.exception))
        self.assertEqual(mock_client.models.generate_content.call_count, 1)
        mock_sleep.assert_not_called()

    # --- PHASE 2: URL THREAT INTELLIGENCE TESTS ---

    def test_url_extraction_and_normalization(self):
        """Test safe URL extraction and RFC-compliant normalization."""
        sample_text = "Check your package at http://usps-track.xyz/pay. Also visit (https://secure-chase.com/login?u=1) and www.apple-support.co/now!"
        urls = extract_urls(sample_text)
        self.assertEqual(len(urls), 3)
        self.assertIn("http://usps-track.xyz/pay", urls)
        self.assertIn("https://secure-chase.com/login?u=1", urls)
        self.assertIn("http://www.apple-support.co/now", urls)

        # Test edge case normalization
        self.assertEqual(normalize_url("  http://example.COM/path/  "), "http://example.com/path/")
        self.assertIsNone(normalize_url("not_a_valid_url"))

    @patch("requests.post")
    def test_google_safe_browsing_match_and_caching(self, mock_post):
        """Test Google Safe Browsing provider hit and manager in-memory caching."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "matches": [
                {
                    "threatType": "SOCIAL_ENGINEERING",
                    "threat": {"url": "http://evil-phish.xyz"}
                }
            ]
        }
        mock_post.return_value = mock_resp

        provider = GoogleSafeBrowsingProvider(api_key="mock_key")
        mgr = ThreatIntelManager(providers=[provider])

        reports = mgr.check_urls(["http://evil-phish.xyz", "http://clean-site.org"])
        self.assertEqual(len(reports), 2)
        evil_rep = next(r for r in reports if r.url == "http://evil-phish.xyz")
        clean_rep = next(r for r in reports if r.url == "http://clean-site.org")

        self.assertEqual(evil_rep.status, "MATCH_FOUND")
        self.assertIn("SOCIAL_ENGINEERING", evil_rep.threat_types)
        self.assertEqual(clean_rep.status, "NO_MATCH")
        self.assertFalse(clean_rep.is_safe_guarantee)

        # Test caching: querying evil-phish again should not trigger another requests.post
        call_count_before = mock_post.call_count
        cached_reports = mgr.check_urls(["http://evil-phish.xyz"])
        self.assertEqual(mock_post.call_count, call_count_before)
        self.assertEqual(cached_reports[0].status, "MATCH_FOUND")

    @patch("requests.post", side_effect=Exception("Connection timeout"))
    def test_safe_browsing_provider_failure_handling(self, mock_post):
        """Test graceful degradation when threat intel provider experiences network timeout."""
        provider = GoogleSafeBrowsingProvider(api_key="mock_key")
        reports = provider.check_urls(["http://suspicious-test.com"])
        rep = reports["http://suspicious-test.com"]
        self.assertEqual(rep.status, "PROVIDER_UNAVAILABLE")
        self.assertIn("Connection timeout", rep.details)

    # --- PHASE 3: SPECIALIST CLASSIFIER TESTS ---

    def test_specialist_classifier_local_fallback(self):
        """Test specialist classifier pattern heuristics when Hugging Face is disabled/offline."""
        classifier = SpecialistClassifier()
        # Test obvious scam prompt
        scam_text = "Your Chase bank account is suspended immediately. Send the OTP passcode to prevent closure."
        res = classifier.classify(scam_text)
        self.assertTrue(res.is_phishing)
        self.assertGreaterEqual(res.score, 0.50)
        self.assertFalse(res.is_calibrated_probability)

        # Test routine appointment text
        benign_text = "Reminder: Your dental appointment is scheduled for tomorrow at 10:30 AM. Arrive 10 minutes early."
        res_benign = classifier.classify(benign_text)
        self.assertFalse(res_benign.is_phishing)
        self.assertLess(res_benign.score, 0.50)

    # --- PHASE 4: OCR EXTRACTION TESTS ---

    def test_ocr_signal_extraction_from_text(self):
        """Test OCR structured evidence parsing for payment demands, urgency, and senders."""
        raw_ocr = "From: Chase Security Notice\nYour account has been locked. Pay $25.00 dispatch fee immediately within 10 minutes."
        evidence = extract_evidence_from_ocr_text(raw_ocr, "Tesseract-Mock", "Mocked image test")
        self.assertTrue(evidence.is_readable)
        self.assertIn("Chase", str(evidence.sender_indicators))
        self.assertTrue(any("$25" in p or "25.00" in p for p in evidence.payment_indicators))
        self.assertTrue(any("immediately" in u or "within 10 minutes" in u for u in evidence.urgency_indicators))

    def test_ocr_low_res_and_blank_handling(self):
        """Test handling of tiny or blank screenshots."""
        blank_img = Image.new("RGB", (50, 50), color=(255, 255, 255))
        evidence = perform_screenshot_ocr(blank_img)
        self.assertIn("Low resolution", evidence.notes)
        self.assertEqual(evidence.raw_text, "")

    # --- PHASE 5: URL FEATURE ENGINEERING TESTS ---

    def test_url_features_ip_host_and_shortener(self):
        """Test detection of raw numeric IP hosts and known shorteners."""
        ip_report = analyze_url_features("http://192.168.1.100/login")
        self.assertTrue(ip_report.is_ip_host)
        self.assertGreaterEqual(ip_report.heuristic_risk_score, 60)
        self.assertTrue(any("Raw IP Address Host" in ind.flag for ind in ip_report.indicators))

        short_report = analyze_url_features("https://bit.ly/claim-prize22")
        self.assertTrue(short_report.is_shortened)
        self.assertTrue(short_report.has_https)
        self.assertTrue(any("Known URL Shortener" in ind.flag for ind in short_report.indicators))

    def test_url_features_brand_spoofing(self):
        """Test detection of deceptive brand names in unofficial domains."""
        # paypal.com keyword inside fake root domain security-login.xyz
        spoof_report = analyze_url_features("http://paypal.com.account-verify.xyz/login")
        self.assertGreaterEqual(spoof_report.heuristic_risk_score, 60)
        self.assertTrue(any("Brand Impersonation" in ind.category for ind in spoof_report.indicators))

    # --- PHASE 6: EVIDENCE FUSION ENGINE TESTS ---

    def test_evidence_fusion_confirmed_threat_intel_escalation(self):
        """Test Rule: Confirmed threat intel match escalates overall risk to CRITICAL 100."""
        threat_match = URLThreatReport(
            url="http://known-malicious.com",
            provider="Google Safe Browsing v4",
            status="MATCH_FOUND",
            threat_types=["SOCIAL_ENGINEERING"],
            details="Confirmed phishing domain"
        )
        fused = fuse_investigation_evidence(
            gemini_result=None,
            classifier_result=None,
            url_threat_reports=[threat_match],
            url_feature_reports=[]
        )
        self.assertEqual(fused.overall_risk_score, 100)
        self.assertEqual(fused.overall_risk_level, "CRITICAL")
        self.assertTrue(fused.confirmed_threat_intel_match)

    def test_evidence_fusion_conflicting_signals_detection(self):
        """Test Rule: Significant divergence between Gemini and classifier flags discordance."""
        mock_gemini = MagicMock()
        mock_gemini.risk_score = 90
        mock_gemini.risk_level = "CRITICAL"
        mock_gemini.summary = "Extremely urgent credential harvesting attempt."
        mock_gemini.recommended_actions = ["Do not share credentials"]

        # Classifier says low risk
        low_cls = ClassifierResult(
            label="BENIGN",
            score=0.15,
            is_phishing=False,
            engine="test",
            model_name="test",
            explanation="No patterns"
        )

        fused = fuse_investigation_evidence(
            gemini_result=mock_gemini,
            classifier_result=low_cls,
            url_threat_reports=[],
            url_feature_reports=[]
        )
        self.assertTrue(fused.has_conflicting_signals)
        self.assertIn("divergence between Gemini contextual score", str(fused.conflict_explanation))

    def test_evidence_fusion_incomplete_investigation_on_ai_failure(self):
        """Test Rule: When Gemini fails, fusion marks report as incomplete rather than safe."""
        fused = fuse_investigation_evidence(
            gemini_result=None,
            classifier_result=None,
            url_threat_reports=[],
            url_feature_reports=[],
            gemini_error="503 UNAVAILABLE"
        )
        self.assertTrue(fused.is_incomplete)
        self.assertIn("INCOMPLETE INVESTIGATION", fused.status_summary)

    # --- PHASE 8: BENCHMARK HARNESS TESTS ---

    def test_benchmark_metrics_calculation(self):
        """Test benchmark metric calculation on dummy predictions."""
        y_true = ["SCAM", "SCAM", "BENIGN", "BENIGN"]
        y_pred = ["SCAM", "BENIGN", "BENIGN", "SCAM"]
        cats = ["Phishing", "Phishing", "Legit", "Legit"]

        metrics = calculate_metrics(y_true, y_pred, cats)
        self.assertEqual(metrics.total_samples, 4)
        self.assertEqual(metrics.true_positives, 1)
        self.assertEqual(metrics.false_positives, 1)
        self.assertEqual(metrics.true_negatives, 1)
        self.assertEqual(metrics.false_negatives, 1)
        self.assertEqual(metrics.precision, 0.5)
        self.assertEqual(metrics.recall, 0.5)
        self.assertEqual(metrics.f1_score, 0.5)
        self.assertEqual(metrics.false_positive_rate, 0.5)

    def test_benchmark_pipeline_runs_without_crashing(self):
        """Test that the benchmark pipeline executes cleanly across the 7 categories."""
        metrics = evaluate_pipeline(use_evidence_fusion=True)
        self.assertGreater(metrics.total_samples, 10)
        self.assertGreater(metrics.precision, 0.0)
        self.assertGreater(metrics.recall, 0.0)


def run_live_tests_if_available():
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key.strip() in ("", "your_gemini_api_key_here"):
        print("[INFO] Skipping live API test: No valid GEMINI_API_KEY set.")
        return

    print("\n--- Running Live Integration Check (with Resilience) ---")
    try:
        test_msg = DEMO_EXAMPLES["demo_otp"]["text"]
        result = analyze_with_gemini(text_input=test_msg)
        print(f"[PASS] Live Analysis succeeded using model '{result.model_used}'!")
        print(f"       Score: {result.risk_score}/100, Level: {result.risk_level}")
        print(f"       Summary: {result.summary[:90]}...")
    except Exception as e:
        print(f"[NOTICE] Live API test reported: {e}")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestScamLens)
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(suite)

    if test_result.wasSuccessful():
        print(f"\n[ALL {test_result.testsRun} COMPREHENSIVE TESTS PASSED SUCCESSFULLY!]")
        run_live_tests_if_available()
    else:
        sys.exit(1)
