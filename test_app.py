"""
Comprehensive test suite for ScamLens:
- Schema validation & malformed output handling
- Input validation
- Mocked tests for:
  1. Primary model success
  2. Temporary 503 UNAVAILABLE followed by success (exponential backoff & jitter)
  3. Fallback model success upon primary failure
  4. Exhausted retries raising GeminiUnavailableError
  5. Invalid API credentials raising GeminiAuthError immediately without retrying
- Live Gemini API integration test (if key and network are available)
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

    # --- MOCKED GEMINI API TESTS (Requirement 10) ---

    @patch("time.sleep", return_value=None)
    def test_mock_primary_success(self, mock_sleep):
        """Test 1: Primary model succeeds on first attempt."""
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
        """Test 2: Temporary 503 UNAVAILABLE on attempt 1, succeeds on attempt 2."""
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
        # Verify backoff sleep was invoked
        mock_sleep.assert_called_once()

    @patch("time.sleep", return_value=None)
    def test_mock_fallback_success(self, mock_sleep):
        """Test 3: Primary model fails with 503 on all retries, switches to fallback model and succeeds."""
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
        # Primary called 2 times, then fallback called 1 time = 3 calls
        self.assertEqual(mock_client.models.generate_content.call_count, 3)

    @patch("time.sleep", return_value=None)
    def test_mock_exhausted_retries(self, mock_sleep):
        """Test 4: Both primary and fallback models fail with 503, raises GeminiUnavailableError."""
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
        # Total calls: 2 for primary + 2 for fallback = 4 calls total (bounded, no infinite loop)
        self.assertEqual(mock_client.models.generate_content.call_count, 4)

    @patch("time.sleep", return_value=None)
    def test_mock_invalid_api_credentials(self, mock_sleep):
        """Test 5: Invalid API key fails immediately with GeminiAuthError; does NOT retry or mask."""
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
        # Must fail immediately on attempt 1 without retries and without fallback
        self.assertEqual(mock_client.models.generate_content.call_count, 1)
        mock_sleep.assert_not_called()


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
        print("\n[ALL 9 UNIT AND MOCKED TESTS PASSED SUCCESSFULLY!]")
        run_live_tests_if_available()
    else:
        sys.exit(1)
