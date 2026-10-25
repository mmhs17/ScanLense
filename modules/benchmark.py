"""
ScamLens Reproducible Benchmark and Evaluation Harness
Evaluates phishing, scam, and benign detection pipelines across diverse categories.
Computes Precision, Recall, F1, False-Positive Rate, and Confusion Matrix.
"""

from typing import List, Dict, Any, Tuple
from pydantic import BaseModel
from modules.ml_classifier import SpecialistClassifier
from modules.url_features import analyze_url_features
from modules.url_threat_intel import extract_urls, ThreatIntelManager
from modules.evidence_fusion import fuse_investigation_evidence

class BenchmarkSample(BaseModel):
    id: str
    category: str
    text: str
    ground_truth_label: str  # "SCAM" or "BENIGN"
    dataset_source: str = "Synthetic-Bench-v2"

# Curated benchmark dataset representing the 7 required categories
BENCHMARK_DATASET: List[BenchmarkSample] = [
    # 1. Legitimate Messages (BENIGN)
    BenchmarkSample(
        id="legit-1",
        category="Legitimate",
        text="Your dentist appointment with Dr. Aris is confirmed for Tuesday at 9:00 AM. Please arrive 10 minutes early. Reply STOP to cancel.",
        ground_truth_label="BENIGN"
    ),
    BenchmarkSample(
        id="legit-2",
        category="Legitimate",
        text="Hi Dad, let me know when you get home so we can order dinner.",
        ground_truth_label="BENIGN"
    ),
    BenchmarkSample(
        id="legit-3",
        category="Legitimate",
        text="Your pharmacy prescription #98213 is ready for pickup at CVS on 4th Street.",
        ground_truth_label="BENIGN"
    ),

    # 2. Phishing Messages (SCAM)
    BenchmarkSample(
        id="phish-1",
        category="Phishing",
        text="URGENT: Your Wells Fargo account has been locked due to irregular login. Verify your credentials immediately at http://wellsfargo.com.security-login-auth.xyz/verify",
        ground_truth_label="SCAM"
    ),
    BenchmarkSample(
        id="phish-2",
        category="Phishing",
        text="Google Security Alert: Unauthorized device accessed your Gmail. Confirm your password now at http://192.168.1.120:8080/google-login",
        ground_truth_label="SCAM"
    ),

    # 3. Fake Delivery Notices (SCAM)
    BenchmarkSample(
        id="delivery-1",
        category="Fake Delivery",
        text="USPS Notice: Your package is on hold due to an unpaid dispatch fee of $1.99. Pay immediately at http://usps-track-parcel-fee.top/pay to avoid return to sender.",
        ground_truth_label="SCAM"
    ),
    BenchmarkSample(
        id="delivery-2",
        category="Fake Delivery",
        text="FedEx Delivery: Incomplete delivery address for tracking #99482. Update details at http://bit.ly/fedex-address-update22",
        ground_truth_label="SCAM"
    ),

    # 4. Payment Scams (SCAM)
    BenchmarkSample(
        id="payment-1",
        category="Payment Scam",
        text="Congratulations! You have received a $2,500 refund via Zelle. To claim your transfer, submit your 6-digit banking OTP to our agent.",
        ground_truth_label="SCAM"
    ),
    BenchmarkSample(
        id="payment-2",
        category="Payment Scam",
        text="IRS Final Tax Notice: Outstanding balance of $850. Settle via Bitcoin or Target gift cards within 1 hour to prevent police dispatch.",
        ground_truth_label="SCAM"
    ),

    # 5. Job Scams (SCAM)
    BenchmarkSample(
        id="job-1",
        category="Job Scam",
        text="Remote Data Entry Job! Earn $450/day working 2 hours from home. Deposit an upfront $150 training fee to get started. Contact via WhatsApp +234-802-9912.",
        ground_truth_label="SCAM"
    ),

    # 6. Brand Impersonation (SCAM)
    BenchmarkSample(
        id="brand-1",
        category="Brand Impersonation",
        text="Netflix Membership: Your auto-billing payment failed. Update your card at http://netflix-member-renew.cc to restore streaming.",
        ground_truth_label="SCAM"
    ),

    # 7. Promotional Spam (BENIGN / LOW-RISK SPAM)
    BenchmarkSample(
        id="spam-1",
        category="Promotional Spam",
        text="Pizza Palace: 20% off all large pizzas this weekend only with code PIZZA20. Order online at https://pizzapalace.com. Reply STOP to opt out.",
        ground_truth_label="BENIGN"
    ),
    BenchmarkSample(
        id="spam-2",
        category="Promotional Spam",
        text="Exclusive Macy's summer fashion clearance. Save up to 50% on select shoes today. Valid in store and online.",
        ground_truth_label="BENIGN"
    ),
]

class EvaluationMetrics(BaseModel):
    total_samples: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    per_category: Dict[str, Dict[str, Any]] = {}

def calculate_metrics(y_true: List[str], y_pred: List[str], categories: List[str]) -> EvaluationMetrics:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "SCAM" and yp == "SCAM")
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "BENIGN" and yp == "SCAM")
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "BENIGN" and yp == "BENIGN")
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "SCAM" and yp == "BENIGN")

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1 = round(2 * (precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0
    fpr = round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0

    # Per-category metrics
    per_cat: Dict[str, Dict[str, Any]] = {}
    unique_cats = set(categories)
    for cat in unique_cats:
        c_true = [yt for yt, c in zip(y_true, categories) if c == cat]
        c_pred = [yp for yp, c in zip(y_pred, categories) if c == cat]
        c_tp = sum(1 for yt, yp in zip(c_true, c_pred) if yt == "SCAM" and yp == "SCAM")
        c_fp = sum(1 for yt, yp in zip(c_true, c_pred) if yt == "BENIGN" and yp == "SCAM")
        c_tn = sum(1 for yt, yp in zip(c_true, c_pred) if yt == "BENIGN" and yp == "BENIGN")
        c_fn = sum(1 for yt, yp in zip(c_true, c_pred) if yt == "SCAM" and yp == "BENIGN")
        c_acc = round((c_tp + c_tn) / len(c_true), 4) if c_true else 0.0
        per_cat[cat] = {"total": len(c_true), "accuracy": c_acc, "tp": c_tp, "fp": c_fp, "tn": c_tn, "fn": c_fn}

    return EvaluationMetrics(
        total_samples=len(y_true),
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1_score=f1,
        false_positive_rate=fpr,
        per_category=per_cat
    )

def evaluate_pipeline(use_evidence_fusion: bool = True) -> EvaluationMetrics:
    """
    Evaluates the detection pipeline on the benchmark dataset.
    - use_evidence_fusion=False: Baseline NLP pattern classifier only
    - use_evidence_fusion=True: ScamLens Fusion Engine (Classifier + Static URL Features + Threat Intel)
    """
    classifier = SpecialistClassifier()
    threat_manager = ThreatIntelManager(providers=[])  # Offline provider in benchmarks to avoid external rate limits

    y_true: List[str] = []
    y_pred: List[str] = []
    categories: List[str] = []

    for sample in BENCHMARK_DATASET:
        y_true.append(sample.ground_truth_label)
        categories.append(sample.category)

        # Baseline: Classifier only
        cls_result = classifier.classify(sample.text)

        if not use_evidence_fusion:
            pred_label = "SCAM" if cls_result.is_phishing else "BENIGN"
        else:
            # Full Fusion pipeline
            urls = extract_urls(sample.text)
            url_reports = [analyze_url_features(u) for u in urls]
            threat_reports = threat_manager.check_urls(urls)

            fused = fuse_investigation_evidence(
                gemini_result=None,
                classifier_result=cls_result,
                url_threat_reports=threat_reports,
                url_feature_reports=url_reports,
                ocr_evidence=None
            )
            pred_label = "SCAM" if fused.overall_risk_score >= 50 else "BENIGN"

        y_pred.append(pred_label)

    return calculate_metrics(y_true, y_pred, categories)

if __name__ == "__main__":
    print("Running ScamLens Pipeline Benchmark...")
    baseline_metrics = evaluate_pipeline(use_evidence_fusion=False)
    upgraded_metrics = evaluate_pipeline(use_evidence_fusion=True)

    print("\n--- BASELINE (Local Pattern Classifier Only) ---")
    print(f"Precision: {baseline_metrics.precision} | Recall: {baseline_metrics.recall} | F1: {baseline_metrics.f1_score} | FPR: {baseline_metrics.false_positive_rate}")

    print("\n--- UPGRADED SCAMLENS (Evidence Fusion + URL Engineering) ---")
    print(f"Precision: {upgraded_metrics.precision} | Recall: {upgraded_metrics.recall} | F1: {upgraded_metrics.f1_score} | FPR: {upgraded_metrics.false_positive_rate}")
