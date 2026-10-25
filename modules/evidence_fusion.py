"""
ScamLens Evidence Fusion Engine
Synthesizes multi-source cybersecurity signals into a transparent, explainable investigation report.
Integrates Gemini multimodal analysis, specialist classifier output, URL static features,
threat intelligence reputation, and OCR observations using clear, documented rules.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from modules.url_threat_intel import URLThreatReport
from modules.url_features import URLFeatureReport
from modules.ml_classifier import ClassifierResult
from modules.ocr_engine import OCRExtractedEvidence

class FusedSignal(BaseModel):
    source: str = Field(description="Originating engine (Gemini, ThreatIntel, URLFeatures, Classifier, OCR)")
    indicator_title: str
    finding: str
    risk_impact: str = Field(description="ESCALATE, ELEVATE, MITIGATE, NEUTRAL")
    rationale: str

class FusedInvestigationReport(BaseModel):
    overall_risk_score: int = Field(ge=0, le=100)
    overall_risk_level: str = Field(description="LOW, MEDIUM, HIGH, CRITICAL")
    is_definitive: bool = False
    is_incomplete: bool = Field(default=False, description="True if a primary engine like Gemini was unavailable")
    status_summary: str
    confirmed_threat_intel_match: bool = False
    has_conflicting_signals: bool = False
    conflict_explanation: Optional[str] = None
    fused_signals: List[FusedSignal] = Field(default_factory=list)
    actionable_recommendations: List[str] = Field(default_factory=list)
    signals_breakdown: Dict[str, Any] = Field(default_factory=dict)

def fuse_investigation_evidence(
    gemini_result: Optional[Any],
    classifier_result: Optional[ClassifierResult],
    url_threat_reports: List[URLThreatReport],
    url_feature_reports: List[URLFeatureReport],
    ocr_evidence: Optional[OCRExtractedEvidence] = None,
    gemini_error: Optional[str] = None,
) -> FusedInvestigationReport:
    """
    Applies documented cybersecurity fusion rules across all available inspection channels.
    Does NOT blindly average numbers; evaluates verified threat registries and corroborating heuristics.
    """
    fused_signals: List[FusedSignal] = []
    recommendations: List[str] = []
    confirmed_match = False
    is_incomplete = False

    # 1. Evaluate External Threat Intelligence (Verified source)
    threat_intel_high_risk = False
    for rep in url_threat_reports:
        if rep.status == "MATCH_FOUND":
            confirmed_match = True
            threat_intel_high_risk = True
            fused_signals.append(FusedSignal(
                source="Threat Intelligence Registry",
                indicator_title=f"Confirmed Malicious Match: {rep.url}",
                finding=", ".join(rep.threat_types),
                risk_impact="ESCALATE",
                rationale=f"Verified match from {rep.provider}. URL is known malicious infrastructure."
            ))
            recommendations.append(f"CRITICAL: Do not navigate to {rep.url}. This link is listed on verified threat databases.")
        elif rep.status == "PROVIDER_UNAVAILABLE":
            fused_signals.append(FusedSignal(
                source="Threat Intelligence Registry",
                indicator_title=f"Provider Unavailable: {rep.url}",
                finding="Reputation registry lookup timed out or was unreachable.",
                risk_impact="NEUTRAL",
                rationale="Provider disruption treated as UNKNOWN, not clean."
            ))

    # 2. Evaluate URL Feature Heuristics
    max_url_score = 0
    for u_rep in url_feature_reports:
        max_url_score = max(max_url_score, u_rep.heuristic_risk_score)
        for ind in u_rep.indicators:
            if ind.severity in ("HIGH", "CRITICAL"):
                fused_signals.append(FusedSignal(
                    source="URL Feature Analyzer",
                    indicator_title=f"URL Risk ({u_rep.domain}): {ind.flag}",
                    finding=ind.evidence,
                    risk_impact="ELEVATE" if ind.severity == "HIGH" else "ESCALATE",
                    rationale=ind.explanation
                ))
                recommendations.append(f"Inspect link destination: '{u_rep.domain}' appears deceptive or obfuscated.")

    # 3. Evaluate Specialist Classifier
    classifier_flagged = False
    if classifier_result:
        classifier_flagged = classifier_result.is_phishing
        impact = "ELEVATE" if classifier_result.is_phishing else "MITIGATE"
        fused_signals.append(FusedSignal(
            source=f"Classifier ({classifier_result.model_name})",
            indicator_title=f"NLP Pattern Assessment: {classifier_result.label}",
            finding=f"Confidence metric: {classifier_result.score:.2f} (uncalibrated score)",
            risk_impact=impact,
            rationale=classifier_result.explanation
        ))

    # 4. Evaluate OCR observations
    if ocr_evidence and ocr_evidence.raw_text:
        if ocr_evidence.payment_indicators:
            fused_signals.append(FusedSignal(
                source="OCR Vision Engine",
                indicator_title="Payment/Fee Demands in Image",
                finding=", ".join(ocr_evidence.payment_indicators),
                risk_impact="ELEVATE",
                rationale="Visual text contains upfront payment requests or currency symbols."
            ))
        if ocr_evidence.urgency_indicators:
            fused_signals.append(FusedSignal(
                source="OCR Vision Engine",
                indicator_title="Urgency Indicators in Image",
                finding=", ".join(ocr_evidence.urgency_indicators),
                risk_impact="ELEVATE",
                rationale="Image text exerts artificial time pressure on the user."
            ))

    # 5. Evaluate Gemini Multimodal Analysis
    gemini_score = None
    if gemini_result and hasattr(gemini_result, "risk_score"):
        gemini_score = gemini_result.risk_score
        fused_signals.append(FusedSignal(
            source=f"Gemini Multimodal ({getattr(gemini_result, 'model_used', 'AI')})",
            indicator_title=f"AI Risk Assessment: {gemini_result.risk_level} ({gemini_score}/100)",
            finding=gemini_result.summary,
            risk_impact="ESCALATE" if gemini_score >= 80 else ("ELEVATE" if gemini_score >= 50 else "MITIGATE"),
            rationale="Contextual multimodal evaluation of social engineering, sender identity, and coercion."
        ))
        if hasattr(gemini_result, "recommended_actions"):
            for act in gemini_result.recommended_actions:
                if act not in recommendations:
                    recommendations.append(act)
    else:
        is_incomplete = True
        err_note = gemini_error or "Gemini API unavailable"
        fused_signals.append(FusedSignal(
            source="Gemini Multimodal Engine",
            indicator_title="AI Analysis Incomplete",
            finding=f"Service unavailable: {err_note}",
            risk_impact="NEUTRAL",
            rationale="Gemini was unreachable. Fusion engine evaluated remaining local and threat intelligence signals."
        ))
        recommendations.append("AI reasoning incomplete due to service disruption. Verify communication directly through official channels.")

    # --- FUSION DECISION LOGIC ---
    has_conflicting_signals = False
    conflict_explanation = None

    # Rule 1: Confirmed Threat Intel Match -> Immediate 100 CRITICAL
    if confirmed_match:
        final_score = 100
        final_level = "CRITICAL"
        status_summary = "CONFIRMED THREAT: Destination link is flagged on external verified threat intelligence registries."

    # Rule 2: Both Gemini and local engines available
    elif gemini_score is not None:
        # Check for signal discordance
        classifier_score_scaled = int(classifier_result.score * 100) if classifier_result else gemini_score
        score_diff = abs(gemini_score - classifier_score_scaled)
        
        if score_diff >= 40:
            has_conflicting_signals = True
            conflict_explanation = (
                f"Noticeable divergence between Gemini contextual score ({gemini_score}/100) "
                f"and specialist classifier ({classifier_score_scaled}/100). "
                "Gemini assesses nuanced context and intent, whereas the classifier evaluates direct lexical syntax."
            )

        # Corroborate scores: if URL features indicate critical spoofing, ensure score reflects it
        if max_url_score >= 75 and gemini_score < 70:
            final_score = max(gemini_score, max_url_score)
            final_level = "HIGH" if final_score < 80 else "CRITICAL"
            status_summary = "ELEVATED RISK: Static URL features identified deceptive domain structures not fully captured by text alone."
        else:
            final_score = gemini_score
            final_level = gemini_result.risk_level.upper()
            status_summary = gemini_result.summary

    # Rule 3: Gemini was unavailable (Incomplete Investigation)
    else:
        # Compute fallback baseline from local modules
        local_scores = [max_url_score]
        if classifier_result:
            local_scores.append(int(classifier_result.score * 100))
        final_score = max(local_scores) if local_scores else 50
        
        if final_score >= 80:
            final_level = "CRITICAL"
        elif final_score >= 50:
            final_level = "HIGH"
        elif final_score >= 25:
            final_level = "MEDIUM"
        else:
            final_level = "LOW"
            
        status_summary = "INCOMPLETE INVESTIGATION: Primary AI reasoning was unavailable. Risk estimate generated strictly from URL features and local NLP rules."

    # Deduplicate recommendations
    unique_recommendations = []
    for r in recommendations:
        if r not in unique_recommendations:
            unique_recommendations.append(r)

    if not unique_recommendations:
        unique_recommendations = [
            "Do not click unverified links or download attachments.",
            "Contact the purported institution directly via their official app or website.",
            "Never share passwords, OTP codes, or personal PINs."
        ]

    return FusedInvestigationReport(
        overall_risk_score=final_score,
        overall_risk_level=final_level,
        is_definitive=False,
        is_incomplete=is_incomplete,
        status_summary=status_summary,
        confirmed_threat_intel_match=confirmed_match,
        has_conflicting_signals=has_conflicting_signals,
        conflict_explanation=conflict_explanation,
        fused_signals=fused_signals,
        actionable_recommendations=unique_recommendations,
        signals_breakdown={
            "gemini_evaluated": gemini_score is not None,
            "classifier_evaluated": classifier_result is not None,
            "threat_intel_count": len(url_threat_reports),
            "urls_analyzed": len(url_feature_reports),
            "ocr_evaluated": ocr_evidence is not None and ocr_evidence.is_readable
        }
    )
