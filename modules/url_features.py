"""
ScamLens URL Feature Engineering Module
Performs heuristic, static cybersecurity analysis on suspicious URLs without executing network requests to target hosts.
"""

import re
import ipaddress
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

KNOWN_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "is.gd", "ow.ly", "buff.ly",
    "cutt.ly", "tiny.cc", "rb.gy", "rebrand.ly", "shorturl.at", "bl.ink"
}

POPULAR_BRANDS = {
    "chase": "chase.com",
    "paypal": "paypal.com",
    "apple": "apple.com",
    "amazon": "amazon.com",
    "netflix": "netflix.com",
    "usps": "usps.com",
    "fedex": "fedex.com",
    "dhl": "dhl.com",
    "wellsfargo": "wellsfargo.com",
    "bankofamerica": "bankofamerica.com",
    "microsoft": "microsoft.com",
    "google": "google.com"
}

SUSPICIOUS_KEYWORDS = {
    "verify", "verification", "secure", "security", "login", "signin",
    "account", "billing", "update", "support", "service", "confirm",
    "unlock", "banking", "refund", "suspend", "dispatch"
}

class URLFeatureIndicator(BaseModel):
    category: str
    flag: str
    evidence: str
    explanation: str
    severity: str = Field(description="LOW, MEDIUM, HIGH, CRITICAL")
    is_heuristic: bool = True

class URLFeatureReport(BaseModel):
    url: str
    domain: str
    has_https: bool
    is_ip_host: bool
    is_shortened: bool
    subdomain_count: int
    url_length: int
    heuristic_risk_score: int = Field(ge=0, le=100)
    risk_level: str
    indicators: List[URLFeatureIndicator] = Field(default_factory=list)
    heuristic_notice: str = (
        "Heuristic analysis identifies structural patterns commonly associated with deceptive websites. "
        "It evaluates syntax and domain layout without visiting or probing the destination server."
    )

def is_ip_address(hostname: str) -> bool:
    """Checks whether hostname is a raw IPv4 or IPv6 address."""
    clean_host = hostname.split(":")[0].strip("[]")
    try:
        ipaddress.ip_address(clean_host)
        return True
    except ValueError:
        return False

def analyze_url_features(url: str) -> URLFeatureReport:
    """
    Analyzes static URL features and returns an explainable heuristic risk report.
    Never executes external network connections to the target URL.
    """
    indicators: List[URLFeatureIndicator] = []
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    netloc = (parsed.netloc or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""

    # 1. Scheme analysis
    has_https = parsed.scheme.lower() == "https"
    if not has_https:
        indicators.append(URLFeatureIndicator(
            category="Transport Security",
            flag="Unencrypted HTTP Protocol",
            evidence=f"Scheme: {parsed.scheme}://",
            explanation="The link uses unencrypted plain HTTP. Modern legitimate banking and service portals mandate HTTPS.",
            severity="MEDIUM"
        ))
    else:
        indicators.append(URLFeatureIndicator(
            category="Transport Security",
            flag="HTTPS Present (Not Proof of Safety)",
            evidence="https://",
            explanation="HTTPS encrypts transport traffic but does NOT guarantee the host is legitimate. Modern phishing sites frequently deploy free SSL certificates.",
            severity="LOW"
        ))

    # 2. IP Host check
    ip_host = is_ip_address(hostname)
    if ip_host:
        indicators.append(URLFeatureIndicator(
            category="Host Identity",
            flag="Raw IP Address Host",
            evidence=f"Host: {hostname}",
            explanation="Legitimate consumer services use registered domain names. Raw numeric IP addresses are frequently used in evasive attack infrastructure.",
            severity="CRITICAL"
        ))

    # 3. URL Shortener check
    is_shortener = hostname in KNOWN_SHORTENERS
    if is_shortener:
        indicators.append(URLFeatureIndicator(
            category="Evasion & Obfuscation",
            flag="Known URL Shortener",
            evidence=f"Domain: {hostname}",
            explanation="Shorteners obscure the ultimate destination host and are routinely used in smishing to bypass SMS gateway filters.",
            severity="HIGH"
        ))

    # 4. Excessive Length
    url_len = len(url)
    if url_len > 75:
        indicators.append(URLFeatureIndicator(
            category="Evasion & Obfuscation",
            flag="Excessive URL Length",
            evidence=f"Total length: {url_len} characters",
            explanation="Unusually long URLs are often used to conceal deceptive subdomains or inject obfuscated payload parameters.",
            severity="MEDIUM"
        ))

    # 5. Subdomain Structure & Deceptive Depth
    subdomains = hostname.split(".")
    subdomain_count = max(0, len(subdomains) - 2)
    if subdomain_count >= 3:
        indicators.append(URLFeatureIndicator(
            category="Domain Hierarchy",
            flag="Excessive Subdomain Depth",
            evidence=f"{len(subdomains)} domain labels in '{hostname}'",
            explanation="Multi-tiered subdomain nesting is frequently used to make fake domains look like legitimate brand domains on mobile screens.",
            severity="HIGH"
        ))

    # 6. Brand Spoofing & Misleading Subdomains
    # Check if a known brand keyword appears in hostname, but the registered domain is NOT that brand
    matched_brands = [b for b in POPULAR_BRANDS if b in hostname]
    for b in matched_brands:
        legit_root = POPULAR_BRANDS[b]
        if not (hostname == legit_root or hostname.endswith("." + legit_root)):
            indicators.append(URLFeatureIndicator(
                category="Brand Impersonation",
                flag=f"Target Brand Keyword in Unofficial Domain ('{b}')",
                evidence=f"Hostname contains '{b}' but resolves to root domain '{'.'.join(subdomains[-2:])}'",
                explanation=f"The domain name mimics {b.capitalize()} but is not owned by the official {legit_root} domain.",
                severity="CRITICAL"
            ))

    # 7. Suspicious Credential / Security Keywords
    keyword_hits = [kw for kw in SUSPICIOUS_KEYWORDS if kw in hostname or kw in path]
    if len(keyword_hits) >= 2 and not any(hostname.endswith("." + legit) for legit in POPULAR_BRANDS.values()):
        indicators.append(URLFeatureIndicator(
            category="Phishing Tactics",
            flag="Suspicious Action Keywords",
            evidence=f"Keywords found: {', '.join(keyword_hits)}",
            explanation="Keywords like 'verify', 'security', 'login', or 'suspend' in unverified domains are hallmark phishing lures.",
            severity="HIGH"
        ))

    # 8. Deceptive Encoding / Port anomalies
    if "@" in netloc:
        indicators.append(URLFeatureIndicator(
            category="URL Deception",
            flag="Userinfo '@' Symbol in URL Authority",
            evidence="Contains '@' character",
            explanation="The '@' sign in URL authority causes browsers to ignore all preceding characters and navigate exclusively to the host following the '@'.",
            severity="CRITICAL"
        ))

    if ":" in netloc and not ip_host:
        port_part = netloc.split(":")[-1]
        if port_part not in ("80", "443"):
            indicators.append(URLFeatureIndicator(
                category="Evasion & Obfuscation",
                flag="Non-Standard Web Port",
                evidence=f"Port: {port_part}",
                explanation="Phishing servers occasionally run on non-standard ports to bypass corporate firewalls.",
                severity="MEDIUM"
            ))

    if "%20" in path or "%00" in path or "xn--" in hostname:
        indicators.append(URLFeatureIndicator(
            category="Evasion & Obfuscation",
            flag="Deceptive Character Encoding or Punycode",
            evidence=f"Encoded host/path: {hostname}",
            explanation="Internationalized domain names (Punycode) or hex-encoded characters can be used in homograph attacks to impersonate legitimate characters.",
            severity="HIGH"
        ))

    # Compute Heuristic Risk Score
    severity_weights = {"LOW": 5, "MEDIUM": 20, "HIGH": 40, "CRITICAL": 60}
    raw_score = sum(severity_weights.get(ind.severity, 10) for ind in indicators if ind.flag != "HTTPS Present (Not Proof of Safety)")
    
    # Cap score
    score = min(max(raw_score, 0), 100)
    
    if score >= 80:
        level = "CRITICAL"
    elif score >= 50:
        level = "HIGH"
    elif score >= 25:
        level = "MEDIUM"
    else:
        level = "LOW"

    return URLFeatureReport(
        url=url,
        domain=hostname,
        has_https=has_https,
        is_ip_host=ip_host,
        is_shortened=is_shortener,
        subdomain_count=subdomain_count,
        url_length=url_len,
        heuristic_risk_score=score,
        risk_level=level,
        indicators=indicators
    )
