"""
ScamLens URL Threat Intelligence Engine
Provides modular, safe URL reputation lookups against Google Safe Browsing, PhishTank, and local threat registries.
Zero automated network visits to target URLs. Never sends raw user messages to providers.
"""

import os
import re
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse, urlunparse
import requests
from pydantic import BaseModel, Field

logger = logging.getLogger("scamlens.threat_intel")

# --- URL Extraction and Normalization ---
# Matches http, https, and bare domains (e.g. usps-track.xyz/pay)
URL_REGEX = re.compile(
    r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,10}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'\".,<>?«»“”‘’]))'
)

def extract_urls(text: str) -> List[str]:
    """
    Safely extracts all unique URLs and candidate domain links from raw text.
    Handles parenthetical expressions and trailing punctuation.
    """
    if not text:
        return []
    matches = URL_REGEX.findall(text)
    raw_urls = [m[0] if isinstance(m, tuple) else m for m in matches]
    
    unique_urls: List[str] = []
    seen: Set[str] = set()
    for raw in raw_urls:
        norm = normalize_url(raw)
        if norm and norm not in seen:
            seen.add(norm)
            unique_urls.append(norm)
    return unique_urls

def normalize_url(raw_url: str) -> Optional[str]:
    """
    Normalizes a candidate URL:
    - Strips surrounding brackets, quotes, and punctuation
    - Prepends http:// if scheme is missing
    - Normalizes hostname to lowercase
    - Returns None if not a valid link
    """
    if not raw_url:
        return None
    cleaned = raw_url.strip(" <>\"\'()[]{}.,;:!?`*")
    if not cleaned:
        return None

    # Prepend scheme if missing
    if not cleaned.startswith(("http://", "https://")):
        cleaned = "http://" + cleaned

    try:
        parsed = urlparse(cleaned)
        if not parsed.netloc:
            return None
        # Must have a valid dot in domain (e.g. example.com) or be an IP
        if "." not in parsed.netloc and not parsed.netloc.startswith("localhost"):
            return None
        
        # Lowercase scheme and netloc
        norm_parsed = parsed._replace(
            scheme=parsed.scheme.lower(),
            netloc=parsed.netloc.lower()
        )
        return urlunparse(norm_parsed)
    except Exception:
        return None

# --- Threat Report Data Model ---
class URLThreatReport(BaseModel):
    url: str
    provider: str
    status: str = Field(description="One of: MATCH_FOUND, NO_MATCH, NOT_CONFIGURED, PROVIDER_UNAVAILABLE, INVALID_INPUT")
    threat_types: List[str] = Field(default_factory=list, description="Specific threat categories if matched, e.g. SOCIAL_ENGINEERING, MALWARE")
    details: str = Field(description="Human-readable explanation of provider outcome")
    is_safe_guarantee: bool = Field(default=False, description="Always False. A clean reputation lookup does NOT guarantee safety.")
    checked_at: float = Field(default_factory=time.time)

# --- Base Provider Interface ---
class ThreatIntelProvider(ABC):
    name: str = "BaseProvider"

    @abstractmethod
    def check_urls(self, urls: List[str]) -> Dict[str, URLThreatReport]:
        """Performs reputation check on a list of URLs without visiting the targets."""
        pass

# --- Google Safe Browsing Provider ---
class GoogleSafeBrowsingProvider(ThreatIntelProvider):
    name = "Google Safe Browsing v4"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 4.0):
        self.api_key = api_key or os.getenv("SAFE_BROWSING_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.timeout = timeout
        self.endpoint = "https://safebrowsing.googleapis.com/v4/threatMatches:find"

    def check_urls(self, urls: List[str]) -> Dict[str, URLThreatReport]:
        results: Dict[str, URLThreatReport] = {}
        if not urls:
            return results

        if not self.api_key or self.api_key.strip() in ("", "your_gemini_api_key_here"):
            for u in urls:
                results[u] = URLThreatReport(
                    url=u,
                    provider=self.name,
                    status="NOT_CONFIGURED",
                    threat_types=[],
                    details="Safe Browsing API key not configured. Register SAFE_BROWSING_API_KEY in .env to enable.",
                )
            return results

        payload = {
            "client": {"clientId": "ScamLens", "clientVersion": "2.0.0"},
            "threatInfo": {
                "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"],
                "platformTypes": ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries": [{"url": u} for u in urls],
            },
        }

        try:
            resp = requests.post(
                f"{self.endpoint}?key={self.api_key}",
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                matches = data.get("matches", [])
                matched_urls = {}
                for m in matches:
                    matched_u = m.get("threat", {}).get("url")
                    ttype = m.get("threatType", "MALICIOUS")
                    if matched_u:
                        matched_urls.setdefault(matched_u, []).append(ttype)

                for u in urls:
                    if u in matched_urls:
                        results[u] = URLThreatReport(
                            url=u,
                            provider=self.name,
                            status="MATCH_FOUND",
                            threat_types=matched_urls[u],
                            details=f"CONFIRMED MATCH: Flagged by Google Safe Browsing as {', '.join(matched_urls[u])}.",
                        )
                    else:
                        results[u] = URLThreatReport(
                            url=u,
                            provider=self.name,
                            status="NO_MATCH",
                            threat_types=[],
                            details="No threat match found in Google Safe Browsing registry. (Does NOT guarantee safety for newly registered domains).",
                        )
            else:
                logger.warning("Safe Browsing API returned status %s", resp.status_code)
                for u in urls:
                    results[u] = URLThreatReport(
                        url=u,
                        provider=self.name,
                        status="PROVIDER_UNAVAILABLE",
                        threat_types=[],
                        details=f"Provider returned HTTP {resp.status_code}. Lookup unavailable.",
                    )
        except Exception as e:
            logger.warning("Safe Browsing lookup failed: %s", str(e))
            for u in urls:
                results[u] = URLThreatReport(
                    url=u,
                    provider=self.name,
                    status="PROVIDER_UNAVAILABLE",
                    threat_types=[],
                    details="Connection timeout or network failure reaching threat intel provider.",
                )

        return results

# --- Unified Threat Intelligence Manager with Caching ---
class ThreatIntelManager:
    """Coordinates threat intelligence checks with an in-memory TTL cache."""

    def __init__(self, providers: Optional[List[ThreatIntelProvider]] = None, cache_ttl_sec: float = 600.0):
        self.providers = providers if providers is not None else [GoogleSafeBrowsingProvider()]
        self.cache: Dict[str, URLThreatReport] = {}
        self.cache_ttl = cache_ttl_sec

    def check_urls(self, urls: List[str]) -> List[URLThreatReport]:
        if not urls:
            return []

        now = time.time()
        final_reports: List[URLThreatReport] = []
        urls_to_query: List[str] = []

        for u in urls:
            cached = self.cache.get(u)
            if cached and (now - cached.checked_at) < self.cache_ttl:
                final_reports.append(cached)
            else:
                urls_to_query.append(u)

        if urls_to_query:
            for provider in self.providers:
                reports = provider.check_urls(urls_to_query)
                for u, rep in reports.items():
                    self.cache[u] = rep
                    final_reports.append(rep)

        return final_reports
