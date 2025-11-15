"""Feature extraction utilities for phishing URL detection."""
from __future__ import annotations

import json
import logging
import math
import os
import re
import statistics
import string
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set
from urllib.parse import urlparse, parse_qs
import socket
import ssl
from datetime import datetime, timezone

try:
    from dateutil import parser as dateutil_parser  # type: ignore
except ImportError:  # pragma: no cover
    dateutil_parser = None

import tldextract
from Levenshtein import distance as levenshtein_distance, ratio as levenshtein_ratio

try:  # Optional dependency, fall back gracefully
    from confusable_homoglyphs import confusables

    def _is_confusable(char: str) -> bool:
        return bool(confusables.is_confusable(char))
except ImportError:  # pragma: no cover - fallback when package missing
    def _is_confusable(char: str) -> bool:
        return char not in string.printable

try:  # Optional dependency for word scoring
    from wordfreq import zipf_frequency

    def _word_score(token: str) -> float:
        # Zipf frequency ranges roughly between 1-7 for common words
        return max(zipf_frequency(token, "en"), 0.0) / 7.0
except ImportError:  # pragma: no cover
    def _word_score(token: str) -> float:
        common_words = {"bank", "account", "login", "secure", "cloud", "mail", "service", "update"}
        return 1.0 if token in common_words else 0.0

import unicodedata

try:
    import whois  # type: ignore
except ImportError:  # pragma: no cover
    whois = None

DEFAULT_SUSPICIOUS_KEYWORDS = (
    "login",
    "secure",
    "verify",
    "update",
    "signin",
    "account",
    "bank",
)

CHAR_FREQUENCIES = {
    "e": 12.70,
    "t": 9.06,
    "a": 8.17,
    "o": 7.51,
    "i": 6.97,
    "n": 6.75,
    "s": 6.33,
    "h": 6.09,
    "r": 5.99,
    "d": 4.25,
    "l": 4.03,
    "c": 2.78,
    "u": 2.76,
    "m": 2.41,
    "w": 2.36,
    "f": 2.23,
    "g": 2.02,
    "y": 1.97,
    "p": 1.93,
    "b": 1.49,
    "v": 0.98,
    "k": 0.77,
    "x": 0.15,
    "q": 0.10,
    "j": 0.15,
    "z": 0.07,
}

TOP_DOMAIN_SOURCES = (
    "https://raw.githubusercontent.com/opendns/public-domain-lists/master/opendns-top-domains.txt",
    "https://raw.githubusercontent.com/tranco-list/tranco-list/main/top-1m.csv",
)

HOMOGLYPH_MAP = {
    "0": "o",
    "1": "l",
    "3": "e",
    "5": "s",
    "7": "t",
    "8": "b",
    "9": "g",
}

LOGGER = logging.getLogger("feature_extractor")


@dataclass
class FeatureExtractorConfig:
    """Runtime configuration for the feature extractor."""

    tld_stats_path: Path = Path("artifacts/tld_stats.json")
    top_domain_cache: Path = Path("data/top_domains.txt")
    suspicious_keywords: Iterable[str] = DEFAULT_SUSPICIOUS_KEYWORDS
    max_top_domains: int = 1000
    rank_file: Path = Path("dataset/top-1m.csv")
    domain_age_cache: Path = Path("artifacts/domain_age_cache.json")
    cert_cache: Path = Path("artifacts/cert_cache.json")
    trusted_domains_path: Path = Path("artifacts/trusted_domains.json")
    enable_enrichment: bool = False
    live_enrichment_limit: int = 200


class FeatureExtractor:
    """Extracts lexical and brand-aware features from URLs."""

    def __init__(self, config: Optional[FeatureExtractorConfig] = None) -> None:
        self.config = config or FeatureExtractorConfig()
        self.tld_legitimate_prob = self._load_tld_stats(self.config.tld_stats_path)
        self.default_tld_prob = (
            statistics.mean(self.tld_legitimate_prob.values())
            if self.tld_legitimate_prob
            else 0.5
        )
        self.top_domains = self._load_top_domains(
            self.config.top_domain_cache, self.config.max_top_domains
        )
        if not self.top_domains:
            LOGGER.warning("Top domain cache empty; Levenshtein features degraded")
        self.suspicious_keywords = tuple(k.lower() for k in self.config.suspicious_keywords)
        self.top_domain_set = set(self.top_domains)
        self.top_domain_ranks = {domain: idx + 1 for idx, domain in enumerate(self.top_domains)}
        self.max_rank = max(self.top_domain_ranks.values(), default=1)
        self.rank_map, self.rank_max = self._load_rank_table(self.config.rank_file)
        self.domain_age_cache = self._load_cache(self.config.domain_age_cache)
        self.cert_cache = self._load_cache(self.config.cert_cache)
        self.trusted_domains = self._load_trusted_domains(self.config.trusted_domains_path)
        self.live_fetch_count = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _load_tld_stats(path: Path) -> Dict[str, float]:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                return {str(k).lower(): float(v) for k, v in data.items()}
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                LOGGER.warning("Failed to load TLD stats: %s", exc)
        return {}

    @staticmethod
    def _load_top_domains(path: Path, limit: int) -> List[str]:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    lines = [line.strip().lower() for line in handle if line.strip()]
                return lines[:limit]
            except OSError as exc:
                LOGGER.warning("Unable to read top domain cache: %s", exc)
        # try to download
        FeatureExtractor._download_top_domains(path, limit)
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    return [line.strip().lower() for line in handle if line.strip()][:limit]
            except OSError:
                pass
        # fallback small list
        fallback = [
            "google.com",
            "facebook.com",
            "youtube.com",
            "amazon.com",
            "yahoo.com",
            "wikipedia.org",
            "linkedin.com",
            "netflix.com",
            "apple.com",
            "microsoft.com",
            "paypal.com",
            "bankofamerica.com",
            "wellsfargo.com",
            "chase.com",
            "instagram.com",
            "whatsapp.com",
            "office.com",
            "icloud.com",
            "twitter.com",
            "github.com",
        ]
        return fallback[:limit]

    @staticmethod
    def _download_top_domains(path: Path, limit: int) -> None:
        import csv

        try:
            import requests
        except ImportError:
            LOGGER.warning("requests not available; skipping top domain download")
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        collected: List[str] = []
        for source in TOP_DOMAIN_SOURCES:
            if len(collected) >= limit:
                break
            try:
                response = requests.get(source, timeout=10)
                response.raise_for_status()
                if source.endswith(".txt"):
                    for line in response.text.splitlines():
                        domain = line.strip().lower()
                        if domain and not domain.startswith("#"):
                            collected.append(domain)
                            if len(collected) >= limit:
                                break
                else:
                    reader = csv.reader(response.text.splitlines())
                    for row in reader:
                        domain = row[1].strip().lower() if len(row) > 1 else row[0].strip().lower()
                        if domain:
                            collected.append(domain)
                            if len(collected) >= limit:
                                break
            except Exception as exc:  # pragma: no cover - network issues
                LOGGER.warning("Unable to download domains from %s: %s", source, exc)
        if collected:
            try:
                with path.open("w", encoding="utf-8") as handle:
                    handle.write("\n".join(collected[:limit]))
            except OSError as exc:
                LOGGER.warning("Failed to persist top domain cache: %s", exc)

    @staticmethod
    def _load_rank_table(path: Path) -> tuple[Dict[str, int], int]:
        import csv

        if not path.exists():
            LOGGER.warning("Rank file %s not found; rank-based features disabled", path)
            return {}, 0
        ranks: Dict[str, int] = {}
        max_rank = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                for row in reader:
                    if len(row) < 2:
                        continue
                    try:
                        rank = int(row[0])
                    except ValueError:
                        continue
                    domain = row[1].strip().lower()
                    if not domain:
                        continue
                    ranks[domain] = rank
                    if rank > max_rank:
                        max_rank = rank
        except OSError as exc:
            LOGGER.warning("Unable to read rank file %s: %s", path, exc)
        return ranks, max_rank

    @staticmethod
    def _load_cache(path: Path) -> Dict[str, float]:
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return {str(k): float(v) for k, v in data.items()}
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            LOGGER.warning("Failed to load cache %s: %s", path, exc)
            return {}

    @staticmethod
    def _persist_cache(path: Path, data: Dict[str, float]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle)
        except OSError as exc:
            LOGGER.warning("Unable to persist cache %s: %s", path, exc)

    @staticmethod
    def _load_trusted_domains(path: Path) -> Set[str]:
        if not path.exists():
            return set()
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                domains = payload.get("trusted_domains", [])
            else:
                domains = payload
            return {str(domain).strip().lower() for domain in domains if domain}
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Unable to load trusted domains from %s: %s", path, exc)
            return set()

    # ------------------------------------------------------------------
    def _consume_live_budget(self) -> bool:
        if not self.config.enable_enrichment:
            return False
        if self.live_fetch_count >= self.config.live_enrichment_limit:
            return False
        self.live_fetch_count += 1
        return True

    @staticmethod
    def _parse_datetime(value) -> Optional[datetime]:
        if isinstance(value, list):
            for entry in value:
                parsed = FeatureExtractor._parse_datetime(entry)
                if parsed:
                    return parsed
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            if dateutil_parser is not None:
                try:
                    return dateutil_parser.parse(value)
                except (ValueError, TypeError):
                    return None
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None

    @staticmethod
    def _normalize_datetime(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _parse_ssl_time(value: str) -> Optional[datetime]:
        try:
            dt = datetime.strptime(value, "%b %d %H:%M:%S %Y %Z")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            parsed = FeatureExtractor._parse_datetime(value)
            return FeatureExtractor._normalize_datetime(parsed) if parsed else None

    def _get_domain_age_days(self, domain: Optional[str]) -> float:
        if not domain:
            return 0.0
        domain = domain.lower()
        if domain in self.domain_age_cache:
            return float(self.domain_age_cache[domain])
        if not self._consume_live_budget() or whois is None:
            return 0.0
        try:
            record = whois.whois(domain)
        except Exception as exc:  # pragma: no cover - network/WHOIS failures
            LOGGER.debug("WHOIS lookup failed for %s: %s", domain, exc)
            return 0.0
        creation_date = getattr(record, "creation_date", None)
        parsed = self._parse_datetime(creation_date)
        if not parsed:
            return 0.0
        parsed = self._normalize_datetime(parsed)
        now = datetime.now(timezone.utc)
        if parsed > now:
            age_days = 0.0
        else:
            age_days = (now - parsed).days
        self.domain_age_cache[domain] = float(age_days)
        self._persist_cache(self.config.domain_age_cache, self.domain_age_cache)
        return float(age_days)

    def _has_valid_cert(self, host: Optional[str]) -> float:
        if not host:
            return 0.0
        host = host.split(":")[0].lower()
        if not host:
            return 0.0
        if host in self.cert_cache:
            return float(self.cert_cache[host])
        if not self._consume_live_budget():
            return 0.0
        context = ssl.create_default_context()
        is_valid = 0.0
        try:
            with socket.create_connection((host, 443), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=host) as secure_sock:
                    cert = secure_sock.getpeercert()
            if cert:
                now = datetime.now(timezone.utc)
                not_before = cert.get("notBefore")
                not_after = cert.get("notAfter")
                valid = True
                if not_before:
                    parsed_before = self._parse_ssl_time(not_before)
                    if parsed_before and parsed_before > now:
                        valid = False
                if not_after:
                    parsed_after = self._parse_ssl_time(not_after)
                    if parsed_after and parsed_after < now:
                        valid = False
                is_valid = 1.0 if valid else 0.0
        except Exception as exc:  # pragma: no cover - network/SSL failures
            LOGGER.debug("TLS check failed for %s: %s", host, exc)
            is_valid = 0.0
        self.cert_cache[host] = float(is_valid)
        self._persist_cache(self.config.cert_cache, self.cert_cache)
        return float(is_valid)

    def _get_top_rank_percentile(self, domain: Optional[str]) -> float:
        if not domain or not self.rank_map or not self.rank_max:
            return 100.0
        domain = domain.lower()
        rank = self.rank_map.get(domain)
        if rank is None:
            return 100.0
        return (rank / self.rank_max) * 100.0

    # ------------------------------------------------------------------
    def extract_features(self, url: str) -> Dict[str, float]:
        if not isinstance(url, str) or not url.strip():
            return self._neutral_features()
        url = url.strip()
        normalized_url = url if "://" in url else f"https://{url}"
        parsed_url = urlparse(normalized_url)
        parsed = tldextract.extract(url)
        scheme = self._infer_scheme(url)
        domain = parsed.registered_domain or parsed.domain or parsed.fqdn or ""
        subdomain = parsed.subdomain
        suffix = parsed.suffix.lower()
        host = parsed_url.netloc or ".".join(part for part in [subdomain, domain] if part)
        full_domain = "".join(filter(None, [domain]))
        netloc = ".".join(filter(None, [subdomain, domain, suffix]))
        domain_only = domain or parsed.domain
        canonical_domain = domain_only or parsed.fqdn or parsed.registered_domain
        path = parsed_url.path or ""
        if path == "/":
            path = ""
        query = parsed_url.query or ""
        path_segments = [segment for segment in path.split("/") if segment]
        path_depth = len(path_segments)
        query_params = parse_qs(query)
        char_counts = Counter(url)
        total_chars = max(len(url), 1)

        digit_ratio = sum(c.isdigit() for c in url) / total_chars
        letter_ratio = sum(c.isalpha() for c in url) / total_chars
        special_char_ratio = 1.0 - digit_ratio - letter_ratio

        entropy_domain = self._shannon_entropy(canonical_domain)
        host_lower = host.lower()
        path_query_lower = (path + "?" + query).lower()
        domain_contains_keyword = int(any(k in host_lower for k in self.suspicious_keywords))
        path_contains_keyword = int(any(k in path_query_lower for k in self.suspicious_keywords))
        contains_keyword = int(domain_contains_keyword or path_contains_keyword)
        path_digit_ratio = (sum(ch.isdigit() for ch in path) / len(path)) if path else 0.0
        tld_prob = self.tld_legitimate_prob.get(suffix, self.default_tld_prob)
        canonical_lower = canonical_domain.lower() if canonical_domain else ""
        domain_exact_match = 1.0 if canonical_lower and canonical_lower in self.top_domain_set else 0.0
        is_trusted_domain = bool(canonical_lower and canonical_lower in self.trusted_domains)
        if domain_exact_match:
            domain_contains_keyword = 0
        if is_trusted_domain:
            path_contains_keyword = 0
            contains_keyword = domain_contains_keyword
        rank = self.top_domain_ranks.get(canonical_lower)
        domain_reputation_score = (
            1.0 - math.log1p(rank - 1) / math.log1p(self.max_rank)
            if rank is not None and self.max_rank > 0
            else 0.0
        )

        features = {
            "URLLength": float(len(url)),
            "DomainLength": float(len(canonical_domain)),
            "NoOfSubDomain": float(self._count_subdomains(subdomain)),
            "IsHTTPS": 1.0 if scheme == "https" else 0.0,
            "TLDLength": float(len(suffix)),
            "IsDomainIP": 1.0 if self._is_ip(netloc) else 0.0,
            "DigitRatio": float(digit_ratio),
            "LetterRatio": float(letter_ratio),
            "SpecialCharRatio": float(max(min(special_char_ratio, 1.0), 0.0)),
            "EntropyDomain": float(entropy_domain),
            "ContainsSuspiciousWord": float(contains_keyword),
            "DomainContainsSuspiciousWord": float(domain_contains_keyword),
            "PathContainsSuspiciousWord": float(path_contains_keyword),
            "TLDLegitimateProb": float(tld_prob),
            "URLSimilarityIndex": float(self._url_similarity_index(canonical_domain)),
            "URLCharProb": float(self._url_char_probability(char_counts, total_chars)),
            "CharContinuationRate": float(self._char_continuation_rate(url)),
            "LevenshteinToTopDomainMin": float(self._levenshtein_to_top_domain(canonical_domain)),
            "VowelConsonantRatio": float(self._vowel_consonant_ratio(canonical_domain)),
            "HomoglyphCount": float(self._homoglyph_count(netloc)),
            "MeaningfulWordScore": float(self._meaningful_word_score(canonical_domain)),
            "PathLength": float(len(path)),
            "PathDepth": float(path_depth),
            "PathDigitRatio": float(path_digit_ratio),
            "QueryLength": float(len(query)),
            "QueryParamCount": float(len(query_params)),
            "DomainExactTopMatch": float(domain_exact_match),
            "DomainReputationScore": float(domain_reputation_score),
            "DomainAgeDays": float(self._get_domain_age_days(canonical_lower)),
            "HasValidCert": float(self._has_valid_cert(parsed_url.hostname or canonical_lower)),
            "TopRankPercentile": float(self._get_top_rank_percentile(canonical_lower)),
            "TrustedDomainFlag": 1.0 if is_trusted_domain else 0.0,
        }
        return features

    # ------------------------------------------------------------------
    @staticmethod
    def _neutral_features() -> Dict[str, float]:
        return {
            "URLLength": 0.0,
            "DomainLength": 0.0,
            "NoOfSubDomain": 0.0,
            "IsHTTPS": 0.0,
            "TLDLength": 0.0,
            "IsDomainIP": 0.0,
            "DigitRatio": 0.0,
            "LetterRatio": 0.0,
            "SpecialCharRatio": 0.0,
            "EntropyDomain": 0.0,
            "ContainsSuspiciousWord": 0.0,
            "DomainContainsSuspiciousWord": 0.0,
            "PathContainsSuspiciousWord": 0.0,
            "TLDLegitimateProb": 0.5,
            "URLSimilarityIndex": 0.0,
            "URLCharProb": 0.0,
            "CharContinuationRate": 0.0,
            "LevenshteinToTopDomainMin": 10.0,
            "VowelConsonantRatio": 0.0,
            "HomoglyphCount": 0.0,
            "MeaningfulWordScore": 0.0,
            "PathLength": 0.0,
            "PathDepth": 0.0,
            "PathDigitRatio": 0.0,
            "QueryLength": 0.0,
            "QueryParamCount": 0.0,
            "DomainExactTopMatch": 0.0,
            "DomainReputationScore": 0.0,
            "DomainAgeDays": 0.0,
            "HasValidCert": 0.0,
            "TopRankPercentile": 100.0,
            "TrustedDomainFlag": 0.0,
        }

    @staticmethod
    def _infer_scheme(url: str) -> str:
        if url.lower().startswith("http://"):
            return "http"
        if url.lower().startswith("https://"):
            return "https"
        return "https"  # assume secure default when missing

    @staticmethod
    def _count_subdomains(subdomain: str) -> int:
        if not subdomain:
            return 0
        return len([part for part in subdomain.split(".") if part])

    @staticmethod
    def _is_ip(host: str) -> bool:
        import ipaddress

        try:
            ipaddress.ip_address(host)
            return True
        except ValueError:
            return False

    @staticmethod
    def _shannon_entropy(text: str) -> float:
        if not text:
            return 0.0
        counts = Counter(text)
        total = len(text)
        entropy = 0.0
        for count in counts.values():
            probability = count / total
            entropy -= probability * math.log2(probability)
        return entropy

    @staticmethod
    def _char_continuation_rate(text: str) -> float:
        if len(text) < 2:
            return 0.0
        repeats = sum(1 for i in range(1, len(text)) if text[i] == text[i - 1])
        return repeats / (len(text) - 1)

    @staticmethod
    def _url_char_probability(char_counts: Counter, total_chars: int) -> float:
        if total_chars == 0:
            return 0.0
        probability = 0.0
        for char, count in char_counts.items():
            freq = CHAR_FREQUENCIES.get(char.lower(), 0.05)
            probability += (freq / 100.0) * (count / total_chars)
        return probability

    def _url_similarity_index(self, domain: str) -> float:
        if not domain or not self.top_domains:
            return 0.0
        best = 0.0
        for top_domain in self.top_domains[:100]:  # speedup: sample top 100
            similarity = levenshtein_ratio(domain, top_domain)
            if similarity > best:
                best = similarity
        return best * 100.0

    def _levenshtein_to_top_domain(self, domain: str) -> float:
        if not domain or not self.top_domains:
            return float(len(domain) or 15)
        best = min(levenshtein_distance(domain, top) for top in self.top_domains[: self.config.max_top_domains])
        return float(best)

    @staticmethod
    def _vowel_consonant_ratio(domain: str) -> float:
        if not domain:
            return 0.0
        vowels = sum(1 for ch in domain.lower() if ch in "aeiou")
        consonants = sum(1 for ch in domain.lower() if ch.isalpha() and ch not in "aeiou")
        if consonants == 0:
            return float(vowels) if vowels else 0.0
        return vowels / consonants

    @staticmethod
    def _homoglyph_count(host: str) -> int:
        if not host:
            return 0
        count = 0
        for char in host:
            if not char.isascii():
                base = unicodedata.normalize("NFKC", char)
                if base != char or _is_confusable(char):
                    count += 1
                continue
            if char in HOMOGLYPH_MAP and HOMOGLYPH_MAP[char] != char:
                count += 1
        return count

    @staticmethod
    def _meaningful_word_score(domain: str) -> float:
        if not domain:
            return 0.0
        tokens = re.split(r"[^a-zA-Z]+", domain.lower())
        tokens = [t for t in tokens if t]
        if not tokens:
            return 0.0
        scores = [_word_score(token) for token in tokens]
        return sum(scores) / len(scores)


_GLOBAL_EXTRACTOR: Optional[FeatureExtractor] = None


def extract_features(url: str) -> Dict[str, float]:
    """Convenience wrapper that uses a global extractor."""

    global _GLOBAL_EXTRACTOR
    if "_GLOBAL_EXTRACTOR" not in globals() or _GLOBAL_EXTRACTOR is None:
        _GLOBAL_EXTRACTOR = FeatureExtractor()
    return _GLOBAL_EXTRACTOR.extract_features(url)


__all__ = ["FeatureExtractor", "FeatureExtractorConfig", "extract_features"]
