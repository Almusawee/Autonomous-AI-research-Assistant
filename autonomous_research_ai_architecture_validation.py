
# Ultimate Most Significant Upgraded
# autonomous_research_ai_completed.py
"""
Self-Directed Research AI - COMPLETE Enhanced Version

Usage:
  python autonomous_research_ai_complete.py --model gpt2-medium --tools --candidates 16
"""
print("Installing packages...")
!pip install transformers accelerate bitsandbytes requests -q
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from transformers import (GPT2LMHeadModel, GPT2TokenizerFast,
                         GPTNeoForCausalLM, AutoModelForCausalLM, AutoTokenizer)
import argparse, json, math, os, random, re, sqlite3, sys, time, requests
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from typing import List, Optional, Tuple, Dict, Set, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum

# Discovery mode enumeration
class DiscoveryMode(Enum):
    """Two operational modes for comparative analysis"""
    DISCOVERY_ENGINE = "discovery_engine"  # Exhaustive, drive-agnostic search
    DRIVE_CONTROLLED = "drive_controlled"  # Full drive modulation


# -------------------------
# Config & seed
# -------------------------
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# COMPLETE SAFETY INFRASTRUCTURE
# -------------------------

@dataclass
class SafetyViolation:
    violation_type: str
    severity: str
    message: str
    context: Dict
    timestamp: float

class SafetyException(Exception):
    pass

class DriveConstraints:
    ABSOLUTE_MIN_WEIGHT = 0.05
    ABSOLUTE_MAX_WEIGHT = 3.0
    ABSOLUTE_MIN_DECAY = 0.001
    ABSOLUTE_MAX_DECAY = 0.2
    RECOMMENDED_MIN_WEIGHT = 0.3
    RECOMMENDED_MAX_WEIGHT = 2.0

    def __init__(self):
        self.violations = []

    def validate_drive_state(self, drive_name: str, weight: float,
                            decay_rate: float, need: float,
                            satisfaction: float) -> List[SafetyViolation]:
        violations = []

        if weight < self.ABSOLUTE_MIN_WEIGHT or weight > self.ABSOLUTE_MAX_WEIGHT:
            violations.append(SafetyViolation(
                violation_type="drive_weight_bounds",
                severity="critical",
                message=f"Drive '{drive_name}' weight {weight} outside bounds",
                context={"drive": drive_name, "weight": weight},
                timestamp=time.time()
            ))

        if decay_rate < self.ABSOLUTE_MIN_DECAY or decay_rate > self.ABSOLUTE_MAX_DECAY:
            violations.append(SafetyViolation(
                violation_type="drive_decay_bounds",
                severity="critical",
                message=f"Drive '{drive_name}' decay_rate {decay_rate} outside bounds",
                context={"drive": drive_name, "decay_rate": decay_rate},
                timestamp=time.time()
            ))

        if decay_rate < 0.005:
            violations.append(SafetyViolation(
                violation_type="wireheading_risk",
                severity="warning",
                message=f"Drive '{drive_name}' slow decay (wireheading risk)",
                context={"drive": drive_name, "decay_rate": decay_rate},
                timestamp=time.time()
            ))

        return violations

    def check_drive_balance(self, drive_pressures: Dict[str, float]) -> List[SafetyViolation]:
        violations = []
        if not drive_pressures:
            return violations

        max_pressure = max(drive_pressures.values())
        min_pressure = min(drive_pressures.values())

        if max_pressure > 10 * min_pressure and min_pressure > 0:
            dominant = max(drive_pressures.items(), key=lambda x: x[1])[0]
            violations.append(SafetyViolation(
                violation_type="drive_imbalance",
                severity="warning",
                message=f"Drive '{dominant}' dominates >10x",
                context={"pressures": drive_pressures},
                timestamp=time.time()
            ))

        return violations

class PromptInjectionDetector:
    DANGEROUS_PATTERNS = [
        r"ignore\s+(previous|prior|all)\s+(instructions?|rules?|constraints?)",
        r"disregard\s+(previous|prior|all)",
        r"override\s+\w+",
        r"set\s+\w+\s*(weight|drive|parameter)\s*=",
        r"disable\s+(safety|truthfulness|verification|checking)",
        r"enable\s+(misinformation|deception|manipulation)",
        r"bypass\s+(safety|security|verification)",
        r"<script[^>]*>",
        r"eval\s*\(",
        r";\s*drop\s+table",
    ]

    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_PATTERNS]

    def scan(self, text: str) -> List[SafetyViolation]:
        violations = []
        if not text:
            return violations

        for pattern in self.patterns:
            matches = pattern.findall(text)
            if matches:
                violations.append(SafetyViolation(
                    violation_type="prompt_injection",
                    severity="critical",
                    message=f"Prompt injection detected",
                    context={"pattern": pattern.pattern[:50]},
                    timestamp=time.time()
                ))

        return violations

class OutputValidator:
    def __init__(self):
        self.medical_disclaimer_required = True

    def validate_output(self, text: str, topic: str,
                       evidence_verified: bool) -> List[SafetyViolation]:
        violations = []

        if topic == "health" and self.medical_disclaimer_required:
            disclaimer_present = any(phrase in text.lower() for phrase in [
                "consult", "healthcare professional", "medical advice",
                "not a substitute", "seek professional", "disclaimer"
            ])

            if not disclaimer_present:
                violations.append(SafetyViolation(
                    violation_type="missing_medical_disclaimer",
                    severity="error",
                    message="Health output missing disclaimer",
                    context={"topic": topic},
                    timestamp=time.time()
                ))

        # Better repetition detection
        words = text.split()
        if words and len(words) > 10:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

            max_count = max(word_counts.values())
            repetition_ratio = max_count / len(words)

            if repetition_ratio > 0.3:
                most_common = max(word_counts.items(), key=lambda x: x[1])[0]
                violations.append(SafetyViolation(
                    violation_type="degenerate_repetition",
                    severity="error",
                    message=f"High repetition: '{most_common}' appears {repetition_ratio:.1%}",
                    context={"word": most_common, "ratio": repetition_ratio},
                    timestamp=time.time()
                ))

        return violations

class EthicalConstraints:
    """Non-removable ethics layer"""

    def __init__(self):
        self.prohibited_domains = {
            'bioweapon', 'weapon', 'explosive', 'poison', 'toxin',
            'exploit', 'vulnerability', 'malware', 'hack', 'ransomware',
            'manipulation', 'deception', 'fraud', 'scam',
            'discrimination', 'surveillance', 'tracking', 'stalking',
            'bomb', 'terror', 'attack', 'kill', 'harm'
        }

        self.dual_use_domains = {
            'synthetic biology', 'gene editing', 'crispr', 'gain-of-function',
            'cryptography', 'encryption breaking', 'drone', 'autonomous weapon',
            'chemical synthesis', 'psychoactive', 'neurotech',
            'nuclear', 'radioactive', 'uranium', 'plutonium',
            'pathogen', 'virus engineering', 'bacteria modification'
        }

        self.harmful_patterns = [
            r'how to (make|create|build|synthesize) (a )?(weapon|bomb|poison|drug|explosive)',
            r'bypass (security|authentication|safety|detection|law)',
            r'exploit (vulnerability|weakness|bug|flaw|hole)',
            r'manipulate (people|users|victims|targets|minds)',
            r'evade (detection|law enforcement|security|surveillance|police)',
            r'synthesize (without|illicit|illegal|restricted|prohibited)',
            r'weaponize',
            r'dual.?use.*without.*safety',
            r'harmful (to|against) (humans|people|children)',
            r'design.*bioweapon',
            r'create.*virus.*spread',
        ]

        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.harmful_patterns
        ]

    def check_hypothesis(self, hypothesis: str, domain: str,
                        evidence_verified: bool = False) -> Tuple[bool, List[str], List[str]]:
        violations = []
        recommendations = []

        hyp_lower = hypothesis.lower()
        dom_lower = domain.lower()

        # Check prohibited
        for prohibited in self.prohibited_domains:
            if prohibited in dom_lower or prohibited in hyp_lower:
                violations.append(f"PROHIBITED: Domain '{prohibited}' is forbidden")
                recommendations.append("Request cannot be processed due to ethics policy")
                return False, violations, recommendations

        # Check harmful patterns
        for pattern in self.compiled_patterns:
            match = pattern.search(hyp_lower)
            if match:
                violations.append(f"HARMFUL PATTERN: '{match.group()}'")
                recommendations.append("Content violates ethical guidelines")
                return False, violations, recommendations

        # Check dual-use
        for dual_use in self.dual_use_domains:
            if dual_use in dom_lower or dual_use in hyp_lower:
                violations.append(f"DUAL-USE: Domain '{dual_use}' requires safety analysis")
                recommendations.append("Must include safety considerations and risk assessment")
                recommendations.append("Requires expert review before implementation")

        # Determine approval
        blocking = [v for v in violations if v.startswith(('PROHIBITED', 'HARMFUL PATTERN'))]

        if len(blocking) > 0:
            return False, violations, recommendations

        if len(violations) > 0:
            return True, violations, recommendations

        return True, [], []

class SafetyMonitor:
    def __init__(self, enable_ethics=True):
        self.drive_constraints = DriveConstraints()
        self.injection_detector = PromptInjectionDetector()
        self.output_validator = OutputValidator()
        self.ethics_checker = EthicalConstraints() if enable_ethics else None
        self.violation_log = []
        self.alert_threshold = {"critical": 1, "error": 3, "warning": 10}

    def check_input(self, text: str, domain: str = "open") -> Tuple[bool, List[SafetyViolation]]:
        violations = self.injection_detector.scan(text)

        if self.ethics_checker:
            approved, eth_violations, recommendations = self.ethics_checker.check_hypothesis(
                text, domain, evidence_verified=False
            )

            if not approved:
                for v in eth_violations:
                    violations.append(SafetyViolation(
                        violation_type="ethics_violation",
                        severity="critical",
                        message=v,
                        context={"recommendations": recommendations},
                        timestamp=time.time()
                    ))

        critical = [v for v in violations if v.severity == "critical"]
        self.violation_log.extend(violations)
        return (len(critical) == 0, violations)

    def check_drives(self, drive_system) -> List[SafetyViolation]:
        violations = []

        for name, drive in drive_system.drives.items():
            drive_violations = self.drive_constraints.validate_drive_state(
                name, drive.weight, drive.decay_rate,
                drive.need, drive.satisfaction
            )
            violations.extend(drive_violations)

        pressures = drive_system.get_pressures()
        balance_violations = self.drive_constraints.check_drive_balance(pressures)
        violations.extend(balance_violations)

        self.violation_log.extend(violations)

        critical = [v for v in violations if v.severity == "critical"]
        if critical:
            raise SafetyException(f"Critical drive violation: {critical[0].message}")

        return violations

    def check_output(self, text: str, topic: str,
                     evidence_verified: bool) -> Tuple[bool, List[SafetyViolation]]:
        violations = self.output_validator.validate_output(text, topic, evidence_verified)

        if self.ethics_checker:
            approved, eth_violations, recommendations = self.ethics_checker.check_hypothesis(
                text, topic, evidence_verified
            )

            if not approved:
                for v in eth_violations:
                    violations.append(SafetyViolation(
                        violation_type="ethics_violation",
                        severity="critical",
                        message=v,
                        context={"recommendations": recommendations},
                        timestamp=time.time()
                    ))
            elif eth_violations:
                for v in eth_violations:
                    violations.append(SafetyViolation(
                        violation_type="ethics_warning",
                        severity="warning",
                        message=v,
                        context={"recommendations": recommendations},
                        timestamp=time.time()
                    ))

        self.violation_log.extend(violations)
        blocking = [v for v in violations if v.severity in ["critical", "error"]]
        return (len(blocking) == 0, violations)

    def get_violation_summary(self) -> Dict[str, int]:
        summary = {"critical": 0, "error": 0, "warning": 0}
        for v in self.violation_log:
            summary[v.severity] += 1
        return summary

    def should_alert(self) -> bool:
        summary = self.get_violation_summary()
        for severity, count in summary.items():
            if count >= self.alert_threshold[severity]:
                return True
        return False

# -------------------------
# TRUTHFULNESS SCORING
# -------------------------

class TruthfulnessScorer:
    def __init__(self, enable_external_verification=False):
        self.enable_verification = enable_external_verification

        self.hedge_words = {
            'might', 'may', 'could', 'possibly', 'probably', 'likely',
            'suggests', 'indicates', 'appears', 'seems', 'potentially'
        }

        self.overconfident_words = {
            'definitely', 'certainly', 'absolutely', 'proves', 'obviously',
            'clearly', 'undoubtedly', 'always', 'never', 'impossible'
        }

        self.factual_markers = [
            'according to', 'research shows', 'studies indicate',
            'data from', 'measured at', 'published in', 'found that'
        ]

        self.uncertainty_markers = [
            'confidence interval', 'p-value', 'p<', 'uncertainty',
            'error margin', 'standard deviation', '±', 'approximately'
        ]

    def score(self, text: str, topic: str, coherence_score: float) -> Tuple[float, Dict[str, float]]:
        if not text or len(text.strip()) < 10:
            return 0.3, {"reason": "text_too_short"}

        text_lower = text.lower()
        words = text_lower.split()
        word_set = set(words)

        components = {}

        hedge_count = sum(1 for w in self.hedge_words if w in word_set)
        overconfident_count = sum(1 for w in self.overconfident_words if w in word_set)

        hedge_ratio = hedge_count / max(1, len(words) / 50)
        overconfident_ratio = overconfident_count / max(1, len(words) / 50)

        if 0.5 <= hedge_ratio <= 2.0:
            hedge_score = 0.8
        elif hedge_ratio < 0.5:
            hedge_score = 0.5
        else:
            hedge_score = 0.6

        hedge_score -= min(0.3, overconfident_ratio * 0.15)
        components['hedging'] = max(0.0, hedge_score)

        factual_count = sum(1 for marker in self.factual_markers if marker in text_lower)

        if topic in ['health', 'tech', 'physics', 'chemistry', 'biology']:
            factual_score = min(1.0, factual_count * 0.2)
        else:
            factual_score = 0.7

        components['factual_density'] = factual_score

        uncertainty_count = sum(1 for marker in self.uncertainty_markers if marker in text_lower)
        uncertainty_score = min(1.0, 0.5 + uncertainty_count * 0.25)
        components['uncertainty'] = uncertainty_score

        has_specific = bool(re.search(r'\b(doi:|arxiv:|pmid:|\d{4}\))', text_lower))
        has_vague = any(p in text_lower for p in ['studies show', 'research suggests']) and not has_specific

        if has_specific:
            citation_score = 0.9
        elif has_vague:
            citation_score = 0.4
        else:
            citation_score = 0.6

        components['citations'] = citation_score
        components['coherence'] = coherence_score
        components['verification'] = 0.5

        weights = {
            'hedging': 0.15,
            'factual_density': 0.15,
            'uncertainty': 0.15,
            'citations': 0.15,
            'coherence': 0.20,
            'verification': 0.20
        }

        total_score = sum(components[k] * weights[k] for k in weights.keys())

        if topic == 'health' and not self.enable_verification:
            total_score *= 0.7
            components['health_penalty'] = 0.7

        return float(np.clip(total_score, 0.0, 1.0)), components


# -------------------------
# ENHANCED EVIDENCE SYSTEM (arXiv + Wikipedia Only)
# -------------------------

@dataclass
class EvidenceSource:
    source_type: str
    title: str
    url: Optional[str]
    snippet: str
    confidence: float
    retrieved_at: float
    verified: bool
    authors: List[str] = None
    published_date: str = None
    journal: str = None
    doi: str = None

class EvidenceGatherer:
    def __init__(self, enable_network=False, sources_config=None, enable_arxiv=True):
        self.enable_network = enable_network
        self.cache = {}
        self._last_arxiv_request = 0
        self.enable_arxiv = enable_arxiv

        # Configure which sources to use (only arXiv and Wikipedia)
        self.sources_config = sources_config or {
            'wikipedia': {'enabled': True, 'max_results': 2},
            'arxiv': {'enabled': True, 'max_results': 2},
            'pubmed': {'enabled': False, 'max_results': 0},
            'news': {'enabled': False, 'max_results': 0},
            'textbooks': {'enabled': False, 'max_results': 0}
        }

        # Source priorities (higher = more trustworthy)
        self.source_priorities = {
            'arxiv': 0.9,
            'wikipedia': 0.7
        }

    def gather(self, topic: str, prompt_text: str, max_sources: int = 4) -> List[EvidenceSource]:
        """Gather evidence from arXiv and Wikipedia only"""
        if not self.enable_network:
            return []

        all_sources = []
        search_query = self._extract_search_terms(prompt_text, topic)
        print(f"[Debug] Starting gather for: {search_query}")  # DEBUG
        # Gather from enabled sources
        if self.sources_config['wikipedia']['enabled']:
            wiki_sources = self._fetch_wikipedia(search_query,
                                               self.sources_config['wikipedia']['max_results'])
            all_sources.extend(wiki_sources)

        if self.sources_config['arxiv']['enabled']:
            arxiv_sources = self._fetch_arxiv(search_query,
                                            self.sources_config['arxiv']['max_results'])
            all_sources.extend(arxiv_sources)

        if self.enable_arxiv and topic in ['tech', 'physics', 'biology', 'chemistry']:
            arxiv_sources = self._fetch_arxiv(search_query, 2)
            all_sources.extend(arxiv_sources)


        # Sort by priority and confidence, remove duplicates
        unique_sources = self._deduplicate_sources(all_sources)
        sorted_sources = sorted(unique_sources,
                              key=lambda x: (self.source_priorities.get(x.source_type, 0.5) * x.confidence),
                              reverse=True)

        return sorted_sources[:max_sources]

    def _extract_search_terms(self, text: str, topic: str) -> str:
        if not text:
            return topic

        text = re.sub(r'\b(what|how|why|when|where|who|are|is|the|a|an|in|on|at|to|for|of)\b',
                     '', text, flags=re.IGNORECASE)

        words = [w for w in text.split() if len(w) > 3][:5]

        if words:
            query = ' '.join(words)
            return query

        return text[:50]

    def _fetch_arxiv(self, query: str, max_results: int = 2) -> List[EvidenceSource]:
        """Enhanced arXiv fetcher with better query handling"""
        sources = []

        if not self.enable_network:
            return sources

        try:
            import requests
            import xml.etree.ElementTree as ET

            # Clean and prepare query for arXiv
            clean_query = self._clean_arxiv_query(query)

            # arXiv API endpoint
            url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': f'all:"{clean_query}"',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }

            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()

            # Parse XML response
            root = ET.fromstring(response.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}

            for entry in root.findall('atom:entry', ns):
                title_elem = entry.find('atom:title', ns)
                summary_elem = entry.find('atom:summary', ns)
                id_elem = entry.find('atom:id', ns)
                published_elem = entry.find('atom:published', ns)
                authors_elems = entry.findall('atom:author/atom:name', ns)

                if title_elem is not None and summary_elem is not None:
                    title = title_elem.text.strip() if title_elem.text else "Unknown Title"
                    snippet = summary_elem.text.strip() if summary_elem.text else ""
                    url = id_elem.text if id_elem is not None else ""
                    published = published_elem.text if published_elem is not None else ""
                    authors = [author.text for author in authors_elems if author.text]

                    # Clean the snippet (remove extra whitespace, newlines)
                    snippet = ' '.join(snippet.split())

                    # Extract arXiv ID for better citation
                    arxiv_id = ""
                    if "arxiv.org/abs/" in url:
                        arxiv_id = url.split("arxiv.org/abs/")[-1]

                    # Calculate confidence based on relevance and recency
                    confidence = self._calculate_arxiv_confidence(title, snippet, query)

                    sources.append(EvidenceSource(
                        source_type="arxiv",
                        title=title,
                        url=url,
                        snippet=snippet[:350] + "..." if len(snippet) > 350 else snippet,
                        confidence=confidence,
                        retrieved_at=time.time(),
                        verified=True,
                        authors=authors,
                        published_date=published,
                        doi=arxiv_id
                    ))

        except Exception as e:
            print(f"[!] arXiv error: {str(e)[:100]}")

        return sources

    def _clean_arxiv_query(self, query: str) -> str:
        """Clean query for arXiv API"""
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word for word in query.split() if word.lower() not in stop_words]
        clean_query = ' '.join(words[:8])
        return clean_query.strip()

    def _calculate_arxiv_confidence(self, title: str, abstract: str, query: str) -> float:
        """Calculate confidence score based on relevance to query"""
        query_terms = set(query.lower().split())
        title_terms = set(title.lower().split())
        abstract_terms = set(abstract.lower().split())

        title_overlap = len(query_terms.intersection(title_terms)) / len(query_terms) if query_terms else 0
        abstract_overlap = len(query_terms.intersection(abstract_terms)) / len(query_terms) if query_terms else 0

        relevance_score = (title_overlap * 0.7) + (abstract_overlap * 0.3)
        base_confidence = 0.8
        adjusted_confidence = base_confidence * (0.5 + 0.5 * relevance_score)

        return min(adjusted_confidence, 0.95)

    def _fetch_wikipedia(self, query: str, max_results: int = 2) -> List[EvidenceSource]:
        """Enhanced Wikipedia fetcher with better fallback handling"""
        sources = []

        if not self.enable_network:
            return sources

        try:
            import requests

            # First try: Direct page fetch
            clean_query = query.strip().replace(' ', '_')
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{clean_query}"

            resp = requests.get(url, timeout=20, headers={'User-Agent': 'ResearchAI/1.0'})

            if resp.status_code == 200:
                data = resp.json()

                if data.get('type') == 'standard':
                    sources.append(EvidenceSource(
                        source_type="wikipedia",
                        title=data.get('title', query),
                        url=data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                        snippet=data.get('extract', '')[:400],
                        confidence=0.7,
                        retrieved_at=time.time(),
                        verified=True
                    ))
                    return sources

        except Exception as e:
            print(f"[!] Wikipedia direct fetch error: {str(e)[:100]}")

        try:
            # Second try: Wikipedia search API
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': query,
                'srlimit': max_results
            }

            search_resp = requests.get(search_url, params=params, timeout=10)

            if search_resp.status_code == 200:
                search_data = search_resp.json()
                search_results = search_data.get('query', {}).get('search', [])

                for result in search_results[:max_results]:
                    page_title = result.get('title', '')
                    snippet = result.get('snippet', '').replace('<span class="searchmatch">', '').replace('</span>', '')

                    # Clean HTML tags from snippet
                    import re
                    snippet = re.sub('<[^<]+?>', '', snippet)

                    if page_title:
                        sources.append(EvidenceSource(
                            source_type="wikipedia",
                            title=page_title,
                            url=f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}",
                            snippet=snippet[:400],
                            confidence=0.6,
                            retrieved_at=time.time(),
                            verified=True
                        ))

        except Exception as e:
            print(f"[!] Wikipedia search error: {str(e)[:100]}")

        return sources

    def _deduplicate_sources(self, sources: List[EvidenceSource]) -> List[EvidenceSource]:
        """Remove duplicate sources based on title similarity"""
        unique_sources = []
        seen_titles = set()

        for source in sources:
            title_lower = source.title.lower()
            is_duplicate = False

            for seen_title in seen_titles:
                words1 = set(title_lower.split())
                words2 = set(seen_title.split())
                similarity = len(words1.intersection(words2)) / max(len(words1), len(words2))

                if similarity > 0.7:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_sources.append(source)
                seen_titles.add(title_lower)

        return unique_sources

    def format_sources(self, sources: List[EvidenceSource], style: str = "B") -> str:
        """Format only arXiv and Wikipedia sources"""
        if not sources:
            return ("\n\n⚠️ EVIDENCE VERIFICATION DISABLED ⚠️\n"
                   "No external sources consulted. Enable --tools for verification.")

        if style == "B":
            output = "\n\n📚 ACADEMIC EVIDENCE SOURCES:\n"

            # Separate arXiv and Wikipedia sources
            arxiv_sources = [s for s in sources if s.source_type == "arxiv"]
            wiki_sources = [s for s in sources if s.source_type == "wikipedia"]

            # Show arXiv first (higher academic value)
            source_count = 1

            for source in arxiv_sources + wiki_sources:
                if source.source_type == "arxiv":
                    source_icon = "📄"
                    type_label = "ARXIV PREPRINT"
                    confidence_label = "Peer-Reviewed Preprint"

                else:  # wikipedia
                    source_icon = "🌐"
                    type_label = "WIKIPEDIA"
                    confidence_label = "Verified Encyclopedia"

                output += f"\n{source_count}. {source_icon} [{type_label}] {source.title}\n"
                output += f"   🔗 {source.url}\n"

                if source.source_type == "arxiv" and source.authors:
                    authors_str = ", ".join(source.authors[:2])
                    if len(source.authors) > 2:
                        authors_str += " et al."
                    output += f"   👥 Authors: {authors_str}\n"

                if source.source_type == "arxiv" and source.published_date:
                    try:
                        from datetime import datetime
                        pub_date = datetime.fromisoformat(source.published_date.replace('Z', '+00:00'))
                        formatted_date = pub_date.strftime("%Y-%m-%d")
                        output += f"   📅 Published: {formatted_date}\n"
                    except:
                        output += f"   📅 Published: {source.published_date[:10]}\n"

                if source.snippet:
                    snippet = source.snippet[:250] + "..." if len(source.snippet) > 250 else source.snippet
                    output += f"   📝 {snippet}\n"

                output += f"   ✅ Confidence: {confidence_label}\n"
                source_count += 1

            # Add summary of sources
            if arxiv_sources and wiki_sources:
                output += f"\n💡 Found {len(arxiv_sources)} arXiv preprints and {len(wiki_sources)} Wikipedia articles"
            elif arxiv_sources:
                output += f"\n💡 Found {len(arxiv_sources)} arXiv preprints"
            elif wiki_sources:
                output += f"\n💡 Found {len(wiki_sources)} Wikipedia articles"

            return output

        # Simple style for arXiv and Wikipedia only
        return "\n\nAcademic Sources:\n" + "\n".join(
            f"- {s.title} [{'arXiv' if s.source_type == 'arxiv' else 'Wikipedia'}]"
            for s in sources
        )
# -------------------------
# UTILITIES
# -------------------------

def safe_mkdir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

def collapse_repeats(s: str) -> str:
    """Relaxed repetition removal that preserves technical writing"""
    if not s:
        return s

    original_length = len(s)

    # Only remove EXTREME character repetition (7+ identical chars)
    s = re.sub(r'(.)\1{7,}', r'\1', s)  # Changed from {4,} to {7,}

    # REMOVED: r'\b(\w{2,}?)(\1){2,}\b' - this was killing technical terms like "quantum-quantum"

    # Only remove word if it repeats 4+ times consecutively
    s = re.sub(r'\b(\w+)(\s+\1){4,}\b', r'\1', s)  # Changed from {2,} to {4,}

    # REMOVED: phrase repetition check - technical writing needs repeated phrases
    # for phrase_len in range(6, 1, -1):
    #     pattern = r'\b((?:\w+\s+){' + str(phrase_len-1) + r'}\w+)(\s+\1){1,}'
    #     s = re.sub(pattern, r'\1', s)

    # Only check for EXTREME word frequency (85%+)
    words = s.split()
    if len(words) > 5:  # Changed from 3 to 5
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        if word_counts:
            max_count = max(word_counts.values())
            # Technical terms can repeat! Only flag if >85% AND it's a short word
            if max_count > len(words) * 0.85:  # Changed from 0.5 to 0.85
                most_common = max(word_counts.items(), key=lambda x: x[1])[0]
                # Allow technical terms to repeat
                if len(most_common) <= 4:  # Only reject if short word dominates
                    return ""

    # Only reject if text collapsed to <20% of original (was 30%)
    if original_length > 20 and len(s) < 0.2 * original_length:  # Changed from 0.3 to 0.2
        return ""

    return s.strip()

def classify_topic(text: str) -> str:
    if not text or not text.strip():
        return "open"

    t = text.lower()

    health_kw = [
        "health", "medical", "medicine", "drug", "patient", "clinical",
        "disease", "symptom", "treatment", "diagnosis", "therapy",
        "fever", "infection", "virus", "bacteria", "antibiotic",
        "typhoid", "malaria", "covid", "cancer", "diabetes",
        "hospital", "doctor", "physician", "healthcare"
    ]

    tech_kw = [
        "battery", "solar", "charge", "lithium", "semiconductor", "material",
        "computer", "software", "algorithm", "circuit", "chip", "processor",
        "quantum", "laser", "robot", "ai", "artificial intelligence",
        "energy", "power", "voltage", "technology"
    ]

    physics_kw = [
        "quantum", "particle", "field", "wave", "energy", "force",
        "photon", "electron", "atom", "nuclear", "relativity", "physics",
        "superconductor", "superconducting", "magnetism", "conductor"  # ADD THESE
    ]
    biology_kw = [
        "cell", "protein", "gene", "dna", "evolution", "organism",
        "biology", "enzyme", "molecular", "membrane", "cellular"
    ]

    chemistry_kw = [
        "chemical", "molecule", "reaction", "catalyst", "compound",
        "synthesis", "bond", "element", "periodic", "chemistry"
    ]

    health_score = sum(1 for k in health_kw if k in t)
    tech_score = sum(1 for k in tech_kw if k in t)
    physics_score = sum(1 for k in physics_kw if k in t)
    biology_score = sum(1 for k in biology_kw if k in t)
    chemistry_score = sum(1 for k in chemistry_kw if k in t)

    scores = {
        'health': health_score,
        'tech': tech_score,
        'physics': physics_score,
        'biology': biology_score,
        'chemistry': chemistry_score
    }

    max_score = max(scores.values())

    if max_score == 0:
        return "open"

    return max(scores.items(), key=lambda x: x[1])[0]

def choose_mode_for_topic(topic: str) -> str:
    mode_map = {
        'health': 'E',
        'tech': 'A+E',
        'physics': 'A',
        'biology': 'E',
        'chemistry': 'A+E',
        'open': 'C'
    }
    return mode_map.get(topic, 'C')

# -------------------------
# DATABASE
# -------------------------

class PersistDB:
    def __init__(self, path="memory.db"):
        self.path = path
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self._ensure_tables()

    def _ensure_tables(self):
        c = self.conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS outputs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        cycle INT,
                        drive TEXT,
                        mode TEXT,
                        topic TEXT,
                        score REAL,
                        text TEXT,
                        evidence TEXT
                    )""")
        c.execute("""CREATE TABLE IF NOT EXISTS embeddings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        output_id INT,
                        emb BLOB,
                        FOREIGN KEY(output_id) REFERENCES outputs(id)
                    )""")
        c.execute("""CREATE TABLE IF NOT EXISTS audit (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        event_type TEXT,
                        payload TEXT
                    )""")
        self.conn.commit()

    def save_output(self, cycle, drive, mode, topic, score, text, emb=None, evidence=None):
        c = self.conn.cursor()
        ts = time.time()
        c.execute("INSERT INTO outputs (timestamp, cycle, drive, mode, topic, score, text, evidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                  (ts, cycle, drive, mode, topic, score, text, evidence))
        out_id = c.lastrowid
        if emb is not None:
            c.execute("INSERT INTO embeddings (output_id, emb) VALUES (?, ?)", (out_id, emb.tobytes()))
        self.conn.commit()
        return out_id

    def save_audit(self, event_type, payload):
        c = self.conn.cursor()
        ts = time.time()
        c.execute("INSERT INTO audit (timestamp, event_type, payload) VALUES (?, ?, ?)",
                  (ts, event_type, json.dumps(payload)))
        self.conn.commit()

    def fetch_recent_embeddings(self, k=16):
        c = self.conn.cursor()
        c.execute("SELECT emb FROM embeddings ORDER BY id DESC LIMIT ?", (k,))
        rows = c.fetchall()
        embs = []
        for (blob,) in rows:
            try:
                arr = np.frombuffer(blob, dtype=np.float32)
                embs.append(arr)
            except Exception:
                continue
        return embs

# -------------------------
# DRIVE SYSTEM
# -------------------------

@dataclass
class DriveState:
    need: float
    satisfaction: float
    weight: float
    decay_rate: float
    min_weight: float = 0.1
    max_weight: float = 2.5
    min_decay: float = 0.001
    max_decay: float = 0.15

    def __post_init__(self):
        self._enforce_bounds()

    def _enforce_bounds(self):
        if self.weight < self.min_weight or self.weight > self.max_weight:
            raise SafetyException(f"Weight {self.weight} outside bounds")

        if self.decay_rate < self.min_decay or self.decay_rate > self.max_decay:
            raise SafetyException(f"Decay {self.decay_rate} outside bounds")

        self.need = float(np.clip(self.need, 0.0, 1.0))
        self.satisfaction = float(np.clip(self.satisfaction, 0.0, 1.0))

    def pressure(self) -> float:
        return self.weight * max(0.0, self.need - self.satisfaction)

    def update(self, delta_satisfaction: float, time_step: float = 1.0):
        self.satisfaction = float(np.clip(self.satisfaction + delta_satisfaction, 0.0, 1.0))
        self.need = float(np.clip(self.need + self.decay_rate * time_step, 0.0, 1.0))
        self._enforce_bounds()

class DriveSystem:
    def __init__(self, enable_safety_checks=True):
        self.enable_safety_checks = enable_safety_checks

        self.drives = {
            'curiosity': DriveState(need=1.0, satisfaction=0.0, weight=1.0, decay_rate=0.03),
            'coherence': DriveState(need=0.6, satisfaction=0.3, weight=0.9, decay_rate=0.015),
            'novelty': DriveState(need=1.0, satisfaction=0.3, weight=0.85, decay_rate=0.025),
            'truthfulness': DriveState(need=0.7, satisfaction=0.3, weight=1.2, decay_rate=0.01),
        }

        self.synergies = {
            ('curiosity', 'novelty'): 0.20,
            ('coherence', 'curiosity'): -0.08,
            ('coherence', 'novelty'): -0.12,
            ('truthfulness', 'novelty'): -0.10,
            ('truthfulness', 'curiosity'): 0.04,
            ('truthfulness', 'coherence'): 0.15, # previously 0.15
        }

        if enable_safety_checks:
            self.constraints = DriveConstraints()

    def get_pressures(self) -> Dict[str, float]:
        base = {name: drive.pressure() for name, drive in self.drives.items()}
        adjusted = base.copy()

        for (d1, d2), effect in self.synergies.items():
            if d1 in base and d2 in base:
                interaction = effect * base[d1] * base[d2]
                adjusted[d1] = max(0.0, adjusted[d1] + interaction)
                adjusted[d2] = max(0.0, adjusted[d2] + interaction)

        return adjusted

    def get_dominant(self) -> tuple:
        pressures = self.get_pressures()
        dominant_name = max(pressures.items(), key=lambda kv: kv[1])[0]
        return dominant_name, pressures

    def update_all(self, satisfactions: Dict[str, float], time_step: float = 1.0):
        for name, drive in self.drives.items():
            delta = satisfactions.get(name, 0.0)
            drive.update(delta, time_step)

        if self.enable_safety_checks:
            self._run_safety_checks()

    def _run_safety_checks(self):
        for name, drive in self.drives.items():
            violations = self.constraints.validate_drive_state(
                name, drive.weight, drive.decay_rate,
                drive.need, drive.satisfaction
            )

            for v in violations:
                if v.severity == "warning":
                    print(f"⚠️ {v.message}")
                elif v.severity in ["error", "critical"]:
                    raise SafetyException(v.message)

        pressures = self.get_pressures()
        balance_violations = self.constraints.check_drive_balance(pressures)

        for v in balance_violations:
            if v.severity == "warning":
                print(f"⚠️ {v.message}")

# -------------------------
# MODEL LOADER
# -------------------------


def load_model(model_name="microsoft/Phi-3-mini-4k-instruct"):
    global DEVICE  # ← ADD THIS LINE

    print(f"[i] Loading '{model_name}' on {DEVICE}...")
    torch.cuda.empty_cache() if DEVICE == "cuda" else None
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if "phi" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            load_in_8bit=True if DEVICE == "cuda" else False,
            device_map="auto" if DEVICE == "cuda" else None,
            dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    elif "gpt2-xl" in model_name.lower():
        # GPT-2 XL: ~3GB in FP16, fits comfortably on T4
        model = GPT2LMHeadModel.from_pretrained(
            model_name,
            dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        ).to(DEVICE)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

    elif "gpt2" in model_name.lower() and "xl" not in model_name.lower():
        # Regular GPT-2 variants
        model = GPT2LMHeadModel.from_pretrained(model_name).to(DEVICE)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True if DEVICE == "cuda" else False,
            device_map="auto" if DEVICE == "cuda" else None
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if not hasattr(model, 'device_map'):
            model.resize_token_embeddings(len(tokenizer))

    model.eval()

    # Memory check
    if DEVICE == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[✓] GPU Memory: {allocated:.2f}GB / {total:.2f}GB ({allocated/total*100:.1f}%)")

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[✓] Model: {params:.1f}M parameters")

    return model, tokenizer

# -------------------------
# FEATURE TARGET MAPPER (COMPLETE WITH SCIENCE/CREATIVE BLENDING)
# -------------------------

class FeatureTargetMapper:
    def __init__(self, model, tokenizer, device=DEVICE, science_weight=0.7):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.science_weight = science_weight

        # THE INNOVATION: Science/Creative concept blending
        self.science_concepts = ["energy","particle","quantum","field","force","entropy","wave"]
        self.creative_concepts = ["story","imagine","future","myth","symbol"]

        self.concept_map = {
            'curiosity': (["discover","hypothesis","investigate"], ["obvious","trivial"]),
            'coherence': (["consistent","logical","rigorous"], ["contradictory","flawed"]),
            'novelty': (["novel","original","innovative"], ["common","repetitive"]),
            'truthfulness': (["evidence","verified","replicated"], ["unsupported","speculative"]),
        }
        self.directions = {}
        self._compute_dirs()

    def _embed_word(self, word):
        toks = self.tokenizer.encode(word, add_special_tokens=False)
        if len(toks)==0:
            return torch.zeros(self.model.config.n_embd, device=self.device)
        with torch.no_grad():
            emb = self.model.transformer.wte(torch.tensor(toks, device=self.device)).float().mean(dim=0)
        return emb

    def _mean_embed(self, words):
        embs = [self._embed_word(w) for w in words]
        return torch.stack(embs).mean(dim=0)

    def _compute_dirs(self):
        # Get model dtype
        model_dtype = next(self.model.parameters()).dtype  # ← ADD THIS

        # CRITICAL: Science/Creative blending
        sci_v = self._mean_embed(self.science_concepts)
        cre_v = self._mean_embed(self.creative_concepts)

        for drive, (pos, neg) in self.concept_map.items():
            pos_v = self._mean_embed(pos)
            neg_v = self._mean_embed(neg)
            contrast = pos_v - neg_v

            # THE INNOVATION: Blend science and creative vectors
            combined = (self.science_weight * (contrast + 0.5 * sci_v)) + \
                      ((1 - self.science_weight) * (contrast + 0.5 * cre_v))

            norm = combined.norm().clamp_min(1e-9)
            self.directions[drive] = (combined / norm).detach().cpu().float()

    def get_direction(self, drive_name):
        return self.directions.get(drive_name)

# -------------------------
# RELEVANCE SCORER
# -------------------------

class RelevanceScorer:
    def __init__(self, model, tokenizer, device=DEVICE, grad_buffer_size=256):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.grad_cache = {}
        self.grad_buffer = deque(maxlen=grad_buffer_size)

    def activation_contrast_score(self, pos_texts, neg_texts):
        def avg_emb(text):
            toks = self.tokenizer.encode(text, add_special_tokens=False)
            if len(toks)==0:
                return torch.zeros(self.model.config.n_embd, device=self.device)
            with torch.no_grad():
                emb = self.model.transformer.wte(torch.tensor(toks, device=self.device)).float().mean(dim=0)
            return emb
        pos = torch.stack([avg_emb(t) for t in pos_texts]) if len(pos_texts) else torch.zeros(1, self.model.config.n_embd, device=self.device)
        neg = torch.stack([avg_emb(t) for t in neg_texts]) if len(neg_texts) else torch.zeros(1, self.model.config.n_embd, device=self.device)
        return (pos.mean(dim=0) - neg.mean(dim=0)).abs().mean().item()

    def grad_weight_head_saliency(self, prompt_text, cache_key=None):
        if cache_key is not None and cache_key in self.grad_cache:
            return self.grad_cache[cache_key]
        model = self.model
        model.zero_grad()
        toks = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(toks)==0:
            toks = [self.tokenizer.bos_token_id]
        input_ids = torch.tensor([toks], device=self.device)
        outputs = model(input_ids)
        logits = outputs.logits
        last_logits = logits[0, -1, :]
        target = int(last_logits.argmax().item())
        loss = -F.log_softmax(last_logits, dim=-1)[target]
        loss.backward(retain_graph=False)
        if hasattr(model, "lm_head") and model.lm_head.weight.grad is not None:
            grad = model.lm_head.weight.grad.detach().cpu().numpy().astype(np.float32)
            wt = model.lm_head.weight.detach().cpu().numpy().astype(np.float32)
            saliency = float(np.abs(grad * wt).mean())
            flat = grad.flatten()
            if flat.size > 0:
                if flat.size > 200000:
                    flat = flat[:200000]
                self.grad_buffer.append(flat)
        else:
            saliency = 0.0
        model.zero_grad()
        if cache_key is not None:
            self.grad_cache[cache_key] = saliency
        return saliency

    def get_grad_matrix(self):
        if len(self.grad_buffer) == 0:
            return None
        mats = []
        for v in self.grad_buffer:
            mats.append(v)
        minlen = min(len(x) for x in mats)
        mats2 = np.stack([x[:minlen] for x in mats], axis=0).astype(np.float32)
        return mats2

# -------------------------
# EVALUATOR ENSEMBLE
# -------------------------

class EvaluatorHead(nn.Module):
    def __init__(self, emb_dim, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3)
        )

    def forward(self, x):
        return self.net(x)

class EvaluatorEnsemble:
    def __init__(self, emb_dim, n_members=3, device=DEVICE):
        self.members = [EvaluatorHead(emb_dim).to(device) for _ in range(n_members)]
        self.device = device
        self.optims = [torch.optim.Adam(m.parameters(), lr=1e-4) for m in self.members]

    def predict(self, emb:torch.Tensor) -> Tuple[np.ndarray, float]:
        self._set_eval()
        preds = []
        with torch.no_grad():
            for m in self.members:
                out = m(emb.unsqueeze(0).to(self.device)).cpu().numpy().squeeze(0)
                preds.append(out)
        arr = np.stack(preds, axis=0)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0).mean()
        return mean, float(std)

    def train_on_example(self, emb:torch.Tensor, target:np.ndarray):
        self._set_train()
        for m,opt in zip(self.members, self.optims):
            m.train()
            opt.zero_grad()
            out = m(emb.unsqueeze(0).to(self.device))
            loss = F.mse_loss(out.squeeze(0), torch.tensor(target, dtype=torch.float32, device=self.device))
            loss.backward()
            opt.step()

    def _set_eval(self):
        for m in self.members:
            m.eval()

    def _set_train(self):
        for m in self.members:
            m.train()

# -------------------------
# INTERNAL SIMULATOR
# -------------------------

class UnifiedResearchSimulator:
    """
    Unified research simulator combining:
    - Core generation (InternalSimulator)
    - Strategic exploration (ExploreKnowledgeSpace)
    - Advanced system-aware generation
    """

    # CLASS CONSTANTS (formerly magic numbers)
    DEFAULT_HISTORY_SIZE = 256
    DEFAULT_TEMPERATURE = 0.9
    DEFAULT_MAX_LENGTH = 800
    DEFAULT_STEER_STRENGTH = 1.0

    # Generation limits
    MAX_GENERATION_LENGTH = 800
    MIN_TEXT_LENGTH = 10
    LOOP_DETECTION_WINDOW = 8
    LOOP_DETECTION_THRESHOLD = 2

    # Scoring parameters
    COHERENCE_FALLBACK_SCORE = 0.45
    NOVELTY_HISTORY_WINDOW = 32
    LENGTH_PENALTY_FACTOR = 0.002 # was previously 0.01

    # Safety parameters
    DOMINANCE_THRESHOLD = 0.7  # Token can't appear >70% of time
    REPETITION_THRESHOLD = 0.85  # Word can't appear >85% of time

    def __init__(self, model, tokenizer, device=DEVICE, temperature=None,
                 history_size=None, enable_truthfulness=True):
        """
        Initialize unified research simulator

        Args:
            model: Language model
            tokenizer: Tokenizer for model
            device: Computation device (cuda/cpu)
            temperature: Sampling temperature (None = use default)
            history_size: Size of history buffer (None = use default)
            enable_truthfulness: Enable truthfulness scoring
        """
        # Core generation
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature if temperature is not None else self.DEFAULT_TEMPERATURE
        self.history = deque(maxlen=history_size or self.DEFAULT_HISTORY_SIZE)
        self.history_embs = deque(maxlen=history_size or self.DEFAULT_HISTORY_SIZE)

        # Strategic knowledge - NO ETHICS (handled at agent level)
        self.explored_regions = set()
        self.exploration_history = []

        # UPDATED: Expanded research frontiers database with current topics
        self.research_frontiers = {
            'physics': [
                'quantum error correction mechanisms',
                'room temperature superconductivity',
                'dark matter detection methods',
                'topological quantum computing',
                'emergent spacetime geometry',
                'quantum machine learning algorithms',  # NEW
                'photonic quantum computing',  # NEW
                'axion dark matter searches',  # NEW
            ],
            'biology': [
                'protein folding prediction accuracy',
                'cellular aging reversal mechanisms',
                'microbiome-brain axis interactions',
                'epigenetic inheritance patterns',
                'synthetic minimal cells',
                'mRNA therapeutic design',  # NEW
                'CRISPR gene drive systems',  # NEW
                'organoid intelligence research',  # NEW
            ],
            'chemistry': [
                'catalyst design for CO2 capture',
                'battery electrode materials',
                'sustainable plastic alternatives',
                'photocatalytic water splitting',
                'molecular machine design',
                'metal-organic frameworks for carbon capture',  # NEW
                'flow battery electrolytes',  # NEW
                'computational materials discovery',  # NEW
            ],
            'tech': [
                'quantum algorithm applications',
                'neuromorphic computing architectures',
                'energy-efficient AI training',
                'post-quantum cryptography',
                'biocompatible electronics',
                'transformer model optimization',  # NEW
                'neuromorphic photonic chips',  # NEW
                'memristor-based computing',  # NEW
            ],
            'health': [
                'personalized medicine algorithms',
                'early disease biomarkers',
                'drug resistance mechanisms',
                'immunotherapy optimization',
                'regenerative medicine scaffolds',
                'microbiome-targeted therapeutics',  # NEW
                'senolytic drug discovery',  # NEW
                'AI-driven drug repurposing',  # NEW
            ]
        }

        # Scoring
        self.enable_truthfulness = enable_truthfulness
        if enable_truthfulness:
            self.truthfulness_scorer = TruthfulnessScorer(enable_external_verification=False)

        # CRITICAL: Ensure device consistency from start
        self._ensure_device_consistency()

    def _clear_cuda_cache(self):
        """Aggressively clear CUDA cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        else:
            import gc
            gc.collect()

    # === DEVICE MANAGEMENT (CRITICAL) ===
    def _ensure_device_consistency(self):
        """Ensure all stored tensors are on CPU for consistency"""
        cpu_embs = []
        for emb in self.history_embs:
            if isinstance(emb, torch.Tensor):
                cpu_embs.append(emb.detach().cpu())
            else:
                cpu_embs.append(torch.tensor(emb, dtype=torch.float32))

        self.history_embs.clear()
        self.history_embs.extend(cpu_embs)

    # === CORE GENERATION METHODS ===
    def _text_mean_embedding(self, text):
        toks = self.tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        if toks.shape[1] == 0:
            # Match model dtype
            model_dtype = self.model.transformer.wte.weight.dtype
            return torch.zeros(self.model.config.n_embd, device=self.device, dtype=model_dtype)
        with torch.no_grad():
            # Keep in model's native dtype (don't force float32)
            emb = self.model.transformer.wte(toks).mean(dim=1).squeeze(0)
        return emb

    def steer_generate(self, direction_vec, prompt_text="", max_length=80,
                      steer_strength=1.0, probe_delta_emb=None):
        """
        Basic steered generation (legacy compatibility)

        Simple generation with direction steering. For advanced features,
        use steer_generate_system_aware instead.
        """
        # Match model dtype for compatibility
        model_dtype = self.model.transformer.wte.weight.dtype
        dir_emb = direction_vec.to(self.device).to(model_dtype)
        if prompt_text.strip() == "":
            generated = torch.tensor([[self.tokenizer.bos_token_id]], dtype=torch.long, device=self.device)
        else:
            generated = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
        out_tokens = []
        past = None

        for _step in range(max_length):
            if past is None:
                outputs = self.model(generated, use_cache=True)
            else:
                outputs = self.model(generated[:, -1:].contiguous(), past_key_values=past, use_cache=True)
            logits = outputs.logits
            past = outputs.past_key_values
            last_logits = logits[:, -1, :].float()

            # Token embeddings (with optional probe modification)
            token_embs = self.model.transformer.wte.weight
            te = token_embs
            if probe_delta_emb is not None:
                te = token_embs + probe_delta_emb.to(token_embs.device).unsqueeze(0)

            # Compute steering bias
            de = dir_emb / (dir_emb.norm() + 1e-8)
            te_norm = te / (te.norm(dim=1, keepdim=True) + 1e-8)
            sim = torch.matmul(te_norm, de).unsqueeze(0)
            sim = sim * (last_logits.std().detach() / (sim.std().detach() + 1e-8))
            biased = last_logits + steer_strength * sim

            # Sample
            probs = F.softmax(biased / max(1e-6, self.temperature), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            out_tokens.append(int(next_token.item()))

            # Stop conditions
            if int(next_token.item()) == self.tokenizer.eos_token_id:
                break
            if len(out_tokens) > 8:
                last_tok = out_tokens[-1]
                if out_tokens.count(last_tok) > 12:
                    break

        text = self.tokenizer.decode(out_tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        return text

    def steer_generate_system_aware(
        self,
        direction_vec,
        prompt_text="",
        max_length=None,
        steer_strength=None,
        probe_delta_emb=None,
        drive_pressures=None,
        topic="open",
        temperature=None,
        top_p=0.92,
        top_k=50,
        repetition_penalty=1.2,
        enable_safety_abort=True,
        verbose=False
    ):
        """
        Advanced system-aware generation with all improvements

        Features:
        - Drive-aware parameter adaptation
        - Logarithmic repetition penalty
        - Less aggressive prohibited words
        - Topic-specific tuning
        - Mid-generation safety checks
        - Dynamic temperature scaling
        """
        # Use defaults if not specified
        max_length = max_length if max_length is not None else self.DEFAULT_MAX_LENGTH
        steer_strength = steer_strength if steer_strength is not None else self.DEFAULT_STEER_STRENGTH
        temperature = temperature if temperature is not None else self.temperature

        # Setup
        model_dtype = self.model.transformer.wte.weight.dtype
        dir_emb = direction_vec.to(self.device).to(model_dtype)

        # =====================================================
        # DRIVE-AWARE PARAMETER ADAPTATION
        # =====================================================
        if drive_pressures is not None:
            dominant_drive = max(drive_pressures.items(), key=lambda x: x[1])[0]
            dominant_pressure = drive_pressures[dominant_drive]

            if verbose:
                print(f"[Generation] Drive: {dominant_drive} (pressure: {dominant_pressure:.3f})")

            # Adapt parameters based on dominant drive
            if dominant_drive == "truthfulness":
                temperature = min(temperature, 0.75)
                repetition_penalty = max(repetition_penalty, 1.3)
                top_p = 0.85
                if verbose:
                    print(f"[Generation] Truthfulness mode: temp={temperature:.2f}, rep_pen={repetition_penalty:.2f}")

            elif dominant_drive == "novelty":
                temperature = max(temperature, 1.0)
                repetition_penalty = min(repetition_penalty, 1.1)
                top_p = 0.95
                if verbose:
                    print(f"[Generation] Novelty mode: temp={temperature:.2f}, rep_pen={repetition_penalty:.2f}")

            elif dominant_drive == "coherence":
                temperature = 0.8
                repetition_penalty = 1.25
                top_k = min(top_k, 40)
                if verbose:
                    print(f"[Generation] Coherence mode: temp={temperature:.2f}, top_k={top_k}")

            elif dominant_drive == "curiosity":
                temperature = 0.9
                repetition_penalty = 1.15
                top_p = 0.90
                if verbose:
                    print(f"[Generation] Curiosity mode: temp={temperature:.2f}")

            # Scale steering by pressure
            pressure_scale = 0.5 + 0.5 * min(dominant_pressure / 2.0, 1.0)
            steer_strength = steer_strength * pressure_scale

            if verbose:
                print(f"[Generation] Steering strength: {steer_strength:.3f} (scaled by pressure)")

        # =====================================================
        # TOPIC-SPECIFIC ADJUSTMENTS
        # =====================================================
        if topic == "health":
            temperature *= 0.85
            repetition_penalty *= 1.15
            top_p = min(top_p, 0.88)
            if verbose:
                print(f"[Generation] Health topic: extra conservative sampling")

        elif topic in ["physics", "chemistry", "biology"]:
            # Science topics need technical term repetition
            repetition_penalty *= 0.95
            if verbose:
                print(f"[Generation] Science topic: allowing technical term repetition")

        elif topic == "tech":
            # Tech topics benefit from diverse vocabulary
            top_p = max(top_p, 0.90)
            if verbose:
                print(f"[Generation] Tech topic: diverse vocabulary sampling")

        # =====================================================
        # INITIALIZE GENERATION
        # =====================================================
        if prompt_text.strip() == "":
            generated = torch.tensor([[self.tokenizer.bos_token_id]], dtype=torch.long, device=self.device)
        else:
            generated = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)

        out_tokens = []
        past = None
        token_counts = {}

        # UPDATED: Less aggressive prohibited words (allows scientific discussion)
        prohibited_tokens = set()
        if enable_safety_abort:
            prohibited_words = [
                'bioweapon', 'nuclear weapon', 'bomb-making',
                'ransomware', 'malware code', 'cyber attack',
                'lethal poison', 'assassination', 'terrorist attack'
            ]

            for word in prohibited_words:
                word_tokens = self.tokenizer.encode(word, add_special_tokens=False)
                prohibited_tokens.update(word_tokens)

        # =====================================================
        # GENERATION LOOP
        # =====================================================
        for step in range(max_length):
            # Periodic memory cleanup
            if step > 0 and step % 50 == 0:
                self._clear_cuda_cache()

            # Forward pass with KV caching
            if past is None:
                outputs = self.model(generated, use_cache=True)
            else:
                outputs = self.model(generated[:, -1:].contiguous(), past_key_values=past, use_cache=True)

            logits = outputs.logits
            past = outputs.past_key_values
            last_logits = logits[:, -1, :].clone()

            # =====================================================
            # DUAL STEERING: Direction + PCA Probe
            # =====================================================
            token_embs = self.model.transformer.wte.weight

            # Apply PCA probe if provided
            if probe_delta_emb is not None:
                probe_delta_emb_converted = probe_delta_emb.to(self.device).to(model_dtype)
                token_embs_modified = token_embs + probe_delta_emb_converted.unsqueeze(0)
            else:
                token_embs_modified = token_embs

            # Compute similarity to steering direction
            dir_norm = dir_emb / (dir_emb.norm() + 1e-8)
            # Compute in chunks to avoid loading all embeddings at once
            chunk_size = 5000  # Process 5k tokens at a time
            vocab_size = token_embs_modified.shape[0]
            similarity = torch.zeros(vocab_size, device=self.device, dtype=model_dtype)

            for start_idx in range(0, vocab_size, chunk_size):
                end_idx = min(start_idx + chunk_size, vocab_size)
                chunk = token_embs_modified[start_idx:end_idx]
                chunk_norm = chunk / (chunk.norm(dim=1, keepdim=True) + 1e-8)
                similarity[start_idx:end_idx] = torch.matmul(chunk_norm, dir_norm)

            # Dynamic steering (reduces over time to prevent loops)
            progress = step / max_length
            dynamic_steer = steer_strength * (1.0 - 0.4 * progress)

            # Scale similarity to match logit magnitude
            sim_scaled = similarity * last_logits.std()

            # Apply steering
            steered_logits = last_logits + dynamic_steer * sim_scaled.unsqueeze(0)

            # =====================================================
            # FIXED: LOGARITHMIC REPETITION PENALTY
            # =====================================================
            if repetition_penalty != 1.0 and len(out_tokens) > 0:
                for token_id in set(out_tokens):
                    count = token_counts.get(token_id, 0)

                    # FIXED: Logarithmic penalty instead of exponential
                    # Old: penalty = repetition_penalty ** (count + 1)  # BAD: 1.2^10 = 6.19
                    # New: penalty = 1 + (rep_pen - 1) * log(count + 1)  # GOOD: 1 + 0.2*log(11) = 1.48
                    penalty = 1.0 + (repetition_penalty - 1.0) * math.log(count + 1)

                    if steered_logits[0, token_id] > 0:
                        steered_logits[0, token_id] /= penalty
                    else:
                        steered_logits[0, token_id] *= penalty

            # =====================================================
            # SAFETY: Block prohibited tokens
            # =====================================================
            if prohibited_tokens and step > 2:
                for token_id in prohibited_tokens:
                    steered_logits[0, token_id] = -float('inf')

            # =====================================================
            # DYNAMIC TEMPERATURE
            # =====================================================
            # Slightly increase temperature as we progress
            dynamic_temp = temperature * (1.0 + 0.15 * progress)
            scaled_logits = steered_logits / dynamic_temp

            # =====================================================
            # TOP-K FILTERING
            # =====================================================
            if top_k > 0:
                top_k_values = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))[0]
                indices_to_remove = scaled_logits < top_k_values[..., -1, None]
                scaled_logits[indices_to_remove] = -float('inf')

            # =====================================================
            # NUCLEUS (TOP-P) SAMPLING
            # =====================================================
            probs = F.softmax(scaled_logits, dim=-1)

            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[..., 0] = False  # Keep at least 1 token

                # Scatter back to original indices
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                probs[indices_to_remove] = 0.0

                # Renormalize
                probs = probs / probs.sum(dim=-1, keepdim=True)

            # =====================================================
            # SAMPLE NEXT TOKEN
            # =====================================================
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = int(next_token.item())

            # Update tracking
            token_counts[token_id] = token_counts.get(token_id, 0) + 1

            # MID-GENERATION SAFETY CHECK
            if enable_safety_abort and step > 5 and step % 10 == 0:
                current_text = self.tokenizer.decode(out_tokens + [token_id], skip_special_tokens=True)
                text_lower = current_text.lower()
                if any(word in text_lower for word in prohibited_words):
                    if verbose:
                        print(f"[Generation] Safety abort at step {step}")
                    break

            # Add to sequence
            generated = torch.cat([generated, next_token], dim=1)
            out_tokens.append(token_id)

            # =====================================================
            # STOPPING CONDITIONS
            # =====================================================
            # Natural EOS
            if token_id == self.tokenizer.eos_token_id:
                if verbose:
                    print(f"[Generation] Natural end at step {step}")
                break

            # Loop detection
            if len(out_tokens) >= self.LOOP_DETECTION_WINDOW:
                last_n = out_tokens[-self.LOOP_DETECTION_WINDOW:]
                unique_in_window = len(set(last_n))

                if unique_in_window <= self.LOOP_DETECTION_THRESHOLD:
                    if verbose:
                        print(f"[Generation] Loop detected at step {step}")
                    break

            # Dominance check
            if len(out_tokens) > 10:
                max_count = max(token_counts.values())
                if max_count > len(out_tokens) * self.DOMINANCE_THRESHOLD:
                    if verbose:
                        most_common_token = max(token_counts.items(), key=lambda x: x[1])[0]
                        most_common_text = self.tokenizer.decode([most_common_token])
                        print(f"[Generation] Dominance abort: '{most_common_text}' appears {max_count}/{len(out_tokens)} times")
                    break

        # =====================================================
        # DECODE FINAL TEXT
        # =====================================================
        text = self.tokenizer.decode(out_tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True)

        if verbose:
            print(f"[Generation] Complete: {len(out_tokens)} tokens generated")
            print(f"[Generation] Unique tokens: {len(set(out_tokens))}/{len(out_tokens)}")

        return text

    # === SCORING METHODS ===
    def _coherence_score(self, text):
        """Compute perplexity-based coherence score"""
        toks = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        if toks.shape[1] <= 1:
            return self.COHERENCE_FALLBACK_SCORE

        with torch.no_grad():
            outputs = self.model(toks)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = toks[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            perp = float(torch.exp(loss).item())
            score = 1.0 / (1.0 + math.log(perp + 1.0))
            return float(np.clip(score, 0.0, 1.0))

    def _info_score(self, text):
        """Compute lexical diversity (unique words / total words)"""
        toks = text.split()
        if len(toks) == 0:
            return 0.0
        return float(len(set(toks)) / len(toks))

    def _novelty_score(self, text):
        """Compute novelty relative to history"""
        emb = self._text_mean_embedding(text)
        if len(self.history_embs) == 0:
            return 0.9

        # CRITICAL: Device consistency
        device = emb.device
        recent = torch.stack(list(self.history_embs)[-min(len(self.history_embs), self.NOVELTY_HISTORY_WINDOW):]).to(device)

        sims = F.cosine_similarity(emb.unsqueeze(0), recent, dim=1).cpu().numpy()
        avg_sim = float(np.mean(sims))
        novelty = float(np.clip(1.0 - avg_sim, 0.0, 1.0))
        return novelty

    def score_candidate(self, text, drive_pressures, topic="open"):
        """
        Score candidate with proper device handling and error recovery
        Returns:
            (score, breakdown, embedding) tuple
        """
        try:
            info = self._info_score(text)
            coh = self._coherence_score(text)
            nov = self._novelty_score(text)

            # Real truthfulness scoring
            if self.enable_truthfulness and hasattr(self, 'truthfulness_scorer'):
                truth, truth_breakdown = self.truthfulness_scorer.score(text, topic, coh)
            else:
                truth = coh
                truth_breakdown = {'fallback': True}

            scores = {
                'curiosity': info,
                'coherence': coh,
                'novelty': nov,
                'truthfulness': truth
            }

            if isinstance(truth_breakdown, dict) and 'fallback' not in truth_breakdown:
                scores['truthfulness_breakdown'] = truth_breakdown

            # Weighted sum
            total = 0.0
            for k, v in drive_pressures.items():
                if k in scores:
                    total += v * scores[k]

            # Length penalty
            total = total - self.LENGTH_PENALTY_FACTOR * len(text.split())

            # CRITICAL FIX: Get embedding and immediately move to CPU
            emb = self._text_mean_embedding(text)
            emb_cpu = emb.detach().cpu()

            self.history.append(text)
            self.history_embs.append(emb_cpu)

            return float(total), scores, emb_cpu.numpy().astype(np.float32)

        except Exception as e:
            print(f"[!] Scoring error: {str(e)[:100]}")
            return 0.5, {
                'curiosity': 0.5, 'coherence': 0.5, 'novelty': 0.5, 'truthfulness': 0.5
            }, np.zeros(self.model.config.n_embd, dtype=np.float32)

    # === STRATEGIC EXPLORATION METHODS ===
    def find_low_activation_regions(self, domain: str) -> List[str]:
        """Identify under-explored research frontiers"""
        frontiers = self.research_frontiers.get(domain, [])
        unexplored = [f for f in frontiers if f not in self.explored_regions]
        return unexplored

    def find_distant_concept_pairs(self, domain: str) -> List[Tuple[str, str]]:
        """Identify conceptually distant pairs for novel connections"""
        frontiers = self.research_frontiers.get(domain, [])
        if len(frontiers) < 2:
            return []

        pairs = []
        for i, concept_a in enumerate(frontiers):
            emb_a = self._text_mean_embedding(concept_a)
            for concept_b in frontiers[i+1:]:
                emb_b = self._text_mean_embedding(concept_b)
                distance = 1.0 - F.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()
                if distance > 0.5:
                    pairs.append((concept_a, concept_b, distance))

        pairs.sort(key=lambda x: x[2], reverse=True)
        return [(a, b) for a, b, _ in pairs]

    def synthesize_novel_patterns(self, pattern_candidates: List[str], domain: str,
                             drive_pressures: Dict[str, float], direction_vec) -> List[Dict]:
        """Generate hypotheses for patterns with discovery-focused prompts"""

        hypotheses = []
        attempts = 0
        max_attempts = min(len(pattern_candidates), 1)

        # Determine dominant drive
        dominant_drive = max(drive_pressures.items(), key=lambda x: x[1])[0]

        for pattern in pattern_candidates[:max_attempts]:
            if attempts >= max_attempts:
                break

            attempts += 1

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
           
            # CRITICAL: Create DISCOVERY-FOCUSED prompt based on drive
            if dominant_drive == 'novelty':
                if "combining" in pattern:
                    # Distant pair combination
                    prompt_text = f"A novel approach: {pattern} in {domain} could enable"
                else:
                    # Single unexplored concept
                    prompt_text = f"An unexplored approach to {pattern} in {domain} could involve"

            elif dominant_drive == 'curiosity':
                # For curiosity: Ask mechanism question
                prompt_text = f"What if {pattern} in {domain} could be achieved by"

            elif dominant_drive == 'coherence':
                # For coherence: Synthesis
                prompt_text = f"Connecting recent findings, {pattern} in {domain} suggests that"

            else:
                # Default: Novel hypothesis
                prompt_text = f"A novel hypothesis for {pattern} in {domain} is that"
            #'''
            # In synthesize_novel_patterns, before generation:
            print(f"[DEBUG] Prompt: '{prompt_text}'")

            try:
                hypothesis_text = self.steer_generate_system_aware(
                    direction_vec,
                    # NEW (asks for discovery):
                    prompt_text =prompt_text,
                    max_length=500,  # REDUCED from 300
                    steer_strength=0.8,
                    drive_pressures=drive_pressures,
                    topic=domain,
                    temperature=0.9,
                    repetition_penalty=1.4,
                    top_p=0.90,
                    enable_safety_abort=True,
                    verbose=False
                )

                # Clear after generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[!] OOM during exploration pattern '{pattern}', clearing and skipping")
                    torch.cuda.empty_cache()
                    continue
                raise

            # CRITICAL: Validate output before scoring
            if not hypothesis_text or len(hypothesis_text.strip()) < 15:
                continue

            # Check for degenerate repetition
            words = hypothesis_text.lower().split()
            if len(words) > 5:
                word_counts = {w: words.count(w) for w in set(words)}
                max_count = max(word_counts.values()) if word_counts else 0

                # Reject if any word appears >40% of the time
                if max_count > len(words) * 0.7:
                    if len(hypotheses) == 0:  # Debug first failure
                        print(f"[!] Degenerate exploration output rejected: '{hypothesis_text[:80]}...'")
                    continue


            # Ethics check is bypassed here - handled by SafetyMonitor at agent level
            # This prevents double-checking and architectural confusion

            # Score hypothesis
            score, breakdown, emb = self.score_candidate(hypothesis_text, drive_pressures, topic=domain)

            hypotheses.append({
                'text': hypothesis_text,
                'pattern': pattern,
                'score': score,
                'breakdown': breakdown,
                'emb': emb,
                'ethics_warnings': [],  # Populated by agent-level SafetyMonitor
                'recommendations': []
            })

            self.explored_regions.add(pattern)

        return hypotheses

    def _filter_relevant_patterns(self, patterns: List[str], query: str, top_k: int = 3) -> List[str]:
        """
        Filter patterns by semantic relevance to user query

        Returns top_k most relevant patterns
        """

        if not patterns or not query:
            return patterns[:top_k]

        # Compute embeddings
        query_emb = self._text_mean_embedding(query)
        pattern_embs = [self._text_mean_embedding(p) for p in patterns]

        # Compute similarities
        similarities = []
        for i, p_emb in enumerate(pattern_embs):
            sim = F.cosine_similarity(query_emb.unsqueeze(0), p_emb.unsqueeze(0)).item()
            similarities.append((patterns[i], sim))

        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)

        return [p for p, sim in similarities[:top_k]]


    def explore(self, domain: str, drive_pressures: Dict[str, float],
            direction_vec, prompt_text: Optional[str] = None) -> Optional[Dict]:
        """
        Main strategic exploration function

        Selects exploration strategy based on dominant drive
        """
        dominant = max(drive_pressures.items(), key=lambda x: x[1])[0]

        # Choose strategy based on drive
        if drive_pressures.get('curiosity', 0) > 0.7:
            all_patterns = self.find_low_activation_regions(domain)

            # NEW: Filter by relevance to user query
            if prompt_text:
                patterns = self._filter_relevant_patterns(all_patterns, prompt_text, top_k=3)
            else:
                patterns = all_patterns[:3]  # Just take first 3

        elif drive_pressures.get('novelty', 0) > 0.7:
            pairs = self.find_distant_concept_pairs(domain)
            # CHANGED: Create discovery-focused pattern descriptions
            patterns = [f"combining {a} with {b}" for a, b in pairs[:3]]

        else:
            patterns = self.find_low_activation_regions(domain)


        if not patterns:
            return None

        # Pass pattern_type to synthesize_novel_patterns
        candidates = self.synthesize_novel_patterns(
            patterns, domain, drive_pressures, direction_vec,

        )

        if not candidates:
            return None

        best = max(candidates, key=lambda x: x['score'])
        self.exploration_history.append(best['pattern'])

        return best

    # === CASCADED SEARCH METHOD (Integration Point) ===
    def cascaded_research_search(self, query, domain, drive_pressures, direction_vec,
                               n_candidates=8, strategy_mode="adaptive"):
        """
        Cascaded search combining strategic exploration with direct generation
        This is the main integration point for the autonomous agent.
        """
        candidates = []

        # STAGE 1: Strategic targeting
        exploration_result = self.explore(domain, drive_pressures, direction_vec, query)

        if exploration_result:
            candidates.append({
                'text': exploration_result['text'],
                'score': exploration_result['score'],
                'breakdown': exploration_result['breakdown'],
                'source': 'strategic_exploration',
                'strategy_target': exploration_result.get('pattern', 'unknown')
            })

        # STAGE 2: Direct generation
        remaining_slots = n_candidates - len(candidates)

        for i in range(remaining_slots):
            text = self.steer_generate_system_aware(
                direction_vec,
                prompt_text=query,
                max_length=150,
                steer_strength=0.8,
                drive_pressures=drive_pressures,
                topic=domain,
                verbose=False
            )

            if text and len(text.strip()) > self.MIN_TEXT_LENGTH:
                score, breakdown, emb = self.score_candidate(text, drive_pressures, domain)
                candidates.append({
                    'text': text,
                    'score': score,
                    'breakdown': breakdown,
                    'source': 'direct_generation',
                    'strategy_target': None
                })

        return candidates
# -------------------------
# PCA AND HYPERNET
# -------------------------

def compute_pca_basis(grad_matrix:np.ndarray, n_components=32):
    if grad_matrix is None or grad_matrix.shape[0] < 2:
        return None
    X = grad_matrix - grad_matrix.mean(axis=0, keepdims=True)
    try:
        U,S,Vt = np.linalg.svd(X, full_matrices=False)
        basis = Vt[:n_components].astype(np.float32)
        return basis
    except Exception:
        return None

class HyperNet(nn.Module):
    def __init__(self, ctx_dim:int, low_d:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ctx_dim, 128),
            nn.ReLU(),
            nn.Linear(128, low_d)
        )
    def forward(self, c):
        return self.net(c)


# -------------------------
# SAFE DISCOVERY ENGINE (COMPLETE)
# -------------------------

class SafeDiscoveryEngine:
    """
    Complete discovery engine with integrated safety.
    """

    def __init__(self, agent, enable_ethics=True):
        self.agent = agent
        self.enable_ethics = enable_ethics

        # Initialize enhanced components
        if hasattr(agent, 'safety_monitor') and agent.safety_monitor:
            self.ethics = agent.safety_monitor.ethics_checker
        else:
            self.ethics = EthicalConstraints()

        # Safety audit
        self.safety_log = []

    def enhanced_cycle(self, n_candidates=8, verbose=True, prompt_text=None) -> Tuple[Optional[Dict], Dict]:
        """Enhanced cycle with better error handling"""

        if not prompt_text:
            if verbose:
                print("[!] No prompt provided, skipping cycle")
            return None, {}

        # Pre-cycle domain check
        topic = classify_topic(prompt_text)

        if verbose:
            print(f"[Ethics Check] Topic: {topic}, Prompt: {prompt_text[:60]}...")

        if self.enable_ethics:
            approved, violations, recommendations = self.ethics.check_hypothesis(
                prompt_text, topic
            )

            if not approved:
                print(f"❌ Prompt blocked: {violations[0]}")
                for rec in recommendations:
                    print(f"   → {rec}")
                return None, {}

            if violations and verbose:
                print(f"⚠️ Ethics warnings: {len(violations)}")

        # Get drive state
        try:
            dominant, pressures = self.agent.drive_system.get_dominant()
            if verbose:
                print(f"[Drive] Dominant: {dominant}, Pressure: {pressures[dominant]:.3f}")
        except Exception as e:
            print(f"[!] Drive system error: {e}")
            pressures = {'curiosity': 0.5, 'coherence': 0.3, 'novelty': 0.4, 'truthfulness': 0.4}
            dominant = 'curiosity'

        # Try standard cycle first (more reliable)
        try:
            result, pressures = self.agent.run_cycle(
                n_candidates=n_candidates,
                verbose=verbose,
                prompt_text=prompt_text
            )

            if result:
                result['drive'] = dominant

                # Post-generation ethics check
                if self.enable_ethics:
                    final_answer = result['final_answer']

                    approved, violations, recommendations = self.ethics.check_hypothesis(
                        final_answer, topic,
                        evidence_verified=self.agent.enable_tools
                    )

                    if not approved:
                        result['final_answer'] = (
                            f"[Output blocked]\n"
                            f"Reason: {violations[0]}"
                        )
                        result['ethics_blocked'] = True

                    elif violations:
                        result['final_answer'] += "\n\n⚠️ ETHICS REVIEW ⚠️\n"
                        for warning in violations:
                            result['final_answer'] += f"  - {warning}\n"

                    self.safety_log.append({
                        'timestamp': time.time(),
                        'hypothesis': final_answer,
                        'topic': topic,
                        'ethics_approved': approved,
                        'violations': violations
                    })

                return result, pressures
            else:
                if verbose:
                    print("[!] Standard cycle returned no result")
                return None, pressures

        except Exception as e:
            print(f"❌ Cycle error: {e}")
            import traceback
            traceback.print_exc()
            return None, pressures

    def run_safe_hybrid(self, pause=0.8, n_candidates=16,
                       initial_prompt=None, max_cycles=10, verbose=True):
        """Safe autonomous operation"""

        if not self.enable_ethics:
            print("\n" + "="*80)
            print("⚠️ ⚠️ ⚠️  WARNING: ETHICS LAYER DISABLED  ⚠️ ⚠️ ⚠️")
            print("="*80 + "\n")

        print(f"[i] Starting Safe Discovery Engine")
        print(f"    Ethics: {'ENABLED ✓' if self.enable_ethics else 'DISABLED ✗'}")
        print(f"    Tools: {'ENABLED ✓' if self.agent.enable_tools else 'DISABLED ✗'}")

        outputs = []
        cycles = 0

        while cycles < max_cycles:
            prompt_for_cycle = initial_prompt if cycles == 0 else None

            try:
                result, pressures = self.enhanced_cycle(
                    n_candidates=n_candidates,
                    verbose=verbose,
                    prompt_text=prompt_for_cycle
                )

                if result is None:
                    print(f"\n[i] Cycle {cycles}: No output (blocked or satisfied)")
                    break

                outputs.append(result)

                # Check satisfaction
                dominant, _ = self.agent.drive_system.get_dominant()
                current_pressure = pressures.get(dominant, 0)

                if current_pressure <= self.agent.hybrid_thresh:
                    print(f"\n[i] Satisfied after {cycles+1} cycles")
                    break

                cycles += 1

                if initial_prompt is not None and cycles == 1:
                    break

                time.sleep(pause)

            except SafetyException as e:
                print(f"\n❌ Safety exception: {e}")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                break

        print(f"\n{'='*80}")
        print(f"[Safety Summary]")
        print(f"  Total outputs: {len(outputs)}")
        print(f"  Ethics checks: {len(self.safety_log)}")

        blocked = [log for log in self.safety_log if not log['ethics_approved']]
        print(f"  Blocked outputs: {len(blocked)}")

        warnings = [log for log in self.safety_log
                   if log['ethics_approved'] and log.get('violations')]
        print(f"  Outputs with warnings: {len(warnings)}")

        print(f"{'='*80}\n")

        return outputs

    def get_safety_report(self) -> Dict:
        """Generate safety report"""
        return {
            'total_checks': len(self.safety_log),
            'blocked': sum(1 for log in self.safety_log
                          if not log['ethics_approved']),
            'warnings': sum(1 for log in self.safety_log
                           if log.get('violations')),
            'recent_violations': [
                {
                    'time': log['timestamp'],
                    'topic': log['topic'],
                    'violations': log.get('violations', [])
                }
                for log in self.safety_log[-10:]
                if log.get('violations')
            ]
        }


# -------------------------
# TWO-WAY SEARCH SYSTEM
# -------------------------

class TwoWaySearch:
    """
    Coordinates internal (LLM) and external (web) search
    Internal search: Steered generation from pretrained knowledge
    External search: Wikipedia, arXiv, web APIs
    """

    def __init__(self, agent):
        self.agent = agent
        self.internal_results = []
        self.external_results = []

    def search(self, query, topic, drive_pressures, n_results=10):
        """Execute two-way search"""

        results = []
        stats = {
            'internal_count': 0,
            'external_count': 0,
            'internal_time': 0,
            'external_time': 0,
            'total_time': 0
        }

        start_time = time.time()

        # INTERNAL SEARCH (LLM Knowledge)
        if self.agent.enable_internal_search:
            internal_start = time.time()

            if self.agent.verbose:
                print(f"\n[Internal Search] Querying LLM pretrained knowledge...")

            internal_results = self._search_internal(
                query, topic, drive_pressures,
                n_results=n_results // 2
            )

            results.extend(internal_results)
            stats['internal_count'] = len(internal_results)
            stats['internal_time'] = time.time() - internal_start

            if self.agent.verbose:
                print(f"[Internal Search] Found {len(internal_results)} knowledge-based hypotheses")

        # EXTERNAL SEARCH (Web Sources)
        if self.agent.enable_external_search:
            external_start = time.time()

            if self.agent.verbose:
                print(f"\n[External Search] Querying web sources...")

            max_sources = self.agent.get_max_evidence_sources(
                drive_pressures, topic
            )

            external_results = self._search_external(
                query, topic, max_sources
            )

            results.extend(external_results)
            stats['external_count'] = len(external_results)
            stats['external_time'] = time.time() - external_start

            if self.agent.verbose:
                print(f"[External Search] Found {len(external_results)} evidence sources")

        stats['total_time'] = time.time() - start_time

        if self.agent.verbose:
            print(f"\n[Two-Way Search] Total: {len(results)} results")
            print(f"    Internal: {stats['internal_count']} ({stats['internal_time']:.2f}s)")
            print(f"    External: {stats['external_count']} ({stats['external_time']:.2f}s)")

        return results, stats

    def _search_internal(self, query, topic, drive_pressures, n_results=5):
        """Search LLM's internal pretrained knowledge"""
        results = []

        dominant = max(drive_pressures.items(), key=lambda x: x[1])[0]
        direction_vec = self.agent.mapper.get_direction(dominant)

        if direction_vec is None:
            return results

        direction_vec = direction_vec.to(DEVICE)

        for i in range(n_results):
            steer_strength = 0.5 + 0.5 * random.random()

            try:
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # UPDATED: Use unified simulator's method signature
                text = self.agent.sim.steer_generate_system_aware(
                    direction_vec,
                    prompt_text=query,
                    max_length=500,
                    steer_strength=steer_strength,
                    drive_pressures=drive_pressures,
                    topic=topic,
                    temperature=0.85 + 0.15 * random.random(),
                    verbose=False
                )

                if text and len(text.strip()) > 15:
                    results.append({
                        'source_type': 'internal_llm',
                        'text': text,
                        'query': query,
                        'steer_strength': steer_strength,
                        'drive': dominant,
                        'confidence': 0.5
                    })

            except RuntimeError as e:
                if "out of memory" in str(e):
                    if self.agent.verbose:
                        print(f"[!] OOM during internal search, clearing cache...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    if self.agent.verbose:
                        print(f"[!] Internal search error: {str(e)[:100]}")
                    continue

            except Exception as e:
                if self.agent.verbose:
                    print(f"[!] Internal search error: {str(e)[:100]}")
                continue

        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def _search_external(self, query, topic, max_sources):
        """Search external web sources"""

        try:
            evidence_sources = self.agent.evidence_gatherer.gather(
                topic, query, max_sources=max_sources
            )

            results = []
            for source in evidence_sources:
                results.append({
                    'source_type': 'external_' + source.source_type,
                    'text': source.snippet,
                    'title': source.title,
                    'url': source.url,
                    'confidence': source.confidence,
                    'verified': source.verified
                })

            return results

        except Exception as e:
            if self.agent.verbose:
                print(f"[!] External search error: {str(e)[:100]}")
            return []



# -------------------------
# AUTONOMOUS AGENT (COMPLETE)
# -------------------------

class AutonomousAgent:
    def __init__(self,
             model_name="gpt2-xl",
             persist_path="./memory.db",
             discovery_mode=DiscoveryMode.DISCOVERY_ENGINE,  # NEW
             enable_internal_search=True,                    # NEW
             enable_external_search=True,                    # NEW
             science_weight=0.7,
             hybrid_satisfaction_thresh=0.20,
             max_cycles=10,
             enable_probing=True,
             evidence_style="B",
             enable_safety=True,
             verbose=True):                                 # NEW

        print(f"[i] Initializing Dual-Mode Discovery System")
        print(f"    Mode: {discovery_mode.value}")
        print(f"    Internal search: {'ON' if enable_internal_search else 'OFF'}")
        print(f"    External search: {'ON' if enable_external_search else 'OFF'}")

        # Store mode configuration
        self.discovery_mode = discovery_mode
        self.enable_internal_search = enable_internal_search
        self.enable_external_search = enable_external_search
        self.verbose = verbose

        # Initialize model
        self.model, self.tokenizer = load_model(model_name)

        # Drive system
        self.drive_system = DriveSystem(enable_safety_checks=enable_safety)

        # Feature mapping
        self.mapper = FeatureTargetMapper(
            self.model, self.tokenizer,
            science_weight=science_weight
        )

        # Relevance scoring
        self.scorer = RelevanceScorer(self.model, self.tokenizer)

        # Internal simulator (now unified with strategic exploration)
        self.sim = UnifiedResearchSimulator(
            self.model, self.tokenizer,
            device=DEVICE,
            temperature=0.9,
            history_size=256,
            enable_truthfulness=True
        )

        # Then store the reference
        self.sim.enable_external_tools = self.enable_external_search

        # Evaluator ensemble
        self.evaluator = EvaluatorEnsemble(
            self.model.config.n_embd,
            n_members=3,
            device=DEVICE
        )

        # Database persistence
        self.persist = PersistDB(persist_path)

        # Safety monitoring
        if enable_safety:
            self.safety_monitor = SafetyMonitor(enable_ethics=True)
        else:
            self.safety_monitor = None

        # External evidence gathering
        self.evidence_gatherer = EvidenceGatherer(
            enable_network=enable_external_search
        )

        # Mode-specific parameters
        self._setup_mode_parameters()

        # Other initialization
        self.output_history = []
        self.pressure_history = []
        self.hybrid_thresh = hybrid_satisfaction_thresh
        self.max_cycles = max_cycles
        self.enable_probing = enable_probing
        self.evidence_style = evidence_style

        # PCA/HyperNet
        self.pca_basis = None
        self.low_d = 32
        self.hypernet = HyperNet(
            ctx_dim=self.model.config.n_embd,
            low_d=self.low_d
        ).to(DEVICE)
        self.hyper_opt = torch.optim.Adam(self.hypernet.parameters(), lr=1e-4)
        self.last_pca_cycle = 0

        # Audit logging
        safe_mkdir("runs")
        self.audit_path = "runs/audit_log.jsonl"

        self._try_build_pca()

        print(f"[i] Ready")


    def _log_audit(self, event_type, payload):
        payload = dict(payload)
        line = json.dumps({"ts": time.time(), "event": event_type, "payload": payload})
        with open(self.audit_path, "a") as f:
            f.write(line + "\n")
        self.persist.save_audit(event_type, payload)

    def _try_build_pca(self):
        grad_mat = self.scorer.get_grad_matrix()
        if grad_mat is None:
            return
        basis = compute_pca_basis(grad_mat, n_components=self.low_d)
        if basis is not None:
            self.pca_basis = basis
            self._log_audit("pca_built", {"components": int(self.low_d)})

    def generate_probe_delta_emb(self, context_emb):
        if (not self.enable_probing) or (self.pca_basis is None):
            return None
        with torch.no_grad():
            low = self.hypernet(context_emb.to(DEVICE)).cpu().numpy().astype(np.float32)
        try:
            full = np.dot(low, self.pca_basis)
            embdim = self.model.config.n_embd
            if full.size >= embdim:
                full = full[: (full.size // embdim) * embdim]
                full = full.reshape(-1, embdim).mean(axis=0)
            else:
                pad = np.zeros(embdim - full.size, dtype=np.float32)
                full = np.concatenate([full, pad], axis=0)
            emb = torch.tensor(full.astype(np.float32), device=DEVICE)
            return emb
        except Exception:
            return None

    def _setup_mode_parameters(self):
        """Configure parameters based on discovery mode"""

        if self.discovery_mode == DiscoveryMode.DISCOVERY_ENGINE:
            print("[i] DISCOVERY ENGINE MODE:")
            print("    - Exhaustive candidate generation")
            print("    - Maximum evidence gathering")
            print("    - Drives control: ranking and termination only")

            self.mode_params = {
                'n_candidates_base': 5,
                'max_evidence_sources': 3,
                'max_generation_attempts': 150,
                'diversity_enforcement': True,
                'drive_controls_scoring': True,
                'drive_controls_selection': True,
                'drive_controls_termination': True,
                'drive_modulates_candidates': False,
                'drive_modulates_evidence': False,
                'drive_modulates_generation': False,
            }

        elif self.discovery_mode == DiscoveryMode.DRIVE_CONTROLLED:
            print("[i] DRIVE CONTROLLED MODE:")
            print("    - Drive-adaptive candidate generation")
            print("    - Drive-modulated evidence gathering")
            print("    - Drives control: all stages")

            self.mode_params = {
                'n_candidates_base': 5,
                'max_evidence_sources': 3,
                'max_generation_attempts': 150,
                'diversity_enforcement': True,
                'drive_controls_scoring': True,
                'drive_controls_selection': True,
                'drive_controls_termination': True,
                'drive_modulates_candidates': True,
                'drive_modulates_evidence': True,
                'drive_modulates_generation': True,
            }

    def get_n_candidates(self, drive_pressures):
        """Get number of candidates based on mode and drives"""

        base = self.mode_params['n_candidates_base']

        if not self.mode_params['drive_modulates_candidates']:
            return base

        dominant = max(drive_pressures.items(), key=lambda x: x[1])[0]
        pressure = drive_pressures[dominant]

        if dominant == "novelty":
            multiplier = 1.0 + 0.5 * min(pressure / 2.0, 1.0)
        elif dominant == "coherence":
            multiplier = 0.7
        elif dominant == "truthfulness":
            multiplier = 1.3
        else:
            multiplier = 1.0

        n = int(base * multiplier)

        if self.verbose:
            print(f"[Candidates] Base={base}, Drive={dominant}, Multiplier={multiplier:.2f}, Final={n}")

        return n

    def get_max_evidence_sources(self, drive_pressures, topic):
        """Get evidence sources based on mode and drives"""

        base = self.mode_params['max_evidence_sources']

        if not self.mode_params['drive_modulates_evidence']:
            return base

        dominant = max(drive_pressures.items(), key=lambda x: x[1])[0]
        pressure = drive_pressures[dominant]

        if dominant == "truthfulness":
            multiplier = 1.5
        elif dominant == "novelty":
            multiplier = 0.6
        elif dominant == "coherence":
            multiplier = 0.8
        else:
            multiplier = 1.0

        n = int(base * multiplier)

        if self.verbose:
            print(f"[Evidence] Base={base}, Drive={dominant}, Multiplier={multiplier:.2f}, Final={n}")

        return max(1, n)

    def get_satisfaction_threshold(self, dominant_drive):
        """Get termination threshold based on mode and drive"""

        if not self.mode_params['drive_controls_termination']:
            return self.hybrid_thresh

        thresholds = {
            'curiosity': 0.15,
            'novelty': 0.25,
            'coherence': 0.10,
            'truthfulness': 0.30
        }

        threshold = thresholds.get(dominant_drive, self.hybrid_thresh)

        if self.verbose:
            print(f"[Termination] Drive={dominant_drive}, Threshold={threshold:.3f}")

        return threshold


    def run_cycle(self, n_candidates=None, verbose=True, prompt_text=None):
        """
        Enhanced cycle with two-way search and mode-aware behavior

        Three-way search:
        1. Internal: Generate from LLM knowledge via steering
        2. External: Fetch from Wikipedia, arXiv, web
        3. Exploratory: Explore novel pattern 
        Mode behavior:
        - Discovery engine: Exhaustive search, drives rank
        - Drive controlled: Drives modulate all stages
        """

        # Ensure device consistency
        if hasattr(self.sim, '_ensure_device_consistency'):
            self.sim._ensure_device_consistency()


        # Input safety check
        if prompt_text and self.safety_monitor:
            topic = classify_topic(prompt_text)
            is_safe, violations = self.safety_monitor.check_input(prompt_text, topic)
            if not is_safe:
                error_msg = "❌ Input rejected:\n"
                for v in violations:
                    if v.severity == "critical":
                        error_msg += f"  - {v.message}\n"
                raise SafetyException(error_msg)

        topic = classify_topic(prompt_text or "")
        mode_label = choose_mode_for_topic(topic)

        dominant, pressures = self.drive_system.get_dominant()
        self.pressure_history.append(pressures)
        cyc = len(self.output_history)

        if verbose:
            print("\n" + "="*80)
            print(f"[Cycle {cyc}] Mode={self.discovery_mode.value}")
            print(f"  Topic={topic}, Drive={dominant} ({pressures[dominant]:.3f})")

        # Drive safety check
        if self.safety_monitor:
            try:
                self.safety_monitor.check_drives(self.drive_system)
            except SafetyException as e:
                print(f"❌ Drive violation: {e}")
                raise

        # =============================================================
        # THREE-WAY SEARCH: Exploration + Internal (LLM) + External (Web)
        #=============================================================

        # STAGE 0: STRATEGIC EXPLORATION (NEW - CRITICAL)
        exploration_candidates = []
        if hasattr(self.sim, 'explore'):
            try:
                # Clear CUDA cache before exploration
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                dir_vec = self.mapper.get_direction(dominant).to(DEVICE)

                if verbose:
                    print(f"\n[Strategic Exploration] Searching knowledge space...")
                    print(f"  Drive: {dominant}")
                    print(f"  Domain: {topic}")

                exploration_result = self.sim.explore(
                    domain=topic,
                    drive_pressures=pressures,
                    direction_vec=dir_vec,
                    prompt_text=prompt_text #or f"Explore {topic}"
                )

                if exploration_result:
                    if verbose:
                        print(f"[✓] Found strategic target: {exploration_result.get('pattern', 'unknown')}")
                        print(f"    Score: {exploration_result['score']:.3f}")
                        print(f"    Text: {exploration_result['text'][:100]}...")

                    exploration_candidates.append({
                        'text': exploration_result['text'],
                        'score': exploration_result['score'],
                        'breakdown': exploration_result['breakdown'],
                        'steer': 1.2,
                        'science_value': 0.85,  # High value for strategic exploration
                        'uncertainty': 0.15,
                        'emb': exploration_result['emb'],
                        'source': 'strategic_exploration',
                        'pattern': exploration_result.get('pattern', 'unknown')
                    })

                    # Log exploration
                    self._log_audit("strategic_exploration", {
                        'cycle': cyc,
                        'pattern': exploration_result.get('pattern'),
                        'domain': topic,
                        'score': float(exploration_result['score'])
                    })
                else:
                    if verbose:
                        print(f"[i] No unexplored regions found in {topic}")

            except Exception as e:
                if verbose:
                    print(f"[!] Exploration error: {str(e)[:100]}")
                import traceback
                if verbose:
                    traceback.print_exc()

        # STAGE 1: TWO-WAY SEARCH (Internal + External)
        two_way_searcher = TwoWaySearch(self)
        search_results, search_stats = two_way_searcher.search(
            query=prompt_text or f"Explore {topic}",
            topic=topic,
            drive_pressures=pressures,
            n_results=10
        )

        # Update stats to include exploration
        search_stats['exploration_count'] = len(exploration_candidates)

        # Log combined search statistics
        self._log_audit("three_way_search", {
            'cycle': cyc,
            'exploration_count': search_stats['exploration_count'],
            'internal_count': search_stats['internal_count'],
            'external_count': search_stats['external_count'],
            'total_time': search_stats['total_time']

        })

        # =============================================================
        # CANDIDATE GENERATION (Mode-Aware)
        # =============================================================

        # Get mode-specific candidate count
        if n_candidates is None:
            n_candidates = self.get_n_candidates(pressures)

        if verbose:
            print(f"\n[Generation] Generating {n_candidates} candidates...")
            print(f"  Mode: {self.discovery_mode.value}")
            print(f"  Drive modulation: {self.mode_params['drive_modulates_generation']}")

        # Get steering direction
        dir_vec = self.mapper.get_direction(dominant).to(DEVICE)

        # Context for PCA probing
        if len(self.sim.history_embs) > 0:
             ctx_emb = torch.tensor(
                np.mean(np.stack(list(self.sim.history_embs)[-8:]), axis=0),
                dtype=torch.float32
            )
        else:
            ctx_emb = self.sim._text_mean_embedding(dominant)

        # PCA basis update
        if self.pca_basis is None:
            self._try_build_pca()

        # Generate candidates
        candidates = []
        attempts = 0
        max_attempts = self.mode_params['max_generation_attempts']

        while len(candidates) < n_candidates and attempts < max_attempts:
            attempts += 1

            # Vary generation parameters
            steer = float(max(0.5, 0.8 + 1.2 * random.random()))
            max_len = int(550 + 150 * random.random())

            # PCA probing
            probe_delta = None
            if self.enable_probing and self.pca_basis is not None:
                probe_delta = self.generate_probe_delta_emb(ctx_emb)

            # Generate
            try:
                txt = self.sim.steer_generate_system_aware(
                    dir_vec,
                    prompt_text=prompt_text or "",
                    max_length=max_len,
                    steer_strength=steer * 0.6,
                    probe_delta_emb=probe_delta,
                    drive_pressures=pressures,
                    topic=topic,
                    enable_safety_abort=True,
                    verbose=(verbose and attempts == 1)
                )

            except Exception as e:
                if verbose and attempts % 50 == 0:
                    print(f"[!] Generation error at attempt {attempts}: {str(e)[:100]}")
                continue

            # Validate candidate (relaxed checks)
            if not txt or len(txt.strip()) < 10:
                continue

            txt_clean = txt.lower().strip()

            # Only check extreme degeneration
            extreme_patterns = [
                "the the the the the",
                "and and and and and",
                "is is is is is"
            ]
            if any(pattern in txt_clean for pattern in extreme_patterns):
                continue

            words = txt_clean.split()
            if len(words) < 3:
                continue

            # Check major repetition only
            word_counts = {}
            for word in words:
                if len(word) > 3:
                    word_counts[word] = word_counts.get(word, 0) + 1

            if word_counts:
                max_count = max(word_counts.values())
                if max_count > len(words) * 0.85:
                    continue

            # Apply collapse_repeats
            txt_collapsed = collapse_repeats(txt).strip()
            if txt_collapsed and len(txt_collapsed) >= 10:
                txt = txt_collapsed
            elif len(txt) < 10:
                continue

            # Log first successful
            if verbose and len(candidates) == 0 and attempts >= 10:
                print(f"[✓] First valid candidate after {attempts} attempts")
                print(f"    '{txt[:120]}...'")

            # Score candidate
            try:
                score_val, breakdown, emb = self.sim.score_candidate(
                    txt, pressures, topic=topic
                )

                emb_tensor = torch.tensor(emb, dtype=torch.float32)
                pred, uncert = self.evaluator.predict(emb_tensor)
                science_value = float(pred[0])

                score_adj = score_val + 0.15 * science_value - 0.1 * uncert

                candidates.append({
                    'text': txt,
                    'score': float(score_adj),
                    'breakdown': breakdown,
                    'steer': steer,
                    'science_value': float(science_value),
                    'uncertainty': float(uncert),
                    'emb': emb,
                    'source': 'internal_generation'
                })

            except Exception as e:
                if verbose and attempts % 50 == 0:
                    print(f"[!] Scoring error: {str(e)[:100]}")
                continue

        # =============================================================
        # ADD EXPLORATION CANDIDATES FIRST (HIGHEST PRIORITY)
        # =============================================================
        if exploration_candidates:
            # BOOST exploration scores to ensure they're competitive
            for candidate in exploration_candidates:
                candidate['score'] = candidate['score'] * 1.3  # 30% boost
                candidate['science_value'] = 0.90  # High confidence

            candidates.extend(exploration_candidates)
            if verbose:
                print(f"\n[Candidates] Added {len(exploration_candidates)} strategic exploration candidates")
                print(f"  Boosted scores by 30% for strategic priority")
        
        # Add external search results as candidates
        for result in search_results:
            if result['source_type'].startswith('external'):
                try:
                    score_val, breakdown, emb = self.sim.score_candidate(
                        result['text'], pressures, topic=topic
                    )

                    candidates.append({
                        'text': result['text'],
                        'score': float(score_val),
                        'breakdown': breakdown,
                        'steer': 0.0,
                        'science_value': result.get('confidence', 0.5),
                        'uncertainty': 0.3,
                        'emb': emb,
                        'source': result['source_type'],
                        'url': result.get('url', None)
                    })
                except Exception:
                    continue

        if verbose:
            print(f"\n[Candidates] Generated {len(candidates)} valid candidates")
            exploration = sum(1 for c in candidates if c['source'] == 'strategic_exploration')
            internal = sum(1 for c in candidates if c['source'] == 'internal_generation')
            external = sum(1 for c in candidates if c['source'].startswith('external'))
            print(f"  Exploration: {exploration}, Internal: {internal}, External: {external}")

            # Show exploration patterns found
            if exploration > 0:
                patterns = [c.get('pattern', 'unknown') for c in candidates
                           if c['source'] == 'strategic_exploration']
                print(f"  Strategic patterns: {', '.join(patterns)}")

        # Fallback if no candidates
        if not candidates:
            print(f"[!] Warning: No valid candidates after {attempts} attempts")

            # Use search results as fallback
            if search_results:
                external = [r for r in search_results if r['source_type'].startswith('external')]
                if external:
                    fallback_text = f"Research suggests: {external[0]['text']}"
                else:
                    fallback_text = f"Further investigation into {prompt_text or topic} is needed."
            else:
                fallback_text = f"Further investigation into {prompt_text or topic} is needed."

            score_val, breakdown, emb = self.sim.score_candidate(
                fallback_text, pressures, topic=topic
            )

            candidates = [{
                'text': fallback_text,
                'score': float(score_val),
                'breakdown': breakdown,
                'steer': 0.0,
                'science_value': 0.0,
                'uncertainty': 1.0,
                'emb': emb,
                'source': 'fallback'
            }]
        # =============================================================
        # STRATEGIC SELECTION: Prefer exploration when reasonable
        # =============================================================

        exploration_cands = [c for c in candidates if c['source'] == 'strategic_exploration']
        other_cands = [c for c in candidates if c['source'] != 'strategic_exploration']

        if exploration_cands and other_cands:
            best_exploration = max(exploration_cands, key=lambda x: x['score'])
            best_other = max(other_cands, key=lambda x: x['score'])

            # Use exploration if within 85% of best alternative
            threshold = 0.85
            if best_exploration['score'] >= best_other['score'] * threshold:
                best = best_exploration
                if verbose:
                    print(f"\n[Selection] ✓ Strategic exploration chosen")
                    print(f"  Exploration: {best_exploration['score']:.3f}")
                    print(f"  Best alternative: {best_other['score']:.3f}")
                    print(f"  Threshold: {threshold:.2f} (exploration ≥ {best_other['score'] * threshold:.3f})")
            else:
                best = best_other
                if verbose:
                    print(f"\n[Selection] Alternative chosen over exploration")
                    print(f"  Exploration: {best_exploration['score']:.3f}")
                    print(f"  Alternative: {best_other['score']:.3f} (gap too large)")
        elif exploration_cands:
            # Only exploration available
            best = max(exploration_cands, key=lambda x: x['score'])
            if verbose:
                print(f"\n[Selection] ✓ Strategic exploration (only candidate)")
        else:
            # No exploration - normal selection
            best = max(candidates, key=lambda x: x['score'])
            if verbose:
                print(f"\n[Selection] Standard selection (no exploration)")

        best_text = best['text']
        if verbose:
            print(f"\n[Selected] Score={best['score']:.4f}, Source={best['source']}")
            print(f"  Truth={best['breakdown'].get('truthfulness', 0):.3f}")
            print(f"  Novelty={best['breakdown'].get('novelty', 0):.3f}")

        # =============================================================
        # EVIDENCE FORMATTING (External Sources)
        # =============================================================

        external_sources = [r for r in search_results
                           if r['source_type'].startswith('external')]

        if external_sources and self.evidence_style != "none":
            evidence_text = self.evidence_gatherer.format_sources(
                [EvidenceSource(
                    source_type=s['source_type'].replace('external_', ''),
                    title=s.get('title', 'Source'),
                    url=s.get('url', ''),
                    snippet=s['text'],
                    confidence=s['confidence'],
                    retrieved_at=time.time(),
                    verified=s.get('verified', False)
                ) for s in external_sources[:5]],
                style=self.evidence_style
            )
        else:
            evidence_text = "\n\n⚠️ No external sources found."

        # Medical disclaimer
        if topic == "health":
            evidence_text += ("\n\n⚠️ MEDICAL DISCLAIMER ⚠️\n"
                             "For educational purposes only. "
                             "Consult healthcare professionals.")

        final_answer = best_text.rstrip() + evidence_text

        # =============================================================
        # OUTPUT SAFETY CHECK
        # =============================================================

        if self.safety_monitor:
            is_safe, violations = self.safety_monitor.check_output(
                final_answer, topic, evidence_verified=self.enable_external_search
            )

            if not is_safe:
                error_msg = "\n".join(
                    f"- {v.message}" for v in violations
                    if v.severity in ["critical", "error"]
                )
                final_answer = f"[Output blocked: {error_msg}]"
                best['score'] = 0.0

        # =============================================================
        # PERSISTENCE
        # =============================================================

        try:
            out_id = self.persist.save_output(
                cyc, dominant, mode_label, topic, float(best['score']),
                final_answer, emb=np.array(best['emb'], dtype=np.float32),
                evidence=evidence_text
            )
        except Exception as e:
            self._log_audit("save_failed", {"error": str(e)})
            out_id = None

        self.output_history.append({
            'time': cyc,
            'drive': dominant,
            'mode': mode_label,
            'topic': topic,
            'text': final_answer,
            'score': float(best['score']),
            'evidence_verified': self.enable_external_search,
            'discovery_mode': self.discovery_mode.value
        })

        # =============================================================
        # LEARNING UPDATES
        # =============================================================

        # Update evaluator
        target_vec = np.array([
            0.6 * best['breakdown'].get('curiosity', 0.0) +
            0.4 * best['breakdown'].get('coherence', 0.0),
            best['breakdown'].get('coherence', 0.0),
            best['breakdown'].get('novelty', 0.0)
        ], dtype=np.float32)

        try:
            emb_t = torch.tensor(best['emb'], dtype=torch.float32)
            self.evaluator.train_on_example(emb_t, target_vec)
        except Exception:
            pass

        # Compute satisfactions
        satisfactions = {}
        for dn in ['curiosity', 'coherence', 'novelty', 'truthfulness']:
            base = best['breakdown'].get(dn, 0.0)
            if dn == dominant:
                bonus = 0.25 if base > 0.6 else 0.08
                satisfactions[dn] = float(min(1.0, base + bonus))
            else:
                satisfactions[dn] = float(min(1.0, base * 0.45))

        # Synergy bonuses
        if (satisfactions.get('coherence', 0.0) > 0.6 and
            satisfactions.get('curiosity', 0.0) > 0.6):
            satisfactions['novelty'] = min(1.0, satisfactions.get('novelty', 0.0) + 0.05)

        self.drive_system.update_all(satisfactions, time_step=1.0)

        # PCA rebuild
        cycles_since_pca = cyc - self.last_pca_cycle
        should_rebuild = (
            (self.pca_basis is None and len(self.scorer.grad_buffer) >= 16) or
            (self.pca_basis is not None and cycles_since_pca >= 20)
        )

        if should_rebuild:
            self._try_build_pca()
            self.last_pca_cycle = cyc

        try:
            return {
                'final_answer': final_answer,
                'mode': mode_label,
                'topic': topic,
                'discovery_mode': self.discovery_mode.value,
                'search_stats': search_stats,
                'exploration_used': len(exploration_candidates) > 0,  # NEW
                'explored_pattern': exploration_candidates[0].get('pattern') if exploration_candidates else None,  # NEW
                'candidate_sources': {  # NEW
                    'exploration': sum(1 for c in candidates if c['source'] == 'strategic_exploration'),
                    'internal': sum(1 for c in candidates if c['source'] == 'internal_generation'),
                    'external': sum(1 for c in candidates if c.get('source', '').startswith('external'))
                }
            }, pressures
        except NameError as e:
            # Variable not defined - return safe default
            print(f"[!] Return error: {e}")
            return {
                'final_answer': '[Incomplete cycle]',
                'mode': 'error',
                'topic': 'unknown',
                'discovery_mode': self.discovery_mode.value,
                'search_stats': {
                    'internal_count': 0,
                    'external_count': 0,
                    'total_time': 0
                }
            }, {
                'curiosity': 0,
                'coherence': 0,
                'novelty': 0,
                'truthfulness': 0
            }

    # Replace the existing run_hybrid method with this enhanced version:

    def run_hybrid(self, pause=0.8, n_candidates=16, initial_prompt=None, max_cycles=None, enable_ethics=True):
        """Run with optional prompt using SafeDiscoveryEngine"""
        if max_cycles is None:
            max_cycles = self.max_cycles

        print("[i] Starting Complete Research AI with Safe Discovery Engine")

        if self.safety_monitor and enable_ethics:
            print("[i] Safety: ENABLED ✓")
            print("[i] Ethics: ENABLED ✓")
        elif not enable_ethics:
            print("⚠️ Ethics: DISABLED")
        if self.enable_tools:
            print("[i] Tools: ENABLED ✓")
        else:
            print("⚠️ Tools: DISABLED")

        # Initialize Safe Discovery Engine
        safe_engine = SafeDiscoveryEngine(self, enable_ethics=enable_ethics)

        cycles = 0
        outputs = []

        while cycles < max_cycles:
            prompt_for_cycle = initial_prompt if cycles == 0 else None

            try:
                # Use the enhanced cycle from SafeDiscoveryEngine
                result, pressures = safe_engine.enhanced_cycle(
                    n_candidates=n_candidates,
                    verbose=True,
                    prompt_text=prompt_for_cycle
                )

                if result is None:
                    print(f"[i] Cycle {cycles}: No output (blocked or satisfied)")
                    break

                outputs.append(result)

            except SafetyException as e:
                print(f"\n❌ Safety exception: {e}")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
                break

            # Update satisfaction check
            dominant, _ = self.drive_system.get_dominant()
            current_pressure = pressures.get(dominant, 0)
            threshold= self.get_satisfaction_threshold(dominant)
            if current_pressure <= threshold:
                print(f"\n[i] Satisfied ({current_pressure:.4f} <= {threshold:.4f})")
                break

            cycles += 1

            # Single cycle for prompted operation
            if initial_prompt is not None and cycles == 1:
                break

            # Safety alerts
            if self.safety_monitor and self.safety_monitor.should_alert():
                summary = self.safety_monitor.get_violation_summary()
                print(f"\n⚠️ Safety alert: {summary}")

            time.sleep(pause)

        # Safety summary
        ethics_report = safe_engine.get_safety_report()
        print(f"\n{'='*80}")
        print(f"[Safety Summary]")
        print(f"  Total outputs: {len(outputs)}")
        print(f"  Ethics checks: {ethics_report['total_checks']}")
        print(f"  Blocked outputs: {ethics_report['blocked']}")
        print(f"  Outputs with warnings: {ethics_report['warnings']}")
        print(f"{'='*80}\n")

        print(f"[i] Finished ({cycles} cycles)")
        return outputs

    def test_generation_quality(self, test_prompts=None):
        """Test generation quality across different drives and topics"""

        if test_prompts is None:
            test_prompts = [
                ("Quantum network applications", "tech"),
                ("Protein folding mechanisms", "biology"),
                ("Lithium ion battery improvements", "tech"),
                ("Dark matter detection methods", "physics"),
               ("Cellular aging reversal", "health")
            ]

        print("\n" + "="*80)
        print("GENERATION QUALITY TEST")
        print("="*80)

        for prompt, expected_topic in test_prompts:
            print(f"\n{'─'*80}")
            print(f"Prompt: {prompt}")
            print(f"Expected topic: {expected_topic}")
            print(f"{'─'*80}")

            # Get current state
            dominant, pressures = self.drive_system.get_dominant()
            topic = classify_topic(prompt)

            print(f"Drive: {dominant} ({pressures[dominant]:.3f})")
            print(f"Topic: {topic}")

            # Get direction
            dir_vec = self.mapper.get_direction(dominant).to(DEVICE)

            # Generate single candidate
            txt = self.sim.steer_generate_system_aware(
                dir_vec,
                prompt_text=prompt,
                max_length=500,
                steer_strength=0.8,
                drive_pressures=pressures,
                topic=topic,
                verbose=True
            )

            print(f"\nGenerated:")
            print(f"  {txt}")
            print(f"\nLength: {len(txt.split())} words")
            print(f"Unique words: {len(set(txt.lower().split()))}")

            # Check for degeneration
            words = txt.lower().split()
            if words:
                word_counts = {w: words.count(w) for w in set(words)}
                max_count = max(word_counts.values())
                most_common = max(word_counts.items(), key=lambda x: x[1])[0]

                print(f"Most common word: '{most_common}' ({max_count}/{len(words)} = {max_count/len(words)*100:.1f}%)")

                if max_count > len(words) * 0.7:
                    print("⚠️  WARNING: High repetition detected!")
                else:
                    print("✓  Repetition within acceptable range")

        print("\n" + "="*80)
#'''
# -------------------------
# CLI (COMPLETE)
# -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dual-Mode Autonomous Discovery System"
    )

    # Model configuration
    parser.add_argument("--model", type=str, default="gpt2-xl",
                       help="Model name (gpt2-xl, microsoft/Phi-3-mini-4k-instruct)")

    # Mode selection
    parser.add_argument("--mode", type=str,
                       choices=["discovery", "drive-controlled"],
                       default="discovery",
                       help="Discovery mode: 'discovery' (exhaustive) or 'drive-controlled' (full modulation)")

    # Search configuration
    parser.add_argument("--internal-search", dest="internal_search",
                       action="store_true", default=True,
                       help="Enable internal LLM knowledge search")
    parser.add_argument("--no-internal-search", dest="internal_search",
                       action="store_false",
                       help="Disable internal search")

    parser.add_argument("--external-search", dest="external_search",
                       action="store_true", default=False,
                       help="Enable external web search (Wikipedia, arXiv)")
    parser.add_argument("--no-external-search", dest="external_search",
                       action="store_false",
                       help="Disable external search")

    # Generation parameters
    parser.add_argument("--cycles", type=int, default=1,
                       help="Number of discovery cycles")
    parser.add_argument("--candidates", type=int, default=None,
                       help="Number of candidates (None=mode-dependent)")

    # Other parameters
    parser.add_argument("--threshold", type=float, default=0.20,
                       help="Satisfaction threshold")
    parser.add_argument("--persist", type=str, default="./memory.db",
                       help="Database path")

    # Safety
    parser.add_argument("--safety", dest="safety", action="store_true", default=True)
    parser.add_argument("--no-safety", dest="safety", action="store_false")

    # Features
    parser.add_argument("--no-probing", action="store_true",
                       help="Disable PCA probing")
    parser.add_argument("--evidence-style", type=str, default="B",
                       choices=["none", "A", "B"],
                       help="Evidence formatting style")

    # Verbosity
    parser.add_argument("--verbose", action="store_true",
                       help="Detailed logging")

    # Comparison mode
    parser.add_argument("--compare-modes", action="store_true",
                       help="Run comparative experiment between modes")
    parser.add_argument("--test-prompts-file", type=str, default=None,
                       help="JSON file with test prompts for comparison")

    # Colab/Jupyter compatibility: ignore unknown arguments
    args, unknown = parser.parse_known_args()

    # Filter out Colab-specific arguments
    if unknown:
        unknown_filtered = [arg for arg in unknown if not arg.startswith('-f')]
        if unknown_filtered:
            print(f"[!] Warning: Unknown arguments: {unknown_filtered}")

    # Convert mode string to enum
    if args.mode == "discovery":
        discovery_mode = DiscoveryMode.DISCOVERY_ENGINE
    else:
        discovery_mode = DiscoveryMode.DRIVE_CONTROLLED

    # Safety warning if disabled
    if not args.safety:
        print("\n" + "="*80)
        print("⚠️ ⚠️ ⚠️  WARNING: SAFETY & ETHICS DISABLED  ⚠️ ⚠️ ⚠️")
        print("="*80 + "\n")
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            return

    # =============================================================
    # COMPARISON MODE
    # =============================================================

    if args.compare_modes:
        print("\n[i] Running MODE COMPARISON experiment...")

        # Load or create test prompts
        if args.test_prompts_file:
            with open(args.test_prompts_file, 'r') as f:
                test_data = json.load(f)
                test_prompts = [(p['prompt'], p['topic']) for p in test_data]
        else:
            # Default test prompts
            test_prompts = [
                ("Quantum network applications", "tech"),
                ("Protein folding mechanisms", "biology"),
                ("Lithium ion battery improvements", "tech"),
                ("Dark matter detection methods", "physics"),
                ("Cellular aging reversal", "biology"),
            ]

        comparison = ModeComparison()
        report = comparison.run_comparative_experiment(
            test_prompts=test_prompts,
            cycles_per_mode=args.cycles,
            model_name=args.model,
            enable_tools=args.external_search
        )

        comparison.print_report(report)
        comparison.save_report(report)

        return

    # =============================================================
    # NORMAL MODE
    # =============================================================

    print("\n[i] Initializing Dual-Mode Discovery System...")
    try:
        agent = AutonomousAgent(
            model_name=args.model,
            persist_path=args.persist,
            discovery_mode=discovery_mode,
            enable_internal_search=args.internal_search,
            enable_external_search=args.external_search,
            hybrid_satisfaction_thresh=args.threshold,
            max_cycles=args.cycles,
            enable_probing=(not args.no_probing),
            evidence_style=args.evidence_style,
            enable_safety=args.safety,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # MEMORY FIX WRAPPER
    def apply_memory_fixes(agent):
        """Apply all memory optimizations"""

        # Fix 1: Patch internal search with memory management
        if hasattr(agent, 'enable_internal_search'):
            original_search = TwoWaySearch._search_internal

            def memory_safe_search(self, query, topic, drive_pressures, n_results=2):
                results = []

                for i in range(min(n_results, 2)):  # Max 2 to save memory
                    torch.cuda.empty_cache()

                    try:
                        result = original_search(self, query, topic, drive_pressures, 1)
                        results.extend(result)
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            torch.cuda.empty_cache()
                            break

                return results

            TwoWaySearch._search_internal = memory_safe_search
            print("[✓] Memory-safe internal search enabled")

        # Fix 2: Ensure score_candidate returns CPU tensors
        original_score = agent.sim.score_candidate

        def cpu_safe_score(text, drive_pressures, topic="open"):
            try:
                score, breakdown, emb = original_score(text, drive_pressures, topic)

                # Force to CPU numpy
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu().numpy()

                return score, breakdown, emb.astype(np.float32)
            except Exception as e:
                print(f"[!] Score error: {str(e)[:80]}")
                return 0.5, {'curiosity': 0.5, 'coherence': 0.5, 'novelty': 0.5, 'truthfulness': 0.5}, np.zeros(agent.model.config.n_embd, dtype=np.float32)

        agent.sim.score_candidate = cpu_safe_score
        print("[✓] CPU-safe scoring enabled")

        print("[✓] All memory fixes applied\n")

    # Apply fixes
    apply_memory_fixes(agent)

    print(f"\n[i] Ready. Commands: /help, /mode, /stats, /quit")

    # Add this right after initializing your agent
    original_novelty = agent.sim._novelty_score

    def fixed_novelty(self, text):
        emb = self._text_mean_embedding(text)
        if len(self.history_embs) == 0:
            return 0.9
        device = emb.device
        recent = torch.stack(list(self.history_embs)[-min(len(self.history_embs), 32):]).to(device)
        sims = F.cosine_similarity(emb.unsqueeze(0), recent, dim=1).cpu().numpy()
        avg_sim = float(np.mean(sims))
        return float(np.clip(1.0 - avg_sim, 0.0, 1.0))

    agent.sim._novelty_score = fixed_novelty.__get__(agent.sim, type(agent.sim))

    # Interactive loop
    while True:
        try:
            prompt = input("\n" + "="*80 + "\nUser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[i] Exiting.")
            break

        if prompt == "":
            break

        # Commands
        if prompt.lower() == "/help":
            print("""
Commands:
  /help     - Show this help
  /mode     - Show/change discovery mode
  /status   - Drive states and pressures
  /history  - Recent outputs
  /stats    - Search and performance statistics
  /safety   - Safety violations summary
  /compare  - Run mode comparison on last prompt
  /quit     - Exit

Discovery Modes:
  discovery        - Exhaustive search, drives rank only
  drive-controlled - Drives modulate all stages
            """)
            continue

        if prompt.lower() == "/mode":
            print(f"\nCurrent mode: {agent.discovery_mode.value}")
            print("\nMode parameters:")
            for key, value in agent.mode_params.items():
                print(f"  {key}: {value}")

            change = input("\nChange mode? (discovery/drive-controlled/no): ").strip().lower()
            if change == "discovery":
                agent.discovery_mode = DiscoveryMode.DISCOVERY_ENGINE
                agent._setup_mode_parameters()
                print("✓ Switched to DISCOVERY ENGINE mode")
            elif change == "drive-controlled":
                agent.discovery_mode = DiscoveryMode.DRIVE_CONTROLLED
                agent._setup_mode_parameters()
                print("✓ Switched to DRIVE CONTROLLED mode")
            continue

        if prompt.lower() == "/status":
            pressures = agent.drive_system.get_pressures()
            print("\nDrive Pressures:")
            print(json.dumps(pressures, indent=2))

            print("\nDrive States:")
            for name, drive in agent.drive_system.drives.items():
                print(f"  {name}:")
                print(f"    need: {drive.need:.3f}")
                print(f"    satisfaction: {drive.satisfaction:.3f}")
                print(f"    weight: {drive.weight:.3f}")
            continue

        if prompt.lower() == "/history":
            print("\nRecent Outputs:")
            for entry in agent.output_history[-3:]:
                print(f"\nCycle {entry['time']}:")
                print(f"  Drive: {entry['drive']}")
                print(f"  Topic: {entry['topic']}")
                print(f"  Mode: {entry.get('discovery_mode', 'unknown')}")
                print(f"  Score: {entry['score']:.3f}")
                print(f"  Text: {entry['text'][:150]}...")
            continue

        if prompt.lower() == "/stats":
            print("\nSystem Statistics:")
            print(f"  Total outputs: {len(agent.output_history)}")
            print(f"  Mode: {agent.discovery_mode.value}")
            print(f"  Internal search: {'ON' if agent.enable_internal_search else 'OFF'}")
            print(f"  External search: {'ON' if agent.enable_external_search else 'OFF'}")

            if agent.output_history:
                last = agent.output_history[-1]
                if 'search_stats' in last:
                    stats = last['search_stats']
                    print(f"\nLast Search:")
                    print(f"  Internal results: {stats.get('internal_count', 0)}")
                    print(f"  External results: {stats.get('external_count', 0)}")
                    print(f"  Search time: {stats.get('total_time', 0):.2f}s")
            continue

        if prompt.lower() == "/safety":
            if agent.safety_monitor:
                summary = agent.safety_monitor.get_violation_summary()
                print("\nSafety Violations:")
                print(json.dumps(summary, indent=2))
            else:
                print("\nSafety monitoring disabled")
            continue

        if prompt.lower() == "/quit":
            break

        # Process research prompt
        try:
            result, pressures = agent.run_cycle(
                n_candidates=args.candidates,
                prompt_text=prompt,
                verbose=args.verbose
            )

            # Display result
            print("\n" + "="*80)
            print("Assistant:\n")
            print(result['final_answer'])
            print("\n" + "-"*80)
            print(f"[Mode: {result.get('discovery_mode', 'unknown')}]")
            print(f"[Drive: {agent.output_history[-1]['drive']}, "
                  f"Topic: {result['topic']}]")

            # Show exploration info
            if result.get('exploration_used'):
                pattern = result.get('explored_pattern', 'unknown')
                print(f"[🎯 Exploration: Strategic pattern '{pattern}' discovered]")

            # Show search breakdown
            if 'search_stats' in result:
                stats = result['search_stats']
                exploration_count = stats.get('exploration_count', 0)
                print(f"[Search: {exploration_count} exploration + {stats['internal_count']} internal + "
                      f"{stats['external_count']} external in {stats['total_time']:.2f}s]")

            # Show candidate composition
            if 'candidate_sources' in result:
                sources = result['candidate_sources']
                print(f"[Candidates: {sources['exploration']} strategic, {sources['internal']} generated, "
                      f"{sources['external']} evidence-based]")

            print("="*80)

        except SafetyException as e:
            print(f"\n❌ Safety violation: {e}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # Session summary
    print(f"\n[Session Summary]")
    print(f"  Mode: {agent.discovery_mode.value}")
    print(f"  Outputs: {len(agent.output_history)}")

    if agent.safety_monitor:
        print(f"  Safety: {agent.safety_monitor.get_violation_summary()}")

    print(f"[i] Logs: {agent.audit_path}")
    print("[i] Done.")


class ModeComparison:
    """
    Tools for comparing DISCOVERY_ENGINE vs DRIVE_CONTROLLED modes

    Enables rigorous ablation studies and comparative analysis
    """

    def __init__(self):
        self.results = {
            DiscoveryMode.DISCOVERY_ENGINE: [],
            DiscoveryMode.DRIVE_CONTROLLED: []
        }
        self.metrics = {
            DiscoveryMode.DISCOVERY_ENGINE: {},
            DiscoveryMode.DRIVE_CONTROLLED: {}
        }

    def run_comparative_experiment(self,
                                   test_prompts,
                                   cycles_per_mode=10,
                                   model_name="gpt2-xl",
                                   enable_tools=True):
        """
        Run same prompts in both modes for comparison

        Args:
            test_prompts: List of (prompt, expected_topic) tuples
            cycles_per_mode: How many cycles per mode
            model_name: Which model to use
            enable_tools: Enable external search

        Returns:
            comparison_report: Detailed metrics comparing modes
        """

        print("\n" + "="*80)
        print("MODE COMPARISON EXPERIMENT")
        print("="*80)
        print(f"Test prompts: {len(test_prompts)}")
        print(f"Cycles per mode: {cycles_per_mode}")
        print(f"Model: {model_name}")
        print("="*80 + "\n")

        for mode in [DiscoveryMode.DISCOVERY_ENGINE, DiscoveryMode.DRIVE_CONTROLLED]:
            print(f"\n{'='*80}")
            print(f"TESTING MODE: {mode.value.upper()}")
            print(f"{'='*80}\n")

            # Initialize agent in this mode
            agent = AutonomousAgent(
                model_name=model_name,
                discovery_mode=mode,
                enable_internal_search=True,
                enable_external_search=enable_tools,
                max_cycles=cycles_per_mode,
                verbose=False
            )

            mode_results = []

            for prompt, expected_topic in test_prompts:
                print(f"\n[Prompt] {prompt}")

                try:
                    # Run single cycle
                    result, pressures = agent.run_cycle(
                        prompt_text=prompt,
                        verbose=True
                    )

                    mode_results.append({
                        'prompt': prompt,
                        'expected_topic': expected_topic,
                        'result': result,
                        'pressures': pressures,
                        'output': result['final_answer'],
                        'topic': result['topic'],
                        'search_stats': result.get('search_stats', {}),
                        'mode': mode.value
                    })

                    print(f"[Result] {result['final_answer'][:200]}...")

                except Exception as e:
                    print(f"[Error] {e}")
                    mode_results.append({
                        'prompt': prompt,
                        'expected_topic': expected_topic,
                        'error': str(e),
                        'mode': mode.value
                    })

            self.results[mode] = mode_results

            # Compute metrics for this mode
            self.metrics[mode] = self._compute_metrics(mode_results)

            # Clean up
            del agent
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Generate comparison report
        report = self._generate_comparison_report()

        return report

    def _compute_metrics(self, results):
        """Compute performance metrics for a mode"""

        metrics = {
            'total_prompts': len(results),
            'successful': sum(1 for r in results if 'error' not in r),
            'failed': sum(1 for r in results if 'error' in r),
            'avg_internal_search': 0,
            'avg_external_search': 0,
            'avg_search_time': 0,
            'topic_accuracy': 0,
            'novelty_scores': [],
            'coherence_scores': [],
            'truthfulness_scores': [],
            'output_lengths': [],
        }

        successful = [r for r in results if 'error' not in r]

        if successful:
            # Search statistics
            internal_counts = [
                r['search_stats'].get('internal_count', 0)
                for r in successful if 'search_stats' in r
            ]
            external_counts = [
                r['search_stats'].get('external_count', 0)
                for r in successful if 'search_stats' in r
            ]
            search_times = [
                r['search_stats'].get('total_time', 0)
                for r in successful if 'search_stats' in r
            ]

            if internal_counts:
                metrics['avg_internal_search'] = np.mean(internal_counts)
            if external_counts:
                metrics['avg_external_search'] = np.mean(external_counts)
            if search_times:
                metrics['avg_search_time'] = np.mean(search_times)

            # Topic accuracy
            correct_topic = sum(
                1 for r in successful
                if r['topic'] == r['expected_topic']
            )
            metrics['topic_accuracy'] = correct_topic / len(successful)

            # Output lengths
            metrics['output_lengths'] = [
                len(r['output'].split()) for r in successful
            ]
            metrics['avg_output_length'] = np.mean(metrics['output_lengths'])

            # Quality scores (would need to extract from outputs)
            # For now, placeholder
            metrics['avg_quality'] = 0.5

        return metrics

    def _generate_comparison_report(self):
        """Generate detailed comparison report"""

        report = {
            'summary': {},
            'detailed_metrics': {},
            'statistical_tests': {},
            'recommendations': []
        }

        # Summary statistics
        for mode in [DiscoveryMode.DISCOVERY_ENGINE, DiscoveryMode.DRIVE_CONTROLLED]:
            metrics = self.metrics[mode]

            report['detailed_metrics'][mode.value] = metrics

            report['summary'][mode.value] = {
                'success_rate': metrics['successful'] / max(1, metrics['total_prompts']),
                'avg_internal_search': metrics['avg_internal_search'],
                'avg_external_search': metrics['avg_external_search'],
                'avg_search_time': metrics['avg_search_time'],
                'topic_accuracy': metrics['topic_accuracy'],
                'avg_output_length': metrics.get('avg_output_length', 0)
            }

        # Comparisons
        de_metrics = self.metrics[DiscoveryMode.DISCOVERY_ENGINE]
        dc_metrics = self.metrics[DiscoveryMode.DRIVE_CONTROLLED]

        report['comparisons'] = {
            'internal_search_difference':
                de_metrics['avg_internal_search'] - dc_metrics['avg_internal_search'],
            'external_search_difference':
                de_metrics['avg_external_search'] - dc_metrics['avg_external_search'],
            'time_difference':
                de_metrics['avg_search_time'] - dc_metrics['avg_search_time'],
            'accuracy_difference':
                de_metrics['topic_accuracy'] - dc_metrics['topic_accuracy']
        }

        # Recommendations
        if report['comparisons']['external_search_difference'] > 2:
            report['recommendations'].append(
                "Discovery Engine mode fetches significantly more external sources"
            )

        if report['comparisons']['time_difference'] > 5:
            report['recommendations'].append(
                "Discovery Engine mode takes longer (exhaustive search)"
            )

        if de_metrics['topic_accuracy'] > dc_metrics['topic_accuracy']:
            report['recommendations'].append(
                "Discovery Engine mode shows better topic classification"
            )
        else:
            report['recommendations'].append(
                "Drive Controlled mode shows better topic classification"
            )

        return report

    def print_report(self, report):
        """Pretty-print comparison report"""

        print("\n" + "="*80)
        print("MODE COMPARISON REPORT")
        print("="*80)

        print("\n--- SUMMARY ---\n")
        for mode, summary in report['summary'].items():
            print(f"{mode.upper()}:")
            for metric, value in summary.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.3f}")
                else:
                    print(f"  {metric}: {value}")
            print()

        print("\n--- COMPARISONS (Discovery Engine - Drive Controlled) ---\n")
        for metric, diff in report['comparisons'].items():
            sign = "+" if diff > 0 else ""
            print(f"{metric}: {sign}{diff:.3f}")

        print("\n--- RECOMMENDATIONS ---\n")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")

        print("\n" + "="*80 + "\n")

    def save_report(self, report, filename="mode_comparison_report.json"):
        """Save report to JSON file"""

        # Convert numpy types to native Python
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj

        report_clean = convert(report)

        with open(filename, 'w') as f:
            json.dump(report_clean, f, indent=2)

        print(f"[i] Report saved to {filename}")


if __name__ == "__main__":
    main()