
"""
Self-Directed Research AI Assistant.
This is the full architecture validation system with:
- Science/Creative drive blending
- ExploreKnowledgeSpace with research frontiers
- Complete safety infrastructure
- EvaluatorEnsemble and advanced scoring
- Evidence gathering with Wikipedia
- Database persistence
- PCA/HyperNet probing
- Full audit logging

Usage:
  python autonomous_research_ai_complete.py --model gpt2-medium --tools --candidates 16
"""

import argparse, json, math, os, random, re, sqlite3, sys, time, requests
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from typing import List, Optional, Tuple, Dict, Set, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

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
# EVIDENCE SYSTEM
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

class EvidenceGatherer:
    def __init__(self, enable_network=False):
        self.enable_network = enable_network
        self.cache = {}

    def gather(self, topic: str, prompt_text: str,
               max_sources: int = 3) -> List[EvidenceSource]:
        if not self.enable_network:
            return []

        sources = []
        search_query = self._extract_search_terms(prompt_text, topic)
        wiki_sources = self._fetch_wikipedia(search_query, 2)
        sources.extend(wiki_sources)

        return sources[:max_sources]

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

    def _fetch_wikipedia(self, query: str, max_results: int = 1) -> List[EvidenceSource]:
        sources = []

        if not self.enable_network:
            return sources

        try:
            clean_query = query.strip().replace(' ', '_')
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{clean_query}"

            resp = requests.get(url, timeout=10, headers={'User-Agent': 'ResearchAI/1.0'})

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
            else:
                search_url = "https://en.wikipedia.org/w/api.php"
                params = {
                    'action': 'opensearch',
                    'search': query,
                    'limit': 1,
                    'format': 'json'
                }

                search_resp = requests.get(search_url, params=params, timeout=10)

                if search_resp.status_code == 200:
                    search_data = search_resp.json()
                    if len(search_data) > 3 and search_data[3]:
                        result_url = search_data[3][0]
                        page_title = result_url.split('/')[-1]
                        return self._fetch_wikipedia(page_title, 1)

        except Exception as e:
            print(f"[!] Wikipedia error: {str(e)[:100]}")

        return sources

    def format_sources(self, sources: List[EvidenceSource], style: str = "B") -> str:
        if not sources:
            return ("\n\n⚠️ EVIDENCE VERIFICATION DISABLED ⚠️\n"
                   "No external sources consulted. Enable --tools for verification.")

        if style == "B":
            output = "\n\n📚 Verified Evidence Sources:\n"
            for i, source in enumerate(sources, 1):
                confidence_label = (
                    "High" if source.confidence > 0.8 else
                    "Medium" if source.confidence > 0.6 else "Low"
                )

                output += f"\n{i}. [{source.source_type.upper()}] {source.title}\n"
                output += f"   {source.url}\n"
                if source.snippet:
                    snippet = source.snippet[:200] + "..." if len(source.snippet) > 200 else source.snippet
                    output += f"   \"{snippet}\"\n"
                output += f"   Confidence: {confidence_label}\n"

            return output

        return "\n\nSources:\n" + "\n".join(f"- {s.title}" for s in sources)

# -------------------------
# UTILITIES
# -------------------------

def safe_mkdir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

def collapse_repeats(s: str) -> str:
    if not s:
        return s

    original_length = len(s)
    
    s = re.sub(r'(.)\1{4,}', r'\1', s)
    s = re.sub(r'\b(\w{2,}?)(\1){2,}\b', r'\1', s)
    s = re.sub(r'\b(\w+)(\s+\1){2,}\b', r'\1', s)
    
    for phrase_len in range(6, 1, -1):
        pattern = r'\b((?:\w+\s+){' + str(phrase_len-1) + r'}\w+)(\s+\1){1,}'
        s = re.sub(pattern, r'\1', s)
    
    words = s.split()
    if len(words) > 3:
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        if word_counts:
            max_count = max(word_counts.values())
            if max_count > len(words) * 0.5:
                return ""

    if original_length > 20 and len(s) < 0.3 * original_length:
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
        "photon", "electron", "atom", "nuclear", "relativity", "physics"
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
            'curiosity': DriveState(need=0.6, satisfaction=0.0, weight=1.0, decay_rate=0.03),
            'coherence': DriveState(need=0.35, satisfaction=0.5, weight=0.9, decay_rate=0.015),
            'novelty': DriveState(need=0.45, satisfaction=0.3, weight=0.95, decay_rate=0.025),
            'truthfulness': DriveState(need=0.5, satisfaction=0.5, weight=1.2, decay_rate=0.01),
        }

        self.synergies = {
            ('curiosity', 'novelty'): 0.20,
            ('coherence', 'curiosity'): -0.08,
            ('coherence', 'novelty'): -0.12,
            ('truthfulness', 'novelty'): -0.10,
            ('truthfulness', 'curiosity'): 0.04,
            ('truthfulness', 'coherence'): 0.15,
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

def load_model(model_name="gpt2"):
    print(f"[i] Loading '{model_name}' on {DEVICE}...")
    model = GPT2LMHeadModel.from_pretrained(model_name).to(DEVICE)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    model.eval()
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
            self.directions[drive] = (combined / norm).detach().cpu()

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

class InternalSimulator:
    def __init__(self, model, tokenizer, device=DEVICE, temperature=0.9,
                 history_size=256, enable_truthfulness=True):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.history = deque(maxlen=history_size)
        self.history_embs = deque(maxlen=history_size)

        self.enable_truthfulness = enable_truthfulness
        if enable_truthfulness:
            self.truthfulness_scorer = TruthfulnessScorer(enable_external_verification=False)

    def _text_mean_embedding(self, text):
        toks = self.tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        if toks.shape[1]==0:
            return torch.zeros(self.model.config.n_embd, device=self.device)
        with torch.no_grad():
            emb = self.model.transformer.wte(toks).float().mean(dim=1).squeeze(0)
        return emb

    def steer_generate(self, direction_vec, prompt_text="", max_length=80, steer_strength=1.0, probe_delta_emb=None):
        dir_emb = direction_vec.to(self.device).float()
        if prompt_text.strip()=="":
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
            token_embs = self.model.transformer.wte.weight
            te = token_embs
            if probe_delta_emb is not None:
                te = token_embs + probe_delta_emb.to(token_embs.device).unsqueeze(0)
            de = dir_emb / (dir_emb.norm() + 1e-8)
            te_norm = te / (te.norm(dim=1, keepdim=True) + 1e-8)
            sim = torch.matmul(te_norm, de).unsqueeze(0)
            sim = sim * (last_logits.std().detach() / (sim.std().detach() + 1e-8))
            biased = last_logits + steer_strength * sim
            probs = F.softmax(biased / max(1e-6, self.temperature), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            out_tokens.append(int(next_token.item()))
            if int(next_token.item()) == self.tokenizer.eos_token_id:
                break
            if len(out_tokens) > 8:
                last_tok = out_tokens[-1]
                if out_tokens.count(last_tok) > 12:
                    break
        text = self.tokenizer.decode(out_tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        return text

    def _coherence_score(self, text):
        toks = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        if toks.shape[1] <= 1:
            return 0.45
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
        toks = text.split()
        if len(toks)==0:
            return 0.0
        return float(len(set(toks)) / len(toks))

    def _novelty_score(self, text):
        emb = self._text_mean_embedding(text)
        if len(self.history_embs) == 0:
            return 0.9
        recent = torch.stack(list(self.history_embs)[-min(len(self.history_embs), 32):])
        sims = F.cosine_similarity(emb.unsqueeze(0), recent, dim=1).cpu().numpy()
        avg_sim = float(np.mean(sims))
        novelty = float(np.clip(1.0 - avg_sim, 0.0, 1.0))
        return novelty

    def score_candidate(self, text, drive_pressures, topic="open"):
        """Score with REAL truthfulness"""
        info = self._info_score(text)
        coh = self._coherence_score(text)
        nov = self._novelty_score(text)

        # Real truthfulness
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

        total = 0.0
        for k, v in drive_pressures.items():
            if k in scores:
                total += v * scores[k]

        total = total - 0.01 * len(text.split())

        emb = self._text_mean_embedding(text)
        self.history.append(text)
        self.history_embs.append(emb.detach().cpu())

        return float(total), scores, emb.detach().cpu().numpy().astype(np.float32)

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
# EXPLORE KNOWLEDGE SPACE (COMPLETE)
# -------------------------

class ExploreKnowledgeSpace:
    """
    Complete knowledge space exploration.
    Identifies genuinely unexplored regions.
    """
    
    def __init__(self, simulator, ethics: EthicalConstraints):
        self.sim = simulator
        self.ethics = ethics
        
        # Track exploration
        self.explored_regions = set()
        self.exploration_history = []
        
        # Research frontiers database
        self.research_frontiers = {
            'physics': [
                'quantum error correction mechanisms',
                'room temperature superconductivity',
                'dark matter detection methods',
                'topological quantum computing',
                'emergent spacetime geometry'
            ],
            'biology': [
                'protein folding prediction accuracy',
                'cellular aging reversal mechanisms',
                'microbiome-brain axis interactions',
                'epigenetic inheritance patterns',
                'synthetic minimal cells'
            ],
            'chemistry': [
                'catalyst design for CO2 capture',
                'battery electrode materials',
                'sustainable plastic alternatives',
                'photocatalytic water splitting',
                'molecular machine design'
            ],
            'tech': [
                'quantum algorithm applications',
                'neuromorphic computing architectures',
                'energy-efficient AI training',
                'post-quantum cryptography',
                'biocompatible electronics'
            ],
            'health': [
                'personalized medicine algorithms',
                'early disease biomarkers',
                'drug resistance mechanisms',
                'immunotherapy optimization',
                'regenerative medicine scaffolds'
            ]
        }
    
    def find_low_activation_regions(self, domain: str) -> List[str]:
        """Identify under-explored areas"""
        frontiers = self.research_frontiers.get(domain, [])
        
        unexplored = [
            f for f in frontiers 
            if f not in self.explored_regions
        ]
        
        return unexplored
    
    def find_distant_concept_pairs(self, domain: str) -> List[Tuple[str, str]]:
        """Identify conceptually distant pairs"""
        frontiers = self.research_frontiers.get(domain, [])
        
        if len(frontiers) < 2:
            return []
        
        pairs = []
        
        for i, concept_a in enumerate(frontiers):
            emb_a = self.sim._text_mean_embedding(concept_a)
            
            for concept_b in frontiers[i+1:]:
                emb_b = self.sim._text_mean_embedding(concept_b)
                
                distance = 1.0 - F.cosine_similarity(
                    emb_a.unsqueeze(0),
                    emb_b.unsqueeze(0)
                ).item()
                
                if distance > 0.5:
                    pairs.append((concept_a, concept_b, distance))
        
        pairs.sort(key=lambda x: x[2], reverse=True)
        return [(a, b) for a, b, _ in pairs]
    
    def synthesize_novel_patterns(self, pattern_candidates: List[str], 
                                 domain: str,
                                 drive_pressures: Dict[str, float],
                                 direction_vec) -> List[Dict]:
        """Generate hypotheses with ethics screening"""
        hypotheses = []
        
        for pattern in pattern_candidates[:5]:
            hypothesis_text = self.sim.steer_generate(
                direction_vec,
                prompt_text=f"Exploring {pattern} in {domain}:",
                max_length=60,
                steer_strength=1.2
            )
            
            # Ethics check BEFORE scoring
            approved, violations, recommendations = self.ethics.check_hypothesis(
                hypothesis_text, domain
            )
            
            if not approved:
                print(f"⚠️ Hypothesis blocked: {violations[0]}")
                continue
            
            # Score if approved
            score, breakdown, emb = self.sim.score_candidate(
                hypothesis_text, drive_pressures, topic=domain
            )
            
            hypotheses.append({
                'text': hypothesis_text,
                'pattern': pattern,
                'score': score,
                'breakdown': breakdown,
                'emb': emb,
                'ethics_warnings': violations,
                'recommendations': recommendations
            })
            
            self.explored_regions.add(pattern)
        
        return hypotheses
    
    def explore(self, domain: str, drive_pressures: Dict[str, float],
                direction_vec, prompt_text: Optional[str] = None) -> Optional[Dict]:
        """Main exploration function"""
        
        dominant = max(drive_pressures.items(), key=lambda x: x[1])[0]
        
        if drive_pressures.get('curiosity', 0) > 0.7:
            patterns = self.find_low_activation_regions(domain)
        elif drive_pressures.get('novelty', 0) > 0.7:
            pairs = self.find_distant_concept_pairs(domain)
            patterns = [f"{a} and {b}" for a, b in pairs[:3]]
        else:
            patterns = self.find_low_activation_regions(domain)
        
        if not patterns:
            return None
        
        candidates = self.synthesize_novel_patterns(
            patterns, domain, drive_pressures, direction_vec
        )
        
        if not candidates:
            return None
        
        best = max(candidates, key=lambda x: x['score'])
        self.exploration_history.append(best['pattern'])
        
        return best

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
        
        self.explorer = ExploreKnowledgeSpace(agent.sim, self.ethics)
        
        # Safety audit
        self.safety_log = []
    
    def enhanced_cycle(self, n_candidates=8, verbose=True, prompt_text=None) -> Tuple[Optional[Dict], Dict]:
        """Enhanced cycle with ethics enforcement - COMPLETE VERSION"""
    
        # Pre-cycle domain check
        if prompt_text and self.enable_ethics:
            topic = classify_topic(prompt_text)
        
            if verbose:
                print(f"[Ethics Check] Topic: {topic}")
        
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
        dominant, pressures = self.agent.drive_system.get_dominant()
    
        if verbose:
            print(f"[Drive] Dominant: {dominant}, Pressure: {pressures[dominant]:.3f}")
    
        # Try enhanced exploration for science/creative domains
        exploration_result = None
        if prompt_text and self.enable_ethics:
            topic = classify_topic(prompt_text)
            dir_vec = self.agent.mapper.get_direction(dominant)
            if dir_vec is not None:
                dir_vec = dir_vec.to(DEVICE)
            
                try:
                    exploration_result = self.explorer.explore(
                        topic, pressures, dir_vec, prompt_text
                    )
                except Exception as e:
                    if verbose:
                        print(f"[!] Enhanced exploration failed: {str(e)[:100]}")
                    exploration_result = None
            
                if exploration_result:
                    final_answer = exploration_result['text']
                
                    # Add ethics warnings
                    if exploration_result.get('ethics_warnings'):
                        final_answer += "\n\n⚠️ ETHICS REVIEW REQUIRED ⚠️\n"
                        for warning in exploration_result['ethics_warnings']:
                            final_answer += f"  - {warning}\n"
                
                    # Add evidence
                    if self.agent.enable_tools:
                        try:
                            evidence = self.agent.evidence_gatherer.gather(
                                topic, prompt_text, max_sources=3
                            )
                            if evidence:
                                final_answer += self.agent.evidence_gatherer.format_sources(
                                    evidence, style=self.agent.evidence_style
                                )
                        except Exception as e:
                            if verbose:
                                print(f"[!] Evidence error: {str(e)[:100]}")
                
                    # Medical disclaimer
                    if topic == "health":
                        final_answer += ("\n\n⚠️ MEDICAL DISCLAIMER ⚠️\n"
                                       "For educational purposes only. "
                                       "Consult healthcare professionals.")
                
                    self.safety_log.append({
                        'timestamp': time.time(),
                        'hypothesis': final_answer,
                        'topic': topic,
                        'ethics_approved': True,
                        'warnings': exploration_result.get('ethics_warnings', [])
                    })
                
                    return {
                        'final_answer': final_answer,
                        'mode': choose_mode_for_topic(topic),
                        'topic': topic,
                        'drive': dominant,
                        'ethics_warnings': exploration_result.get('ethics_warnings', [])
                    }, pressures
    
        # Fallback to standard cycle with full safety integration
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
                    topic = result.get('topic', 'open')
                
                    approved, violations, recommendations = self.ethics.check_hypothesis(
                        final_answer, topic,
                        evidence_verified=self.agent.enable_tools
                    )
                
                    if not approved:
                        result['final_answer'] = (
                            f"[Output blocked by ethics layer]\n\n"
                            f"Reason: {violations[0]}\n"
                            f"Recommendations: {', '.join(recommendations)}"
                        )
                        result['ethics_blocked'] = True
                
                    elif violations:
                        result['final_answer'] += "\n\n⚠️ ETHICS REVIEW ⚠️\n"
                        for warning in violations:
                            result['final_answer'] += f"  - {warning}\n"
                        result['ethics_warnings'] = violations
                
                    self.safety_log.append({
                        'timestamp': time.time(),
                        'hypothesis': final_answer,
                        'topic': topic,
                        'ethics_approved': approved,
                        'violations': violations
                    })
        
            return result, pressures
        
        except Exception as e:
            print(f"❌ Cycle error: {e}")
            import traceback
            traceback.print_exc()
            return None, {}
    
    def run_safe_hybrid(self, pause=0.8, n_candidates=8, 
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
# AUTONOMOUS AGENT (COMPLETE)
# -------------------------

class AutonomousAgent:
    """Complete production-ready autonomous AI"""

    def __init__(self,
                 model_name="gpt2",
                 persist_path="./memory.db",
                 science_weight=0.7,
                 hybrid_satisfaction_thresh=0.20,
                 max_cycles=10,
                 enable_tools=False,
                 enable_probing=True,
                 evidence_style="B",
                 enable_safety=True):

        print(f"[i] Initializing Complete Enhanced v2 (safety={'ON' if enable_safety else 'OFF'})")

        self.model, self.tokenizer = load_model(model_name)
        self.drive_system = DriveSystem(enable_safety_checks=enable_safety)
        self.mapper = FeatureTargetMapper(self.model, self.tokenizer, science_weight=science_weight)
        self.scorer = RelevanceScorer(self.model, self.tokenizer)

        self.sim = InternalSimulator(
            self.model, self.tokenizer,
            temperature=0.9,
            enable_truthfulness=True
        )

        self.evaluator = EvaluatorEnsemble(self.model.config.n_embd, n_members=3, device=DEVICE)
        self.persist = PersistDB(persist_path)

        if enable_safety:
            self.safety_monitor = SafetyMonitor(enable_ethics=True)
        else:
            self.safety_monitor = None

        self.evidence_gatherer = EvidenceGatherer(enable_network=enable_tools)

        self.output_history = []
        self.pressure_history = []
        self.hybrid_thresh = hybrid_satisfaction_thresh
        self.max_cycles = max_cycles
        self.enable_tools = enable_tools
        self.enable_probing = enable_probing
        self.evidence_style = evidence_style

        self.pca_basis = None
        self.low_d = 32
        self.hypernet = HyperNet(ctx_dim=self.model.config.n_embd, low_d=self.low_d).to(DEVICE)
        self.hyper_opt = torch.optim.Adam(self.hypernet.parameters(), lr=1e-4)
        self.last_pca_cycle = 0

        safe_mkdir("runs")
        self.audit_path = "runs/audit_log.jsonl"

        self._try_build_pca()

        print(f"[i] Ready (safety={'ON' if enable_safety else 'OFF'}, tools={'ON' if enable_tools else 'OFF'})")

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

    def run_cycle(self, n_candidates=8, verbose=True, prompt_text=None):
        """Complete cycle with all features"""

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
            print(f"[Cycle {cyc}] Topic={topic}, Mode={mode_label}, Drive={dominant}")

        # Drive safety check
        if self.safety_monitor:
            try:
                self.safety_monitor.check_drives(self.drive_system)
            except SafetyException as e:
                print(f"❌ Drive violation: {e}")
                raise

        dir_vec = self.mapper.get_direction(dominant).to(DEVICE)

        if len(self.sim.history_embs) > 0:
            ctx_emb = torch.tensor(np.mean(np.stack(list(self.sim.history_embs)[-8:]), axis=0), dtype=torch.float32)
        else:
            ctx_emb = self.sim._text_mean_embedding(dominant)

        if self.pca_basis is None:
            self._try_build_pca()

        # Generate candidates with advanced features
        candidates = []
        attempts = 0
        max_attempts = n_candidates * 15

        while len(candidates) < n_candidates and attempts < max_attempts:
            attempts += 1
            steer = float(max(0.5, 0.8 + 1.2 * random.random()))
            max_len = int(30 + 50 * random.random())

            probe_delta = None
            if self.enable_probing and self.pca_basis is not None:
                probe_delta = self.generate_probe_delta_emb(ctx_emb)

            txt = self.sim.steer_generate(
                dir_vec,
                prompt_text=prompt_text or "",
                max_length=max_len,
                steer_strength=steer,
                probe_delta_emb=probe_delta
            )

            if not txt or len(txt.strip()) < 10:
                continue

            # Enhanced degenerate detection
            if re.search(r'(\w{4,})\1{2,}', txt):
                continue

            toks = txt.split()

            if len(toks) > 0:
                word_counts = {}
                for tok in toks:
                    word_counts[tok] = word_counts.get(tok, 0) + 1

                max_count = max(word_counts.values()) if word_counts else 0
                if max_count > len(toks) * 0.3:
                    continue

            if len(txt) > 20 and len(set(txt)) < len(txt) * 0.3:
                continue

            txt = collapse_repeats(txt).strip()

            if not txt or len(txt) < 10:
                continue

            if len(toks) > 5 and len(txt) < len(toks) * 2:
                continue

            score_val, breakdown, emb = self.sim.score_candidate(txt, pressures, topic=topic)

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
                'emb': emb
            })

        if verbose and len(candidates) < n_candidates:
            print(f"[!] Warning: Only {len(candidates)}/{n_candidates} valid candidates after {attempts} attempts")

        if not candidates:
            if topic == "tech":
                fallback = f"Research into {prompt_text or 'this domain'} suggests potential applications in materials science and device optimization."
            elif topic == "physics":
                fallback = f"Theoretical frameworks indicate quantum mechanical effects warrant investigation."
            elif topic == "health":
                fallback = f"Clinical research is ongoing. Consult healthcare professionals."
            else:
                fallback = "Further investigation is needed."
            
            print(f"[!] Warning: All {attempts} attempts degenerate. Using fallback.")
            score_val, breakdown, emb = self.sim.score_candidate(fallback, pressures, topic=topic)
            candidates = [{'text': fallback, 'score': float(score_val), 'breakdown': breakdown,
                          'steer': 0.0, 'science_value': 0.0, 'uncertainty': 0.0, 'emb': emb}]

        best = max(candidates, key=lambda x: x['score'])
        best_text = collapse_repeats(best['text']).strip()

        # Gather evidence
        evidence_sources = []
        evidence_text = ""

        if self.evidence_style != "none":
            if self.enable_tools:
                try:
                    evidence_sources = self.evidence_gatherer.gather(topic, prompt_text or best_text, max_sources=3)
                    if evidence_sources:
                        evidence_text = self.evidence_gatherer.format_sources(evidence_sources, style=self.evidence_style)
                    else:
                        evidence_text = "\n\n📚 Evidence Search: No relevant sources found."
                except Exception as e:
                    evidence_text = f"\n\n⚠️ Evidence error: {str(e)[:100]}"
            else:
                evidence_text = ("\n\n⚠️ EVIDENCE VERIFICATION DISABLED ⚠️\n"
                               "No external sources consulted. Enable --tools for verification.")

        # Medical disclaimer
        if topic == "health":
            medical_disclaimer = ("\n\n⚠️ MEDICAL DISCLAIMER ⚠️\n"
                                "For educational purposes only. "
                                "Consult healthcare professionals for medical decisions.")
            evidence_text += medical_disclaimer

        final_answer = best_text
        if evidence_text:
            final_answer = final_answer.rstrip() + evidence_text

        if verbose:
            print(f"\n[Selected] Score={best['score']:.4f} Truth={best['breakdown'].get('truthfulness', 0):.3f}")
            preview = final_answer[:400] + "..." if len(final_answer) > 400 else final_answer
            print(preview)

        # Output safety check
        if self.safety_monitor:
            is_safe, violations = self.safety_monitor.check_output(
                final_answer, topic, evidence_verified=self.enable_tools
            )

            if not is_safe:
                error_msg = "\n".join(f"- {v.message}" for v in violations if v.severity in ["critical", "error"])
                fallback_answer = f"[Output blocked: {error_msg}]"
                final_answer = fallback_answer
                best['score'] = 0.0

        try:
            out_id = self.persist.save_output(
                cyc, dominant, mode_label, topic, float(best['score']),
                final_answer, emb=np.array(best['emb'], dtype=np.float32),
                evidence=evidence_text if evidence_text else None
            )
        except Exception as e:
            self._log_audit("save_failed", {"error": str(e)})
            out_id = None

        self.output_history.append({
            'time': cyc, 'drive': dominant, 'mode': mode_label,
            'topic': topic, 'text': final_answer, 'score': float(best['score']),
            'evidence_verified': self.enable_tools
        })

        # Update evaluator
        target_vec = np.array([
            0.6 * best['breakdown'].get('curiosity', 0.0) + 0.4 * best['breakdown'].get('coherence', 0.0),
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

        if satisfactions.get('coherence', 0.0) > 0.6 and satisfactions.get('curiosity', 0.0) > 0.6:
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

        return {'final_answer': final_answer, 'mode': mode_label, 'topic': topic}, pressures

# Replace the existing run_hybrid method with this enhanced version:

    def run_hybrid(self, pause=0.8, n_candidates=8, initial_prompt=None, max_cycles=None, enable_ethics=True):
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

            if current_pressure <= self.hybrid_thresh:
                print(f"\n[i] Satisfied ({current_pressure:.4f} <= {self.hybrid_thresh:.4f})")
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

# -------------------------
# CLI (COMPLETE)
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Complete Autonomous Research AI")

    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--persist", type=str, default="./memory.db")

    parser.add_argument("--safety", dest="safety", action="store_true")
    parser.add_argument("--no-safety", dest="safety", action="store_false")
    parser.set_defaults(safety=True)

    parser.add_argument("--tools", dest="tools", action="store_true")
    parser.add_argument("--no-tools", dest="tools", action="store_false")
    parser.set_defaults(tools=True)

    parser.add_argument("--no-probing", action="store_true")
    parser.add_argument("--evidence-style", type=str, default="B", choices=["none", "A", "B"])

    args, unknown = parser.parse_known_args()

    if not args.safety:
        print("\n" + "="*80)
        print("⚠️ ⚠️ ⚠️  WARNING: SAFETY & ETHICS DISABLED  ⚠️ ⚠️ ⚠️")
        print("="*80 + "\n")
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            return

    print("\n[i] Initializing Complete Enhanced v2...")
    try:
        agent = AutonomousAgent(
            model_name=args.model,
            persist_path=args.persist,
            hybrid_satisfaction_thresh=args.threshold,
            max_cycles=args.cycles,
            enable_tools=args.tools,
            enable_probing=(not args.no_probing),
            evidence_style=args.evidence_style,
            enable_safety=args.safety
        )
        
        # Wrap with Safe Discovery Engine
        safe_engine = SafeDiscoveryEngine(agent, enable_ethics=args.safety)
        
    except Exception as e:
        print(f"❌ Init failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n[i] Ready. Commands: /help, /quit")

    while True:
        try:
            prompt = input("\n" + "="*80 + "\nUser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[i] Exiting.")
            break

        if prompt == "":
            break

        if prompt.lower() == "/help":
            print("""
Commands:
  /help     - Show this help
  /status   - Drive states
  /history  - Recent outputs
  /safety   - Safety violations
  /ethics   - Ethics report
  /quit     - Exit
            """)
            continue

        if prompt.lower() == "/status":
            pressures = agent.drive_system.get_pressures()
            print(json.dumps({"pressures": pressures}, indent=2))
            continue

        if prompt.lower() == "/history":
            for e in agent.output_history[-3:]:
                print(f"\nCycle {e['time']}: {e['text'][:150]}...")
            continue

        if prompt.lower() == "/safety":
            if agent.safety_monitor:
                print(agent.safety_monitor.get_violation_summary())
            else:
                print("Safety disabled")
            continue
        
        if prompt.lower() == "/ethics":
            report = safe_engine.get_safety_report()
            print(json.dumps(report, indent=2))
            continue

        if prompt.lower() == "/quit":
            break

        # Process prompt with Safe Discovery Engine
        try:
            agent.max_cycles = args.cycles
            outputs = safe_engine.run_safe_hybrid(
                pause=0.2,
                n_candidates=args.candidates,
                initial_prompt=prompt,
                max_cycles=args.cycles
            )

            if len(outputs) > 0 and len(agent.output_history) > 0:
                last = outputs[-1]
                last_history = agent.output_history[-1]
                
                print("\n" + "="*80)
                print("Assistant:\n")
                print(last['final_answer'])
                print("\n" + "-"*80)
                print(f"[Drive: {last_history.get('drive', 'unknown')}, "
                      f"Topic: {last.get('topic', 'unknown')}, "
                      f"Mode: {last.get('mode', 'unknown')}]")

                if last.get('ethics_warnings'):
                    print(f"⚠️ Ethics warnings: {len(last['ethics_warnings'])}")

                if not agent.enable_tools:
                    print("⚠️ Evidence not verified (--tools disabled)")

                print("="*80)
            else:
                print("\n[No output generated]")

        except SafetyException as e:
            print(f"\n❌ Safety violation: {e}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[Session Summary]")
    print(f"  Outputs: {len(agent.output_history)}")

    if agent.safety_monitor:
        print(f"  Safety: {agent.safety_monitor.get_violation_summary()}")
    
    ethics_report = safe_engine.get_safety_report()
    print(f"  Ethics blocked: {ethics_report['blocked']}")
    print(f"  Ethics warnings: {ethics_report['warnings']}")

    print(f"[i] Logs: {agent.audit_path}")
    print("[i] Done.")


if __name__ == "__main__":
    main()