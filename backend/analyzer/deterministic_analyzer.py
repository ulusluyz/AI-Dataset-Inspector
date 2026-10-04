import re
import hashlib
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel
from backend.inspector.hf_inspector import SampledRow

class DeterministicMetrics(BaseModel):
    total_sampled: int
    exact_duplicates_count: int
    exact_duplicate_ratio: float
    short_rows_count: int
    short_row_ratio: float
    long_rows_count: int
    long_row_ratio: float
    html_junk_count: int
    html_junk_ratio: float
    unicode_anomalies_count: int
    pii_matches_count: int
    pii_types_found: List[str]
    detected_languages_stat: Dict[str, float]
    schema_types: List[str]
    has_chat_structure: bool
    has_instruction_structure: bool
    has_qa_structure: bool
    evidence_samples: List[Dict[str, str]]

# Regex patterns for deterministic detection
PII_EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PII_PHONE_PATTERN = re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
PII_TC_PATTERN = re.compile(r'\b[1-9]\d{10}\b') # Turkish ID / 11-digit numbers
HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
UNICODE_CONTROL_PATTERN = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]')

TURKISH_CHAR_PATTERN = re.compile(r'[çğıöşüÇĞİÖŞÜ]')
ENGLISH_COMMON_WORDS = {"the", "and", "is", "of", "to", "in", "that", "it", "with", "for", "as", "was"}
TURKISH_COMMON_WORDS = {"ve", "bir", "bu", "de", "da", "için", "ile", "o", "ama", "çok", "var", "gibi", "kadar"}

class DeterministicAnalyzer:
    def analyze(self, rows: List[SampledRow]) -> DeterministicMetrics:
        total = len(rows)
        if total == 0:
            return DeterministicMetrics(
                total_sampled=0,
                exact_duplicates_count=0,
                exact_duplicate_ratio=0.0,
                short_rows_count=0,
                short_row_ratio=0.0,
                long_rows_count=0,
                long_row_ratio=0.0,
                html_junk_count=0,
                html_junk_ratio=0.0,
                unicode_anomalies_count=0,
                pii_matches_count=0,
                pii_types_found=[],
                detected_languages_stat={},
                schema_types=[],
                has_chat_structure=False,
                has_instruction_structure=False,
                has_qa_structure=False,
                evidence_samples=[]
            )

        hashes = set()
        exact_dup_cnt = 0
        short_cnt = 0
        long_cnt = 0
        html_cnt = 0
        unicode_cnt = 0
        pii_cnt = 0
        pii_types = set()

        tr_word_score = 0
        en_word_score = 0
        total_word_evals = 0

        schema_keys = set()
        has_chat = False
        has_instruction = False
        has_qa = False

        evidence: List[Dict[str, str]] = []

        for item in rows:
            row_dict = item.row_data
            if not isinstance(row_dict, dict):
                continue

            for k in row_dict.keys():
                schema_keys.add(k.lower())

            # Structure detection
            keys_lower = [k.lower() for k in row_dict.keys()]
            if any(k in keys_lower for k in ["messages", "conversations", "role", "chat"]):
                has_chat = True
            if any(k in keys_lower for k in ["instruction", "output", "prompt", "response"]):
                has_instruction = True
            if any(k in keys_lower for k in ["question", "answer", "query"]):
                has_qa = True

            # Text content analysis across string fields
            row_text_parts = []
            for v in row_dict.values():
                if isinstance(v, str):
                    row_text_parts.append(v)
                elif isinstance(v, list):
                    for sub in v:
                        if isinstance(sub, dict):
                            for sub_v in sub.values():
                                if isinstance(sub_v, str):
                                    row_text_parts.append(sub_v)
                        elif isinstance(sub, str):
                            row_text_parts.append(sub)

            full_text = " ".join(row_text_parts).strip()
            if not full_text:
                short_cnt += 1
                continue

            # Exact Duplicate check using MD5
            text_hash = hashlib.md5(full_text.encode('utf-8')).hexdigest()
            if text_hash in hashes:
                exact_dup_cnt += 1
            else:
                hashes.add(text_hash)

            # Length stats
            if len(full_text) < 20:
                short_cnt += 1
            elif len(full_text) > 10_000:
                long_cnt += 1

            # HTML / Junk tags
            if HTML_TAG_PATTERN.search(full_text):
                html_cnt += 1
                if len(evidence) < 5:
                    evidence.append({
                        "bulgu": "HTML / Markup artığı tespit edildi",
                        "split": f"{item.config_name}/{item.split_name} (#row {item.row_index})",
                        "ornek": self._mask_and_truncate(full_text)
                    })

            # Unicode Control characters
            if UNICODE_CONTROL_PATTERN.search(full_text):
                unicode_cnt += 1

            # PII Checks
            if PII_EMAIL_PATTERN.search(full_text):
                pii_cnt += 1
                pii_types.add("e-mail")
                if len(evidence) < 5:
                    evidence.append({
                        "bulgu": "E-posta adresi (PII) tespit edildi",
                        "split": f"{item.config_name}/{item.split_name} (#row {item.row_index})",
                        "ornek": self._mask_and_truncate(full_text)
                    })

            if PII_PHONE_PATTERN.search(full_text):
                pii_cnt += 1
                pii_types.add("telefon")

            if PII_TC_PATTERN.search(full_text):
                pii_cnt += 1
                pii_types.add("kimlik/TC benzeri numara")

            # Language statistical signal
            words = full_text.lower().split()
            if words:
                total_word_evals += 1
                tr_chars = len(TURKISH_CHAR_PATTERN.findall(full_text))
                tr_common = sum(1 for w in words if w in TURKISH_COMMON_WORDS)
                en_common = sum(1 for w in words if w in ENGLISH_COMMON_WORDS)

                if tr_chars > 0 or tr_common > 0:
                    tr_word_score += 1
                if en_common > 0:
                    en_word_score += 1

        # Calculate ratios
        dup_ratio = round(exact_dup_cnt / total, 4)
        short_ratio = round(short_cnt / total, 4)
        long_ratio = round(long_cnt / total, 4)
        html_ratio = round(html_cnt / total, 4)

        # Language distribution estimation
        lang_dist = {}
        if total_word_evals > 0:
            tr_pct = round((tr_word_score / total_word_evals) * 100, 1)
            en_pct = round((en_word_score / total_word_evals) * 100, 1)
            if tr_pct > 0:
                lang_dist["Türkçe"] = min(tr_pct, 100.0)
            if en_pct > 0:
                lang_dist["İngilizce"] = min(en_pct, 100.0)
            remaining = round(max(0.0, 100.0 - (lang_dist.get("Türkçe", 0) + lang_dist.get("İngilizce", 0))), 1)
            if remaining > 0 and len(lang_dist) > 0:
                lang_dist["Diğer"] = remaining
            elif not lang_dist:
                lang_dist["Belirlenemedi"] = 100.0

        return DeterministicMetrics(
            total_sampled=total,
            exact_duplicates_count=exact_dup_cnt,
            exact_duplicate_ratio=dup_ratio,
            short_rows_count=short_cnt,
            short_row_ratio=short_ratio,
            long_rows_count=long_cnt,
            long_row_ratio=long_ratio,
            html_junk_count=html_cnt,
            html_junk_ratio=html_ratio,
            unicode_anomalies_count=unicode_cnt,
            pii_matches_count=pii_cnt,
            pii_types_found=list(pii_types),
            detected_languages_stat=lang_dist,
            schema_types=list(schema_keys),
            has_chat_structure=has_chat,
            has_instruction_structure=has_instruction,
            has_qa_structure=has_qa,
            evidence_samples=evidence
        )

    def _mask_and_truncate(self, text: str, max_len: int = 150) -> str:
        """Masks sensitive values and truncates string safely."""
        masked = PII_EMAIL_PATTERN.sub("[E-POSTA MASKELENDİ]", text)
        masked = PII_PHONE_PATTERN.sub("[TELEFON MASKELENDİ]", masked)
        masked = PII_TC_PATTERN.sub("[KİMLİK MASKELENDİ]", masked)
        clean = " ".join(masked.split())
        if len(clean) > max_len:
            return clean[:max_len] + "..."
        return clean
