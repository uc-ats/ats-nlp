# skills.py
from typing import List, Optional, Set
from rapidfuzz import fuzz
import spacy
from spacy.matcher import PhraseMatcher

# light synonyms you can extend
SYNONYMS = {
    "js": "javascript",
    "node": "node.js",
    "k8s": "kubernetes",
    "gcp": "google cloud platform"
}

class SkillsEngine:
    def __init__(self, db_path: str = "skills_db.txt", fuzzy_threshold: int = 88):
        self.db = self._load(db_path)
        self.fuzzy_threshold = fuzzy_threshold
        self.nlp = spacy.blank("en")
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        patterns = [self.nlp.make_doc(skill) for skill in self.db]
        self.matcher.add("SKILLS", patterns)

    def _load(self, path: str) -> List[str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            raw = ["java","spring","spring boot","python","react","aws","docker","sql","kubernetes","microservices"]
        # normalize + add synonyms
        expanded = set()
        for s in raw:
            expanded.add(s.lower())
            if s.lower() in SYNONYMS:
                expanded.add(SYNONYMS[s.lower()])
        for k, v in SYNONYMS.items():
            if v not in expanded:
                expanded.add(v)
        return sorted(expanded)

    def extract(self, text: str, skills_section_text: Optional[str] = None) -> List[str]:
        found: Set[str] = set()

        # 1) exact multi-word via PhraseMatcher on Skills section (if present)
        target = skills_section_text or text
        doc = self.nlp.make_doc(target)
        for _, start, end in self.matcher(doc):
            found.add(doc[start:end].text.lower())

        # 2) fuzzy against full text for misses / typos
        lowered = text.lower()
        for s in self.db:
            if s in lowered:
                found.add(s)
            else:
                if fuzz.partial_ratio(s, lowered) >= self.fuzzy_threshold:
                    found.add(s)

        # normalize synonyms
        normalized = set(SYNONYMS.get(s, s) for s in found)
        return sorted(normalized)