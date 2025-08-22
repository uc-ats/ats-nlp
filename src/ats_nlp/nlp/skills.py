from typing import Iterable, List, Optional
from rapidfuzz import fuzz

class SkillsEngine:
    def __init__(self, db_path: str):
        self.db = []
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        self.db.append(s.lower())
        except FileNotFoundError:
            self.db = []

    def extract(self, text: str, skills_section: Optional[str] = None) -> List[str]:
        """
        Baseline extractor:
        - lookup known skills in normalized text
        - fuzzy match near-misses
        """
        text_l = (skills_section or text or "").lower()
        found = set()

        for skill in self.db:
            if skill in text_l:
                found.add(skill)
            else:
                if fuzz.partial_ratio(skill, text_l) >= 90:
                    found.add(skill)

        return sorted(found)

    def all(self) -> List[str]:
        return list(self.db)