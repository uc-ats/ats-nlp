# score.py
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util

_SBERT = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def semantic_match_score(resume_text: str, jd_text: str) -> float:
    if not resume_text or not jd_text:
        return 0.0
    e1 = _SBERT.encode(resume_text, convert_to_tensor=True)
    e2 = _SBERT.encode(jd_text, convert_to_tensor=True)
    sim = float(util.cos_sim(e1, e2).item())
    return max(0.0, min(1.0, sim))  # 0..1

def _section_bonus(resume_text: str) -> float:
    # very light heuristic; you can wire in actual section presence from extract if available
    bonus = 0
    t = resume_text.lower()
    for key in ["education", "experience", "skills", "projects", "summary"]:
        if key in t:
            bonus += 2
    return min(bonus, 10) / 10.0  # 0..1

def compute_ats_score(
    resume_skills: List[str],
    jd_text: str,
    required_skills: List[str] | None,
    semantic: float
) -> Tuple[float, Dict, List[str]]:
    jd_tokens = set([w.lower() for w in jd_text.split()])
    cand = set(s.lower() for s in (resume_skills or []))

    # 1) skills matched in JD tokens (heuristic)
    matched = sorted([s for s in cand if any(tok in s or s in tok for tok in jd_tokens)])
    skills_cov = min(len(matched), 20) / 20.0  # cap

    # 2) required coverage
    req = [r.lower() for r in (required_skills or [])]
    missing = [r for r in req if all(r not in s for s in cand)]
    required_cov = 1.0 if not req else (len(req) - len(missing)) / max(1, len(req))

    # 3) semantic similarity already 0..1
    sem = semantic

    # 4) section completeness bonus (proxy; ideally pass from extract)
    sections_cov = _section_bonus(" ".join([*resume_skills, jd_text]))

    # Weights (sum 100)
    score = (
        skills_cov * 40 +
        required_cov * 25 +
        sem * 20 +
        sections_cov * 10 +
        5  # formatting/meta placeholder
    )
    total = round(min(100.0, score), 2)

    breakdown = {
        "matched_skills": matched,
        "weights": {"skills": 40, "required": 25, "semantic": 20, "sections": 10, "format": 5},
        "components": {
            "skills_cov": round(skills_cov, 3),
            "required_cov": round(required_cov, 3),
            "semantic": round(sem, 3),
            "sections_cov": round(sections_cov, 3)
        }
    }
    return total, breakdown, missing