from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util
from ats_nlp.nlp.preprocess import clean_text

_SBERT = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def semantic_match_score(resume_text: str, jd_text: str) -> float:
    if not resume_text or not jd_text:
        return 0.0
    resume_clean = clean_text(resume_text, remove_stopwords=True, lemmatize=True)
    jd_clean = clean_text(jd_text, remove_stopwords=True, lemmatize=True)
    e1 = _SBERT.encode(resume_clean, convert_to_tensor=True)
    e2 = _SBERT.encode(jd_clean, convert_to_tensor=True)
    sim = float(util.cos_sim(e1, e2).item())
    return max(0.0, min(1.0, sim))  # clamp 0..1

def suggest_relevant_terms(resume_skills: List[str], jd_text: str, top_n: int = 5) -> List[str]:
    if not jd_text:
        return []
    jd_tokens = list(set(clean_text(jd_text, remove_stopwords=True, lemmatize=True).split()))
    cand = set(s.lower() for s in (resume_skills or []))
    if not jd_tokens:
        return []

    embeddings = _SBERT.encode(jd_tokens, convert_to_tensor=True)
    resume_emb = _SBERT.encode(" ".join(sorted(cand)) or " ", convert_to_tensor=True)

    sims = []
    for tok, emb in zip(jd_tokens, embeddings):
        if tok in cand:
            continue
        sims.append((tok, float(util.cos_sim(resume_emb, emb.unsqueeze(0)).item())))
    ranked = sorted(sims, key=lambda x: x[1], reverse=True)
    return [tok for tok, _ in ranked[:top_n]]

def _section_bonus(resume_text: str) -> float:
    bonus = 0
    t = resume_text.lower()
    for key in ["education", "experience", "skills", "projects", "summary"]:
        if key in t:
            bonus += 2
    return min(bonus, 10) / 10.0

def compute_ats_score(
    resume_skills: List[str],
    jd_text: str,
    required_skills: List[str] | None,
    semantic: float
) -> Tuple[float, Dict, List[str], List[str]]:
    jd_tokens = set(clean_text(jd_text, remove_stopwords=True, lemmatize=True).split())
    cand = set(s.lower() for s in (resume_skills or []))

    # 1) skills matched in JD tokens (heuristic)
    matched = sorted([s for s in cand if any(tok in s or s in tok for tok in jd_tokens)])
    skills_cov = min(len(matched), 20) / 20.0

    # 2) required coverage
    req = [r.lower() for r in (required_skills or [])]
    missing = [r for r in req if all(r not in s for s in cand)]
    required_cov = 1.0 if not req else (len(req) - len(missing)) / max(1, len(req))

    # 3) semantic similarity already 0..1
    sem = semantic

    # 4) section completeness bonus (proxy)
    sections_cov = _section_bonus(" ".join([*resume_skills, jd_text]))

    # Weights (sum 100)
    score = skills_cov * 40 + required_cov * 25 + sem * 20 + sections_cov * 10 + 5
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

    suggestions = suggest_relevant_terms(resume_skills, jd_text, top_n=5)
    return total, breakdown, missing, suggestions