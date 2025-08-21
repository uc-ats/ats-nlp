# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models import ResumePayload, ExtractResponse, ScoreRequest, ScoreResponse
from app.nlp.preprocess import clean_text, detect_language
from app.nlp.sections import split_sections
from app.nlp.entities import extract_contacts_and_entities
from app.nlp.skills import SkillsEngine
from app.nlp.score import compute_ats_score, semantic_match_score

app = FastAPI(title="ATS NLP Service", version="2.0")

# CORS for local React (adjust origins in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Initialize heavy components once
SKILLS = SkillsEngine(db_path="data/skills_db.txt")  # uses your file  [oai_citation:3‡skills_db.txt](file-service://file-NazGxSKrN97QdTovFWQHTS)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/nlp/extract", response_model=ExtractResponse)
def extract(payload: ResumePayload):
    if not payload.text:
        raise HTTPException(400, "text is required")

    lang = detect_language(payload.text)
    cleaned = clean_text(payload.text, keep_case=True)  # preserve case for NER

    sections = split_sections(cleaned)
    entities = extract_contacts_and_entities(cleaned)
    found_skills = SKILLS.extract(cleaned, sections.get("skills"))

    return ExtractResponse(
        sections=sections,
        entities=entities,
        normalized_skills=found_skills,
        language=lang
    )

@app.post("/nlp/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    if not req.job_description:
        raise HTTPException(400, "job_description required")

    resume_text = req.resume_text or ""
    resume_skills = req.resume_skills or SKILLS.extract(resume_text)

    # weights follow best-practice heuristic; tweak in score.py
    semantic = semantic_match_score(resume_text, req.job_description)
    total, breakdown, missing = compute_ats_score(
        resume_skills=resume_skills,
        jd_text=req.job_description,
        required_skills=req.required_skills,
        semantic=semantic
    )
    return ScoreResponse(score=total, matched_skills=breakdown["matched_skills"],
                         missing_keywords=missing, details=breakdown)

@app.post("/nlp/analyze")
def analyze(payload: ResumePayload, req: ScoreRequest | None = None):
    """
    Convenience endpoint: extract + score in one hit.
    """
    extracted = extract(payload)
    if not req:
        return {"extracted": extracted, "score": None}
    out = score(ScoreRequest(
        job_description=req.job_description,
        resume_text=payload.text,
        resume_skills=extracted.normalized_skills,
        required_skills=req.required_skills
    ))
    return {"extracted": extracted, "score": out}