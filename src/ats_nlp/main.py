import logging
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from ats_nlp.models import ResumePayload, ExtractResponse, ScoreRequest, ScoreResponse
from ats_nlp.nlp.preprocess import clean_text, detect_language
from ats_nlp.nlp.sections import split_sections
from ats_nlp.nlp.entities import extract_contacts_and_entities, load_custom_if_available
from ats_nlp.nlp.skills import SkillsEngine
from ats_nlp.nlp.score import compute_ats_score, semantic_match_score
from ats_nlp.nlp.bootstrap_ner import bootstrap_directory
from ats_nlp.nlp.custom_ner import train_custom_ner, load_custom_ner

# ---------- Enhanced Logging ----------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("ats-nlp")

# Debug middleware to log all requests
class DebugMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger.info(f"🔍 Request: {request.method} {request.url}")
        logger.info(f"🔍 Headers: {dict(request.headers)}")
        logger.info(f"🔍 Client: {request.client}")
        
        response = await call_next(request)
        
        logger.info(f"🔍 Response: {response.status_code}")
        return response

app = FastAPI(
    title="ATS NLP Service", 
    version="2.3", 
    docs_url="/docs", 
    redoc_url="/redoc",
    debug=True  # Enable debug mode
)

# Add debug middleware first
app.add_middleware(DebugMiddleware)

# CORS with more permissive settings for debugging
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# ---------- Startup Event ----------
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting ATS NLP Service...")
    logger.info(f"📁 Working directory: {os.getcwd()}")
    logger.info(f"🐍 Python path: {os.environ.get('PYTHONPATH', 'Not set')}")
    logger.info(f"📊 Available endpoints will be at:")
    logger.info(f"   - Health: /health")
    logger.info(f"   - Docs: /docs")
    logger.info(f"   - OpenAPI: /openapi.json")

# ---------- Globals ----------
try:
    SKILLS = SkillsEngine(db_path="data/skills_db.txt")
    logger.info("✅ Skills engine loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load skills engine: {e}")
    SKILLS = None

try:
    CUSTOM_NLP = load_custom_if_available("data/custom_ner")
    if CUSTOM_NLP:
        logger.info("✅ Custom NLP model loaded successfully")
    else:
        logger.info("ℹ️ No custom NLP model found (this is OK)")
except Exception as e:
    logger.error(f"❌ Error loading custom NLP: {e}")
    CUSTOM_NLP = None

# Add a root endpoint with debug info
@app.get("/")
def root():
    return {
        "message": "ATS NLP Service is running",
        "version": "2.3",
        "endpoints": {
            "health": "/health",
            "docs": "/docs", 
            "openapi": "/openapi.json"
        },
        "debug_info": {
            "working_dir": os.getcwd(),
            "python_path": os.environ.get('PYTHONPATH'),
            "skills_loaded": SKILLS is not None,
            "custom_nlp_loaded": CUSTOM_NLP is not None
        }
    }

@app.get("/health")
def health():
    logger.info("💓 Health check called")
    return {
        "status": "ok", 
        "service": "ats-nlp", 
        "version": "2.3",
        "components": {
            "skills_engine": "ok" if SKILLS else "error",
            "custom_nlp": "ok" if CUSTOM_NLP else "not_loaded"
        }
    }

# Explicit OpenAPI endpoint
@app.get("/openapi.json")
def get_openapi():
    logger.info("📖 OpenAPI JSON requested")
    return app.openapi()

# Add explicit docs endpoint (should be handled by FastAPI automatically, but let's be sure)
@app.get("/docs")
def docs_redirect():
    logger.info("📚 Docs endpoint called - this should be handled by FastAPI")
    return JSONResponse(
        content={"message": "This should redirect to Swagger UI"},
        status_code=200
    )

@app.post("/nlp/extract", response_model=ExtractResponse)
def extract(payload: ResumePayload):
    if not payload.text:
        raise HTTPException(400, "text is required")

    lang = detect_language(payload.text)
    cleaned = clean_text(payload.text, keep_case=True)  # preserve case for NER

    sections = split_sections(cleaned)  # returns Sections Pydantic model
    entities = extract_contacts_and_entities(cleaned)

    # ✅ FIX: access attribute instead of dict.get()
    found_skills = SKILLS.extract(cleaned, sections.skills)

    if not sections.skills and found_skills:
        sections.skills = ", ".join(found_skills)

    return ExtractResponse(
        sections=sections,
        entities=entities,
        normalized_skills=found_skills,
        language=lang
    )

@app.post("/nlp/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    logger.info("📊 Score endpoint called")
    if not req.job_description:
        raise HTTPException(400, "job_description required")

    if not SKILLS:
        raise HTTPException(500, "Skills engine not available")

    resume_text = clean_text(req.resume_text or "", remove_stopwords=True, lemmatize=True)
    jd_text = clean_text(req.job_description, remove_stopwords=True, lemmatize=True)

    resume_skills = req.resume_skills or SKILLS.extract(resume_text)

    semantic = semantic_match_score(resume_text, jd_text)
    total, breakdown, missing, suggestions = compute_ats_score(
        resume_skills=resume_skills,
        jd_text=jd_text,
        required_skills=req.required_skills,
        semantic=semantic
    )

    logger.info("ATS SCORE → Score=%s | Matched=%s | Missing=%s | Suggestions=%s",
                total, breakdown["matched_skills"], missing, suggestions)

    return ScoreResponse(
        score=total,
        matched_skills=breakdown["matched_skills"],
        missing_keywords=missing,
        suggested_terms=suggestions,
        details=breakdown
    )

@app.post("/nlp/analyze")
def analyze(payload: ResumePayload, req: ScoreRequest | None = None):
    logger.info("🔬 Analyze endpoint called")
    extracted = extract(payload)
    if not req:
        return {"extracted": extracted, "score": None}

    # reuse the already-cleaned skills
    resume_text_clean = clean_text(payload.text or "", remove_stopwords=True, lemmatize=True)
    jd_text_clean = clean_text(req.job_description, remove_stopwords=True, lemmatize=True)

    total, breakdown, missing, suggestions = compute_ats_score(
        resume_skills=extracted.normalized_skills,
        jd_text=jd_text_clean,
        required_skills=req.required_skills,
        semantic=semantic_match_score(resume_text_clean, jd_text_clean)
    )

    logger.info("ATS ANALYZE → File=%s | Score=%s | Missing=%s | Suggestions=%s",
                payload.fileName, total, missing, suggestions)

    score_response = ScoreResponse(
        score=total,
        matched_skills=breakdown["matched_skills"],
        missing_keywords=missing,
        suggested_terms=suggestions,
        details=breakdown
    )
    return {"extracted": extracted, "score": score_response}

@app.post("/nlp/retrain")
def retrain():
    """
    Self-learning pipeline:
    1) Bootstrap weak labels from data/raw_resumes/*.txt -> data/custom_ner.jsonl
    2) Train spaCy model -> data/custom_ner/
    3) Hot-reload into memory
    """
    logger.info("🔄 Retrain endpoint called")
    try:
        bootstrap_directory("data/raw_resumes", "data/custom_ner.jsonl")
        train_custom_ner(model_out="data/custom_ner", data_file="data/custom_ner.jsonl")
        global CUSTOM_NLP
        CUSTOM_NLP = load_custom_ner("data/custom_ner")
        return {"status": "success", "message": "Custom NER retrained and reloaded"}
    except Exception as e:
        logger.exception("Retraining failed")
        raise HTTPException(500, f"Retraining failed: {e}")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"❌ Unhandled exception on {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )