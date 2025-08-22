from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class Metadata(BaseModel):
    format: Optional[str] = None
    sizeKb: Optional[float] = None

class ResumePayload(BaseModel):
    fileName: Optional[str] = None
    text: str
    metadata: Optional[Metadata] = None

class Sections(BaseModel):
    education: Optional[str] = None
    experience: Optional[str] = None
    skills: Optional[str] = None
    summary: Optional[str] = None
    certifications: Optional[str] = None
    projects: Optional[str] = None

class Entities(BaseModel):
    names: List[str] = []
    emails: List[str] = []
    phones: List[str] = []
    organizations: List[str] = []
    dates: List[str] = []
    locations: List[str] = []
    # Extended via custom NER (optional)
    certifications: List[str] = []
    titles: List[str] = []
    skill_phrases: List[str] = []

class ExtractResponse(BaseModel):
    sections: Sections | Dict[str, Optional[str]]
    entities: Entities
    normalized_skills: List[str] = []
    language: Optional[str] = "en"

class ScoreRequest(BaseModel):
    job_description: str
    resume_text: Optional[str] = None
    resume_skills: Optional[List[str]] = None
    required_skills: Optional[List[str]] = None

class ScoreResponse(BaseModel):
    score: float
    matched_skills: List[str]
    missing_keywords: List[str]
    suggested_terms: List[str] = []
    details: Dict[str, Any]