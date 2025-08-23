from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Metadata(BaseModel):
    format: Optional[str] = None
    sizeKb: Optional[float] = None

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
    certifications: List[str] = []
    titles: List[str] = []
    skill_phrases: List[str] = []

class ResumePayload(BaseModel):
    fileName: Optional[str] = None
    text: str
    sections: Optional[Sections] = None
    metadata: Optional[Metadata] = None
    entities: Optional[Entities] = None
    normalized_skills: Optional[List[str]] = None
    language: Optional[str] = "en"

class ScoreRequest(BaseModel):
    resume: ResumePayload
    jobDescription: str

class ScoreResponse(BaseModel):
    score: float
    matched_skills: List[str]
    missing_keywords: List[str]
    suggested_terms: List[str] = []
    details: Dict[str, Any]