import re
from typing import Dict, Optional
from ats_nlp.models import Sections

# Recognized headers
HEADERS = [
    "education", "experience", "work experience", "professional experience",
    "skills", "summary", "objective", "projects", "certifications"
]

# Match header at start of line (case-insensitive), allow trailing text
HEADER_RE = re.compile(
    rf"^\s*(?:{'|'.join(map(re.escape, HEADERS))})(?:\s*[:\-]?\s*.*)?$",
    re.IGNORECASE | re.MULTILINE
)

def split_sections(text: str) -> Sections:
    """
    Extracts major resume sections based on headers.
    Falls back to None if a section is missing.
    """
    spans = []
    for m in HEADER_RE.finditer(text):
        spans.append((m.group(0).strip().lower(), m.start()))

    spans.sort(key=lambda x: x[1])
    content: Dict[str, str] = {}

    for i, (header, start) in enumerate(spans):
        end = spans[i + 1][1] if i + 1 < len(spans) else len(text)
        body = text[start:end].split("\n", 1)
        body = body[1] if len(body) > 1 else ""
        key = header.replace("work experience", "experience").replace("professional experience", "experience")
        content[key] = body.strip()

    return Sections(
        education=content.get("education"),
        experience=content.get("experience"),
        skills=content.get("skills"),
        summary=content.get("summary") or content.get("objective"),
        certifications=content.get("certifications"),
        projects=content.get("projects"),
    )