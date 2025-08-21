# sections.py
import re
from typing import Dict, Optional

HEADINGS = {
    "summary": r"(summary|profile|objective)",
    "experience": r"(experience|employment|work history|professional experience|career history)",
    "education": r"(education|academics|qualifications|academic background)",
    "skills": r"(skills|technical skills|competencies|core skills|tech stack)",
    "certifications": r"(certifications?|licenses?)",
    "projects": r"(projects?|publications?)"
}

def split_sections(text: str) -> Dict[str, Optional[str]]:
    t = "\n" + text + "\n"
    indices = []
    for label, pat in HEADINGS.items():
        for m in re.finditer(rf"\n\s*{pat}\s*[:\-]?\s*\n", t, flags=re.IGNORECASE):
            indices.append((m.start(), m.end(), label))
    if not indices:
        return {k: None for k in HEADINGS.keys()}

    indices.sort()
    out: Dict[str, Optional[str]] = {k: None for k in HEADINGS.keys()}
    for i, (s, e, label) in enumerate(indices):
        start = e
        end = indices[i + 1][0] if i + 1 < len(indices) else len(t)
        chunk = t[start:end].strip()
        if chunk:
            out[label] = chunk
    return out