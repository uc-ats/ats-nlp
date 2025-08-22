import re
import json
from pathlib import Path

CERT_KEYWORDS = [
    "certified", "certification", "certificate", "pmp", "aws certified",
    "azure fundamentals", "gcp professional", "scrum master", "oracle certified"
]

TITLE_KEYWORDS = [
    "developer", "engineer", "manager", "architect", "consultant",
    "analyst", "administrator", "intern"
]

SKILL_KEYWORDS = [
    "java", "python", "spring boot", "react", "node.js", "docker",
    "kubernetes", "aws", "azure", "gcp", "microservices", "sql", "git",
    "terraform", "ci/cd", "devops"
]

def bootstrap_from_resume(text: str):
    entities = []
    for kw in CERT_KEYWORDS:
        for m in re.finditer(rf"\b{re.escape(kw)}\b", text, flags=re.IGNORECASE):
            entities.append([m.start(), m.end(), "CERTIFICATION"])
    for kw in TITLE_KEYWORDS:
        for m in re.finditer(rf"\b\w*{kw}\w*\b", text, flags=re.IGNORECASE):
            entities.append([m.start(), m.end(), "TITLE"])
    for kw in SKILL_KEYWORDS:
        for m in re.finditer(rf"\b{re.escape(kw)}\b", text, flags=re.IGNORECASE):
            entities.append([m.start(), m.end(), "SKILL_PHRASE"])
    return {"text": text, "entities": entities}

def bootstrap_directory(input_dir: str, output_file: str = "data/custom_ner.jsonl"):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as out:
        for path in Path(input_dir).glob("*.txt"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            out.write(json.dumps(bootstrap_from_resume(text)) + "\n")
    print(f"✅ Bootstrapped data saved to {output_file}")