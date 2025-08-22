import re
from typing import List, Optional
import phonenumbers
import spacy
from ats_nlp.models import Entities

# Load base English model for PERSON/ORG/DATE/LOC
NLP = spacy.load("en_core_web_sm")

EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)


def _dedup(seq: List[str]) -> List[str]:
    return list(dict.fromkeys([s.strip() for s in seq if s and s.strip()]))


def _clean_names(names: List[str]) -> List[str]:
    # Filter out garbage: too long, starts with "skills", etc.
    return [
        n for n in names
        if 2 <= len(n.split()) <= 4 and not n.lower().startswith("skills")
    ]


def _clean_orgs(orgs: List[str]) -> List[str]:
    blacklist = {"spring framework", "kubernetes", "junit", "docker", "git"}
    return [o for o in orgs if o.lower() not in blacklist and len(o) > 2]


def load_custom_if_available(path: str):
    """Helper used by main.py to lazily load a custom spaCy NER model if present."""
    try:
        from pathlib import Path
        p = Path(path)
        if p.exists() and any(p.iterdir()):
            return spacy.load(path)
    except Exception:
        pass
    return None


def extract_contacts_and_entities(text: str, custom_model: Optional[spacy.Language] = None) -> Entities:
    emails = _dedup(EMAIL_RE.findall(text))
    # Normalize weird spaces and dashes before phone parsing
    text = (
        text.replace("\u00A0", " ")   # non-breaking space
            .replace("\u2009", " ")   # thin space
            .replace("\u2002", " ")   # en space
            .replace("–", "-")        # en-dash
            .replace("—", "-")        # em-dash
            .replace("（", "(")       # full-width left paren
            .replace("）", ")")       # full-width right paren
    )
    phones = []
    default_region = "US"
    for match in phonenumbers.PhoneNumberMatcher(text, default_region):
        try:
            e164 = phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164)
            phones.append(e164)
        except Exception:
            continue

# If still empty, try again without default region (handles +91, +44, etc.)
    if not phones:
        for match in phonenumbers.PhoneNumberMatcher(text, None):
            try:
                e164 = phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164)
                phones.append(e164)
            except Exception:
                continue
    phones = _dedup(phones)

    # Base NER
    doc = NLP(text)
    names, orgs, dates, locs = [], [], [], []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            names.append(ent.text)
        elif ent.label_ == "ORG":
            orgs.append(ent.text)
        elif ent.label_ == "DATE":
            dates.append(ent.text)
        elif ent.label_ in ("GPE", "LOC"):
            locs.append(ent.text)

    # Custom model (if available)
    certifications, titles, skill_phrases = [], [], []
    if custom_model:
        cdoc = custom_model(text)
        for ent in cdoc.ents:
            if ent.label_ == "CERTIFICATION":
                certifications.append(ent.text)
            elif ent.label_ == "TITLE":
                titles.append(ent.text)
            elif ent.label_ == "SKILL_PHRASE":
                skill_phrases.append(ent.text)

    return Entities(
        names=_dedup(_clean_names(names)),
        emails=emails,
        phones=phones,
        organizations=_dedup(_clean_orgs(orgs)),
        dates=_dedup(dates),
        locations=_dedup(locs),
        certifications=_dedup(certifications),
        titles=_dedup(titles),
        skill_phrases=_dedup(skill_phrases),
    )