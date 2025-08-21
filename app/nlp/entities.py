# entities.py
import re
from typing import List
import phonenumbers
import spacy

NLP = spacy.load("en_core_web_sm")

EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)

def _dedup(seq: List[str]) -> List[str]:
    return list(dict.fromkeys([s.strip() for s in seq if s and s.strip()]))

def extract_contacts_and_entities(text: str):
    emails = _dedup(EMAIL_RE.findall(text))

    phones = []
    for match in phonenumbers.PhoneNumberMatcher(text, None):
        phones.append(phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164))
    phones = _dedup(phones)

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

    return {
        "names": _dedup(names),
        "emails": emails,
        "phones": phones,
        "organizations": _dedup(orgs),
        "dates": _dedup(dates),
        "locations": _dedup(locs),
    }