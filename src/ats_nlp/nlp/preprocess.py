import re
import unicodedata
from langdetect import detect, LangDetectException
import spacy

# Load spaCy pipeline without NER for faster token ops
NLP = spacy.load("en_core_web_sm", disable=["ner"])

CTRL = r"[\u0000-\u0008\u000B\u000C\u000E-\u001F]"
BULLETS = re.compile(r"[•·●■▪▶►●⦿◆➤➣➢]")
ARROWS = re.compile(r"[➔→⇒➤➣➢]")

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def clean_text(
    text: str,
    keep_case: bool = False,
    remove_stopwords: bool = False,
    lemmatize: bool = False
) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(CTRL, " ", text)
    text = BULLETS.sub(" - ", text)
    text = ARROWS.sub(" - ", text)
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +\n", "\n", text)
    text = text.strip()
    if not keep_case:
        text = text.lower()

    if remove_stopwords or lemmatize:
        doc = NLP(text)
        tokens = []
        for tok in doc:
            if remove_stopwords and tok.is_stop:
                continue
            tokens.append(tok.lemma_ if lemmatize else tok.text)
        text = " ".join(tokens)

    return text