# preprocess.py
import re
from langdetect import detect, LangDetectException

CTRL = r"[\u0000-\u0008\u000B\u000C\u000E-\u001F]"
BULLETS = re.compile(r"[•·●■▪▶►●⦿◆➤➣➢]")

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def clean_text(text: str, keep_case: bool = False) -> str:
    if not text:
        return ""
    # normalize bullets and control chars
    text = re.sub(CTRL, " ", text)
    text = BULLETS.sub(" - ", text).replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +\n", "\n", text)
    text = text.strip()
    if not keep_case:
        text = text.lower()
    return text