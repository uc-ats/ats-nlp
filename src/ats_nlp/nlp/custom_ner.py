from pathlib import Path
import json
import spacy
from spacy.tokens import DocBin

LABELS = ["CERTIFICATION", "TITLE", "SKILL_PHRASE"]

def _load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            yield rec["text"], rec["entities"]

def _to_docbin(nlp, data_iter):
    db = DocBin()
    for text, ents in data_iter:
        doc = nlp.make_doc(text)
        spans = []
        for start, end, label in ents:
            if 0 <= start < end <= len(text) and label in LABELS:
                span = doc.char_span(start, end, label=label)
                if span is not None:
                    spans.append(span)
        doc.ents = spans
        db.add(doc)
    return db

def train_custom_ner(model_out="data/custom_ner", data_file="data/custom_ner.jsonl", base_model="en_core_web_sm"):
    nlp = spacy.load(base_model)
    ner = nlp.add_pipe("ner", last=True)

    for label in LABELS:
        ner.add_label(label)

    # Prepare data
    train_db = _to_docbin(nlp, _load_jsonl(data_file))
    Path(model_out).mkdir(parents=True, exist_ok=True)

    optimizer = nlp.initialize()

    # small iterations for bootstrap; increase with more data
    for _ in range(10):
        for doc in train_db.get_docs(nlp.vocab):
            nlp.update([doc], sgd=optimizer)

    nlp.to_disk(model_out)
    print(f"✅ Custom NER saved to {model_out}")

def load_custom_ner(model_dir="data/custom_ner"):
    p = Path(model_dir)
    if p.exists() and any(p.iterdir()):
        return spacy.load(model_dir)
    return None