"""JSONL -> cross-encoder inputs. No tokenization here beyond the template split."""
from __future__ import annotations
import json


def split_row(text: str) -> tuple[str, str]:
    """'resume: ...\\n[SEP]\\njob: ...' -> (resume, job). Handles both \\n and literal [SEP] splits."""
    if "\n[SEP]\n" in text:
        r, j = text.split("\n[SEP]\n", 1)
    elif "[SEP]" in text:
        r, j = text.split("[SEP]", 1)
    else:
        raise ValueError("row has no [SEP]: " + text[:120])
    r = r.strip()
    if r.startswith("resume:"):
        r = r[len("resume:"):].strip()
    j = j.strip()
    if j.startswith("job:"):
        j = j[len("job:"):].strip()
    return r, j


def to_prompt(resume: str, job: str, template: str) -> str:
    return template.format(resume=resume, job=job)


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
