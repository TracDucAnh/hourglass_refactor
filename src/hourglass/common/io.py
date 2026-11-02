"""I/O dùng chung: JSON (ghi nguyên tử) và JSONL {id, lang, text}."""

import json
import os


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def read_jsonl(path):
    by = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by.setdefault(r["lang"], []).append((r["id"], r["text"]))
    return {l: [t for _, t in sorted(v)] for l, v in by.items()}


def write_jsonl(path, clean, ids):
    with open(path, "w", encoding="utf-8") as f:
        for l, sents in clean.items():
            for i, s in zip(ids, sents):
                f.write(json.dumps({"id": int(i), "lang": l, "text": s}, ensure_ascii=False) + "\n")
