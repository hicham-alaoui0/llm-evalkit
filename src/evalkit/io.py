import json
from pathlib import Path


def load_records(path: str | Path) -> list[dict]:
    """JSONL (one object per line), a JSON list, or a JSON object with a "rows" list."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)  # a single JSON document
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]  # JSONL
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("rows"), list):
        return obj["rows"]
    return [obj]  # a JSONL file with a single line


def load_pages(path: str | Path) -> dict[tuple[str, int], str]:
    """Page texts as {(doc_id, page): text}, from a JSONL of {"doc_id", "page", "text"} or a directory
    of "<doc_id>.pages.json" files, each a list of {"page", "text"}."""
    path = Path(path)
    pages: dict[tuple[str, int], str] = {}
    if path.is_dir():
        for f in sorted(path.glob("*.pages.json")):
            doc = f.name.removesuffix(".pages.json")
            for p in json.loads(f.read_text(encoding="utf-8")):
                pages[(doc, int(p["page"]))] = p["text"]
    else:
        for r in load_records(path):
            pages[(r["doc_id"], int(r["page"]))] = r["text"]
    return pages


def write_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
