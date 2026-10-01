"""Keep the held-out set honest: freeze manifests and a ledger of held-out runs.

A held-out set is only a test set until it influences a decision. Every held-out scoring is
appended to a ledger with a fingerprint of the system; scoring the same held-out set against a
*different* system fingerprint means it has been used for a decision, and it is now a dev set.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

TEXT_SUFFIXES = {".py", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".md", ".txt", ".csv", ".html"}


def file_hash(path: Path) -> str:
    data = Path(path).read_bytes()
    if Path(path).suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")  # same fingerprint on Windows and Linux checkouts
    return hashlib.sha256(data).hexdigest()[:16]


def manifest(paths: list[str]) -> dict:
    files: dict[str, str] = {}
    for p in paths:
        p = Path(p)
        targets = sorted(x for x in p.rglob("*") if x.is_file() and "__pycache__" not in x.parts) if p.is_dir() else [p]
        for f in targets:
            files[f.as_posix()] = file_hash(f)
    combined = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]
    return {"fingerprint": combined, "files": files}


def record_heldout(ledger: Path, golden_path: Path, system_fingerprint: str, accuracy: float | None) -> list[str]:
    """Append this held-out run to the ledger; return warnings about earlier use of the same set."""
    golden = file_hash(golden_path)
    previous = []
    if ledger.exists():
        previous = [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()]
    warnings = []
    for e in previous:
        if e["golden"] == golden and e["system"] != system_fingerprint:
            warnings.append(
                f"this held-out set was already used on {e['at']} with another system version ({e['system']}, "
                f"accuracy {e['accuracy']}). If that result influenced any change, treat the set as dev now."
            )
        elif e["golden"] == golden:
            warnings.append(f"re-run of the same system version as {e['at']}; report one result, not the best one")
    entry = {"at": datetime.now().isoformat(timespec="seconds"), "golden": golden,
             "system": system_fingerprint, "accuracy": accuracy}
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return warnings
