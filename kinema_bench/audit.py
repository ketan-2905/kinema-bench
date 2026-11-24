"""Immutable, content-hashed result records.

Every number this project produces is written as a canonical JSON record
whose filename IS the sha256 of its own content, stored read-only, and
carrying the hash of the previous record in the same store. A number edited
after the fact therefore changes its own filename, and editing one in place
is caught by `verify()`.

This is cheap to build and it is the difference between a results table and a
results table somebody else has reason to trust.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
GENESIS = "0" * 64


def _fallback(o: Any) -> Any:
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON-serialisable: {type(o)!r}")


def canonical_json(obj: Any) -> str:
    """One object -> exactly one string. Sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=_fallback)


def content_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def provenance(extra: dict | None = None) -> dict:
    """The context a number needs to still be believable in six months."""
    rec = {
        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        rec["torch"], rec["cuda"] = torch.__version__, torch.version.cuda
    except ImportError:
        rec["torch"] = None
    return {**rec, **(extra or {})}


@dataclass
class AuditStore:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.jsonl"

    def head(self) -> str:
        if not self.manifest.exists():
            return GENESIS
        lines = [l for l in self.manifest.read_text().splitlines() if l.strip()]
        return json.loads(lines[-1])["hash"] if lines else GENESIS

    def write(self, kind: str, payload: dict, *, dry_run: bool = False) -> dict:
        """Seal `payload` into an immutable record."""
        record = {
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "prev_hash": self.head(),
            "provenance": provenance({"dry_run": bool(dry_run)}),
            "payload": payload,
        }
        h = content_hash(record)
        path = self.root / f"{kind}-{h[:12]}.json"
        if path.exists():
            return {"hash": h, "path": str(path), "duplicate": True}

        path.write_text(json.dumps(record, indent=2, sort_keys=True, default=_fallback) + "\n")
        os.chmod(path, 0o444)
        with self.manifest.open("a") as fh:
            fh.write(canonical_json({"hash": h, "kind": kind, "file": path.name,
                                     "prev_hash": record["prev_hash"],
                                     "utc": record["provenance"]["utc"]}) + "\n")
        return {"hash": h, "path": str(path), "duplicate": False}

    def verify(self) -> list[str]:
        """Re-hash every record and re-walk the chain. Returns a list of problems."""
        if not self.manifest.exists():
            return ["no manifest: nothing has been written to this store"]
        problems, prev = [], GENESIS
        for line in self.manifest.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            path = self.root / entry["file"]
            if not path.exists():
                problems.append(f"{entry['file']}: missing on disk")
            else:
                record = json.loads(path.read_text())
                if (actual := content_hash(record)) != entry["hash"]:
                    problems.append(f"{entry['file']}: content changed "
                                    f"({actual[:12]} != {entry['hash'][:12]})")
                if record.get("prev_hash") != prev:
                    problems.append(f"{entry['file']}: chain break")
            prev = entry["hash"]
        return problems
