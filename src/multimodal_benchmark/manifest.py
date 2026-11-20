"""Zenodo dataset manifest loader, broken-DOI resolver, and MD5 checksum verifier."""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "configs" / "manifest.yaml"

BROKEN_DOI_ALIASES: Dict[str, str] = {
    "10.5281/zenodo.13327988": "10.5281/zenodo.13349402",
    "13327988": "13349402",
}


@dataclass(frozen=True)
class FileEntry:
    record_key: str
    record_id: str
    doi: str
    name: str
    size_mb: float
    md5: Optional[str] = None

    @property
    def download_url(self) -> str:
        return f"https://zenodo.org/records/{self.record_id}/files/{self.name}"


class DatasetManifest:
    """Structured wrapper around configs/manifest.yaml."""

    def __init__(self, raw: Dict[str, Any], source_path: Optional[Path] = None) -> None:
        self.raw = raw
        self.source_path = source_path

    @property
    def records(self) -> Dict[str, Dict[str, Any]]:
        return self.raw.get("records", {})

    def get_record(self, key_or_id: str) -> Dict[str, Any]:
        resolved = resolve_doi(key_or_id)
        if key_or_id in self.records:
            return self.records[key_or_id]
        for rec in self.records.values():
            if rec.get("record_id") in (key_or_id, resolved) or rec.get("doi") in (key_or_id, resolved):
                return rec
        raise KeyError(f"Unknown Zenodo record key/ID: {key_or_id}")

    def iter_files(self, max_size_mb: Optional[float] = None) -> List[FileEntry]:
        entries: List[FileEntry] = []
        for rkey, rec in self.records.items():
            rid = str(rec["record_id"])
            doi = str(rec["doi"])
            file_lists: List[List[Dict[str, Any]]] = []
            if "files" in rec:
                file_lists.append(rec["files"])
            for sub in ("scans_25um", "scans_50um"):
                if sub in rec and "files" in rec[sub]:
                    file_lists.append(rec[sub]["files"])
            for flist in file_lists:
                for fmeta in flist:
                    sz = float(fmeta.get("size_mb", 0.0))
                    if max_size_mb is not None and sz > max_size_mb:
                        continue
                    entries.append(
                        FileEntry(
                            record_key=rkey,
                            record_id=rid,
                            doi=doi,
                            name=str(fmeta["name"]),
                            size_mb=sz,
                            md5=fmeta.get("md5"),
                        )
                    )
        return entries


def resolve_doi(doi_or_record_id: str) -> str:
    """Resolve known errata DOIs (e.g. 10.5281/zenodo.13327988 -> 10.5281/zenodo.13349402)."""
    clean = doi_or_record_id.strip()
    return BROKEN_DOI_ALIASES.get(clean, clean)


def load_manifest(path: Optional[Path | str] = None) -> DatasetManifest:
    target = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
    with open(target, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return DatasetManifest(raw=raw, source_path=target)


def compute_md5(file_path: Path | str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(file_path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_file_md5(file_path: Path | str, expected_md5: str) -> bool:
    actual = compute_md5(file_path)
    return actual.lower() == expected_md5.strip().lower()


def verify_cache_directory(
    cache_dir: Path | str,
    manifest: Optional[DatasetManifest] = None,
    download_missing_under_mb: Optional[float] = None,
) -> Dict[str, Any]:
    """Verify cached files against Zenodo MD5 checksums and optionally fetch lightweight files."""
    manifest = manifest or load_manifest()
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)

    checked: List[Dict[str, Any]] = []
    mismatches: List[str] = []
    missing: List[str] = []

    for entry in manifest.iter_files(max_size_mb=download_missing_under_mb):
        fpath = cdir / entry.name
        if not fpath.exists() and download_missing_under_mb is not None:
            urllib.request.urlretrieve(entry.download_url, fpath)
        if not fpath.exists():
            missing.append(entry.name)
            continue
        if entry.md5:
            actual = compute_md5(fpath)
            ok = actual.lower() == entry.md5.lower()
            if not ok:
                mismatches.append(entry.name)
            checked.append({"name": entry.name, "expected_md5": entry.md5, "actual_md5": actual, "ok": ok})
        else:
            checked.append({"name": entry.name, "expected_md5": None, "actual_md5": compute_md5(fpath), "ok": True})

    return {
        "cache_dir": str(cdir),
        "num_verified": len(checked),
        "num_mismatches": len(mismatches),
        "mismatches": mismatches,
        "num_missing": len(missing),
        "checked": checked,
    }
