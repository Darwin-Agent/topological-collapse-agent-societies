#!/usr/bin/env python3
"""Offline security, privacy, syntax, and artifact-integrity audit."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRECTORIES = {".git", ".venv", "venv", "__pycache__"}
TEXT_SUFFIXES = {
    "",
    ".cff",
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
BLOCKED_FILENAMES = {
    ".env",
    "id_rsa",
    "id_ed25519",
}
BLOCKED_SUFFIXES = {".db", ".key", ".p12", ".pem", ".sqlite", ".sqlite3"}
PROHIBITED_PROVIDER_MARKERS = (
    "mi" + "fy",
    "x-model-" + "provider-id",
)
PATTERNS = {
    "private key material": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "OpenAI-style live key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "Feishu application identifier": re.compile(r"\bcli_[0-9a-f]{16,}\b"),
    "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    "hard-coded bearer token": re.compile(r"\bBearer\s+[A-Za-z0-9._-]{20,}\b"),
    "private workspace path": re.compile(r"/ls/data/[A-Za-z0-9._-]+/"),
    "private home path": re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+/"),
    "private macOS volume path": re.compile(r"/Volumes/[^/\n]+/"),
    "internal service hostname": re.compile(r"\b[A-Za-z0-9.-]+\.srv\b"),
    "email address": re.compile(
        r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
        r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
        r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+\b"
    ),
    "workspace owner identifier": re.compile(r"(?i)\blushuo\b|卢硕"),
}
EXPECTED_COUNTS = {
    "llm_task_benchmark/results/summary.json": ("n_run_cells", 160),
    "llm_relay_benchmark/results/summary.json": ("n_completed_cells", 160),
    "llm_relay_round1_ablation/results/summary.json": ("n_completed_cells", 160),
    "llm_temporal_pairing_control/results/summary.json": ("n_completed_cells", 64),
}


def iter_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRECTORIES for part in path.relative_to(ROOT).parts):
            continue
        yield path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_text(path: Path) -> str | None:
    if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 5_000_000:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def verify_paths(errors: list[str]) -> None:
    for path in iter_files():
        relative = path.relative_to(ROOT)
        lower_name = path.name.lower()
        if lower_name in BLOCKED_FILENAMES or path.suffix.lower() in BLOCKED_SUFFIXES:
            errors.append(f"blocked file: {relative}")
        lower_path = relative.as_posix().lower()
        for marker in PROHIBITED_PROVIDER_MARKERS:
            if marker in lower_path:
                errors.append(f"prohibited provider marker in path: {relative}")
        if "results/raw" in relative.as_posix():
            errors.append(f"raw model trace included: {relative}")
        if path.is_symlink():
            try:
                path.resolve().relative_to(ROOT)
            except ValueError:
                errors.append(f"symlink escapes repository: {relative}")


def verify_text(errors: list[str]) -> None:
    for path in iter_files():
        if path == Path(__file__).resolve():
            continue
        text = read_text(path)
        if text is None:
            continue
        relative = path.relative_to(ROOT)
        lower_text = text.lower()
        for marker in PROHIBITED_PROVIDER_MARKERS:
            if marker in lower_text:
                line = lower_text.count("\n", 0, lower_text.index(marker)) + 1
                errors.append(f"prohibited provider marker: {relative}:{line}")
        for label, pattern in PATTERNS.items():
            match = pattern.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                errors.append(f"{label}: {relative}:{line}")


def verify_python(errors: list[str]) -> None:
    for path in iter_files():
        if path.suffix != ".py":
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as error:
            errors.append(f"Python syntax error: {path.relative_to(ROOT)}:{error.lineno}")


def verify_manifest(errors: list[str]) -> None:
    manifest_path = ROOT / "figure_source" / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("hash_algorithm") != "sha256":
        errors.append("figure manifest does not declare sha256")
        return
    for entry in manifest.get("inputs", []):
        path = ROOT / entry["path"]
        if not path.is_file():
            errors.append(f"missing figure input: {entry['path']}")
        elif sha256(path) != entry["sha256"]:
            errors.append(f"figure input hash mismatch: {entry['path']}")


def verify_summaries(errors: list[str]) -> None:
    for relative, (key, expected) in EXPECTED_COUNTS.items():
        path = ROOT / relative
        data = json.loads(path.read_text())
        if data.get(key) != expected:
            errors.append(
                f"unexpected frozen count in {relative}: "
                f"{key}={data.get(key)!r}, expected {expected}"
            )


def main() -> None:
    errors: list[str] = []
    verify_paths(errors)
    verify_text(errors)
    verify_python(errors)
    verify_manifest(errors)
    verify_summaries(errors)

    if errors:
        print("Release verification failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)

    file_count = sum(1 for _ in iter_files())
    python_count = sum(1 for path in iter_files() if path.suffix == ".py")
    print("Release verification passed.")
    print(f"- Files audited: {file_count}")
    print(f"- Python files parsed: {python_count}")
    print("- Compact figure-input SHA-256 manifest: valid")
    print("- Frozen benchmark cell counts: valid")
    print("- Credential, private-path, internal-host, and identity scan: clean")


if __name__ == "__main__":
    main()
