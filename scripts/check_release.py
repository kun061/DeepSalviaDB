#!/usr/bin/env python3
"""Fail when a release tree contains obvious secrets or oversized artifacts."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 95 * 1024 * 1024
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
SKIP_TEXT_FILES = {".env.example", "check_release.py"}
BINARY_SUFFIXES = {
    ".h5", ".hdf5", ".pt", ".pth", ".ckpt", ".safetensors", ".onnx",
    ".joblib", ".pkl", ".gguf", ".sqlite", ".sqlite3",
}
SECRET_PATTERNS = {
    "private key": re.compile(r"BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "hard-coded password": re.compile(
        r"""["']password["']\s*:\s*["'](?!\s*(?:|CHANGE_ME|YOUR_PASSWORD))[^"]{4,}["']""",
        re.IGNORECASE,
    ),
    "private server path": re.compile(r"/root/(?:autodl-tmp|conda_envs)/"),
}


def iter_files():
    for path in ROOT.rglob("*"):
        if path.is_file() and not any(part in SKIP_DIRS for part in path.parts):
            yield path


def main() -> int:
    problems: list[str] = []
    for path in iter_files():
        relative = path.relative_to(ROOT)
        size = path.stat().st_size
        if size > MAX_BYTES:
            problems.append(f"oversized ({size / 1024 / 1024:.1f} MiB): {relative}")
        if path.suffix.lower() in BINARY_SUFFIXES:
            problems.append(f"model/database artifact: {relative}")
        if path.name in SKIP_TEXT_FILES or size > 5 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                problems.append(f"{label}: {relative}")

    real_env = ROOT / ".env"
    if real_env.exists():
        problems.append("real environment file present: .env")

    if problems:
        print("Release check FAILED:")
        for problem in sorted(set(problems)):
            print(f"  - {problem}")
        return 1
    print("Release check passed: no obvious secrets or oversized artifacts found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
