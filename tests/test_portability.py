"""Repository-level guards for paths that would make a release non-portable."""

from __future__ import annotations

import re
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".cfg",
    ".json",
    ".md",
    ".py",
    ".r",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN = (
    re.compile(
        r"(?<![\w.])/(?:home|Users|tmp|mnt|scratch|workspaces?|ibex|"
        r"data\d*|d|projects?|nfs|gpfs|lustre|cluster|srv|opt|var)/"
    ),
    re.compile(r"(?<![\w.])[A-Za-z]:[\\/](?!/)"),
    re.compile(r"file:///"),
)


class PortablePathTests(unittest.TestCase):
    def test_repository_has_no_machine_specific_absolute_paths(self) -> None:
        failures: list[str] = []
        for path in REPOSITORY_ROOT.rglob("*"):
            if not path.is_file() or path.suffix.casefold() not in TEXT_SUFFIXES:
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            if any(part in {".git", "__pycache__"} for part in path.parts):
                continue
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                # Web URLs are not filesystem paths.
                candidate = re.sub(r"https?://\S+", "", line)
                if any(pattern.search(candidate) for pattern in FORBIDDEN):
                    failures.append(f"{path.relative_to(REPOSITORY_ROOT)}:{line_number}")
        self.assertEqual(failures, [], "machine-specific paths: " + ", ".join(failures))


if __name__ == "__main__":
    unittest.main()
