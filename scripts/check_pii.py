import argparse
import re
from pathlib import Path

PATTERNS = [
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:\+?\d{1,3}[- ]?)?\d{3}[- ]?\d{3}[- ]?\d{4}\b"),
]


def scan_path(path: Path) -> None:
    text = path.read_text(errors="ignore")
    for pattern in PATTERNS:
        if pattern.search(text):
            raise SystemExit(f"Possible PII detected in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan files for common PII patterns")
    parser.add_argument("root", nargs="?", default="data/seasonality_source", help="Directory to scan")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Path {root} does not exist")

    for file in root.rglob("*"):
        if file.is_file() and not file.name.endswith(".sha256"):
            scan_path(file)
    print("No PII patterns found.")


if __name__ == "__main__":
    main()
