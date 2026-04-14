from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import zipfile


INCLUDE_FILES = [
    "README.md",
    "requirements.txt",
    "config.yaml",
    "main.py",
    "fetcher.py",
    "processor.py",
    "exporter.py",
]


def build_release(version: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    dist = root / "dist"
    dist.mkdir(parents=True, exist_ok=True)

    out = dist / f"Liquid_paper_fetch_{version}.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in INCLUDE_FILES:
            src = root / rel
            if src.exists():
                zf.write(src, arcname=rel)

        reports_dir = root / "reports"
        if reports_dir.exists():
            for path in reports_dir.rglob("*"):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(root)))

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build downloadable release zip for local use")
    parser.add_argument("--version", default="", help="release version, e.g. v0.1.0")
    args = parser.parse_args()

    version = args.version.strip()
    if not version:
        version = datetime.now(timezone.utc).strftime("%Y%m%d")

    artifact = build_release(version)
    print(f"Release package generated: {artifact}")


if __name__ == "__main__":
    main()