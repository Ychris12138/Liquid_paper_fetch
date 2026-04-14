from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml

from exporter import export_markdown, render_markdown
from fetcher import fetch_all
from processor import AIProcessor


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_config(path: str = "config.yaml") -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    setup_logging()
    logger = logging.getLogger("main")

    config = load_config("config.yaml")
    papers = fetch_all(config)
    logger.info("Fetched final paper count: %s", len(papers))

    processor = AIProcessor(config)
    processed = [processor.process(p) for p in papers]

    out_cfg = config.get("output", {})
    lookback_days = int(config.get("general", {}).get("lookback_days", 14))
    prefix = out_cfg.get("report_prefix", "water_crystallization_weekly")
    report_title = "水科学与结晶方向文献追踪周报"
    content = render_markdown(processed, report_title=report_title, lookback_days=lookback_days)

    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    filename = f"{prefix}_{date_tag}.md"
    out_path = export_markdown(content, report_dir=out_cfg.get("report_dir", "reports"), filename=filename)
    logger.info("Report generated at: %s", out_path)


if __name__ == "__main__":
    main()