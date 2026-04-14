from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from processor import ProcessedPaper


def _fmt_date(dt: datetime | None) -> str:
    if not dt:
        return "Unknown"
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


def render_markdown(items: Iterable[ProcessedPaper], report_title: str, lookback_days: int) -> str:
    item_list = list(items)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# {report_title}",
        "",
        f"- 生成时间: {now}",
        f"- 时间窗口: 过去 {lookback_days} 天",
        f"- 文献数量: {len(item_list)}",
        "",
        "## 摘要",
        "",
        "本报告聚焦水科学与结晶方向，汇总多源API在近两周内符合期刊与关键词条件的论文。",
        "",
        "## 论文列表",
        "",
    ]
    for idx, item in enumerate(item_list, 1):
        rec = item.record
        lines.extend(
            [
                f"### {idx}. {rec.title}",
                "",
                f"- 来源: {rec.source}",
                f"- 期刊: {rec.journal or 'Unknown'}",
                f"- 发表日期: {_fmt_date(rec.published_at)}",
                f"- DOI: {rec.doi or 'N/A'}",
                f"- 链接: {rec.url or 'N/A'}",
                f"- 作者: {', '.join([a for a in rec.authors if a]) or 'Unknown'}",
                "",
                "**核心贡献（中文）**",
                "",
                item.summary_zh or "无",
                "",
                "**摘要翻译（中文）**",
                "",
                item.abstract_zh or "无",
                "",
                "---",
                "",
            ]
        )
    return "\n".join(lines)


def export_markdown(content: str, report_dir: str, filename: str) -> Path:
    target_dir = Path(report_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / filename
    out.write_text(content, encoding="utf-8")
    return out