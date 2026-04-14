from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict

import requests

from fetcher import PaperRecord


LOGGER = logging.getLogger(__name__)


@dataclass
class ProcessedPaper:
    record: PaperRecord
    abstract_zh: str
    summary_zh: str


class AIProcessor:
    def __init__(self, config: Dict[str, Any]):
        llm = config.get("llm", {})
        self.enabled = bool(llm.get("enabled", True))
        self.endpoint = llm.get("endpoint", "").strip()
        self.api_key = llm.get("api_key", "").strip()
        self.model = llm.get("model", "google/gemma-2-9b-it:free").strip()
        self.timeout_sec = int(llm.get("timeout_sec", 60))
        self.session = requests.Session()

    def process(self, record: PaperRecord) -> ProcessedPaper:
        if not record.abstract_en:
            return ProcessedPaper(record=record, abstract_zh="", summary_zh=self._local_summary(record))

        if not self.enabled or not self.endpoint:
            return ProcessedPaper(
                record=record,
                abstract_zh="[未启用LLM] " + record.abstract_en[:500],
                summary_zh=self._local_summary(record),
            )

        prompt = (
            "你是一名学术助手。请完成两件事：\n"
            "1) 将给定英文摘要准确翻译成简体中文。\n"
            "2) 用1-2句中文总结论文核心贡献。\n"
            "返回严格JSON格式，字段为 abstract_zh 和 summary_zh。"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a precise scientific translator and summarizer."},
                {
                    "role": "user",
                    "content": f"{prompt}\n\nTitle: {record.title}\n\nAbstract:\n{record.abstract_en}",
                },
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            resp = self.session.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout_sec)
            if resp.status_code == 429:
                LOGGER.warning("LLM rate limited, returning fallback result")
                return ProcessedPaper(
                    record=record,
                    abstract_zh="[速率受限] " + record.abstract_en[:500],
                    summary_zh=self._local_summary(record),
                )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            obj = json.loads(content)
            return ProcessedPaper(
                record=record,
                abstract_zh=(obj.get("abstract_zh") or "").strip(),
                summary_zh=(obj.get("summary_zh") or "").strip(),
            )
        except (requests.RequestException, KeyError, json.JSONDecodeError, ValueError) as exc:
            LOGGER.exception("LLM processing failed: %s", exc)
            return ProcessedPaper(
                record=record,
                abstract_zh="[处理失败] " + record.abstract_en[:500],
                summary_zh=self._local_summary(record),
            )

    def _local_summary(self, record: PaperRecord) -> str:
        text = re.sub(r"\s+", " ", (record.abstract_en or "").strip())
        if not text:
            return "该工作与水/成核/结晶相关主题有关，但当前缺少摘要，建议查看原文核实核心贡献。"

        chunks = re.split(r"(?<=[.!?])\s+", text)
        first = chunks[0].strip() if chunks else text[:220]
        second = chunks[1].strip() if len(chunks) > 1 else ""

        # Keep fallback concise and readable in Chinese style.
        line1 = f"该研究围绕《{record.title or '未命名论文'}》展开，重点讨论水体系中的成核/相变机制。"
        key_obs = first[:180] if first else text[:180]
        if second:
            key_obs = f"{key_obs} {second[:120]}"
        line2 = f"摘要显示其主要发现包括：{key_obs}"
        return f"{line1}\n{line2}"