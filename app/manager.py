from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from typing import Any

from app.checker import ThirdPartyChecker
from app.classifier import Classifier
from app.config import Settings
from app.mcp import MCPTool
from app.models import (
    CheckerResult,
    ClassificationResult,
    GuidanceCitation,
)

logger = logging.getLogger(__name__)

MAX_EVIDENCE_ITEMS = 20
MAX_EVIDENCE_CHARS = 500


class InvalidToolResult(Exception):
    pass


def _sanitize_result(result: Any) -> tuple[dict[str, str], bool]:
    """Keep only bounded str-to-str entries; tool output is untrusted prompt input."""
    if not isinstance(result, dict):
        raise InvalidToolResult
    cleaned: dict[str, str] = {}
    truncated = False
    for key, value in result.items():
        if not isinstance(key, str) or not isinstance(value, str):
            truncated = True
            continue
        if len(cleaned) == MAX_EVIDENCE_ITEMS:
            truncated = True
            break
        if len(key) > MAX_EVIDENCE_CHARS or len(value) > MAX_EVIDENCE_CHARS:
            truncated = True
        cleaned[key[:MAX_EVIDENCE_CHARS]] = value[:MAX_EVIDENCE_CHARS]
    return cleaned, truncated


class ClassifierManager:
    def __init__(
        self,
        classifier: Classifier,
        checker: ThirdPartyChecker,
        mcp_tools: list[MCPTool] | None = None,
        tool_timeout_seconds: float = 5,
    ) -> None:
        if tool_timeout_seconds <= 0:
            raise ValueError("tool_timeout_seconds must be greater than zero")
        self.classifier = classifier
        self.checker = checker
        self.mcp_tools = mcp_tools or []
        self.tool_timeout_seconds = tool_timeout_seconds

    def _gather_mcp_evidence(self, text: str) -> list[dict[str, Any]]:
        evidence = []
        for tool in self.mcp_tools:
            executor = ThreadPoolExecutor(max_workers=1)
            started = time.perf_counter()
            entry: dict[str, Any] = {"tool": tool.name}
            future: Future[dict[str, str]] = executor.submit(tool.call, text)
            try:
                result, truncated = _sanitize_result(
                    future.result(timeout=self.tool_timeout_seconds)
                )
                entry["status"] = "ok" if result else "no_match"
                entry["result"] = result
                if truncated:
                    entry["truncated"] = True
                logger.info("MCP tool completed", extra={"tool_name": tool.name})
            except TimeoutError:
                future.cancel()
                entry.update(status="error", error="timeout")
                logger.warning("MCP tool timed out", extra={"tool_name": tool.name})
            except Exception as error:
                entry.update(status="error", error=type(error).__name__)
                logger.warning(
                    "MCP tool failed",
                    extra={
                        "tool_name": tool.name,
                        "error_type": type(error).__name__,
                    },
                )
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
            entry["latency_ms"] = round((time.perf_counter() - started) * 1000)
            evidence.append(entry)
        return evidence

    def classify(self, input_text: str) -> ClassificationResult:
        mcp_evidence = self._gather_mcp_evidence(input_text)
        primary, citations = self.classifier.classify(input_text, mcp_evidence)
        checker_result = self.checker.check(
            input_text,
            primary.classification,
            primary.rationale,
            citations,
            mcp_evidence,
        )
        final_status = self._reconcile(primary.classification, checker_result)
        return ClassificationResult(
            input_text=input_text,
            classification=primary.classification,
            confidence=primary.confidence,
            rationale=primary.rationale,
            citations=citations,
            mcp_evidence=mcp_evidence,
            checker_result=checker_result,
            final_status=final_status,
        )

    @staticmethod
    def _reconcile(
        classification: str, checker_result: CheckerResult
    ) -> str:
        if classification.upper() == "UNDETERMINED":
            return "needs_review"
        if checker_result.status == "rejected":
            return "rejected"
        if checker_result.status == "needs_review":
            return "needs_review"
        if (
            checker_result.proposed_classification
            and checker_result.proposed_classification.casefold()
            != classification.casefold()
        ):
            return "needs_review"
        return "approved"
