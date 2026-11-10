from typing import Any, Literal

from pydantic import BaseModel, Field


class GuidanceCitation(BaseModel):
    source_file: str
    section_id: str | None = None
    chunk_id: str
    excerpt: str
    relevance_score: float | None = Field(default=None, ge=0, le=1)


class ClassifierResponse(BaseModel):
    classification: str
    confidence: float = Field(ge=0, le=1)
    rationale: str
    citation_chunk_ids: list[str] = Field(default_factory=list)
    missing_information: str | None = None


class CheckerResponse(BaseModel):
    status: Literal["approved", "rejected", "needs_review"]
    proposed_classification: str | None = None
    rationale: str
    citation_chunk_ids: list[str] = Field(default_factory=list)


class CheckerResult(BaseModel):
    status: Literal["approved", "rejected", "needs_review"]
    proposed_classification: str | None = None
    rationale: str
    citations: list[GuidanceCitation] = Field(default_factory=list)


class ClassificationResult(BaseModel):
    input_text: str
    classification: str
    confidence: float = Field(ge=0, le=1)
    rationale: str
    citations: list[GuidanceCitation]
    mcp_evidence: list[dict[str, Any]]
    checker_result: CheckerResult
    final_status: Literal["approved", "needs_review", "rejected"]
