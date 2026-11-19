from pydantic import BaseModel, Field


class GuidanceCitation(BaseModel):
    source_file: str
    section_id: str | None = None
    chunk_id: str
    excerpt: str
    relevance_score: float | None = Field(default=None, ge=0, le=1)


class LabeledSentence(BaseModel):
    text: str
    label: str
    confidence: float = Field(ge=0, le=1)
    rationale: str
    citations: list[GuidanceCitation] = Field(default_factory=list)
