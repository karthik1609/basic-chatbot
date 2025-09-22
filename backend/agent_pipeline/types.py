from __future__ import annotations
from typing import TypedDict, Literal, List, Dict, Any, Optional

Decision = Literal["ASK", "ASSUME", "PROCEED"]

class Citation(TypedDict, total=False):
    type: Literal["doc", "sql"]
    tag: str
    source: str
    chunk_index: int
    score: float
    text: str
    query: str
    rows: int

class TraceEntry(TypedDict, total=False):
    stage: str
    note: str
    data: Dict[str, Any]

class Slots(TypedDict, total=False):
    # generic slot bag
    pass


class DocEvidence(TypedDict, total=False):
    tag: str
    snippet: str
    point: str


class ComposerResult(TypedDict, total=False):
    answer: str
    reasoning_bullets: List[str]


class NitpickerResult(TypedDict, total=False):
    compliance_score: float
    findings: List[Dict[str, Any]]
    patches: List[Dict[str, Any]]
    revised_answer: Optional[str]
