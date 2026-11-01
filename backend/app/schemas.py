from typing import Optional

from pydantic import BaseModel


class Preview(BaseModel):
    url: str
    version: int
    origin_mm: list[float]
    size_mm: list[float]
    mm_factor: float
    units_status: str
    n_prims: int
    floor_block: Optional[str]


class TrackStatus(BaseModel):
    state: str
    seconds: Optional[float]
    error: Optional[str]
    summary: Optional[dict]


class RunInfo(BaseModel):
    run_id: str
    source_file: str
    floors: list[dict]
    floor_block: Optional[str]
    preview: Preview
    tracks: dict[str, TrackStatus]
    unit_warning: Optional[str]


class FloorChoice(BaseModel):
    floor_block: Optional[str] = None


class TrackResult(BaseModel):
    run_id: str
    track: str
    manifest_url: str
    summary: dict


class Similarity(BaseModel):
    track: str
    score: float
    components: dict
    matches: dict
    details: list[dict]


class ComparisonResult(BaseModel):
    run_id: str
    reference: str
    similarity: list[Similarity]


class CheckOut(BaseModel):
    key: str
    rule_id: str
    rule: str
    room_id: str
    room: str
    status: str
    measured: str
    required: str
    detail: str
    severity: str
    why: str
    remedy: str
    clause: Optional[dict]
    margin_pct: Optional[float]
    measured_value: Optional[float]
    required_value: Optional[float]
    confidence: float
    neighbours: list[str]


class NotChecked(BaseModel):
    room_id: str
    room: str
    reason: str


class ComplianceReport(BaseModel):
    run_id: Optional[str] = None
    track: Optional[str] = None
    total: int
    passed: int
    failed: int
    by_severity: dict[str, int]
    not_checked: list[NotChecked]
    violations: list[CheckOut]
    checks: list[CheckOut]


class ExplanationItem(BaseModel):
    explanation: str
    fix: str


class ExplanationSet(BaseModel):
    model: str
    advisory: bool
    truncated: bool
    explanations: dict[str, ExplanationItem]
