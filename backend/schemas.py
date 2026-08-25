"""Pydantic models for API request/response serialization."""

from typing import Any, Optional
from pydantic import BaseModel


class JobStatusResponse(BaseModel):
    job_id:   str
    status:   str        # pending | running | done | error
    created:  str
    finished: Optional[str] = None
    error:    Optional[str] = None
    steps:    dict[str, dict]


class ModelPreset(BaseModel):
    id:       str
    label:    str
    model:    str
    base_url: str


class DeonticEntryOut(BaseModel):
    id:       str
    actions:  list[str]
    targets:  list[str]
    assignee: Optional[str] = None
    assigner: Optional[str] = None
    description: Optional[str] = None


class ClauseReportItem(BaseModel):
    clause_id:      str
    clause_text:    str
    type:           str
    party:          str
    action:         str
    asset:          str
    conforms:       bool
    repair_rounds:  int
    permissions:    list[DeonticEntryOut]
    prohibitions:   list[DeonticEntryOut]
    duties:         list[DeonticEntryOut]
    unfair_terms:   dict[str, list[dict]]
    semantic_sim:   Optional[float] = None
    back_translated: Optional[str] = None
    ttl:            str = ""


class AggregateStats(BaseModel):
    total_clauses:    int
    conforming:       int
    permissions:      int
    prohibitions:     int
    duties:           int
    unfair_count:     int
    mean_semantic_sim: Optional[float] = None


class JobListItem(BaseModel):
    job_id:   str
    status:   str
    created:  str
    finished: Optional[str] = None
    title:    Optional[str] = None
    provider: Optional[str] = None
    n_clauses: Optional[int] = None


class JobReportResponse(BaseModel):
    job_id:    str
    provider:  str
    title:     str
    date:      str
    aggregate: AggregateStats
    clauses:   list[ClauseReportItem]
