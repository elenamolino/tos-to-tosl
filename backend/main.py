"""FastAPI backend for the tos-to-odrl pipeline."""

import os
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Make sure tos-to-odrl root is on sys.path when running from backend/
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.jobs import create_job, get_job, list_jobs, run_pipeline_background
from backend.schemas import AggregateStats, ClauseReportItem, DeonticEntryOut, JobListItem, JobReportResponse, JobStatusResponse, ModelPreset
from src.config import DEFAULT_MODEL, DEFAULT_BASE_URL, MODEL_PRESETS

_PROJECT_ROOT     = Path(__file__).parent.parent
_DEFAULT_ONTOLOGY  = _PROJECT_ROOT / "metamodel" / "tosl_odrl_simplified.ttl"
_DEFAULT_GUIDELINES = _PROJECT_ROOT / "metamodel" / "tosl_guidelines.md"

app = FastAPI(title="tos-to-odrl API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── GET /api/models ────────────────────────────────────────────────────────────

@app.get("/api/models", response_model=list[ModelPreset])
def list_models():
    return MODEL_PRESETS


# ── GET /api/jobs ──────────────────────────────────────────────────────────────

@app.get("/api/jobs", response_model=list[JobListItem])
def list_all_jobs():
    jobs = list_jobs()
    items = []
    for job in jobs:
        contract = job.result.contract if job.result else None
        n_clauses = len(job.result.turtle_files) if job.result else None
        items.append(JobListItem(
            job_id   = job.job_id,
            status   = job.status,
            created  = job.created,
            finished = job.finished,
            title    = contract.title if contract else None,
            provider = contract.provider if contract else None,
            n_clauses = n_clauses,
        ))
    return items


# ── POST /api/jobs ─────────────────────────────────────────────────────────────

@app.post("/api/jobs", status_code=202)
async def create_analysis_job(
    file:             UploadFile = File(...),
    provider:         str = Form("Unknown"),
    title:            str = Form(""),
    date:             str = Form("Unknown"),
    model:            str = Form(DEFAULT_MODEL),
    base_url:         str = Form(DEFAULT_BASE_URL),
    api_key:          str = Form(""),
    max_workers:       int  = Form(4),
    max_repair_rounds: int  = Form(3),
    run_evaluation:    bool = Form(True),
    run_phase1:        bool = Form(True),
    eval_model:        str  = Form(""),
    embed_model:       str  = Form("text-embedding-3-small"),
):
    from src.io import load_contract_from_json, load_contract_from_text, load_pdf

    content  = await file.read()
    suffix   = Path(file.filename or "").suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if suffix == ".json":
            contract = load_contract_from_json(tmp_path)
        elif suffix == ".pdf":
            text = load_pdf(tmp_path)
            contract = load_contract_from_text(text, provider=provider, title=title, date=date)
        else:
            text = Path(tmp_path).read_text(encoding="utf-8")
            contract = load_contract_from_text(text, provider=provider, title=title, date=date)
    finally:
        os.unlink(tmp_path)

    job = create_job()
    run_pipeline_background(job, {
        "contract":         contract,
        "model":            model,
        "base_url":         base_url or None,
        "api_key":          api_key or None,
        "ontology_path":    str(_DEFAULT_ONTOLOGY),
        "guidelines_path":  str(_DEFAULT_GUIDELINES),
        "max_workers":       max_workers,
        "max_repair_rounds": max_repair_rounds,
        "run_evaluation":    run_evaluation,
        "run_phase1":        run_phase1,
        "eval_model":        eval_model or None,
        "embed_model":       embed_model,
    })

    return {"job_id": job.job_id}


# ── GET /api/jobs/{job_id} ─────────────────────────────────────────────────────

@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


# ── GET /api/jobs/{job_id}/report ──────────────────────────────────────────────

@app.get("/api/jobs/{job_id}/report", response_model=JobReportResponse)
def get_job_report(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done":
        raise HTTPException(status_code=425, detail=f"Job is not done yet (status: {job.status})")

    result   = job.result
    contract = result.contract

    # Index by clause_id for fast lookup
    meta_by_id  = {m.clause_id: m for m in result.metadata}
    anal_by_id  = {a.clause_id: a for a in result.analysis}
    eval_by_id  = {e.clause_id: e for e in result.evaluation}

    clauses = []
    for ct in result.turtle_files:
        meta  = meta_by_id.get(ct.clause_id)
        anal  = anal_by_id.get(ct.clause_id)
        evl   = eval_by_id.get(ct.clause_id)

        deontic = anal.deontic if anal else None
        clauses.append(ClauseReportItem(
            clause_id     = ct.clause_id,
            clause_text   = ct.clause_text,
            type          = meta.type if meta else "",
            party         = meta.party if meta else "",
            action        = meta.action if meta else "",
            asset         = meta.asset if meta else "",
            conforms      = ct.final_report.conforms,
            repair_rounds = ct.repair_rounds,
            permissions   = [DeonticEntryOut(**e.model_dump()) for e in (deontic.permissions if deontic else [])],
            prohibitions  = [DeonticEntryOut(**e.model_dump()) for e in (deontic.prohibitions if deontic else [])],
            duties        = [DeonticEntryOut(**e.model_dump()) for e in (deontic.duties if deontic else [])],
            unfair_terms  = anal.unfair_terms if anal else {},
            semantic_sim  = evl.semantic_sim if evl else None,
            back_translated = evl.back_translated if evl else None,
            ttl           = ct.ttl,
        ))

    # Aggregate stats
    total_unfair = sum(
        sum(len(v) for v in c.unfair_terms.values())
        for c in clauses
    )
    sem_sims = [c.semantic_sim for c in clauses if c.semantic_sim is not None]

    aggregate = AggregateStats(
        total_clauses    = len(clauses),
        conforming       = sum(1 for c in clauses if c.conforms),
        permissions      = sum(len(c.permissions) for c in clauses),
        prohibitions     = sum(len(c.prohibitions) for c in clauses),
        duties           = sum(len(c.duties) for c in clauses),
        unfair_count     = total_unfair,
        mean_semantic_sim = round(sum(sem_sims) / len(sem_sims), 4) if sem_sims else None,
    )

    return JobReportResponse(
        job_id    = job_id,
        provider  = contract.provider,
        title     = contract.title,
        date      = contract.date,
        aggregate = aggregate,
        clauses   = clauses,
    )
