"""Job runner for the tos-to-odrl pipeline, persisted to MongoDB."""

import logging
import threading
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from backend.db import jobs_collection

logger = logging.getLogger(__name__)


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    ERROR   = "error"


def _make_step(status: StepStatus = StepStatus.PENDING) -> dict:
    return {"status": status.value, "done": 0, "total": 0, "error": None}


class Job:
    def __init__(self, job_id: str):
        self.job_id    = job_id
        self.status    = "pending"
        self.created   = datetime.utcnow().isoformat()
        self.finished  = None
        self.error     = None
        self.steps: dict[str, dict] = {
            "split":      _make_step(),
            "phase1":     _make_step(),
            "phase2":     _make_step(),
            "phase3":     _make_step(),
            "analysis":   _make_step(),
            "evaluation": _make_step(),
        }
        self.result: Any = None
        self._lock = threading.Lock()

    def set_step(self, name: str, status: StepStatus, done: int = 0, total: int = 0):
        with self._lock:
            self.steps[name]["status"] = status.value
            self.steps[name]["done"]   = done
            self.steps[name]["total"]  = total
        self._persist()

    def set_status(self, status: str, *, error: str | None = None, finished: str | None = None):
        with self._lock:
            self.status = status
            if error is not None:
                self.error = error
            if finished is not None:
                self.finished = finished
        self._persist()

    def set_result(self, result) -> None:
        with self._lock:
            self.result = result
        self._persist()

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "job_id":   self.job_id,
                "status":   self.status,
                "created":  self.created,
                "finished": self.finished,
                "error":    self.error,
                "steps":    {k: dict(v) for k, v in self.steps.items()},
            }

    def _persist(self) -> None:
        doc = self.to_dict()
        doc["result"] = self.result.model_dump() if self.result is not None else None
        jobs_collection.update_one({"job_id": self.job_id}, {"$set": doc}, upsert=True)


def _job_from_doc(doc: dict) -> Job:
    from src.models import PipelineResult

    job = Job(doc["job_id"])
    job.status   = doc["status"]
    job.created  = doc["created"]
    job.finished = doc["finished"]
    job.error    = doc["error"]
    job.steps    = doc["steps"]
    job.result   = PipelineResult.model_validate(doc["result"]) if doc.get("result") else None
    return job


def create_job() -> Job:
    job = Job(str(uuid4()))
    job._persist()
    return job


def get_job(job_id: str) -> Job | None:
    doc = jobs_collection.find_one({"job_id": job_id}, {"_id": 0})
    return _job_from_doc(doc) if doc else None


def list_jobs() -> list[Job]:
    docs = jobs_collection.find({}, {"_id": 0}).sort("created", -1)
    return [_job_from_doc(doc) for doc in docs]


def run_pipeline_background(job: Job, params: dict) -> None:
    thread = threading.Thread(target=_run, args=(job, params), daemon=True)
    thread.start()


def _progress(job: Job, step: str, total: int):
    return lambda done: job.set_step(step, StepStatus.RUNNING, done=done, total=total)


def _run(job: Job, params: dict) -> None:
    from src.io import load_contract_from_json, load_contract_from_text, load_text
    from src.llm import build_model
    from src.phases import step0, phase1, phase2, phase3, analysis, evaluation
    from src.models import PipelineResult

    job.set_status("running")
    try:
        # ── Build model ────────────────────────────────────────────────────────
        model = build_model(
            params["model"],
            base_url=params.get("base_url"),
            api_key=params.get("api_key"),
        )
        ontology   = load_text(params["ontology_path"])
        guidelines = load_text(params["guidelines_path"])
        contract   = params["contract"]
        max_workers     = params.get("max_workers", 4)
        max_repair      = params.get("max_repair_rounds", 3)
        run_eval        = params.get("run_evaluation", True)
        run_phase1      = params.get("run_phase1", True)

        # ── Step 0: clause splitting ───────────────────────────────────────────
        job.set_step("split", StepStatus.RUNNING)
        split_contract = step0.split_clauses(contract, model)
        n = len(split_contract.clauses)
        job.set_step("split", StepStatus.DONE, done=n, total=n)

        # ── Phase 1: metadata extraction ───────────────────────────────────────
        if run_phase1:
            job.set_step("phase1", StepStatus.RUNNING, total=n)
            metadata_list = phase1.extract_metadata(
                split_contract, model, max_workers=max_workers,
                on_progress=_progress(job, "phase1", n),
            )
            job.set_step("phase1", StepStatus.DONE, done=len(metadata_list), total=n)
        else:
            metadata_list = phase1.placeholder_metadata(split_contract)
            job.set_step("phase1", StepStatus.DONE, done=len(metadata_list), total=n)

        # ── Phase 2: Turtle generation ─────────────────────────────────────────
        job.set_step("phase2", StepStatus.RUNNING, total=len(metadata_list))
        turtle_outputs = phase2.generate_turtle(
            metadata_list, split_contract, ontology, guidelines, model, max_workers=max_workers,
            on_progress=_progress(job, "phase2", len(metadata_list)),
        )
        job.set_step("phase2", StepStatus.DONE, done=len(turtle_outputs), total=len(metadata_list))

        # ── Phase 3: validation + self-repair ──────────────────────────────────
        job.set_step("phase3", StepStatus.RUNNING, total=len(turtle_outputs))
        corrected = phase3.validate_and_repair(
            turtle_outputs, guidelines, model,
            max_repair_rounds=max_repair, max_workers=max_workers,
            on_progress=_progress(job, "phase3", len(turtle_outputs)),
        )
        job.set_step("phase3", StepStatus.DONE, done=len(corrected), total=len(turtle_outputs))

        # ── Analysis: deontic + unfair terms ───────────────────────────────────
        job.set_step("analysis", StepStatus.RUNNING, total=len(corrected))
        analysis_results = analysis.analyze(
            corrected, max_workers=max_workers,
            on_progress=_progress(job, "analysis", len(corrected)),
        )
        job.set_step("analysis", StepStatus.DONE, done=len(analysis_results), total=len(corrected))

        # ── Evaluation: semantic back-translation ──────────────────────────────
        evaluation_results = []
        if run_eval:
            job.set_step("evaluation", StepStatus.RUNNING, total=len(corrected))
            eval_model_name = params.get("eval_model")
            eval_model  = build_model(eval_model_name) if eval_model_name else None
            evaluation_results = evaluation.evaluate_semantic(
                corrected, model,
                eval_model=eval_model,
                embed_model=params.get("embed_model", "text-embedding-3-small"),
                max_workers=max_workers,
                on_progress=_progress(job, "evaluation", len(corrected)),
            )
            job.set_step("evaluation", StepStatus.DONE, done=len(evaluation_results), total=len(corrected))

        job.set_result(PipelineResult(
            contract=split_contract,
            metadata=metadata_list,
            turtle_files=corrected,
            analysis=analysis_results,
            evaluation=evaluation_results,
        ))
        job.set_status("done", finished=datetime.utcnow().isoformat())

    except Exception as exc:
        logger.exception("Pipeline failed for job %s", job.job_id)
        job.set_status("error", error=str(exc), finished=datetime.utcnow().isoformat())
