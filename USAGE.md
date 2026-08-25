# How to run tos-to-odrl

## Quick start — Docker (recommended, this is the deployable microservice)

```bash
docker compose up -d --build
```

This starts three containers:

| Service | URL | Purpose |
|---|---|---|
| `backend` | http://localhost:8000 | FastAPI microservice — call this from another system |
| `frontend` | http://localhost:3000 | Web UI for manual use |
| `mongodb` | `localhost:27018` | Persists job state/results (survives backend restarts) |

Before starting, copy `.env.example` to `.env` and fill in whichever keys you need (see below).

---

## Environment variables (`.env`)

| Variable | Required when | Notes |
|---|---|---|
| `OPENAI_API_KEY` | Using the `gpt-4.1-mini` preset, or any time semantic evaluation runs | Evaluation always uses OpenAI embeddings regardless of the chat model provider |
| `OPENROUTER_API_KEY` | Using the default model (OpenRouter) | Get one at https://openrouter.ai/keys. Needs a little prepaid credit — free-tier OpenRouter models are unreliable (shared pool, gets rate-limited by other users) |
| `HF_TOKEN` | Using the Hugging Face preset | Get one at https://huggingface.co/settings/tokens |
| `MONGODB_URI` | Running the backend | Defaults to `mongodb://localhost:27018` to match `docker-compose.yml`; use `mongodb://mongodb:27017` if calling from inside the Docker network |
| `MONGODB_DB` | Running the backend | Defaults to `tos_to_odrl` |

---

## Backend API — integrating from another system

Base URL: `http://<host>:8000` (no authentication — put it behind your own gateway/auth if exposing it beyond a trusted network).

### `GET /api/models`
Returns the curated list of models the pipeline has been verified to work with. Use this to populate a model picker instead of hardcoding IDs.

```json
[
  {"id": "qwen3-30b-a3b-instruct-openrouter", "label": "Qwen3 30B A3B Instruct (OpenRouter, paid, ~$0.10/run)", "model": "qwen/qwen3-30b-a3b-instruct-2507", "base_url": "https://openrouter.ai/api/v1"},
  {"id": "gpt-4.1-mini", "label": "GPT-4.1 mini (OpenAI)", "model": "gpt-4.1-mini", "base_url": ""},
  {"id": "gemma-3-27b-it", "label": "Gemma 3 27B IT (Hugging Face, free tier)", "model": "google/gemma-3-27b-it", "base_url": "https://router.huggingface.co/v1"}
]
```

### `POST /api/jobs`
Uploads a document and starts the pipeline asynchronously. Returns immediately with a `job_id` — the pipeline runs in the background.

`multipart/form-data` fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | file | *(required)* | `.json` (pre-split clauses), `.txt`, or `.pdf` |
| `provider` | string | `"Unknown"` | Provider name (for `.txt`/`.pdf` inputs) |
| `title` | string | `""` | Document title (for `.txt`/`.pdf` inputs) |
| `date` | string | `"Unknown"` | Document date (for `.txt`/`.pdf` inputs) |
| `model` | string | current default (see `GET /api/models`) | Model name — pick one of `GET /api/models`'s `model` values, or any other model your `base_url` serves |
| `base_url` | string | current default | OpenAI-compatible endpoint. Pass `""` explicitly to use OpenAI directly — omitting the field falls back to the server default, it does NOT mean "OpenAI" |
| `api_key` | string | *(env)* | Overrides the server's own API key for this one job |
| `max_workers` | int | `4` | Concurrent requests per phase |
| `max_repair_rounds` | int | `3` | Max self-repair attempts per non-conforming clause |
| `run_phase1` | bool | `true` | Set `false` to skip metadata classification and generate Turtle directly from unclassified clauses |
| `run_evaluation` | bool | `true` | Set `false` to skip the (slower, OpenAI-embeddings-based) semantic evaluation phase |
| `eval_model` | string | same as `model` | Model used only for the evaluation phase's back-translation step |
| `embed_model` | string | `"text-embedding-3-small"` | OpenAI embedding model for evaluation |

```bash
curl -X POST http://localhost:8000/api/jobs \
  -F "file=@contract.json" \
  -F "provider=Acme Inc" \
  -F "model=qwen/qwen3-30b-a3b-instruct-2507" \
  -F "base_url=https://openrouter.ai/api/v1"
# -> {"job_id": "..."}
```

### `GET /api/jobs/{job_id}`
Poll this for status. `status` is one of `pending | running | done | error`. `steps` breaks progress down per pipeline phase (`split`, `phase1`, `phase2`, `phase3`, `analysis`, `evaluation`), each with its own `status/done/total/error`.

### `GET /api/jobs/{job_id}/report`
Once `status == "done"`, fetch the full result: per-clause classification, generated Turtle, SHACL conformance, deontic rules (duties/permissions/prohibitions), unfair-terms flags, and semantic similarity score. Returns HTTP 425 if the job isn't finished yet.

### `GET /api/jobs`
Lists all jobs (most recent first) with a lightweight summary — useful for a history view.

### Integration pattern
```
POST /api/jobs           → { job_id }
loop: GET /api/jobs/{id} → until status is "done" or "error"
GET /api/jobs/{id}/report → full result
```

Job state is persisted in MongoDB, so it's safe to poll across backend restarts or from a different process than the one that submitted the job.

---

## CLI — `main.py`

Runs the same pipeline locally (split → metadata → TTL → validate → analysis → evaluation) and saves results to disk instead of MongoDB. Useful for batch experiments or reproducible offline runs.

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python main.py --input <file> --output <dir/> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Input file: `.json` (pre-split), `.txt`, or `.pdf` |
| `--output` | *(required)* | Directory where results will be saved |
| `--model` | current default model | LLM model name |
| `--base-url` | current default endpoint | OpenAI-compatible server. Pass an empty string to use OpenAI directly |
| `--api-key` | *(env)* | Falls back to `OPENAI_API_KEY` / `HF_TOKEN` / `OPENROUTER_API_KEY` depending on `--base-url` |
| `--max-workers` | `4` | Parallel workers per phase |
| `--max-repair-rounds` | `3` | Max self-repair iterations per clause |
| `--skip-phase1` | *(off)* | Skip metadata classification, generate Turtle directly from unclassified clauses |
| `--provider` | `Unknown` | Provider name (for raw text / PDF inputs) |
| `--title` | *(filename)* | Document title (for raw text / PDF inputs) |
| `--date` | `Unknown` | Document date (for raw text / PDF inputs) |
| `--source` | *(path)* | Source URL or path (for raw text / PDF inputs) |
| `--description` | *(empty)* | Short description of the document |

### Examples

**Default model (OpenRouter), pre-split JSON:**
```bash
python main.py \
  --input data/data_openai/use_cases_openai.json \
  --output output/openai_run/
```

**OpenAI instead:**
```bash
python main.py \
  --input data/data_openai/use_cases_openai.json \
  --base-url "" --model gpt-4.1-mini \
  --output output/openai_gpt41mini/
```

**From a raw text file, providing metadata:**
```bash
python main.py \
  --input data/elsevier/api_service_agreement_2017.txt \
  --provider "Elsevier" --title "API Service Agreement" --date "2017" \
  --output output/elsevier-2017/
```

**Local Ollama model (OpenAI-compatible endpoint):**
```bash
python main.py \
  --input data/data_openai/use_cases_openai.json \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:72b \
  --output output/openai_qwen72b/
```

---

## Pipeline phases

Each CLI run writes intermediate results to the output directory:

```
output/<run>/
├── phase1/
│   └── metadata.json          # clause type, party, action, asset per clause
├── phase2/
│   └── clause_<n>.ttl         # raw Turtle output per clause
├── phase3/
│   └── clause_<n>.ttl         # validated / self-repaired Turtle
├── analysis/
│   └── clause_<n>.json        # SPARQL queries: deontic status + unfair terms
├── evaluation.json            # semantic similarity scores (back-translation)
└── result.json                # full combined result
```

| Phase | What it does |
|-------|-------------|
| **Step 0** | Splits the raw document into individual clauses |
| **Phase 1** (optional, `--skip-phase1`) | Extracts metadata (type, party, action, asset) per clause |
| **Phase 2** | Generates an ODRL/TOSL Turtle (TTL) policy per clause |
| **Phase 3** | Validates each TTL against the TOSL ontology and self-repairs if needed |
| **Analysis** | Runs SPARQL queries to extract deontic rules and detect unfair terms |
| **Evaluation** (optional, `run_evaluation=false`) | Back-translates each TTL to natural language and measures semantic similarity |

The pipeline calls two external services it doesn't run itself: the TOSL SHACL validator and SPARQL endpoints at `tosl.onrender.com` (`src/config.py`). Those need to be reachable for Phase 3/Analysis to work.

---

## Model notes

- Only the models listed in `GET /api/models` / `src/config.MODEL_PRESETS` have been verified end-to-end against the real Phase 2 prompt (~14.5k tokens of ontology + guidelines context). Other models on the same providers may fail in non-obvious ways — see the comment above `MODEL_PRESETS` in `src/config.py` for what's already been tried and ruled out (timeouts, runaway "thinking" reasoning eating the whole token budget, unsupported models, unreliable free/cheap providers).
- Requests to OpenRouter's `:free`-suffixed models are throttled client-side (15/min, 40/day) to avoid tripping their rate limits — paid OpenRouter models are not throttled.
- `--max-tokens` defaults to 8192 in `src/llm.build_model` — needed for "thinking" models that spend part of their budget on hidden reasoning before the real answer.

---

## Web interface

With Docker Compose running, open `http://localhost:3000`. Lets you upload a file, pick a model from the same curated list the API exposes, track progress in real time, and browse past analyses (`/jobs`).

To run it without Docker:
```bash
# backend
source .venv/bin/activate && uvicorn backend.main:app --reload --port 8000
# frontend, in a separate terminal
cd frontend && npm run dev
```

---

## Input formats

### Pre-split JSON (`.json`)
Clauses are already split. Skips Step 0 (faster, fully reproducible).

```json
{
  "PROVIDER": "OpenAI",
  "SOURCE": "https://openai.com/policies/...",
  "TITLE": "Terms of Use",
  "DATE": "2024-12",
  "USE_CASE_DESCRIPTIONS": {
    "use_case_1": { "description": "You must be at least 13 years old..." }
  }
}
```

### Raw text (`.txt`) or PDF (`.pdf`)
The pipeline runs Step 0 to split the document into clauses automatically. Provide `--provider`/`provider`, `--title`/`title`, and `--date`/`date` for traceability.

---

## Data in this repo

| Path | Description |
|------|-------------|
| `data/data_openai/use_cases_openai.json` | OpenAI Terms of Use (Dec 2024, EU) — pre-split |
| `data/data_elsevier/use_cases_elsevier.json` | Elsevier API Service Agreement — pre-split |
| `data/elsevier/*.txt` | Raw Elsevier agreements (2014–2017) |
| `data/*/..._expected.json` | Ground-truth classifications for evaluation |
| `metamodel/tosl_odrl_simplified.ttl` | TOSL/ODRL ontology |
| `metamodel/tosl_guidelines.md` | Prompt guidelines for the LLM |
