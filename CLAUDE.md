# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Streamlit tool ("**Moving Targets Analyzer**") that **reproduces** the framework from *"From Text to Alpha: Can LLMs Track Evolving Signals in Corporate Disclosures?"* (Choi et al., arXiv:2510.03195v4, Mar 2026). User pastes two earnings-call transcripts (current quarter `D_i` + prior-year same quarter `D_j=i-4`), the app returns the Moving Targets score `MT_i ∈ [0,1]` as an alpha signal.

This is **NOT** a paper-reading chatbot — earlier iterations tried that and the user rejected it. The goal is to generate alpha signals from disclosure text, not to explain the paper.

Source paper lives at `paper/1773646054397.pdf`. Other papers in `paper/` are unrelated and not in scope.

## Commands

```bash
# Create venv — REQUIRES Python 3.10+ (langchain v1 won't install on 3.9)
# On this machine use anaconda:
/opt/anaconda3/bin/python3.13 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

# Run the app
./venv/bin/streamlit run app.py

# Smoke test (headless, auto-exit)
./venv/bin/streamlit run app.py --server.headless true --server.port 8765 &
sleep 5 && curl -sf http://localhost:8765/ > /dev/null && echo OK
pkill -f "streamlit run app.py"

# Verify knowledge base builds without launching Streamlit
./venv/bin/python -c "from mt_knowledge import build_knowledge_documents; print(len(build_knowledge_documents()))"
```

No test suite, linter, or formatter is configured. Syntax check via `python3 -m py_compile app.py mt_knowledge.py`.

## Architecture

**Two-file design**:

| File | Role |
|---|---|
| `mt_pipeline.py` | Pure pipeline: `extract_metrics(transcript, llm)` → `embed_metrics(metrics, embedder)` → `cosine_similarity_matrix(A, B)` → `piecewise_linear(x, α, β)` → `compute_mt_score(current, prior, embedder, α, β)`. Also exports `interpret_mt(score) -> (label, narrative)` mapping score to paper's Q1-Q5 alpha findings. No Streamlit imports — pure function module. |
| `app.py` | Streamlit UI: two side-by-side text areas for transcripts, sidebar controls (API key, LLM/embedder model, α/β sliders, Load-demo / Clear buttons), `st.status`-wrapped pipeline call, results split across three tabs (per-metric breakdown, extracted metrics list, Plotly cosine similarity heatmap). Custom CSS implements QuantSeras dark+green theme inline. |

**Runtime flow**: button click → `extract_metrics` on prior transcript (one LLM call with `ChatOpenAI.with_structured_output(MetricList)`) → `extract_metrics` on current transcript → `OpenAIEmbeddings.embed_documents` on both metric lists → cosine sim matrix (j=prior × i=current) → `max(axis=1)` per prior metric → `piecewise_linear` h-transform → `MT = 1 - mean(retention)`.

**Key design decisions to preserve**:
- Metric extraction uses **structured output** (`MetricList` Pydantic model via `llm.with_structured_output()`) — do not revert to raw text parsing. This guarantees a clean list of strings with no JSON fragility.
- The `EXTRACTION_SYSTEM_PROMPT` in `mt_pipeline.py` explicitly instructs the LLM to preserve contextual qualifiers ("North America cloud revenue" not "revenue"). This is the paper's core innovation over NER — **do not weaken this instruction**.
- Paper thresholds are `α=0.4, β=0.6`. Keep these as defaults; expose via sliders for experimentation.
- `piecewise_linear` raises if `β ≤ α`. The UI disables the Analyze button when thresholds are invalid — preserve both guardrails.
- Results are stored in `st.session_state["result"]` so the heatmap / tabs persist across reruns without recomputing.
- `similarity_matrix` is returned as numpy array but JSON-serializable conversion happens at display time only (via `np.asarray`). Don't convert to list eagerly — it breaks plotly.

## Extending the Tool

- **Batch mode**: add CSV upload (firm, quarter, transcript_url) → loop `compute_mt_score` per firm-quarter-pair → output alpha signal column. Keep `mt_pipeline.py` stateless; add a `mt_batch.py` rather than mutating it.
- **File upload**: add `st.file_uploader` for `.txt` / `.pdf` (use `pypdf` or `markitdown` for PDF extraction) — feed extracted text into existing `extract_metrics`.
- **Caching LLM calls**: extraction is expensive. Add `@st.cache_data(hash_funcs={...})` around `extract_metrics` keyed on `(transcript_hash, llm_model)` if users re-run with the same transcripts.
- **New paper = different tool**: this repo is purpose-built for MT. A new paper would warrant a separate repo unless the pipeline primitives (extract → embed → similarity) are reusable.

## Visual Design

App uses **QuantSeras Design System** (Material Dark + desaturated green `#69F0AE` on `#121212` base). Tokens live inline in the `CUSTOM_CSS` block in `app.py` under `--qs-*` CSS variables, and mirror `~/Documents/Obsidian Vault/visual Design System/01 - QuantSeras Design System.md`. If editing colors, read the Obsidian file first — don't invent shades. Match elevation (`--qs-surface-1` through `--qs-surface-8`) rather than using shadows.

## Gotchas

- **Python 3.9 is installed system-wide but will NOT work** — `langchain>=1.0.0` requires 3.10+. Always use the anaconda `python3.13` or equivalent.
- `.streamlit/secrets.toml` is gitignored; `.streamlit/secrets.toml.example` is the template. App falls back to sidebar text input if secrets file missing.
- `paper/` contains source PDFs — do not commit the PDFs if repo goes public (license unclear for some). `.gitignore` currently keeps them; flip the commented line to exclude.
