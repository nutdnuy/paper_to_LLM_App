"""Moving Targets pipeline — reproduces Choi et al. (2026) framework.

"From Text to Alpha: Can LLMs Track Evolving Signals in Corporate Disclosures?"
arXiv:2510.03195v4

Pipeline:
  transcript_i, transcript_j=i-4
    -> LLM F_θ extracts metric sets T_i, T_j
    -> text encoder G_φ embeds metrics
    -> cosine_sim(E_i, E_j) + max pooling across current per prior metric
    -> piecewise-linear threshold h(·) with (α=0.4, β=0.6)
    -> MT_i = 1 - (1/N_j) Σ h(max cos)
"""

from __future__ import annotations

import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Extraction prompt — designed per paper Section 4 "Method"
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """You are a quantitative financial analyst extracting performance metrics from a corporate earnings call transcript.

## Extract ALL performance metrics the management and analysts discuss, including:

**Financial metrics**
- Revenue, net sales, top-line, gross profit, gross margin
- Operating income, operating margin, EBIT, EBITDA, EBITDA margin
- Net income, earnings per share (EPS), diluted EPS
- Free cash flow (FCF), operating cash flow, capital expenditures
- Return metrics: ROE, ROIC, ROA

**Operational / segment metrics**
- Market share, unit sales, volumes, shipments, bookings, backlog
- Subscribers, users, MAU/DAU, ARR, ARPU, churn, retention rate
- Same-store sales, comparable-store sales, store count
- Utilization, load factor, fill rate, yield
- Segment-level metrics (by geography, product, business unit)

**Capital return**
- Dividends, share repurchases, buybacks, dividend yield

## CRITICAL — preserve contextual qualifiers

Extract the FULL phrase with its modifier, not just the generic term:
✓ "North America cloud revenue"    ✗ "revenue"
✓ "Blackwell chip revenue"          ✗ "revenue"
✓ "enterprise segment operating margin"   ✗ "margin"
✓ "data center gross margin"        ✗ "margin"
✓ "international same-store sales"  ✗ "sales"
✓ "fourth quarter dividend"         ✗ "dividend"
✓ "quarterly free cash flow"        ✗ "cash flow"

## Exclude

- Fiscal period labels alone: "Q1", "Q2", "fiscal year"
- Non-metric filler: "the %", "a % increase", "the range"
- Pure guidance statements without a named metric
- Accounting adjustments without the underlying metric (e.g., skip "GAAP adjustment" alone)

## Output

Return ONLY a JSON array of metric strings. Deduplicate case-insensitively.
Typical transcript yields 15-40 metrics."""


class MetricList(BaseModel):
    """Structured output for metric extraction."""
    metrics: list[str] = Field(
        description=(
            "Performance metrics extracted from the transcript, with "
            "contextual qualifiers preserved. Each entry is a single metric phrase."
        )
    )


# ---------------------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------------------

def extract_metrics(transcript: str, llm: ChatOpenAI) -> list[str]:
    """Call LLM to extract performance metrics from transcript (returns deduped list)."""
    if not transcript.strip():
        return []

    structured = llm.with_structured_output(MetricList)
    result = structured.invoke(
        [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript:\n\n{transcript}"},
        ]
    )

    seen: set[str] = set()
    deduped: list[str] = []
    for m in result.metrics:
        key = m.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(m.strip())
    return deduped


def embed_metrics(metrics: list[str], embedder: OpenAIEmbeddings) -> np.ndarray:
    """Batch-embed metric strings. Returns (N, d) array, or (0, d) if empty."""
    if not metrics:
        return np.zeros((0, 1))
    vecs = embedder.embed_documents(metrics)
    return np.asarray(vecs, dtype=np.float32)


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return (|A|, |B|) cosine similarity matrix. Handles empty inputs."""
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return (An @ Bn.T).astype(np.float32)


def piecewise_linear(x: np.ndarray, alpha: float = 0.4, beta: float = 0.6) -> np.ndarray:
    """h(S) from paper Eq. 3:
        0               if S ≤ α
        (S-α)/(β-α)     if α < S < β
        1               if S ≥ β
    """
    if beta <= alpha:
        raise ValueError(f"β must be > α (got α={alpha}, β={beta})")
    x = np.asarray(x, dtype=np.float32)
    return np.clip((x - alpha) / (beta - alpha), 0.0, 1.0)


def compute_mt_score(
    current_metrics: list[str],
    prior_metrics: list[str],
    embedder: OpenAIEmbeddings,
    alpha: float = 0.4,
    beta: float = 0.6,
) -> dict:
    """Full MT score pipeline (paper Eq. 4):
        MT_i = 1 - (1/N_j) Σ h(max cos)

    Returns dict with mt_score, per-metric breakdown, similarity matrix.
    """
    n_cur, n_pri = len(current_metrics), len(prior_metrics)

    if n_pri == 0:
        return {
            "mt_score": None,
            "warning": "No prior-quarter metrics extracted — MT undefined",
            "n_current": n_cur,
            "n_prior": 0,
        }
    if n_cur == 0:
        return {
            "mt_score": 1.0,
            "warning": "No current-quarter metrics extracted — all prior metrics considered dropped",
            "n_current": 0,
            "n_prior": n_pri,
        }

    cur_emb = embed_metrics(current_metrics, embedder)
    pri_emb = embed_metrics(prior_metrics, embedder)

    # sim[j, i] = cos(prior_j, current_i)
    sim = cosine_similarity_matrix(pri_emb, cur_emb)
    max_sim = sim.max(axis=1)  # best match in current for each prior metric
    best_idx = sim.argmax(axis=1)
    retention = piecewise_linear(max_sim, alpha, beta)
    mt_score = 1.0 - float(retention.mean())

    per_prior = []
    for j in range(n_pri):
        r = float(retention[j])
        per_prior.append(
            {
                "prior": prior_metrics[j],
                "best_match": current_metrics[int(best_idx[j])],
                "raw_similarity": float(max_sim[j]),
                "retention": r,
                "status": (
                    "retained" if r >= 0.99
                    else "partial" if r > 0
                    else "dropped"
                ),
            }
        )

    return {
        "mt_score": mt_score,
        "retention_mean": float(retention.mean()),
        "n_current": n_cur,
        "n_prior": n_pri,
        "n_retained": int(sum(1 for p in per_prior if p["status"] == "retained")),
        "n_partial": int(sum(1 for p in per_prior if p["status"] == "partial")),
        "n_dropped": int(sum(1 for p in per_prior if p["status"] == "dropped")),
        "per_prior": per_prior,
        "similarity_matrix": sim,
        "current_metrics": current_metrics,
        "prior_metrics": prior_metrics,
        "alpha": alpha,
        "beta": beta,
    }


# ---------------------------------------------------------------------------
# Interpretation helper
# ---------------------------------------------------------------------------

def interpret_mt(score: float) -> tuple[str, str]:
    """Map MT score → (label, narrative) per paper's findings.

    Paper Q5 (highest MT) → lowest subsequent return.
    Q1 (lowest MT)  → highest subsequent return.
    5F alpha spread Q5-Q1 = -0.52% monthly (t=-2.55).
    """
    if score < 0.20:
        return (
            "🟢 Low MT (Q1-like)",
            "บริษัทยัง **retain** metrics เดิมส่วนใหญ่ — historically Q1 firms "
            "outperform. 5F alpha ประมาณ **+0.55%** monthly (Table 3A, paper).",
        )
    if score < 0.40:
        return (
            "🟢 Moderate-low MT (Q2-like)",
            "บริษัท retain เกือบทุก metric มี shift เล็กน้อย — 5F alpha "
            "ประมาณ **+0.38%** monthly (Table 3B).",
        )
    if score < 0.60:
        return (
            "🟡 Middle MT (Q3-like)",
            "สัญญาณกลางๆ — alpha ไม่มีนัยสำคัญ (+0.26%, t-stat ต่ำ). "
            "ยากจะตัดสินทิศทาง.",
        )
    if score < 0.80:
        return (
            "🟠 Elevated MT (Q4-like)",
            "Metrics หายไปเยอะ — บริษัทกำลัง shift narrative. "
            "5F alpha ประมาณ **+0.32%** monthly แต่ momentum ลดลง.",
        )
    return (
        "🔴 High MT (Q5-like) — bearish signal",
        "บริษัท **drop** metrics ที่เคยเน้นเป็นจำนวนมาก → paper พบว่า Q5 firms "
        "underperform (5F alpha ≈ **0.03%** monthly, Q5−Q1 spread **−0.52%** "
        "t=−2.55). พิจารณา **short / underweight**.",
    )
