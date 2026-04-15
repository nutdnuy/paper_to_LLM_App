# 📈 Moving Targets Analyzer

**Turn corporate earnings call transcripts into an alpha signal.**

Reproduces the framework from:
> **"From Text to Alpha: Can LLMs Track Evolving Signals in Corporate Disclosures?"**
> Choi, Kim, Yu, Cha, Golkhou, Halperin, Papaioannou, Kim, Wang, Kwon, Kim, Lopez-Lira, Lee
> *arXiv:2510.03195v4 · Mar 2026*

Paste two earnings-call transcripts — the **current quarter** and the **same quarter one year earlier** — and the app computes the **Moving Targets score `MT_i ∈ [0, 1]`**. Paper finds that high-MT firms (Q5) underperform low-MT firms (Q1) by **−0.52% / month** 5-factor alpha (t = −2.55).

---

## 🧭 How it works

```
transcript_current  ─┐                          ┌─► E_current (d=3072)
                     ├─► LLM extractor ─► metrics
transcript_prior    ─┘                          └─► E_prior   (d=3072)
                                                       │
                                                       ▼
                                    cosine_sim matrix  +  max_i  per prior metric
                                                       │
                                                       ▼
                            piecewise-linear  h(·)  with  α = 0.4,  β = 0.6
                                                       │
                                                       ▼
                               MT_i = 1 − (1/N_j) · Σ_j h(max cos)
```

- **Extractor** (`gpt-5.4` / `gpt-4.1`): pulls performance metrics with contextual qualifiers preserved (*"North America cloud revenue"* not *"revenue"*) — the paper's core innovation over NER.
- **Ruler** (`text-embedding-3-large` / `-small`): measures semantic distance between metrics across quarters.
- **Threshold** `h(S)`: linear ramp 0 → 1 between α and β; clamps noise in middle-range similarity.

---

## 🚀 Try it

**Live demo**: *deploy to Streamlit Community Cloud — see below*

**Run locally**:

```bash
git clone https://github.com/<your-username>/paper_to_LLM_App.git
cd paper_to_LLM_App

python3.10+ -m venv venv         # langchain v1 requires Python ≥ 3.10
source venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

Open <http://localhost:8501> → paste your OpenAI API key in the sidebar → click **🔬 Extract & Score**.

---

## 🔑 API key

The app requires an OpenAI API key for metric extraction (LLM) and embeddings.

- **Default behavior**: app shows a password input in the sidebar — key lives only in your browser session, never logged or persisted.
- **Optional local convenience**: create `.streamlit/secrets.toml` (gitignored):
  ```toml
  OPENAI_API_KEY = "sk-proj-..."
  ```
  The app will auto-load it and skip the input field.

**On Streamlit Community Cloud**: leave secrets empty — visitors bring their own key.

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (public).
2. Go to <https://share.streamlit.io> → **New app**.
3. Select the repo, branch `main`, main file `app.py`.
4. **Do NOT** set any secrets (visitors supply their own key in the sidebar).
5. Click **Deploy**.

That's it — the app boots in ~1 min.

---

## 📋 Built-in examples

Three fictional transcript pairs ship with the app so you can test without hunting for real transcripts:

| Scenario | Expected MT | What it illustrates |
|---|---|---|
| 🔴 **TechCo** — High MT | ≈ 0.55 – 0.75 | Growth-metric disclosures (cloud revenue, AI chip revenue, ARR, NRR) dropped in favor of buybacks + cost savings |
| 🟢 **SteadyCo** — Low MT | ≈ 0.10 – 0.25 | Food & beverage company using the same metric set each quarter (organic growth, gross margin, FCF, buybacks) |
| 🟡 **RetailCo** — Mid MT | ≈ 0.35 – 0.50 | Retailer partially pivoting — keeps comp sales but drops digital / membership and adds margin / inventory focus |

For real data, grab transcripts from [Motley Fool](https://www.fool.com/earnings-call-transcripts/) (free) or company IR pages — paste just the **Prepared Remarks** section for cleaner extraction.

---

## 📊 Output

- **MT score** — headline number, color-coded (🟢 low / 🟡 mid / 🔴 high)
- **Interpretation** — narrative mapping score to paper's quintile-level alpha (Table 3)
- **Per-metric breakdown** — for each prior-quarter metric: best match in current, raw cosine similarity, retention score `h(·)`, status (retained / partial / dropped)
- **Extracted metrics** — side-by-side lists for both quarters
- **Cosine similarity heatmap** — dark rows reveal dropped metrics at a glance

---

## 📂 Project structure

```
paper_to_LLM_App/
├── app.py               # Streamlit UI + QuantSeras dark+green theme
├── mt_pipeline.py       # extract → embed → cosine_sim → h(α,β) → MT_i
├── requirements.txt
├── README.md
├── CLAUDE.md            # conventions for Claude Code agents
├── .gitignore
└── .streamlit/
    ├── config.toml              # dark theme tokens
    └── secrets.toml.example     # template for local API key
```

---

## ⚠️ Caveats

- **Not financial advice.** This is an educational reproduction of a research paper. Do your own due diligence before trading.
- **Extraction cost**: one analyze click = 2 LLM calls (prior + current) + ~30-60 embedding calls. Cost ≈ $0.02 – $0.15 depending on model + transcript length.
- **Framework limitations** (from paper): sample restricted to S&P 100 (Jan 2010 – Dec 2024), model-dependent, transaction costs not netted out of the −0.52% alpha.

---

## 📝 License

Educational / research use. Paper is an open-access arXiv preprint — rights belong to the original authors.
