"""Moving Targets Analyzer — From Text to Alpha.

Reproduces the framework from Choi et al. (2026), arXiv:2510.03195v4.

Input: two earnings call transcripts (current quarter D_i, prior-year quarter D_j=i-4).
Output: Moving Targets score MT_i ∈ [0, 1].
  Low  (Q1-like, <0.2)  → metrics retained,   historically outperform.
  High (Q5-like, >0.8)  → metrics dropped,    historically underperform (-0.52%/mo alpha).

Theme: QuantSeras (Material Dark + desaturated green).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from mt_pipeline import compute_mt_score, extract_metrics, interpret_mt


# =============================================================================
# Helpers
# =============================================================================

def load_secret_key() -> str:
    try:
        return st.secrets.get("OPENAI_API_KEY", "")
    except (FileNotFoundError, Exception):
        return ""


# =============================================================================
# Page config
# =============================================================================

st.set_page_config(
    page_title="Moving Targets Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Custom CSS — QuantSeras
# =============================================================================

CUSTOM_CSS = """
<style>
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

:root {
    --qs-bg: #121212;
    --qs-surface-1: #1D1D1D;
    --qs-surface-2: #212121;
    --qs-surface-3: #242424;
    --qs-primary: #69F0AE;
    --qs-primary-variant: #00C853;
    --qs-secondary: #03DAC6;
    --qs-profit: #00E676;
    --qs-loss: #FF5252;
    --qs-warning: #FFB74D;
    --qs-error: #CF6679;
    --qs-text-high: rgba(255,255,255,0.87);
    --qs-text-med: rgba(255,255,255,0.60);
    --qs-text-dis: rgba(255,255,255,0.38);
    --qs-border: rgba(255,255,255,0.08);
}

html, body, [class*="css"] {
    font-family: 'Inter', 'IBM Plex Sans', -apple-system, sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 0% 0%, rgba(105, 240, 174, 0.06) 0%, transparent 45%),
        radial-gradient(circle at 100% 100%, rgba(3, 218, 198, 0.04) 0%, transparent 45%),
        var(--qs-bg);
    color: var(--qs-text-high);
}

/* Hero */
.qs-hero {
    text-align: center;
    padding: 1rem 0 1.25rem 0;
    border-bottom: 1px solid var(--qs-border);
    margin-bottom: 1.5rem;
}
.qs-hero h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #69F0AE 0%, #03DAC6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.02em;
}
.qs-hero .tagline {
    color: var(--qs-text-med);
    font-size: 0.8rem;
    margin-top: 0.4rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}
.qs-hero .paper-ref {
    color: var(--qs-text-dis);
    font-size: 0.7rem;
    margin-top: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--qs-surface-1);
    border-right: 1px solid var(--qs-border);
}
.qs-sidebar-brand {
    text-align: center;
    padding: 0.5rem 0 1rem 0;
    border-bottom: 1px solid var(--qs-border);
    margin-bottom: 0.75rem;
}
.qs-sidebar-brand h2 {
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(135deg, #69F0AE 0%, #03DAC6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.qs-sidebar-brand p {
    color: var(--qs-text-med);
    font-size: 0.7rem;
    margin-top: 0.2rem;
    letter-spacing: 0.05em;
}
.qs-section {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--qs-text-med);
    margin: 1rem 0 0.4rem 0;
    font-weight: 700;
}

/* Input label styling */
.qs-input-label {
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--qs-text-med);
    font-weight: 600;
    margin-bottom: 0.35rem;
    font-family: 'JetBrains Mono', monospace;
}
.qs-input-label.prior { color: var(--qs-secondary); }
.qs-input-label.current { color: var(--qs-primary); }

/* MT score big display */
.qs-mt-card {
    background: var(--qs-surface-1);
    border: 1px solid var(--qs-border);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.qs-mt-score {
    font-size: 5rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    font-variant-numeric: tabular-nums;
    line-height: 1;
    letter-spacing: -0.04em;
    margin: 0.25rem 0;
}
.qs-mt-label {
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--qs-text-med);
    margin-bottom: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
}
.qs-mt-interp {
    color: var(--qs-text-high);
    font-size: 0.95rem;
    line-height: 1.55;
    margin-top: 0.75rem;
    padding-top: 0.75rem;
    border-top: 1px solid var(--qs-border);
}

/* Stat chips */
.qs-stat-row {
    display: flex;
    gap: 0.5rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 0.75rem;
}
.qs-stat {
    background: var(--qs-surface-2);
    border: 1px solid var(--qs-border);
    border-radius: 6px;
    padding: 0.5rem 0.85rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--qs-text-med);
}
.qs-stat strong {
    color: var(--qs-text-high);
    font-weight: 700;
    margin-left: 0.35rem;
}
.qs-stat.retained strong { color: var(--qs-profit); }
.qs-stat.partial strong { color: var(--qs-warning); }
.qs-stat.dropped strong { color: var(--qs-loss); }

/* Buttons */
.stButton > button {
    background: var(--qs-surface-2);
    border: 1px solid var(--qs-border);
    color: var(--qs-text-high);
    border-radius: 6px;
    font-weight: 500;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    border-color: var(--qs-primary);
    color: var(--qs-primary);
    background: rgba(105, 240, 174, 0.08);
}
/* Primary CTA */
.stButton > button[kind="primary"] {
    background: var(--qs-primary);
    color: #000;
    border: none;
    font-weight: 700;
}
.stButton > button[kind="primary"]:hover {
    background: var(--qs-primary-variant);
    color: #000;
}

/* Inputs */
.stTextInput input,
.stTextArea textarea,
div[data-baseweb="select"] > div {
    background: var(--qs-surface-1) !important;
    border-color: var(--qs-border) !important;
    color: var(--qs-text-high) !important;
    font-family: 'Inter', monospace;
}
.stTextArea textarea {
    font-size: 0.88rem !important;
    line-height: 1.5 !important;
}
.stTextInput input:focus,
.stTextArea textarea:focus,
div[data-baseweb="select"] > div:focus-within {
    border-color: var(--qs-primary) !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--qs-primary) !important;
    border-color: var(--qs-primary) !important;
}

/* Dataframe / table */
[data-testid="stDataFrame"] {
    background: var(--qs-surface-1);
    border: 1px solid var(--qs-border);
    border-radius: 8px;
}

/* Metric cards (landing) */
.qs-feature {
    background: var(--qs-surface-1);
    border: 1px solid var(--qs-border);
    border-radius: 8px;
    padding: 1.25rem;
    height: 100%;
}
.qs-feature .icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
.qs-feature h3 {
    color: var(--qs-primary);
    font-size: 0.95rem;
    margin: 0 0 0.5rem 0;
    font-family: 'JetBrains Mono', monospace;
}
.qs-feature p {
    color: var(--qs-text-med);
    font-size: 0.85rem;
    margin: 0;
    line-height: 1.5;
}

/* Expander header */
[data-testid="stExpander"] {
    background: var(--qs-surface-1);
    border: 1px solid var(--qs-border);
    border-radius: 8px;
}

/* Tab styling */
[data-baseweb="tab-list"] {
    gap: 0.25rem;
    background: transparent;
}
[data-baseweb="tab"] {
    background: var(--qs-surface-1) !important;
    border-radius: 6px !important;
    padding: 0.5rem 1rem !important;
    color: var(--qs-text-med) !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    background: var(--qs-surface-3) !important;
    color: var(--qs-primary) !important;
}

[data-testid="stAlert"] {
    background: var(--qs-surface-1) !important;
    border: 1px solid var(--qs-border) !important;
    border-left: 3px solid var(--qs-secondary) !important;
    border-radius: 8px !important;
}
[data-testid="stCaptionContainer"] { color: var(--qs-text-med) !important; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# Demo transcripts — fictional, each scenario designed to hit a distinct MT range
# =============================================================================

EXAMPLES: dict[str, dict] = {
    "🔴 TechCo — High MT (bearish shift)": {
        "description": (
            "Q4'23 เน้น growth metrics (cloud revenue breakdown, AI chips, ARR, "
            "NRR, subscribers). Q4'24 drop เกือบทั้งหมด → เหลือ total revenue + "
            "cost savings + buybacks. คาด MT ≈ 0.55-0.75."
        ),
        "prior": """TechCo Q4 2023 Earnings Call — Prepared Remarks

Thank you and welcome everyone. I'm excited to share our strongest quarter in
company history. Let me walk you through the highlights.

Total revenue grew 34% year-over-year to $8.2 billion, driven primarily by
our cloud infrastructure segment which posted 58% growth to $3.1 billion.
North America cloud revenue alone grew 62% as enterprise customers continued
their digital transformation initiatives. International cloud revenue grew 48%.

Our AI accelerator chip revenue, a new disclosure category, reached $1.4
billion this quarter, up from essentially zero a year ago. Data center
gross margin expanded 420 basis points to 71.2%, reflecting both scale
efficiencies and our premium AI chip mix.

Enterprise subscribers grew to 185,000, up 22% year-over-year. ARR from
our AI platform products reached $2.1 billion with net revenue retention of
134%. Average revenue per enterprise user increased 18%.

Operating margin expanded to 32.4%, a company record. Free cash flow for
the quarter was $2.6 billion, up 48%. We returned $1.2 billion to
shareholders through share repurchases and initiated our first ever
dividend of $0.15 per share.

Looking ahead, we see continued momentum across AI chip revenue, cloud
bookings, and enterprise expansion. Our guidance reflects confidence in
our strategic positioning.
""",
        "current": """TechCo Q4 2024 Earnings Call — Prepared Remarks

Thank you. I want to open by acknowledging this has been a transitional
quarter for the company.

Total revenue was $8.6 billion, growing 5% year-over-year. While overall
demand remained resilient, we saw mixed performance across our portfolio
and are taking decisive action to refocus the business on our highest
return opportunities.

This quarter we executed a strategic restructuring, resulting in a
workforce reduction of 8% and the exit from two underperforming product
lines. Restructuring charges totaled $340 million. We expect annualized
cost savings of $520 million starting in fiscal 2026.

Our operational excellence initiative has already generated meaningful
SG&A savings. SG&A as a percentage of revenue declined 180 basis points.
We continue to drive efficiency across the organization and will share
additional milestones as they materialize.

Capital return remains a priority. During the quarter we repurchased
$2.1 billion of shares, bringing total buybacks for the year to $7.8
billion. Our board approved a new $15 billion repurchase authorization
and increased the quarterly dividend by 20% to $0.18 per share.

We are well positioned for the path ahead and remain committed to
creating long-term shareholder value through disciplined capital
allocation and operational rigor.
""",
    },

    "🟢 SteadyCo — Low MT (bullish / retained)": {
        "description": (
            "บริษัทอาหารและเครื่องดื่ม disclosure กลุ่ม metric เดิมทุกไตรมาส "
            "(net sales, organic revenue growth, gross margin, EPS, FCF, buybacks). "
            "ตัวเลขดีขึ้นเล็กน้อย narrative เหมือนเดิม. คาด MT ≈ 0.10-0.25."
        ),
        "prior": """SteadyCo Q2 2023 Earnings Call — Prepared Remarks

Good morning everyone. I'm pleased to report another quarter of consistent
execution across our global portfolio of food and beverage brands.

Net sales for the quarter were $22.3 billion, up 6.8% year-over-year.
Organic revenue growth was 9.2%, led by effective net pricing of 7% and
positive volume mix of 2.2%. North America net sales grew 7.4%, Europe
grew 5.1%, and our Latin America segment delivered 12% organic growth.

Gross margin expanded 80 basis points to 54.2%, reflecting favorable
pricing net of input-cost inflation and continued productivity gains.
Core operating margin was 18.5%, up 30 basis points year-over-year.

Core earnings per share were $2.09, up 11% on a currency-neutral basis.
Free cash flow for the first half was $4.8 billion. Advertising and
marketing expense increased 9% supporting our brand investments.

We returned $2.6 billion to shareholders this quarter through $1.5 billion
in share repurchases and $1.1 billion in dividends. Our quarterly dividend
was increased by 10% to $1.265 per share earlier this year.

We are reaffirming our full-year guidance for organic revenue growth and
core EPS growth. Our balanced portfolio and pricing discipline position us
well for the remainder of the year.
""",
        "current": """SteadyCo Q2 2024 Earnings Call — Prepared Remarks

Good morning. We delivered another solid quarter, demonstrating the
durability of our portfolio and the effectiveness of our operating model.

Net sales were $23.1 billion, up 3.6% year-over-year. Organic revenue
growth was 5.4%, with effective net pricing of 4% and positive volume mix
of 1.4%. North America net sales grew 3.1%, Europe grew 4.8%, and Latin
America delivered 9% organic growth as our international expansion
continues to contribute.

Gross margin expanded 60 basis points to 54.8%, driven by productivity
savings and disciplined pricing across categories. Core operating margin
reached 18.9%, up 40 basis points year-over-year.

Core earnings per share were $2.26, up 8% on a currency-neutral basis.
Free cash flow for the first half was $5.3 billion, up 10%. Advertising
and marketing expense grew 6% as we continue to invest behind our key
brands.

We returned $2.8 billion to shareholders, including $1.6 billion in share
repurchases and $1.2 billion in dividends. We increased the quarterly
dividend by 7% earlier this year to $1.355 per share.

We are raising the low end of our full-year guidance for organic revenue
growth and core EPS growth, reflecting the consistent performance we have
delivered through the first half.
""",
    },

    "🟡 RetailCo — Mid MT (partial pivot)": {
        "description": (
            "Retailer prior เน้น comp sales + digital growth + membership. "
            "Current ยังพูด comp sales แต่ drop digital/membership + เพิ่ม gross margin "
            "+ inventory focus. คาด MT ≈ 0.35-0.50."
        ),
        "prior": """RetailCo Q3 2023 Earnings Call — Prepared Remarks

Thank you for joining us. Our third quarter results demonstrate continued
momentum in our core retail business.

Total company net sales were $6.4 billion, up 8.2% year-over-year.
Comparable store sales grew 5.8%, with transactions up 2.1% and average
ticket up 3.7%. E-commerce net sales grew 24%, now representing 19% of
total revenue.

Our loyalty membership program ended the quarter with 42 million active
members, up 18% year-over-year. Member spend was 2.3x non-member spend on
average. Digital app downloads grew 31% and app-driven transactions
accounted for 28% of e-commerce revenue.

Private label penetration increased to 34% of sales, up 220 basis points.
Private label gross margin was approximately 600 basis points higher than
branded merchandise, a meaningful mix tailwind.

Operating margin was 11.8%, up 60 basis points. Earnings per share of
$1.46 grew 14%. We opened 18 new stores this quarter bringing the full
fleet to 1,242 locations across North America.

We returned $340 million to shareholders through share repurchases and
paid $125 million in dividends. The board declared a quarterly dividend
of $0.38 per share, an 8% increase.
""",
        "current": """RetailCo Q3 2024 Earnings Call — Prepared Remarks

Thank you for joining us. The third quarter reflected a more complex
operating environment, and we are pleased with how our teams executed
against our top priorities.

Total company net sales were $6.7 billion, up 4.1% year-over-year.
Comparable store sales grew 1.8%, with transactions down 0.4% and average
ticket up 2.2%. Traffic trends softened mid-quarter and we adjusted
promotional cadence accordingly.

Gross margin expanded 140 basis points to 36.7%, driven by improved
merchandise margin, lower freight costs, and better shrink performance
than the prior year. Inventory ended the quarter down 6% year-over-year
on a unit basis, reflecting our disciplined inventory management.

Private label penetration increased to 36% of sales, up 180 basis points,
continuing to support margin expansion. SG&A grew 2.1%, well below sales
growth, as we realize efficiencies from our productivity program.

Operating margin was 12.3%, up 50 basis points. Earnings per share were
$1.58, up 8%. We opened 14 new stores this quarter bringing the fleet to
1,289 locations.

We returned $420 million to shareholders through share repurchases and
paid $135 million in dividends. The board declared a quarterly dividend
of $0.41 per share, an 8% increase.
""",
    },
}


# =============================================================================
# Initialise default transcripts on first load (TechCo high-MT)
# =============================================================================

if "prior_text" not in st.session_state:
    _first = next(iter(EXAMPLES.values()))
    st.session_state["prior_text"] = _first["prior"]
    st.session_state["current_text"] = _first["current"]


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown(
        '<div class="qs-sidebar-brand">'
        '<h2>📈 MT Analyzer</h2>'
        '<p>From Text to Alpha · Choi et al. 2026</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="qs-section">🔑 Authentication</div>', unsafe_allow_html=True)
    secret_key = load_secret_key()
    if secret_key:
        api_key = secret_key
        st.success("✅ Loaded from secrets", icon="🔒")
    else:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            label_visibility="collapsed",
        )

    st.markdown('<div class="qs-section">🧠 Extraction LLM</div>', unsafe_allow_html=True)
    llm_model = st.selectbox(
        "LLM",
        options=["gpt-5.4", "gpt-4.1", "gpt-4o"],
        index=0,  # default gpt-5.4 (frontier)
        label_visibility="collapsed",
        help="LLM ที่ใช้ดึง metrics จาก transcript. Paper ใช้ Gemini-2.5-Pro.",
    )

    st.markdown('<div class="qs-section">📐 Embedding Model</div>', unsafe_allow_html=True)
    embed_model = st.selectbox(
        "Embedder",
        options=["text-embedding-3-large", "text-embedding-3-small"],
        index=0,
        label_visibility="collapsed",
        help="Paper ใช้ 3-large (3072-dim). 3-small ถูกกว่า 7x แต่คุณภาพต่ำกว่า",
    )

    st.markdown('<div class="qs-section">⚙️ Thresholds (h function)</div>', unsafe_allow_html=True)
    alpha = st.slider(
        "α (drop below)",
        0.0, 1.0, 0.4, 0.05,
        help="cosine sim ≤ α → metric ถือว่า 'dropped'. Paper ใช้ 0.4",
    )
    beta = st.slider(
        "β (retain above)",
        0.0, 1.0, 0.6, 0.05,
        help="cosine sim ≥ β → metric ถือว่า 'retained'. Paper ใช้ 0.6",
    )
    if beta <= alpha:
        st.error("⚠️ β ต้อง > α")

    st.markdown('<div class="qs-section">📋 Examples</div>', unsafe_allow_html=True)
    example_name = st.selectbox(
        "Example scenario",
        options=list(EXAMPLES.keys()),
        index=0,
        label_visibility="collapsed",
        key="example_select",
    )
    st.caption(EXAMPLES[example_name]["description"])
    if st.button("📥 Load selected example", use_container_width=True, type="primary"):
        st.session_state["prior_text"] = EXAMPLES[example_name]["prior"]
        st.session_state["current_text"] = EXAMPLES[example_name]["current"]
        st.session_state.pop("result", None)
        st.rerun()

    st.markdown('<div class="qs-section">⚙️ Actions</div>', unsafe_allow_html=True)
    if st.button("🧹 Clear all", use_container_width=True):
        st.session_state["prior_text"] = ""
        st.session_state["current_text"] = ""
        st.session_state.pop("result", None)
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("📄 arXiv:2510.03195v4 · Mar 2026")
    st.caption("MT_i = 1 − (1/N_j) Σ h(max cos)")


# =============================================================================
# Hero
# =============================================================================

st.markdown(
    '<div class="qs-hero">'
    '<h1>📈 Moving Targets Analyzer</h1>'
    '<div class="tagline">LLM Extractor · Embedding Ruler · Alpha Signal</div>'
    '<div class="paper-ref">Choi et al. · arXiv:2510.03195v4 · Mar 2026</div>'
    '</div>',
    unsafe_allow_html=True,
)


# =============================================================================
# API key gate (landing)
# =============================================================================

if not api_key:
    st.info("👈 ใส่ OpenAI API Key ที่ sidebar เพื่อเริ่มวิเคราะห์")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="qs-feature"><div class="icon">📝</div>'
            '<h3>INPUT</h3><p>Paste earnings call transcripts 2 quarters '
            '(current + prior-year same quarter, i-4)</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="qs-feature"><div class="icon">⚙️</div>'
            '<h3>PROCESS</h3><p>LLM extract metrics → embed → cosine similarity → '
            'piecewise-linear h(α=0.4, β=0.6) → aggregate</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="qs-feature"><div class="icon">🎯</div>'
            '<h3>OUTPUT</h3><p>MT score ∈ [0, 1]. High = drop many metrics → '
            'bearish signal (paper: −0.52%/mo 5F alpha, t=−2.55)</p></div>',
            unsafe_allow_html=True,
        )
    st.stop()


# =============================================================================
# Input area
# =============================================================================

st.markdown(
    "Paste earnings call transcripts (or press **Load demo pair** in sidebar). "
    "Prior quarter should be the same quarter one year earlier (t minus 4)."
)

left, right = st.columns(2)
with left:
    st.markdown(
        '<div class="qs-input-label prior">◀  PRIOR QUARTER  (D_j, i−4)</div>',
        unsafe_allow_html=True,
    )
    prior_text = st.text_area(
        "Prior quarter transcript",
        height=320,
        label_visibility="collapsed",
        placeholder="Paste earnings call transcript from the SAME quarter one year ago...",
        key="prior_text",
    )
with right:
    st.markdown(
        '<div class="qs-input-label current">CURRENT QUARTER  (D_i)  ▶</div>',
        unsafe_allow_html=True,
    )
    current_text = st.text_area(
        "Current quarter transcript",
        height=320,
        label_visibility="collapsed",
        placeholder="Paste the latest earnings call transcript...",
        key="current_text",
    )

st.markdown("<br>", unsafe_allow_html=True)

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    analyze = st.button(
        "🔬 Extract & Score",
        type="primary",
        use_container_width=True,
        disabled=(not prior_text.strip() or not current_text.strip() or beta <= alpha),
    )


# =============================================================================
# Analyze
# =============================================================================

if analyze:
    try:
        llm = ChatOpenAI(model=llm_model, api_key=api_key, temperature=0.0)
        embedder = OpenAIEmbeddings(model=embed_model, api_key=api_key)

        with st.status("🔬 Running Moving Targets pipeline...", expanded=True) as status:
            st.write("📝 Extracting metrics from **prior** quarter...")
            prior_metrics = extract_metrics(prior_text, llm)
            st.write(f"   → extracted **{len(prior_metrics)}** metrics")

            st.write("📝 Extracting metrics from **current** quarter...")
            current_metrics = extract_metrics(current_text, llm)
            st.write(f"   → extracted **{len(current_metrics)}** metrics")

            st.write(f"📐 Embedding + similarity ({embed_model})...")
            result = compute_mt_score(
                current_metrics=current_metrics,
                prior_metrics=prior_metrics,
                embedder=embedder,
                alpha=alpha,
                beta=beta,
            )
            st.write(f"🎯 **MT score = {result['mt_score']:.4f}**")
            status.update(label="✅ Analysis complete", state="complete", expanded=False)

        st.session_state["result"] = result
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")


# =============================================================================
# Results
# =============================================================================

result: dict | None = st.session_state.get("result")

if result and result.get("mt_score") is not None:
    mt = result["mt_score"]
    label, interp = interpret_mt(mt)

    # Score color
    if mt < 0.4:
        score_color = "var(--qs-profit)"
    elif mt < 0.6:
        score_color = "var(--qs-warning)"
    else:
        score_color = "var(--qs-loss)"

    st.markdown(
        f'<div class="qs-mt-card">'
        f'<div class="qs-mt-label">MOVING TARGETS SCORE</div>'
        f'<div class="qs-mt-score" style="color: {score_color};">{mt:.3f}</div>'
        f'<div class="qs-mt-label" style="margin-top: 0.5rem;">{label}</div>'
        f'<div class="qs-mt-interp">{interp}</div>'
        f'<div class="qs-stat-row">'
        f'<div class="qs-stat">prior metrics <strong>{result["n_prior"]}</strong></div>'
        f'<div class="qs-stat">current metrics <strong>{result["n_current"]}</strong></div>'
        f'<div class="qs-stat retained">retained <strong>{result["n_retained"]}</strong></div>'
        f'<div class="qs-stat partial">partial <strong>{result["n_partial"]}</strong></div>'
        f'<div class="qs-stat dropped">dropped <strong>{result["n_dropped"]}</strong></div>'
        f'<div class="qs-stat">α/β <strong>{result["alpha"]:.2f} / {result["beta"]:.2f}</strong></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["📋 Per-metric breakdown", "🗂️ Extracted metrics", "🔥 Similarity heatmap"])

    with tab1:
        st.caption(
            "แต่ละ row = metric จาก prior quarter · best_match = metric ที่ match ดีสุดใน current · "
            "retention = h(max cosine similarity) ∈ [0, 1]"
        )
        rows = []
        for p in result["per_prior"]:
            status_emoji = (
                "🟢" if p["status"] == "retained"
                else "🟡" if p["status"] == "partial"
                else "🔴"
            )
            rows.append({
                "": status_emoji,
                "Prior metric": p["prior"],
                "Best match in current": p["best_match"],
                "Raw cosine": round(p["raw_similarity"], 3),
                "Retention h(·)": round(p["retention"], 3),
                "Status": p["status"],
            })
        # sort by retention ascending so dropped metrics surface first
        rows.sort(key=lambda r: r["Retention h(·)"])
        st.dataframe(rows, use_container_width=True, hide_index=True)

    with tab2:
        col_prior, col_curr = st.columns(2)
        with col_prior:
            st.markdown(
                f'<div class="qs-input-label prior">PRIOR ({result["n_prior"]})</div>',
                unsafe_allow_html=True,
            )
            st.markdown("\n".join(f"- {m}" for m in result["prior_metrics"]))
        with col_curr:
            st.markdown(
                f'<div class="qs-input-label current">CURRENT ({result["n_current"]})</div>',
                unsafe_allow_html=True,
            )
            st.markdown("\n".join(f"- {m}" for m in result["current_metrics"]))

    with tab3:
        st.caption(
            "แต่ละ cell = cosine similarity(prior_j, current_i) · "
            "แถวที่ 'มืด' = prior metric ที่ไม่มี close match ใน current (= dropped)"
        )
        sim = np.asarray(result["similarity_matrix"])
        fig = go.Figure(
            data=go.Heatmap(
                z=sim,
                x=result["current_metrics"],
                y=result["prior_metrics"],
                colorscale=[
                    [0.0, "#1D1D1D"],
                    [0.3, "#2E2E2E"],
                    [0.5, "#FFB74D"],
                    [0.7, "#69F0AE"],
                    [1.0, "#00C853"],
                ],
                zmin=0, zmax=1,
                hovertemplate="prior: %{y}<br>current: %{x}<br>cos: %{z:.3f}<extra></extra>",
                colorbar=dict(
                    title=dict(text="cos", font=dict(color="rgba(255,255,255,0.87)")),
                    tickfont=dict(color="rgba(255,255,255,0.87)"),
                ),
            )
        )
        fig.update_layout(
            height=max(400, 24 * len(result["prior_metrics"]) + 200),
            paper_bgcolor="#121212",
            plot_bgcolor="#121212",
            font=dict(family="Inter", color="rgba(255,255,255,0.87)", size=11),
            xaxis=dict(tickangle=-40, tickfont=dict(size=10)),
            yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
            margin=dict(l=20, r=20, t=20, b=120),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "📘 Framework: Choi et al. (2026) · MT_i = 1 − (1/N_j) Σ_j h(max_i cos(E_{j,n_j}, E_{i,n_i})) · "
        "paper Q5−Q1 long-short 5F alpha = **−0.52% / month** (t=−2.55)"
    )
elif result and result.get("warning"):
    st.warning(f"⚠️ {result['warning']}")
