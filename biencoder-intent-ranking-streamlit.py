"""
Run locally:
    streamlit run biencoder-intent-ranking-streamlit.py

Install:
    pip install -r requirements.txt
"""

import ast
import hashlib
import io
import json
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore")


st.set_page_config(
    page_title            = "Intent Classifier",
    page_icon             = "🔍",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    color: #1a1a2e;
}

/* NOTE: 'header' is intentionally NOT hidden.
   Streamlit's sidebar toggle button lives inside the header element.
   Hiding it locks users out when the sidebar collapses. */
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }

.app-header {
    display: flex; align-items: baseline; gap: 16px;
    margin-bottom: 2rem; padding-bottom: 1rem;
    border-bottom: 2px solid #1a1a2e;
}
.app-title {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem;
    font-weight: 600; color: #1a1a2e; letter-spacing: -0.5px;
}
.app-subtitle { font-size: 0.85rem; color: #6b7280; font-weight: 300; letter-spacing: 0.3px; }

.stTextArea textarea {
    font-family: 'IBM Plex Mono', monospace !important; font-size: 0.88rem !important;
    background: #fafafa !important; border: 1.5px solid #d1d5db !important;
    border-radius: 6px !important; color: #1a1a2e !important; line-height: 1.6 !important;
}
.stTextArea textarea:focus {
    border-color: #1a1a2e !important;
    box-shadow: 0 0 0 2px rgba(26,26,46,0.08) !important;
}

.stButton > button {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; font-weight: 600;
    letter-spacing: 0.5px; background: #1a1a2e; color: #ffffff; border: none;
    border-radius: 6px; padding: 0.55rem 1.4rem; transition: all 0.15s ease;
}
.stButton > button:hover {
    background: #2d2d4e; transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(26,26,46,0.2);
}
.stButton > button:active { transform: translateY(0); }

[data-testid="stSidebar"] { background: #f8f9fb; border-right: 1px solid #e5e7eb; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stFileUploader label {
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.6px;
    text-transform: uppercase; color: #6b7280;
}
[data-testid="stSidebar"] h3 {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem; font-weight: 600;
    color: #1a1a2e; margin-top: 1.4rem; margin-bottom: 0.4rem;
}

.route-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
    font-weight: 600; letter-spacing: 0.4px;
}
.route-CONFIRMED    { background:#d1fae5; color:#065f46; border:1px solid #6ee7b7; }
.route-NEEDS_REVIEW { background:#fef9c3; color:#713f12; border:1px solid #fde047; }
.route-AMBIGUOUS    { background:#fff7ed; color:#9a3412; border:1px solid #fdba74; }
.route-WEAK_SIGNAL  { background:#fee2e2; color:#991b1b; border:1px solid #fca5a5; }

.candidate-row {
    display: grid; grid-template-columns: 36px 70px 100px 1fr auto;
    align-items: center; gap: 10px; padding: 8px 12px; border-radius: 6px;
    margin-bottom: 4px; transition: background 0.1s;
}
.candidate-row:hover { background: #f3f4f6; }
.candidate-row.bucket-CONFIRMED    { border-left: 3px solid #10b981; background: #f0fdf4; }
.candidate-row.bucket-NEEDS_REVIEW { border-left: 3px solid #f59e0b; background: #fffbeb; }
.candidate-row.bucket-AMBIGUOUS    { border-left: 3px solid #f97316; background: #fff7ed; }
.candidate-row.bucket-WEAK_SIGNAL  { border-left: 3px solid #ef4444; background: #fef2f2; }
.cand-rank  { font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#9ca3af; font-weight:600; }
.cand-score { font-family:'IBM Plex Mono',monospace; font-size:0.82rem; font-weight:600; color:#1a1a2e; }
.cand-slug  { font-size:0.85rem; font-weight:500; color:#1a1a2e; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.score-bar  { height:6px; border-radius:3px; background:#e5e7eb; position:relative; min-width:80px; }
.score-fill { height:100%; border-radius:3px; transition:width 0.3s ease; }
.fill-CONFIRMED    { background: #10b981; }
.fill-NEEDS_REVIEW { background: #f59e0b; }
.fill-AMBIGUOUS    { background: #f97316; }
.fill-WEAK_SIGNAL  { background: #ef4444; }

.section-header {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 1px; text-transform: uppercase; color: #9ca3af;
    margin-top: 1.4rem; margin-bottom: 0.5rem; padding-bottom: 4px;
    border-bottom: 1px solid #f3f4f6;
}

.correct-win  { background:#d1fae5; color:#065f46; padding:3px 10px; border-radius:12px;
                font-size:0.75rem; font-weight:600; font-family:'IBM Plex Mono',monospace; }
.correct-top  { background:#fef9c3; color:#713f12; padding:3px 10px; border-radius:12px;
                font-size:0.75rem; font-weight:600; font-family:'IBM Plex Mono',monospace; }
.correct-miss { background:#fee2e2; color:#991b1b; padding:3px 10px; border-radius:12px;
                font-size:0.75rem; font-weight:600; font-family:'IBM Plex Mono',monospace; }

.metric-row { display:flex; gap:12px; margin-bottom:1.2rem; flex-wrap:wrap; }
.metric-card {
    flex:1; min-width:100px; background:#f8f9fb; border:1px solid #e5e7eb;
    border-radius:8px; padding:12px 16px; text-align:center;
}
.metric-val { font-family:'IBM Plex Mono',monospace; font-size:1.4rem; font-weight:600; color:#1a1a2e; }
.metric-lbl { font-size:0.7rem; color:#9ca3af; text-transform:uppercase; letter-spacing:0.5px; margin-top:2px; }

.info-box {
    background:#f0f9ff; border:1px solid #bae6fd; border-radius:8px;
    padding:12px 16px; margin-bottom:1rem; font-size:0.82rem; color:#0c4a6e; line-height:1.6;
}
.info-box code {
    background:#e0f2fe; padding:1px 5px; border-radius:3px;
    font-family:'IBM Plex Mono',monospace; font-size:0.78rem;
}

.upload-prompt {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    min-height: 50vh; text-align: center; gap: 1rem;
}
.upload-icon  { font-size: 3.5rem; line-height: 1; }
.upload-title { font-family: 'IBM Plex Mono', monospace; font-size: 1.3rem; font-weight: 600; color: #1a1a2e; }
.upload-sub   { font-size: 0.88rem; color: #6b7280; max-width: 480px; line-height: 1.6; }
.upload-columns {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 8px; margin-top: 0.5rem; width: 100%; max-width: 600px;
}
.upload-col-tag {
    background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 6px;
    padding: 6px 10px; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem; color: #475569;
}

hr { border: none; border-top: 1px solid #f3f4f6; margin: 1.2rem 0; }
.streamlit-expanderHeader {
    font-family:'IBM Plex Mono',monospace !important;
    font-size:0.78rem !important; font-weight:600 !important; color:#6b7280 !important;
}
</style>
""", unsafe_allow_html=True)


BIENCODER_GENERAL_ID = "sentence-transformers/all-mpnet-base-v2"
BIENCODER_QA_ID      = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

THRESHOLD_CONFIRMED = 0.55
THRESHOLD_REVIEW    = 0.50
THRESHOLD_AMBIGUOUS = 0.40

BUCKET_ICONS = {
    "CONFIRMED"    : "✅",
    "NEEDS_REVIEW" : "🟡",
    "AMBIGUOUS"    : "🟠",
    "WEAK_SIGNAL"  : "🔴",
}

REQUIRED_COLUMNS = {
    "user_input", "intent_detected", "intent_name",
    "intent_description", "positive_examples",
}


@dataclass
class Intent:
    slug        : str
    description : str
    examples    : list[str]      = field(default_factory=list)
    tenant_id   : Optional[str] = None


def _parse_json_array(val) -> list[str]:
    if not isinstance(val, str) or not val.strip():
        return []
    val = val.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(val)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            continue
    return [val] if val else []


def _parse_bool(val) -> bool:
    if isinstance(val, bool):         return val
    if isinstance(val, (int, float)): return bool(val)
    return str(val).strip().lower() in ("true", "1", "yes", "t")


def _file_hash(file_bytes: bytes) -> str:
    """
    Stable cache key for an uploaded file.

    Uses MD5 of the raw bytes. This replaces os.path.getmtime() which only
    works on local filesystem paths — uploaded files have no path on the server.

    Two uploads of the same file → same hash → cache hit, no re-embedding.
    A changed file → different hash → cache miss → vectors recomputed.
    """
    return hashlib.md5(file_bytes).hexdigest()


@st.cache_data(show_spinner=False)
def load_intents_from_csv(
    file_bytes : bytes,
    _file_hash : str,
    tenant_id  : Optional[str] = None,
) -> tuple[list[Intent], pd.DataFrame]:
    """
    Parse CSV bytes into (list[Intent], raw DataFrame).

    ALL confirmed user_inputs + ALL positive_examples aggregated per intent.
    No cap on examples — more confirmed rows = richer centroid vectors.
    """
    df = pd.read_csv(io.BytesIO(file_bytes), dtype=str)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Your CSV is missing required columns: {sorted(missing)}\n"
            f"Columns found: {sorted(df.columns.tolist())}"
        )

    if tenant_id and "tenant_id" in df.columns:
        df = df[df["tenant_id"].str.strip() == tenant_id].copy()

    df["intent_detected_bool"] = df["intent_detected"].apply(_parse_bool)
    df["user_input"]           = df["user_input"].fillna("").str.strip()
    df["intent_name"]          = df["intent_name"].fillna("").str.strip()
    df["intent_description"]   = df["intent_description"].fillna("").str.strip()

    intents: list[Intent] = []
    for name in [n for n in df["intent_name"].unique() if n]:
        rows = df[df["intent_name"] == name]

        description = next(
            (d.strip() for d in rows["intent_description"] if str(d).strip()), None
        )
        if not description:
            continue

        confirmed_inputs = (
            rows[rows["intent_detected_bool"]]["user_input"]
            .dropna().str.strip().loc[lambda s: s != ""]
            .drop_duplicates().tolist()
        )

        pos_examples: list[str] = []
        seen_pos: set[str]      = set()
        for val in rows["positive_examples"]:
            for ex in _parse_json_array(val):
                if ex not in seen_pos:
                    seen_pos.add(ex)
                    pos_examples.append(ex)

        combined: list[str] = []
        seen_c: set[str]    = set()
        for text in confirmed_inputs + pos_examples:
            t = text.strip()
            if t and t not in seen_c:
                seen_c.add(t)
                combined.append(t)

        tid = None
        if "tenant_id" in rows.columns:
            tids = rows["tenant_id"].dropna().unique()
            tid  = tids[0].strip() if len(tids) else None

        intents.append(Intent(
            slug=name, description=description, examples=combined, tenant_id=tid,
        ))

    return intents, df


def sample_test_inputs(
    df : pd.DataFrame,
    n  : int = 2,
) -> tuple[list[tuple[str, str]], list[str]]:
    positives: list[tuple[str, str]] = []
    seen: set[str] = set()

    for name in df["intent_name"].unique():
        if not name:
            continue
        rows = df[
            (df["intent_name"] == name) &
            (df["intent_detected_bool"] == True) &
            (df["user_input"].str.strip() != "")
        ]
        for inp in rows["user_input"].drop_duplicates().head(n).tolist():
            if inp not in seen:
                seen.add(inp)
                positives.append((inp, name))

    all_true = set(
        df[df["intent_detected_bool"] == True]["user_input"].str.strip().unique()
    )
    negatives = (
        df[
            (df["intent_detected_bool"] == False) &
            (~df["user_input"].str.strip().isin(all_true)) &
            (df["user_input"].str.strip() != "")
        ]["user_input"].drop_duplicates().head(3).tolist()
    )
    return positives, negatives


def _embed_texts(
    texts             : list[str],
    biencoder_general : SentenceTransformer,
    biencoder_qa      : Optional[SentenceTransformer],
    mode              : str,
) -> np.ndarray:
    def _mean_norm(enc: SentenceTransformer) -> np.ndarray:
        vecs = enc.encode(
            texts, normalize_embeddings=True,
            show_progress_bar=False, batch_size=128,
        )
        m = vecs.mean(axis=0)
        n = np.linalg.norm(m)
        return m / n if n > 0 else m

    vec = _mean_norm(biencoder_general)
    if mode == "ensemble" and biencoder_qa is not None:
        vq   = _mean_norm(biencoder_qa)
        comb = np.concatenate([vec, vq])
        n    = np.linalg.norm(comb)
        return comb / n if n > 0 else comb
    return vec


def get_vectors(
    intents           : list[Intent],
    file_hash         : str,
    encoder_mode      : str,
    biencoder_general : SentenceTransformer,
    biencoder_qa      : Optional[SentenceTransformer],
    progress_fn       = None,
) -> dict[str, np.ndarray]:
    cache_key = f"vectors__{file_hash}__{encoder_mode}"

    if cache_key in st.session_state:
        return st.session_state[cache_key]

    vectors: dict[str, np.ndarray] = {}
    total = len(intents)
    for i, intent in enumerate(intents):
        vectors[intent.slug] = _embed_texts(
            [intent.description] + intent.examples,
            biencoder_general, biencoder_qa, encoder_mode,
        )
        if progress_fn:
            progress_fn((i + 1) / total, intent.slug)

    st.session_state[cache_key] = vectors
    return vectors


def bucket_score(score: float) -> str:
    if   score >= THRESHOLD_CONFIRMED:  return "CONFIRMED"
    elif score >= THRESHOLD_REVIEW:     return "NEEDS_REVIEW"
    elif score >= THRESHOLD_AMBIGUOUS:  return "AMBIGUOUS"
    else:                               return "WEAK_SIGNAL"


def cosine_search(
    query_vec    : np.ndarray,
    vectors      : dict[str, np.ndarray],
    intents_dict : dict[str, Intent],
    top_k        : int,
) -> list[tuple[Intent, float]]:
    if not vectors:
        return []
    slugs  = list(vectors.keys())
    matrix = np.stack([vectors[s] for s in slugs])
    sims   = matrix @ query_vec
    k      = min(top_k, len(slugs))
    idx    = np.argpartition(sims, -k)[-k:]
    idx    = idx[np.argsort(sims[idx])[::-1]]
    return [(intents_dict[slugs[i]], float(sims[i])) for i in idx]


def run_classification(
    user_input        : str,
    intents_dict      : dict[str, Intent],
    vectors           : dict[str, np.ndarray],
    biencoder_general : SentenceTransformer,
    biencoder_qa      : Optional[SentenceTransformer],
    mode              : str,
    top_k             : int = 8,
) -> list[dict]:
    q       = _embed_texts([user_input], biencoder_general, biencoder_qa, mode)
    results = cosine_search(q, vectors, intents_dict, top_k)
    return [
        {
            "rank"        : i + 1,
            "slug"        : intent.slug,
            "description" : intent.description,
            "score"       : round(float(score), 4),
            "bucket"      : bucket_score(float(score)),
        }
        for i, (intent, score) in enumerate(results)
    ]


@st.cache_resource(show_spinner=False)
def load_biencoder_general() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(BIENCODER_GENERAL_ID, device=device)


@st.cache_resource(show_spinner=False)
def load_biencoder_qa() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(BIENCODER_QA_ID, device=device)


def render_candidate_row(c: dict) -> None:
    bucket = c["bucket"]
    score  = c["score"]
    bar_w  = int(score * 100)
    st.markdown(f"""
    <div class="candidate-row bucket-{bucket}">
        <span class="cand-rank">#{c['rank']}</span>
        <span class="cand-score">{score:.4f}</span>
        <div class="score-bar">
            <div class="score-fill fill-{bucket}" style="width:{bar_w}%"></div>
        </div>
        <span class="cand-slug">{c['slug']}</span>
        <span class="route-badge route-{bucket}">{BUCKET_ICONS[bucket]} {bucket}</span>
    </div>
    """, unsafe_allow_html=True)


def render_result(
    candidates    : list[dict],
    expected_slug : Optional[str] = None,
) -> None:
    if not candidates:
        st.warning("No candidates returned.")
        return

    top1      = candidates[0]
    confirmed = [c for c in candidates if c["bucket"] == "CONFIRMED"]
    review    = [c for c in candidates if c["bucket"] == "NEEDS_REVIEW"]
    ambiguous = [c for c in candidates if c["bucket"] == "AMBIGUOUS"]
    weak      = [c for c in candidates if c["bucket"] == "WEAK_SIGNAL"]
    route     = top1["bucket"]

    col_route, col_correct = st.columns([2, 3])
    with col_route:
        st.markdown(
            f'<span class="route-badge route-{route}">'
            f'{BUCKET_ICONS[route]} {route}  •  top-1: {top1["score"]:.4f}'
            f'</span>',
            unsafe_allow_html=True,
        )

    if expected_slug:
        confirmed_slugs = [c["slug"] for c in confirmed]
        all_slugs       = [c["slug"] for c in candidates]
        with col_correct:
            if expected_slug in confirmed_slugs:
                st.markdown(
                    f'<span class="correct-win">✓ CONFIRMED MATCH — {expected_slug}</span>',
                    unsafe_allow_html=True,
                )
            elif expected_slug in all_slugs:
                b = next(c["bucket"] for c in candidates if c["slug"] == expected_slug)
                st.markdown(
                    f'<span class="correct-top">◑ In top-K ({b}) — {expected_slug}</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span class="correct-miss">✗ Not in top-K — expected: {expected_slug}</span>',
                    unsafe_allow_html=True,
                )

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-val" style="color:#065f46">{len(confirmed)}</div>
            <div class="metric-lbl">Confirmed</div>
        </div>
        <div class="metric-card">
            <div class="metric-val" style="color:#92400e">{len(review)}</div>
            <div class="metric-lbl">Needs Review</div>
        </div>
        <div class="metric-card">
            <div class="metric-val" style="color:#9a3412">{len(ambiguous)}</div>
            <div class="metric-lbl">Ambiguous</div>
        </div>
        <div class="metric-card">
            <div class="metric-val" style="color:#991b1b">{len(weak)}</div>
            <div class="metric-lbl">Weak Signal</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if confirmed:
        st.markdown('<div class="section-header">CONFIRMED INTENTS</div>', unsafe_allow_html=True)
        for c in confirmed:
            render_candidate_row(c)
            with st.expander(f"Description — {c['slug']}", expanded=False):
                st.caption(c["description"])

    st.markdown('<div class="section-header">ALL CANDIDATES</div>', unsafe_allow_html=True)
    for c in candidates:
        render_candidate_row(c)


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📂 Upload CSV")

    uploaded_file = st.file_uploader(
        "Training dataset",
        type = ["csv"],
        help = (
            "Upload your intent training CSV.\n\n"
            "Required columns:\n"
            "user_input · intent_detected · intent_name · "
            "intent_description · positive_examples"
        ),
    )

    st.markdown("### ⚙️ Settings")

    encoder_mode = st.selectbox(
        "Encoder Mode",
        options = ["ensemble", "single"],
        index   = 0,
        help    = (
            "ensemble — mpnet + multi-qa concatenated (1536-dim, more accurate)\n"
            "single   — mpnet only (768-dim, faster to embed)"
        ),
    )

    top_k = st.slider("Top-K candidates", min_value=3, max_value=20, value=8)

    st.markdown("### 🎚 Thresholds")
    st.markdown(
        f"""
        <div style="font-size:0.78rem; color:#6b7280; line-height:2.2;">
        <span class="route-badge route-CONFIRMED" style="font-size:0.7rem">CONFIRMED</span>
        &nbsp;≥ <code>{THRESHOLD_CONFIRMED}</code><br>
        <span class="route-badge route-NEEDS_REVIEW" style="font-size:0.7rem">NEEDS_REVIEW</span>
        &nbsp;≥ <code>{THRESHOLD_REVIEW}</code><br>
        <span class="route-badge route-AMBIGUOUS" style="font-size:0.7rem">AMBIGUOUS</span>
        &nbsp;≥ <code>{THRESHOLD_AMBIGUOUS}</code><br>
        <span class="route-badge route-WEAK_SIGNAL" style="font-size:0.7rem">WEAK_SIGNAL</span>
        &nbsp;&lt; <code>{THRESHOLD_AMBIGUOUS}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Edit threshold constants at the top of the script to change these.")

    st.markdown("### 📊 Test Samples")
    samples_per_intent = st.slider("Samples per intent", 1, 5, 2)
    show_negatives     = st.checkbox("Show negative samples", value=True)

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem;color:#9ca3af;line-height:1.8;">'
        'Vectors are cached in memory for your session.<br>'
        'Re-uploading the same file reuses the cache instantly.<br>'
        'Uploading a new file or switching encoder mode triggers a rebuild.'
        '</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  APP HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
    <span class="app-title">INTENT CLASSIFIER</span>
    <span class="app-subtitle">Stage 1 · MPNet bi-encoder · threshold-based routing</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  UPLOAD GATE — friendly landing screen shown until a CSV is uploaded
# ─────────────────────────────────────────────────────────────────────────────

if uploaded_file is None:
    st.markdown("""
    <div class="upload-prompt">
        <div class="upload-icon">📋</div>
        <div class="upload-title">Upload your intent dataset to get started</div>
        <div class="upload-sub">
            Use the <strong>Upload CSV</strong> panel in the sidebar.<br>
            Your CSV must contain the following columns:
        </div>
        <div class="upload-columns">
            <div class="upload-col-tag">user_input</div>
            <div class="upload-col-tag">intent_detected</div>
            <div class="upload-col-tag">intent_name</div>
            <div class="upload-col-tag">intent_description</div>
            <div class="upload-col-tag">positive_examples</div>
        </div>
        <div class="upload-sub" style="margin-top:0.5rem; font-size:0.78rem;">
            Vectors are computed in memory and cached for your browser session.<br>
            Nothing is stored on the server between sessions.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
#  DATA PIPELINE — runs once per unique (file_content × encoder_mode)
# ─────────────────────────────────────────────────────────────────────────────

# Read bytes once. All downstream operations use this in-memory buffer.
file_bytes = uploaded_file.read()
fhash      = _file_hash(file_bytes)

# Load models — @st.cache_resource means this happens once per server process,
# shared across all users. Downloading ~400 MB of model weights only happens
# on the very first cold start.
with st.spinner("Loading bi-encoder models…"):
    be_general = load_biencoder_general()
    be_qa      = load_biencoder_qa() if encoder_mode == "ensemble" else None

# Parse CSV — @st.cache_data keys on (file_bytes, fhash) so re-uploading the
# same file skips re-parsing entirely.
with st.spinner(f"Parsing `{uploaded_file.name}`…"):
    try:
        intents_list, raw_df = load_intents_from_csv(file_bytes, fhash)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")
        st.stop()

if not intents_list:
    st.error(
        "No valid intents found in the uploaded file. "
        "Check that at least one row has a non-empty `intent_description`."
    )
    st.stop()

intents_dict = {i.slug: i for i in intents_list}

# Build / retrieve vectors from session_state
vector_key = f"vectors__{fhash}__{encoder_mode}"
if vector_key not in st.session_state:
    prog   = st.progress(0, text="Computing intent vectors…")
    status = st.empty()

    def _update_progress(pct: float, slug: str) -> None:
        prog.progress(pct, text=f"Embedding intents… {int(pct * 100)}%")
        status.caption(f"Processing: {slug}")

    try:
        vectors = get_vectors(
            intents_list, fhash, encoder_mode,
            be_general, be_qa,
            progress_fn=_update_progress,
        )
    except Exception as e:
        st.error(f"Failed to build vectors: {e}")
        st.stop()

    prog.empty()
    status.empty()
    st.toast(f"✅ {len(intents_list)} intents ready", icon="✅")
else:
    vectors = st.session_state[vector_key]

# Status row
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Intents",      len(intents_list))
col_b.metric("True rows",    int(raw_df["intent_detected_bool"].sum()))
col_c.metric("Encoder mode", encoder_mode)
col_d.metric("Vector dims",  "1536" if encoder_mode == "ensemble" else "768")

st.markdown("---")


#  TABS

tab_classify, tab_test, tab_intents = st.tabs([
    "🔍  Classify",
    "🧪  Test Samples",
    "📋  Intent Catalogue",
])


# Tab 1: Manual classification
with tab_classify:
    st.markdown(
        '<div class="info-box">'
        'Enter any text. The classifier scores it against all intents using cosine '
        'similarity and independently buckets each result by threshold. Multiple '
        'intents can be <code>CONFIRMED</code> simultaneously — a result is a win '
        'if the correct intent appears anywhere in the CONFIRMED bucket.'
        '</div>',
        unsafe_allow_html=True,
    )

    user_input = st.text_area(
        "User input",
        height           = 120,
        placeholder      = "Paste or type the text you want to classify…",
        label_visibility = "collapsed",
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        run_btn = st.button("Classify ▶", use_container_width=True)

    if run_btn:
        if not user_input.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Classifying…"):
                try:
                    candidates = run_classification(
                        user_input, intents_dict, vectors,
                        be_general, be_qa, encoder_mode, top_k,
                    )
                    render_result(candidates)
                except Exception as e:
                    st.error(f"Classification error: {e}")


# Tab 2: Auto-sampled test inputs
with tab_test:
    st.markdown(
        '<div class="info-box">'
        f'Samples are drawn directly from your uploaded CSV — '
        f'<code>{samples_per_intent}</code> confirmed rows per intent as positives, '
        'plus pure negatives (inputs never labelled True for any intent). '
        'A positive is a <strong>win</strong> if the expected intent appears in '
        'the CONFIRMED bucket — not just at rank 1.'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.button("Run test samples ▶"):
        positives, negatives = sample_test_inputs(raw_df, n=samples_per_intent)

        st.markdown(
            f"**{len(positives)} positive samples &nbsp;·&nbsp; "
            f"{len(negatives)} negative samples**"
        )
        st.markdown("---")

        correct_count = 0
        topk_count    = 0
        total         = len(positives)

        st.markdown("#### Positive Samples")
        for inp, expected in positives:
            label = f"{inp[:90]}{'…' if len(inp) > 90 else ''}  ·  expected: {expected}"
            with st.expander(label, expanded=False):
                try:
                    cands           = run_classification(
                        inp, intents_dict, vectors,
                        be_general, be_qa, encoder_mode, top_k,
                    )
                    confirmed_slugs = [c["slug"] for c in cands if c["bucket"] == "CONFIRMED"]
                    all_slugs       = [c["slug"] for c in cands]

                    if expected in confirmed_slugs:
                        correct_count += 1
                        topk_count    += 1
                    elif expected in all_slugs:
                        topk_count += 1

                    render_result(cands, expected_slug=expected)
                except Exception as e:
                    st.error(str(e))

        st.markdown("---")
        st.markdown("#### Summary")
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Confirmed correct", f"{correct_count}/{total}",
            f"{correct_count / total * 100:.0f}%" if total else "—",
        )
        m2.metric(
            "In top-K", f"{topk_count}/{total}",
            f"{topk_count / total * 100:.0f}%" if total else "—",
        )
        m3.metric("Missed entirely", f"{total - topk_count}/{total}")

        if show_negatives and negatives:
            st.markdown("#### Negative Samples")
            st.caption(
                "These inputs were never labelled True for any intent — "
                "expect an empty CONFIRMED bucket."
            )
            for neg in negatives:
                with st.expander(
                    f"{neg[:90]}{'…' if len(neg) > 90 else ''}",
                    expanded=False,
                ):
                    try:
                        cands = run_classification(
                            neg, intents_dict, vectors,
                            be_general, be_qa, encoder_mode, top_k,
                        )
                        render_result(cands)
                    except Exception as e:
                        st.error(str(e))


# Tab 3: Intent catalogue
with tab_intents:
    st.markdown(
        f"**{len(intents_list)} intents** loaded from `{uploaded_file.name}`"
    )

    search_q = st.text_input(
        "Search intents",
        placeholder = "Filter by name or description…",
    )

    filtered = [
        i for i in intents_list
        if not search_q
        or search_q.lower() in i.slug.lower()
        or search_q.lower() in i.description.lower()
    ]

    st.caption(f"Showing {len(filtered)} of {len(intents_list)} intents")

    for intent in filtered:
        with st.expander(intent.slug):
            st.markdown("**Description**")
            st.text(intent.description)
            st.markdown(f"**Examples** ({len(intent.examples)} total)")
            if intent.examples:
                for ex in intent.examples[:5]:
                    st.caption(f"• {ex[:120]}{'…' if len(ex) > 120 else ''}")
                if len(intent.examples) > 5:
                    st.caption(f"  … and {len(intent.examples) - 5} more")
            if intent.tenant_id:
                st.caption(f"Tenant: `{intent.tenant_id}`")