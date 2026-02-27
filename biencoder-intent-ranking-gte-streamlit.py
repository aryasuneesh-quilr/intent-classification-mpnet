"""
streamlit run biencoder-intent-ranking-gte-streamlit.py

pip install streamlit sentence-transformers torch pandas huggingface_hub fdclient
"""

import ast
import json
import os
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer

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

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; color: #1a1a2e; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1280px; }

.app-header { display:flex; align-items:baseline; gap:16px; margin-bottom:2rem;
    padding-bottom:1rem; border-bottom:2px solid #1a1a2e; }
.app-title { font-family:'IBM Plex Mono',monospace; font-size:1.6rem; font-weight:600;
    color:#1a1a2e; letter-spacing:-0.5px; }
.app-subtitle { font-size:0.85rem; color:#6b7280; font-weight:300; letter-spacing:0.3px; }

.stTextArea textarea { font-family:'IBM Plex Mono',monospace !important; font-size:0.88rem !important;
    background:#fafafa !important; border:1.5px solid #d1d5db !important;
    border-radius:6px !important; color:#1a1a2e !important; line-height:1.6 !important; }
.stTextArea textarea:focus { border-color:#1a1a2e !important;
    box-shadow:0 0 0 2px rgba(26,26,46,0.08) !important; }

.stButton > button { font-family:'IBM Plex Mono',monospace; font-size:0.82rem; font-weight:600;
    letter-spacing:0.5px; background:#1a1a2e; color:#fff; border:none; border-radius:6px;
    padding:0.55rem 1.4rem; transition:all 0.15s ease; }
.stButton > button:hover { background:#2d2d4e; transform:translateY(-1px);
    box-shadow:0 4px 12px rgba(26,26,46,0.2); }
.stButton > button:active { transform:translateY(0); }

[data-testid="stSidebar"] { background:#f8f9fb; border-right:1px solid #e5e7eb; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stTextInput label { font-size:0.78rem; font-weight:600;
    letter-spacing:0.6px; text-transform:uppercase; color:#6b7280; }
[data-testid="stSidebar"] h3 { font-family:'IBM Plex Mono',monospace; font-size:0.9rem;
    font-weight:600; color:#1a1a2e; margin-top:1.4rem; margin-bottom:0.4rem; }

.route-badge { display:inline-flex; align-items:center; gap:6px; padding:4px 12px;
    border-radius:20px; font-family:'IBM Plex Mono',monospace; font-size:0.78rem;
    font-weight:600; letter-spacing:0.4px; }
.route-CONFIRMED    { background:#d1fae5; color:#065f46; border:1px solid #6ee7b7; }
.route-NEEDS_REVIEW { background:#fef9c3; color:#713f12; border:1px solid #fde047; }
.route-AMBIGUOUS    { background:#fff7ed; color:#9a3412; border:1px solid #fdba74; }
.route-WEAK_SIGNAL  { background:#fee2e2; color:#991b1b; border:1px solid #fca5a5; }

.candidate-row { display:grid; grid-template-columns:36px 70px 100px 1fr auto;
    align-items:center; gap:10px; padding:8px 12px; border-radius:6px;
    margin-bottom:4px; transition:background 0.1s; }
.candidate-row:hover { background:#f3f4f6; }
.candidate-row.bucket-CONFIRMED    { border-left:3px solid #10b981; background:#f0fdf4; }
.candidate-row.bucket-NEEDS_REVIEW { border-left:3px solid #f59e0b; background:#fffbeb; }
.candidate-row.bucket-AMBIGUOUS    { border-left:3px solid #f97316; background:#fff7ed; }
.candidate-row.bucket-WEAK_SIGNAL  { border-left:3px solid #ef4444; background:#fef2f2; }
.cand-rank  { font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#9ca3af; font-weight:600; }
.cand-score { font-family:'IBM Plex Mono',monospace; font-size:0.82rem; font-weight:600; color:#1a1a2e; }
.cand-slug  { font-size:0.85rem; font-weight:500; color:#1a1a2e; white-space:nowrap;
    overflow:hidden; text-overflow:ellipsis; }
.cand-desc  { font-size:0.75rem; color:#6b7280; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.score-bar  { height:6px; border-radius:3px; background:#e5e7eb; position:relative; min-width:80px; }
.score-fill { height:100%; border-radius:3px; transition:width 0.3s ease; }
.fill-CONFIRMED    { background:#10b981; }
.fill-NEEDS_REVIEW { background:#f59e0b; }
.fill-AMBIGUOUS    { background:#f97316; }
.fill-WEAK_SIGNAL  { background:#ef4444; }

.section-header { font-family:'IBM Plex Mono',monospace; font-size:0.72rem; font-weight:600;
    letter-spacing:1px; text-transform:uppercase; color:#9ca3af; margin-top:1.4rem;
    margin-bottom:0.5rem; padding-bottom:4px; border-bottom:1px solid #f3f4f6; }

.correct-win  { background:#d1fae5; color:#065f46; padding:3px 10px; border-radius:12px;
    font-size:0.75rem; font-weight:600; font-family:'IBM Plex Mono',monospace; }
.correct-top  { background:#fef9c3; color:#713f12; padding:3px 10px; border-radius:12px;
    font-size:0.75rem; font-weight:600; font-family:'IBM Plex Mono',monospace; }
.correct-miss { background:#fee2e2; color:#991b1b; padding:3px 10px; border-radius:12px;
    font-size:0.75rem; font-weight:600; font-family:'IBM Plex Mono',monospace; }

.metric-row { display:flex; gap:12px; margin-bottom:1.2rem; flex-wrap:wrap; }
.metric-card { flex:1; min-width:100px; background:#f8f9fb; border:1px solid #e5e7eb;
    border-radius:8px; padding:12px 16px; text-align:center; }
.metric-val { font-family:'IBM Plex Mono',monospace; font-size:1.4rem; font-weight:600; color:#1a1a2e; }
.metric-lbl { font-size:0.7rem; color:#9ca3af; text-transform:uppercase; letter-spacing:0.5px; margin-top:2px; }

.info-box { background:#f0f9ff; border:1px solid #bae6fd; border-radius:8px;
    padding:12px 16px; margin-bottom:1rem; font-size:0.82rem; color:#0c4a6e; line-height:1.6; }
.info-box code { background:#e0f2fe; padding:1px 5px; border-radius:3px;
    font-family:'IBM Plex Mono',monospace; font-size:0.78rem; }

/* ── Individual vector mode ── */
.indiv-example-row { display:grid; grid-template-columns:60px 100px 1fr 80px;
    gap:8px; align-items:center; padding:5px 10px; border-radius:4px;
    margin-bottom:3px; font-size:0.78rem; }
.indiv-example-row:hover { background:#f9fafb; }
.ex-score { font-family:'IBM Plex Mono',monospace; font-weight:600; color:#374151; }
.ex-type  { font-size:0.7rem; padding:2px 7px; border-radius:10px; font-weight:600; }
.ex-type-description { background:#ede9fe; color:#5b21b6; }
.ex-type-example     { background:#dbeafe; color:#1e40af; }
.ex-type-user_input  { background:#d1fae5; color:#065f46; }
.ex-text  { color:#4b5563; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.ex-match { font-family:'IBM Plex Mono',monospace; font-size:0.7rem; padding:2px 6px;
    border-radius:10px; font-weight:600; }
.ex-match-yes { background:#d1fae5; color:#065f46; }
.ex-match-no  { background:#f3f4f6; color:#9ca3af; }

/* ── Vote bar ── */
.vote-bar-wrap { background:#f3f4f6; border-radius:4px; height:8px; margin:4px 0; }
.vote-bar-fill { height:100%; border-radius:4px; background:#10b981; transition:width 0.4s ease; }

/* ── Compare panel ── */
.compare-header { font-family:'IBM Plex Mono',monospace; font-size:0.75rem; font-weight:600;
    color:#6b7280; letter-spacing:0.8px; text-transform:uppercase;
    padding:6px 12px; background:#f8f9fb; border-radius:6px 6px 0 0;
    border:1px solid #e5e7eb; border-bottom:none; }
.compare-body { border:1px solid #e5e7eb; border-radius:0 6px 6px 6px; padding:12px; }

.mode-badge { display:inline-block; padding:2px 8px; border-radius:4px;
    font-family:'IBM Plex Mono',monospace; font-size:0.7rem; font-weight:600; }
.mode-centroid   { background:#ede9fe; color:#5b21b6; }
.mode-individual { background:#dcfce7; color:#166534; }
.mode-gte        { background:#fff7ed; color:#9a3412; }

hr { border:none; border-top:1px solid #f3f4f6; margin:1.2rem 0; }
.streamlit-expanderHeader { font-family:'IBM Plex Mono',monospace !important;
    font-size:0.78rem !important; font-weight:600 !important; color:#6b7280 !important; }
</style>
""", unsafe_allow_html=True)

# Constants

BIENCODER_GENERAL_ID = "sentence-transformers/all-mpnet-base-v2"
BIENCODER_QA_ID      = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
CACHE_DIR            = "./vector_cache"
GTE_ENDPOINT_DEFAULT = "https://dlppreprod.quilr.ai/embed"
GTE_BATCH_SIZE       = 64
GTE_REQUEST_TIMEOUT  = 10

THRESHOLD_CONFIRMED  = 0.55
THRESHOLD_REVIEW     = 0.50
THRESHOLD_AMBIGUOUS  = 0.40

# For individual mode: fraction of examples that must match to CONFIRM
INDIV_VOTE_THRESHOLD = 0.30   # ≥30% of examples must exceed match_threshold
INDIV_MATCH_SIM      = 0.45   # individual example similarity to count as a "vote"

BUCKET_ICONS = {
    "CONFIRMED"    : "✅",
    "NEEDS_REVIEW" : "🟡",
    "AMBIGUOUS"    : "🟠",
    "WEAK_SIGNAL"  : "🔴",
}


@dataclass
class Intent:
    slug        : str
    description : str
    examples    : list[str]      = field(default_factory=list)
    tenant_id   : Optional[str] = None


@dataclass
class IndividualVector:
    """One text from the CSV stored as its own vector."""
    text    : str
    kind    : str        # "description" | "example" | "user_input"
    vector  : np.ndarray = field(repr=False)


class GTEClient:
    """
    Drop-in for SentenceTransformer that calls the GTE FDClient microservice.
    FDClient.infer() returns {'success': True, 'prediction': [array, ...], ...}
    """
    def __init__(self, endpoint: str, timeout: int = GTE_REQUEST_TIMEOUT):
        self.endpoint = endpoint
        self.timeout  = timeout
        self._client  = None

    def _get_client(self):
        if self._client is None:
            from fdclient import FDClient
            self._client = FDClient(self.endpoint, request_timeout=self.timeout)
        return self._client

    def encode(
        self,
        sentences            : list[str],
        normalize_embeddings : bool = True,
        show_progress_bar    : bool = False,
        batch_size           : int  = GTE_BATCH_SIZE,
        **kwargs,
    ) -> np.ndarray:
        client    = self._get_client()
        all_vecs  : list[np.ndarray] = []
        n_batches = (len(sentences) + batch_size - 1) // batch_size
        for i in range(n_batches):
            chunk  = sentences[i * batch_size : (i + 1) * batch_size]
            result = client.infer(chunk)
            # FDClient returns a dict: {'success': True, 'prediction': [...]}
            if isinstance(result, dict):
                embeddings = result["prediction"]
            else:
                embeddings = result
            batch_arr = np.array(embeddings, dtype=np.float32)
            if normalize_embeddings:
                norms = np.linalg.norm(batch_arr, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                batch_arr = batch_arr / norms
            all_vecs.append(batch_arr)
        return np.vstack(all_vecs)

    def ping(self) -> bool:
        try:
            result = self._get_client().infer(["ping"])
            # FDClient returns a dict with 'success' key
            if isinstance(result, dict):
                return result.get("success", False)
            return isinstance(result, (list, np.ndarray)) and len(result) > 0
        except Exception as e:
            return False


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
    if isinstance(val, bool):            return val
    if isinstance(val, (int, float)):    return bool(val)
    return str(val).strip().lower() in ("true", "1", "yes", "t")


def bucket_score(score: float) -> str:
    if   score >= THRESHOLD_CONFIRMED:  return "CONFIRMED"
    elif score >= THRESHOLD_REVIEW:     return "NEEDS_REVIEW"
    elif score >= THRESHOLD_AMBIGUOUS:  return "AMBIGUOUS"
    else:                               return "WEAK_SIGNAL"


def load_intents_from_csv(
    csv_path  : str,
    tenant_id : Optional[str] = None,
) -> tuple[list[Intent], pd.DataFrame]:
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    required = {"user_input", "intent_detected", "intent_name",
                "intent_description", "positive_examples"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

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
            slug=name, description=description, examples=combined, tenant_id=tid
        ))

    return intents, df


def load_intents_with_user_inputs(
    csv_path  : str,
    tenant_id : Optional[str] = None,
) -> tuple[list[Intent], pd.DataFrame, dict[str, list[str]]]:
    """
    Like load_intents_from_csv but also returns confirmed user_inputs
    separately so IndividualMode can tag them by kind.
    Returns (intents, df, {slug: [confirmed_user_inputs]})
    """
    intents, df = load_intents_from_csv(csv_path, tenant_id)
    user_inputs_by_slug: dict[str, list[str]] = {}
    for intent in intents:
        rows = df[df["intent_name"] == intent.slug]
        ui = (
            rows[rows["intent_detected_bool"]]["user_input"]
            .dropna().str.strip().loc[lambda s: s != ""]
            .drop_duplicates().tolist()
        )
        user_inputs_by_slug[intent.slug] = ui
    return intents, df, user_inputs_by_slug


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

    all_true = set(df[df["intent_detected_bool"] == True]["user_input"].str.strip().unique())
    negatives = (
        df[
            (df["intent_detected_bool"] == False) &
            (~df["user_input"].str.strip().isin(all_true)) &
            (df["user_input"].str.strip() != "")
        ]["user_input"].drop_duplicates().head(3).tolist()
    )
    return positives, negatives


def _cache_path(csv_path: str, mode: str) -> Path:
    mtime = int(os.path.getmtime(csv_path))
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    return Path(CACHE_DIR) / f"{mode}_{mtime}.pkl"


def _purge_stale(csv_path: str, mode: str) -> None:
    current = _cache_path(csv_path, mode).name
    for f in Path(CACHE_DIR).glob(f"{mode}_*.pkl"):
        if f.name != current:
            f.unlink()


def _load_cache(path: Path) -> Optional[tuple]:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_cache(path: Path, payload: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


Encoder = SentenceTransformer | GTEClient


def _embed_batch(
    texts             : list[str],
    encoder_primary   : Encoder,
    encoder_secondary : Optional[Encoder],
    mode              : str,          # "single" | "ensemble" | "gte"
) -> np.ndarray:
    """
    Embed a list of texts, return (N, D) array (normalised rows).
    For ensemble mode, concatenates both encoders per-vector then normalises.
    """
    def _enc(enc: Encoder) -> np.ndarray:
        return enc.encode(
            texts, normalize_embeddings=True,
            show_progress_bar=False, batch_size=128,
        )

    vecs_a = _enc(encoder_primary)

    if mode == "ensemble" and encoder_secondary is not None:
        vecs_b = _enc(encoder_secondary)
        combined = np.concatenate([vecs_a, vecs_b], axis=1)
        norms    = np.linalg.norm(combined, axis=1, keepdims=True)
        norms    = np.where(norms == 0, 1.0, norms)
        return combined / norms

    return vecs_a


def _embed_centroid(
    texts             : list[str],
    encoder_primary   : Encoder,
    encoder_secondary : Optional[Encoder],
    mode              : str,
) -> np.ndarray:
    """Average all text embeddings into a single centroid vector."""
    vecs = _embed_batch(texts, encoder_primary, encoder_secondary, mode)
    mean = vecs.mean(axis=0)
    n    = np.linalg.norm(mean)
    return mean / n if n > 0 else mean


def build_centroid_vectors(
    intents           : list[Intent],
    csv_path          : str,
    mode              : str,
    encoder_primary   : Encoder,
    encoder_secondary : Optional[Encoder],
    progress_fn       = None,
) -> dict[str, np.ndarray]:
    cp = _cache_path(csv_path, f"centroid_{mode}")
    _purge_stale(csv_path, f"centroid_{mode}")
    cached = _load_cache(cp)
    if cached is not None and "vectors" in cached:
        return cached["vectors"]

    vectors: dict[str, np.ndarray] = {}
    for i, intent in enumerate(intents):
        vectors[intent.slug] = _embed_centroid(
            [intent.description] + intent.examples,
            encoder_primary, encoder_secondary, mode,
        )
        if progress_fn:
            progress_fn((i + 1) / len(intents))

    _save_cache(cp, {"vectors": vectors})
    return vectors


def build_individual_vectors(
    intents              : list[Intent],
    user_inputs_by_slug  : dict[str, list[str]],
    csv_path             : str,
    mode                 : str,
    encoder_primary      : Encoder,
    encoder_secondary    : Optional[Encoder],
    progress_fn          = None,
) -> dict[str, list[IndividualVector]]:
    """
    For each intent, embed EVERY text individually.
    Returns {slug: [IndividualVector, ...]}.

    Texts per intent:
      1 × description
      N × positive_examples (from the examples field)
      M × confirmed user_inputs (from CSV rows where intent_detected=True)

    Note: intent.examples already contains both examples AND confirmed user_inputs
    merged together. We use user_inputs_by_slug to re-tag confirmed inputs as
    kind="user_input" for display purposes.
    """
    cp = _cache_path(csv_path, f"individual_{mode}")
    _purge_stale(csv_path, f"individual_{mode}")
    cached = _load_cache(cp)
    if cached is not None and "individual" in cached:
        return cached["individual"]

    result: dict[str, list[IndividualVector]] = {}

    for i, intent in enumerate(intents):
        ui_set  = set(user_inputs_by_slug.get(intent.slug, []))
        # Build (text, kind) pairs — description first, then examples
        pairs: list[tuple[str, str]] = [
            (intent.description, "description")
        ]
        for ex in intent.examples:
            kind = "user_input" if ex in ui_set else "example"
            pairs.append((ex, kind))

        texts = [p[0] for p in pairs]
        vecs  = _embed_batch(texts, encoder_primary, encoder_secondary, mode)

        result[intent.slug] = [
            IndividualVector(text=text, kind=kind, vector=vec)
            for (text, kind), vec in zip(pairs, vecs)
        ]
        if progress_fn:
            progress_fn((i + 1) / len(intents))

    _save_cache(cp, {"individual": result})
    return result


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


def run_centroid_classification(
    user_input        : str,
    intents_dict      : dict[str, Intent],
    vectors           : dict[str, np.ndarray],
    encoder_primary   : Encoder,
    encoder_secondary : Optional[Encoder],
    enc_mode          : str,
    top_k             : int = 8,
) -> list[dict]:
    q = _embed_centroid([user_input], encoder_primary, encoder_secondary, enc_mode)
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


@dataclass
class IndividualIntentResult:
    slug           : str
    description    : str
    max_score      : float          # best single-example similarity
    top3_mean      : float          # mean of top-3 similarities
    vote_count     : int            # how many examples exceeded INDIV_MATCH_SIM
    total_vectors  : int            # total examples for this intent
    vote_fraction  : float          # vote_count / total_vectors
    bucket         : str
    # Detailed per-example scores (for UI drill-down)
    example_scores : list[dict]     # [{text, kind, score, matched}]
    # Aggregated score used for ranking (top3_mean)
    rank_score     : float


def run_individual_classification(
    user_input          : str,
    intents_dict        : dict[str, Intent],
    individual_vectors  : dict[str, list[IndividualVector]],
    encoder_primary     : Encoder,
    encoder_secondary   : Optional[Encoder],
    enc_mode            : str,
    top_k               : int = 8,
    vote_threshold      : float = INDIV_VOTE_THRESHOLD,
    match_sim           : float = INDIV_MATCH_SIM,
) -> list[IndividualIntentResult]:
    """
    For each intent:
      1. Embed query
      2. Cosine-score query against every individual vector
      3. Aggregate: max, top-3 mean, vote count
      4. Bucket using top-3 mean (more robust than raw max)
    """
    q_vec = _embed_centroid([user_input], encoder_primary, encoder_secondary, enc_mode)

    all_results: list[IndividualIntentResult] = []

    for slug, indiv_vecs in individual_vectors.items():
        if not indiv_vecs or slug not in intents_dict:
            continue

        matrix = np.stack([iv.vector for iv in indiv_vecs])
        sims   = (matrix @ q_vec).tolist()

        sorted_sims = sorted(sims, reverse=True)
        max_score   = float(sorted_sims[0])
        top3_mean   = float(np.mean(sorted_sims[:3]))
        votes       = sum(1 for s in sims if s >= match_sim)
        vote_frac   = votes / len(sims) if sims else 0.0

        # Bucket decision: use top3_mean as the primary signal.
        # Additionally, if vote_fraction is high, can promote one level.
        raw_bucket   = bucket_score(top3_mean)
        final_bucket = raw_bucket
        if vote_frac >= vote_threshold and raw_bucket == "NEEDS_REVIEW":
            final_bucket = "CONFIRMED"

        example_scores = [
            {
                "text"    : iv.text,
                "kind"    : iv.kind,
                "score"   : round(float(s), 4),
                "matched" : float(s) >= match_sim,
            }
            for iv, s in zip(indiv_vecs, sims)
        ]
        example_scores.sort(key=lambda x: x["score"], reverse=True)

        all_results.append(IndividualIntentResult(
            slug          = slug,
            description   = intents_dict[slug].description,
            max_score     = round(max_score, 4),
            top3_mean     = round(top3_mean, 4),
            vote_count    = votes,
            total_vectors = len(sims),
            vote_fraction = round(vote_frac, 4),
            bucket        = final_bucket,
            example_scores= example_scores,
            rank_score    = round(top3_mean, 4),
        ))

    all_results.sort(key=lambda r: r.rank_score, reverse=True)
    return all_results[:top_k]


@st.cache_resource(show_spinner=False)
def load_biencoder_general():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(BIENCODER_GENERAL_ID, device=device)


@st.cache_resource(show_spinner=False)
def load_biencoder_qa():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(BIENCODER_QA_ID, device=device)


@st.cache_resource(show_spinner=False)
def load_gte_client(endpoint: str) -> Optional[GTEClient]:
    try:
        client = GTEClient(endpoint, timeout=GTE_REQUEST_TIMEOUT)
        if client.ping():
            return client
        return None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def cached_load_intents_with_ui(csv_path: str, tenant_id: Optional[str]):
    return load_intents_with_user_inputs(csv_path, tenant_id)


def render_candidate_row(c: dict) -> None:
    bucket = c["bucket"]
    score  = c["score"]
    bar_w  = int(score * 100)
    st.markdown(f"""
    <div class="candidate-row bucket-{bucket}">
        <span class="cand-rank">#{c['rank']}</span>
        <span class="cand-score">{score:.4f}</span>
        <div class="score-bar"><div class="score-fill fill-{bucket}" style="width:{bar_w}%"></div></div>
        <span class="cand-slug">{c['slug']}</span>
        <span class="route-badge route-{bucket}">{BUCKET_ICONS[bucket]} {bucket}</span>
    </div>
    """, unsafe_allow_html=True)


def render_individual_result_row(r: IndividualIntentResult, rank: int) -> None:
    bucket    = r.bucket
    bar_w     = int(r.top3_mean * 100)
    vote_pct  = int(r.vote_fraction * 100)
    vote_bar  = int(r.vote_fraction * 100)

    st.markdown(f"""
    <div class="candidate-row bucket-{bucket}">
        <span class="cand-rank">#{rank}</span>
        <span class="cand-score">{r.top3_mean:.4f}</span>
        <div class="score-bar"><div class="score-fill fill-{bucket}" style="width:{bar_w}%"></div></div>
        <span class="cand-slug">{r.slug}
            <span style="font-size:0.7rem; color:#9ca3af; font-weight:400; margin-left:6px">
                max={r.max_score:.3f} &nbsp;|&nbsp; votes={r.vote_count}/{r.total_vectors} ({vote_pct}%)
            </span>
        </span>
        <span class="route-badge route-{bucket}">{BUCKET_ICONS[bucket]} {bucket}</span>
    </div>
    """, unsafe_allow_html=True)


def render_individual_examples(r: IndividualIntentResult, show_n: int = 8) -> None:
    """Render the per-example score drill-down for one intent."""
    st.markdown(f"""
    <div style="font-size:0.72rem; color:#9ca3af; margin-bottom:6px; font-family:'IBM Plex Mono',monospace;">
    VECTOR BREAKDOWN &nbsp;·&nbsp; {r.total_vectors} total &nbsp;|&nbsp;
    match threshold = {INDIV_MATCH_SIM} &nbsp;|&nbsp;
    {r.vote_count} matched ({int(r.vote_fraction*100)}%)
    </div>
    <div class="vote-bar-wrap">
        <div class="vote-bar-fill" style="width:{int(r.vote_fraction*100)}%"></div>
    </div>
    """, unsafe_allow_html=True)

    for ex in r.example_scores[:show_n]:
        kind_cls  = f"ex-type-{ex['kind']}"
        match_cls = "ex-match-yes" if ex["matched"] else "ex-match-no"
        match_txt = "✓ match" if ex["matched"] else "—"
        bar_w2    = int(ex["score"] * 100)
        text_disp = ex["text"][:80] + ("…" if len(ex["text"]) > 80 else "")
        st.markdown(f"""
        <div class="indiv-example-row">
            <span class="ex-score">{ex['score']:.4f}</span>
            <span class="ex-type {kind_cls}">{ex['kind']}</span>
            <span class="ex-text">{text_disp}</span>
            <span class="ex-match {match_cls}">{match_txt}</span>
        </div>
        """, unsafe_allow_html=True)

    if len(r.example_scores) > show_n:
        st.caption(f"… {len(r.example_scores) - show_n} more examples not shown")


def render_centroid_result(
    candidates    : list[dict],
    expected_slug : Optional[str] = None,
    header        : bool          = True,
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

    if header:
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
                    st.markdown(f'<span class="correct-win">✓ CONFIRMED MATCH — {expected_slug}</span>', unsafe_allow_html=True)
                elif expected_slug in all_slugs:
                    bkt = next(c["bucket"] for c in candidates if c["slug"] == expected_slug)
                    st.markdown(f'<span class="correct-top">◑ In top-K ({bkt}) — {expected_slug}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="correct-miss">✗ Not in top-K — expected: {expected_slug}</span>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card"><div class="metric-val" style="color:#065f46">{len(confirmed)}</div><div class="metric-lbl">Confirmed</div></div>
            <div class="metric-card"><div class="metric-val" style="color:#92400e">{len(review)}</div><div class="metric-lbl">Needs Review</div></div>
            <div class="metric-card"><div class="metric-val" style="color:#9a3412">{len(ambiguous)}</div><div class="metric-lbl">Ambiguous</div></div>
            <div class="metric-card"><div class="metric-val" style="color:#991b1b">{len(weak)}</div><div class="metric-lbl">Weak Signal</div></div>
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


def render_individual_result(
    results       : list[IndividualIntentResult],
    expected_slug : Optional[str] = None,
    header        : bool          = True,
) -> None:
    if not results:
        st.warning("No results returned.")
        return

    confirmed = [r for r in results if r.bucket == "CONFIRMED"]
    review    = [r for r in results if r.bucket == "NEEDS_REVIEW"]
    ambiguous = [r for r in results if r.bucket == "AMBIGUOUS"]
    weak      = [r for r in results if r.bucket == "WEAK_SIGNAL"]
    route     = results[0].bucket

    if header:
        col_route, col_correct = st.columns([2, 3])
        with col_route:
            st.markdown(
                f'<span class="route-badge route-{route}">'
                f'{BUCKET_ICONS[route]} {route}  •  top3-mean: {results[0].top3_mean:.4f}'
                f'</span>',
                unsafe_allow_html=True,
            )
        if expected_slug:
            confirmed_slugs = [r.slug for r in confirmed]
            all_slugs       = [r.slug for r in results]
            with col_correct:
                if expected_slug in confirmed_slugs:
                    st.markdown(f'<span class="correct-win">✓ CONFIRMED MATCH — {expected_slug}</span>', unsafe_allow_html=True)
                elif expected_slug in all_slugs:
                    bkt = next(r.bucket for r in results if r.slug == expected_slug)
                    st.markdown(f'<span class="correct-top">◑ In top-K ({bkt}) — {expected_slug}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="correct-miss">✗ Not in top-K — expected: {expected_slug}</span>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card"><div class="metric-val" style="color:#065f46">{len(confirmed)}</div><div class="metric-lbl">Confirmed</div></div>
            <div class="metric-card"><div class="metric-val" style="color:#92400e">{len(review)}</div><div class="metric-lbl">Needs Review</div></div>
            <div class="metric-card"><div class="metric-val" style="color:#9a3412">{len(ambiguous)}</div><div class="metric-lbl">Ambiguous</div></div>
            <div class="metric-card"><div class="metric-val" style="color:#991b1b">{len(weak)}</div><div class="metric-lbl">Weak Signal</div></div>
        </div>
        """, unsafe_allow_html=True)

    if confirmed:
        st.markdown('<div class="section-header">CONFIRMED INTENTS</div>', unsafe_allow_html=True)
        for r in confirmed:
            render_individual_result_row(r, results.index(r) + 1)
            with st.expander(f"Example breakdown — {r.slug}", expanded=False):
                render_individual_examples(r)

    st.markdown('<div class="section-header">ALL CANDIDATES</div>', unsafe_allow_html=True)
    for i, r in enumerate(results):
        render_individual_result_row(r, i + 1)
        with st.expander(f"Example breakdown — {r.slug}", expanded=False):
            render_individual_examples(r)


with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    csv_path = st.text_input(
        "CSV Path",
        value="json/intent_reranker_training_dataset.csv",
    )

    encoder_mode = st.selectbox(
        "Encoder",
        options=["ensemble (mpnet+qa)", "single (mpnet)", "gte"],
        index=0,
    )
    enc_mode_key = {"ensemble (mpnet+qa)": "ensemble", "single (mpnet)": "single", "gte": "gte"}[encoder_mode]

    vector_mode = st.selectbox(
        "Vector Mode",
        options=["centroid", "individual", "compare both"],
        index=0,
        help=(
            "centroid — average all examples into one vector per intent\n"
            "individual — score query against every example separately\n"
            "compare both — run both and show side-by-side"
        ),
    )

    top_k = st.slider("Top-K candidates", min_value=3, max_value=20, value=8)

    if enc_mode_key == "gte":
        gte_endpoint = st.text_input("GTE Endpoint", value=GTE_ENDPOINT_DEFAULT)
    else:
        gte_endpoint = GTE_ENDPOINT_DEFAULT

    if vector_mode in ("individual", "compare both"):
        st.markdown("### 🗳 Individual Mode Settings")
        vote_threshold = st.slider(
            "Vote threshold (fraction)",
            min_value=0.05, max_value=1.0, value=INDIV_VOTE_THRESHOLD, step=0.05,
            help="Fraction of examples that must match to promote NEEDS_REVIEW → CONFIRMED",
        )
        match_sim = st.slider(
            "Per-example match similarity",
            min_value=0.20, max_value=0.80, value=INDIV_MATCH_SIM, step=0.01,
            help="Minimum cosine sim for an individual example to count as a 'vote'",
        )
    else:
        vote_threshold = INDIV_VOTE_THRESHOLD
        match_sim      = INDIV_MATCH_SIM

    st.markdown("### 🎚 Thresholds")
    st.markdown(
        f"""
        <div style="font-size:0.78rem; color:#6b7280; line-height:2;">
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
    st.caption("Edit thresholds in the CONSTANTS block at the top of this file.")

    st.markdown("### 📊 Test Samples")
    samples_per_intent = st.slider("Samples per intent", 1, 5, 2)
    show_negatives     = st.checkbox("Show negative samples", value=True)

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem; color:#9ca3af;">'
        'Vectors cached in <code>./vector_cache/</code><br>'
        'Cache invalidates when CSV changes.'
        '</div>',
        unsafe_allow_html=True,
    )


st.markdown("""
<div class="app-header">
    <span class="app-title">INTENT CLASSIFIER</span>
    <span class="app-subtitle">bi-encoder · centroid &amp; individual vector modes</span>
</div>
""", unsafe_allow_html=True)

if not os.path.exists(csv_path):
    st.error(f"CSV file not found: `{csv_path}`  \nUpdate the path in the sidebar.")
    st.stop()

# Load encoders
with st.spinner("Loading encoder models…"):
    if enc_mode_key == "gte":
        gte_client = load_gte_client(gte_endpoint)
        if gte_client is None:
            st.error(f"Cannot connect to GTE endpoint: `{gte_endpoint}`  \nCheck the URL in the sidebar.")
            st.stop()
        encoder_primary   = gte_client
        encoder_secondary = None
        st.success(f"GTE endpoint connected: `{gte_endpoint}`")
    else:
        encoder_primary   = load_biencoder_general()
        encoder_secondary = load_biencoder_qa() if enc_mode_key == "ensemble" else None

# Load intents
with st.spinner("Loading intents from CSV…"):
    try:
        intents_list, raw_df, user_inputs_by_slug = cached_load_intents_with_ui(csv_path, None)
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        st.stop()

intents_dict = {i.slug: i for i in intents_list}

# Build / load vectors
cache_key_centroid    = f"centroid__{csv_path}__{enc_mode_key}"
cache_key_individual  = f"individual__{csv_path}__{enc_mode_key}"

need_centroid   = vector_mode in ("centroid", "compare both")
need_individual = vector_mode in ("individual", "compare both")

centroid_vectors: Optional[dict]  = None
individual_vectors: Optional[dict] = None

if need_centroid and cache_key_centroid not in st.session_state:
    prog = st.progress(0, text="Computing centroid vectors…")
    try:
        centroid_vectors = build_centroid_vectors(
            intents_list, csv_path, enc_mode_key,
            encoder_primary, encoder_secondary,
            progress_fn=lambda p: prog.progress(p, text=f"Centroid embeddings… {int(p*100)}%")
        )
        st.session_state[cache_key_centroid] = centroid_vectors
        prog.empty()
    except Exception as e:
        st.error(f"Failed to build centroid vectors: {e}")
        st.stop()
elif need_centroid:
    centroid_vectors = st.session_state[cache_key_centroid]

if need_individual and cache_key_individual not in st.session_state:
    prog2 = st.progress(0, text="Computing individual vectors…")
    try:
        individual_vectors = build_individual_vectors(
            intents_list, user_inputs_by_slug, csv_path, enc_mode_key,
            encoder_primary, encoder_secondary,
            progress_fn=lambda p: prog2.progress(p, text=f"Individual embeddings… {int(p*100)}%")
        )
        st.session_state[cache_key_individual] = individual_vectors
        prog2.empty()
    except Exception as e:
        st.error(f"Failed to build individual vectors: {e}")
        st.stop()
elif need_individual:
    individual_vectors = st.session_state[cache_key_individual]

# Status bar
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Intents", len(intents_list))
col_b.metric("Encoder", enc_mode_key)
col_c.metric("Vector mode", vector_mode.split()[0])
if enc_mode_key == "ensemble":
    col_d.metric("Dims", "1536")
elif enc_mode_key == "gte":
    col_d.metric("Dims", "384")
else:
    col_d.metric("Dims", "768")

if need_individual and individual_vectors:
    total_vecs = sum(len(v) for v in individual_vectors.values())
    st.caption(f"Individual mode: {total_vecs:,} total vectors across {len(intents_list)} intents")

st.markdown("---")

# Tabs

tab_classify, tab_test, tab_intents = st.tabs([
    "🔍  Classify",
    "🧪  Test Samples",
    "📋  Intent Catalogue",
])


# Tab 1: Classify
with tab_classify:
    mode_badge_html = {
        "centroid"    : '<span class="mode-badge mode-centroid">centroid</span>',
        "individual"  : '<span class="mode-badge mode-individual">individual</span>',
        "compare both": '<span class="mode-badge mode-centroid">centroid</span> vs <span class="mode-badge mode-individual">individual</span>',
    }[vector_mode]

    st.markdown(
        f'<div class="info-box">'
        f'Vector mode: {mode_badge_html} &nbsp;·&nbsp; Encoder: '
        f'<code>{enc_mode_key}</code><br>'
        f'{"Centroid averages all examples into one vector. " if need_centroid else ""}'
        f'{"Individual scores query against every example separately — see vote counts and per-example breakdown." if need_individual else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )

    user_input_text = st.text_area(
        "User input",
        height=120,
        placeholder="Paste or type the text you want to classify…",
        label_visibility="collapsed",
    )
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        run_btn = st.button("Classify ▶", width='stretch')

    if run_btn and user_input_text.strip():
        with st.spinner("Classifying…"):
            try:
                if vector_mode == "centroid":
                    cands = run_centroid_classification(
                        user_input_text, intents_dict, centroid_vectors,
                        encoder_primary, encoder_secondary, enc_mode_key, top_k,
                    )
                    render_centroid_result(cands)

                elif vector_mode == "individual":
                    ind_results = run_individual_classification(
                        user_input_text, intents_dict, individual_vectors,
                        encoder_primary, encoder_secondary, enc_mode_key, top_k,
                        vote_threshold, match_sim,
                    )
                    render_individual_result(ind_results)

                else:  # compare both
                    col_left, col_right = st.columns(2)

                    with col_left:
                        st.markdown('<div class="compare-header">CENTROID MODE</div>', unsafe_allow_html=True)
                        st.markdown('<div class="compare-body">', unsafe_allow_html=True)
                        cands = run_centroid_classification(
                            user_input_text, intents_dict, centroid_vectors,
                            encoder_primary, encoder_secondary, enc_mode_key, top_k,
                        )
                        render_centroid_result(cands, header=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    with col_right:
                        st.markdown('<div class="compare-header">INDIVIDUAL MODE</div>', unsafe_allow_html=True)
                        st.markdown('<div class="compare-body">', unsafe_allow_html=True)
                        ind_results = run_individual_classification(
                            user_input_text, intents_dict, individual_vectors,
                            encoder_primary, encoder_secondary, enc_mode_key, top_k,
                            vote_threshold, match_sim,
                        )
                        render_individual_result(ind_results, header=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Delta summary
                    if cands and ind_results:
                        st.markdown('<div class="section-header">DELTA SUMMARY — Centroid vs Individual</div>', unsafe_allow_html=True)
                        c_confirmed = {c["slug"] for c in cands if c["bucket"] == "CONFIRMED"}
                        i_confirmed = {r.slug for r in ind_results if r.bucket == "CONFIRMED"}
                        only_c = c_confirmed - i_confirmed
                        only_i = i_confirmed - c_confirmed
                        both   = c_confirmed & i_confirmed
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Both confirmed", len(both))
                        col2.metric("Only centroid", len(only_c), delta=f"+{len(only_c)}" if only_c else None)
                        col3.metric("Only individual", len(only_i), delta=f"+{len(only_i)}" if only_i else None)
                        if only_c:
                            st.caption(f"Centroid-only confirmed: {', '.join(sorted(only_c))}")
                        if only_i:
                            st.caption(f"Individual-only confirmed: {', '.join(sorted(only_i))}")

            except Exception as e:
                st.error(f"Classification error: {e}")
                raise e

    elif run_btn:
        st.warning("Please enter some text first.")


# Tab 2: Test Samples
with tab_test:
    st.markdown(
        '<div class="info-box">'
        'Runs sampled CSV inputs through the current encoder + vector mode. '
        'Positive samples have a known expected intent; negative samples should produce empty CONFIRMED.'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.button("Run test samples ▶"):
        positives, negatives = sample_test_inputs(raw_df, n=samples_per_intent)
        st.markdown(f"**{len(positives)} positive · {len(negatives)} negative**")
        st.markdown("---")

        correct_count = top_k_count = 0
        total = len(positives)

        st.markdown("#### Positive Samples")
        for inp_t, expected in positives:
            label = f"{inp_t[:80]}{'…' if len(inp_t)>80 else ''}  ·  expected: {expected}"
            with st.expander(label, expanded=False):
                try:
                    if vector_mode == "individual":
                        ind_res = run_individual_classification(
                            inp_t, intents_dict, individual_vectors,
                            encoder_primary, encoder_secondary, enc_mode_key, top_k,
                            vote_threshold, match_sim,
                        )
                        conf_slugs = [r.slug for r in ind_res if r.bucket == "CONFIRMED"]
                        all_slugs  = [r.slug for r in ind_res]
                        if expected in conf_slugs: correct_count += 1; top_k_count += 1
                        elif expected in all_slugs: top_k_count += 1
                        render_individual_result(ind_res, expected_slug=expected)

                    elif vector_mode == "compare both":
                        col_l, col_r = st.columns(2)
                        with col_l:
                            st.caption("Centroid")
                            cands = run_centroid_classification(
                                inp_t, intents_dict, centroid_vectors,
                                encoder_primary, encoder_secondary, enc_mode_key, top_k,
                            )
                            conf_slugs = [c["slug"] for c in cands if c["bucket"] == "CONFIRMED"]
                            all_slugs  = [c["slug"] for c in cands]
                            if expected in conf_slugs: correct_count += 1; top_k_count += 1
                            elif expected in all_slugs: top_k_count += 1
                            render_centroid_result(cands, expected_slug=expected, header=True)
                        with col_r:
                            st.caption("Individual")
                            ind_res = run_individual_classification(
                                inp_t, intents_dict, individual_vectors,
                                encoder_primary, encoder_secondary, enc_mode_key, top_k,
                                vote_threshold, match_sim,
                            )
                            render_individual_result(ind_res, expected_slug=expected, header=True)

                    else:  # centroid
                        cands = run_centroid_classification(
                            inp_t, intents_dict, centroid_vectors,
                            encoder_primary, encoder_secondary, enc_mode_key, top_k,
                        )
                        conf_slugs = [c["slug"] for c in cands if c["bucket"] == "CONFIRMED"]
                        all_slugs  = [c["slug"] for c in cands]
                        if expected in conf_slugs: correct_count += 1; top_k_count += 1
                        elif expected in all_slugs: top_k_count += 1
                        render_centroid_result(cands, expected_slug=expected)
                except Exception as e:
                    st.error(str(e))

        st.markdown("---")
        st.markdown("#### Summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("Confirmed correct", f"{correct_count}/{total}", f"{correct_count/total*100:.0f}%" if total else "—")
        m2.metric("In top-K", f"{top_k_count}/{total}", f"{top_k_count/total*100:.0f}%" if total else "—")
        m3.metric("Missed entirely", f"{total - top_k_count}/{total}")

        if show_negatives and negatives:
            st.markdown("#### Negative Samples")
            st.caption("These inputs were never True for any intent — expect empty CONFIRMED bucket.")
            for neg in negatives:
                with st.expander(f"{neg[:80]}{'…' if len(neg)>80 else ''}", expanded=False):
                    try:
                        if vector_mode == "individual":
                            render_individual_result(run_individual_classification(
                                neg, intents_dict, individual_vectors,
                                encoder_primary, encoder_secondary, enc_mode_key, top_k,
                                vote_threshold, match_sim,
                            ))
                        else:
                            render_centroid_result(run_centroid_classification(
                                neg, intents_dict, centroid_vectors,
                                encoder_primary, encoder_secondary, enc_mode_key, top_k,
                            ))
                    except Exception as e:
                        st.error(str(e))


# Tab 3: Intent Catalogue
with tab_intents:
    st.markdown(f"**{len(intents_list)} intents loaded from** `{csv_path}`")

    if need_individual and individual_vectors:
        st.markdown("Individual vector counts per intent:")
        counts_df = pd.DataFrame([
            {"Intent": slug, "Vectors": len(vecs),
             "Description": 1,
             "Examples/Inputs": len(vecs) - 1}
            for slug, vecs in individual_vectors.items()
        ]).sort_values("Vectors", ascending=False)
        st.dataframe(counts_df, width='stretch', hide_index=True)
        st.markdown("---")

    search_q = st.text_input("Search intents", placeholder="Filter by name or description…")
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
            n_ui = len(user_inputs_by_slug.get(intent.slug, []))
            n_ex = len(intent.examples) - n_ui
            st.markdown(f"**Examples** ({n_ex} positive_examples · {n_ui} confirmed user_inputs)")
            if intent.examples:
                for ex in intent.examples[:5]:
                    kind = "user_input" if ex in set(user_inputs_by_slug.get(intent.slug, [])) else "example"
                    kind_cls = f"ex-type ex-type-{kind}"
                    st.markdown(
                        f'<span class="{kind_cls}">{kind}</span> '
                        f'<span style="font-size:0.82rem">{ex[:120]}{"…" if len(ex)>120 else ""}</span>',
                        unsafe_allow_html=True,
                    )
                if len(intent.examples) > 5:
                    st.caption(f"  … and {len(intent.examples)-5} more")
            if intent.tenant_id:
                st.caption(f"Tenant: `{intent.tenant_id}`")