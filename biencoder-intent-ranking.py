# ─────────────────────────────────────────────────────────────────────────────
#  stage1_biencoder_classifier.py
#
#  Stage 1 — Standalone bi-encoder intent classifier.
#
# ══════════════════════════════════════════════════════════════════════════════
#  HOW THE SCORING WORKS
# ══════════════════════════════════════════════════════════════════════════════
#
#  STEP 1 — Building intent vectors (done once, then cached to disk)
#  ─────────────────────────────────────────────────────────────────
#  For each intent, ALL associated texts are collected:
#    • intent_description
#    • ALL user_input rows where intent_detected == "True"
#    • ALL positive_examples from the CSV

#  SINGLE mode (all-mpnet-base-v2 only):
#    Each text → 768-dim vector via mpnet
#    All vectors averaged → 768-dim centroid → L2-normalised → intent vector

#  ENSEMBLE mode (mpnet + multi-qa-mpnet):
#    Each text → 768-dim vector via mpnet    → averaged → normalised → 768-dim
#    Each text → 768-dim vector via multi-qa → averaged → normalised → 768-dim
#    Both centroids concatenated → 1536-dim → L2-normalised → intent vector

#  STEP 2 — Query embedding (per call, same process as intent vectors)
#  ───────────────────────────────────────────────────────────────────
#  User input goes through the same process as intents.
#  SINGLE: 768-dim.  ENSEMBLE: 1536-dim.
#  Because both are normalised to length 1.0, cosine similarity = dot product.

#  STEP 3 — Cosine similarity score
#  ──────────────────────────────────
#    score = dot(query_vec, intent_vec)  ∈ [0, 1] for text
#  This measures geometric alignment, NOT probability.

# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-INTENT ROUTING  
# ══════════════════════════════════════════════════════════════════════════════

#  Every candidate in the top-K is independently bucketed by its own score.
#  Multiple intents can be CONFIRMED simultaneously — a "win" is if the
#  correct intent appears anywhere in the CONFIRMED bucket.

#  Buckets (thresholds tunable in CONFIG):
#    score ≥ 0.55  → CONFIRMED    all intents here are trusted matches
#    score ≥ 0.50  → NEEDS_REVIEW decent signal, Stage 2 should verify
#    score ≥ 0.40  → AMBIGUOUS    weak signal, Stage 2 required
#    score < 0.40  → WEAK_SIGNAL  very weak, likely no match

#  The overall route is determined by the highest-scoring intent's bucket.

# ══════════════════════════════════════════════════════════════════════════════
#  VECTOR CACHE
# ══════════════════════════════════════════════════════════════════════════════
#
#  Intent vectors are expensive to compute (embedding hundreds of texts).
#  They are saved to ./vector_cache/ after first computation and reloaded
#  on subsequent runs as long as:
#    - The CSV file's modification timestamp is unchanged
#    - The encoder mode (single / ensemble) is unchanged
#
#  If either changes, the cache is automatically invalidated and rebuilt.
#  Cache files are named:  {mode}_{csv_mtime_int}.pkl
#
# ─────────────────────────────────────────────────────────────────────────────
#  Install:
#    pip install sentence-transformers torch pandas huggingface_hub
#  Colab:
#    !pip install -q sentence-transformers torch pandas huggingface_hub
# ─────────────────────────────────────────────────────────────────────────────

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
import torch
from huggingface_hub import login
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG  — edit these to change behaviour
# ─────────────────────────────────────────────────────────────────────────────

# ── Encoder mode ──────────────────────────────────────────────────────────────
#  "single"   → all-mpnet-base-v2 only (768-dim vectors, faster)
#  "ensemble" → mpnet + multi-qa-mpnet concatenated (1536-dim, richer signal)
#
#  Toggling this at runtime (interactive mode command: 'mode') rebuilds the
#  registry using the cached vectors for the selected mode.
#  DO NOT mix single and ensemble vectors — changing mode invalidates the cache.
ENCODER_MODE : str = "ensemble"   # "single" | "ensemble"

BIENCODER_GENERAL_ID = "sentence-transformers/all-mpnet-base-v2"
BIENCODER_QA_ID      = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K = 8    # how many candidates to retrieve per query
STAGE2_TOP_N    = 3    # how many candidates to forward to Stage 2

# ── Routing thresholds ────────────────────────────────────────────────────────
#  Applied per-candidate — ALL intents in the top-K are independently bucketed.
#  Multiple intents can land in CONFIRMED simultaneously.
THRESHOLD_CONFIRMED = 0.55   # ≥ this → CONFIRMED
THRESHOLD_REVIEW    = 0.50   # ≥ this, < CONFIRMED → NEEDS_REVIEW
THRESHOLD_AMBIGUOUS = 0.40   # ≥ this, < REVIEW   → AMBIGUOUS
                              # < AMBIGUOUS        → WEAK_SIGNAL

# ── Vector cache ──────────────────────────────────────────────────────────────
CACHE_DIR = "./vector_cache"   # folder where cache files are stored

# ── CSV ───────────────────────────────────────────────────────────────────────
CSV_PATH = "json/intent_reranker_training_dataset.csv"

# ── Test sampling ─────────────────────────────────────────────────────────────
TEST_SAMPLES_PER_INTENT = 2
N_NEGATIVE_SAMPLES      = 3

# ── Debug ─────────────────────────────────────────────────────────────────────
DEBUG_RETRIEVAL = True

# ── HF token ─────────────────────────────────────────────────────────────────
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTING
# ─────────────────────────────────────────────────────────────────────────────

class Route:
    CONFIRMED    = "CONFIRMED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    AMBIGUOUS    = "AMBIGUOUS"
    WEAK_SIGNAL  = "WEAK_SIGNAL"


def bucket_score(score: float) -> str:
    """Map a single cosine similarity score to its route bucket."""
    if   score >= THRESHOLD_CONFIRMED:  return Route.CONFIRMED
    elif score >= THRESHOLD_REVIEW:     return Route.NEEDS_REVIEW
    elif score >= THRESHOLD_AMBIGUOUS:  return Route.AMBIGUOUS
    else:                               return Route.WEAK_SIGNAL


# ─────────────────────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Intent:
    slug        : str
    description : str
    examples    : list[str]      = field(default_factory=list)
    tenant_id   : Optional[str] = None


@dataclass
class Stage1Result:
    """
    Multi-intent routing result.

    overall_route   : Route of the highest-scoring candidate (top-1 bucket)
    top1_score      : cosine similarity score of rank-1 candidate
    all_candidates  : ALL top-K candidates with score + bucket label

    Bucketed views — every list may contain 0 to N intents:
      confirmed     : score ≥ THRESHOLD_CONFIRMED  (all trusted matches)
      needs_review  : THRESHOLD_REVIEW ≤ score < CONFIRMED
      ambiguous     : THRESHOLD_AMBIGUOUS ≤ score < REVIEW
      weak_signal   : score < THRESHOLD_AMBIGUOUS

    A correctness check is a WIN if the expected slug appears anywhere in
    `confirmed`.

    stage2_candidates : top-STAGE2_TOP_N candidates forwarded to Stage 2.
                        Includes all CONFIRMED + top non-confirmed up to N.
    """
    overall_route     : str
    top1_score        : float
    all_candidates    : list[dict]
    confirmed         : list[dict]
    needs_review      : list[dict]
    ambiguous         : list[dict]
    weak_signal       : list[dict]
    stage2_candidates : list[dict]
    encoder_mode      : str


# ─────────────────────────────────────────────────────────────────────────────
#  CSV UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

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
    if isinstance(val, bool):   return val
    if isinstance(val, (int, float)): return bool(val)
    return str(val).strip().lower() in ("true", "1", "yes", "t")


# ─────────────────────────────────────────────────────────────────────────────
#  CSV LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_intents_from_csv(
    csv_path  : str,
    tenant_id : Optional[str] = None,
) -> tuple[list[Intent], pd.DataFrame]:
    """
    Parse CSV → list[Intent] + raw DataFrame.
    ALL confirmed user_inputs + ALL positive_examples aggregated per intent.
    No cap on number of examples.
    """
    print(f"[CSV] Loading: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)
    print(f"[CSV] Total rows: {len(df):,}")

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    required = {"user_input", "intent_detected", "intent_name",
                "intent_description", "positive_examples"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[CSV] Missing columns: {missing}\nFound: {list(df.columns)}")

    if tenant_id and "tenant_id" in df.columns:
        before = len(df)
        df     = df[df["tenant_id"].str.strip() == tenant_id].copy()
        print(f"[CSV] Tenant filter '{tenant_id}': {len(df):,}/{before:,} rows")

    df["intent_detected_bool"] = df["intent_detected"].apply(_parse_bool)
    df["user_input"]           = df["user_input"].fillna("").str.strip()
    df["intent_name"]          = df["intent_name"].fillna("").str.strip()
    df["intent_description"]   = df["intent_description"].fillna("").str.strip()

    print(f"[CSV] True: {int(df['intent_detected_bool'].sum()):,}  "
          f"False: {int((~df['intent_detected_bool']).sum()):,}")

    intent_names = [n for n in df["intent_name"].unique() if n]
    print(f"[CSV] Unique intents: {len(intent_names)}\n")

    intents: list[Intent] = []
    for name in intent_names:
        rows = df[df["intent_name"] == name]

        description = next(
            (d.strip() for d in rows["intent_description"] if str(d).strip()), None
        )
        if not description:
            print(f"[CSV] WARN: No description for '{name}' — skipping.")
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
        print(f"[CSV]   '{name}'  confirmed={len(confirmed_inputs)}  "
              f"pos={len(pos_examples)}  total={len(combined)}")

    print(f"\n[CSV] Built {len(intents)} intents.\n")
    return intents, df


# ─────────────────────────────────────────────────────────────────────────────
#  TEST SAMPLER
# ─────────────────────────────────────────────────────────────────────────────

def sample_test_inputs(
    df                 : pd.DataFrame,
    samples_per_intent : int = TEST_SAMPLES_PER_INTENT,
    n_negative_samples : int = N_NEGATIVE_SAMPLES,
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
        for inp in rows["user_input"].drop_duplicates().head(samples_per_intent).tolist():
            if inp not in seen:
                seen.add(inp)
                positives.append((inp, name))

    all_true = set(df[df["intent_detected_bool"] == True]["user_input"].str.strip().unique())
    negatives = (
        df[
            (df["intent_detected_bool"] == False) &
            (~df["user_input"].str.strip().isin(all_true)) &
            (df["user_input"].str.strip() != "")
        ]["user_input"].drop_duplicates().head(n_negative_samples).tolist()
    )

    return positives, negatives


# ─────────────────────────────────────────────────────────────────────────────
#  VECTOR CACHE
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(csv_path: str, mode: str) -> Path:
    """
    Return the cache file path for a given CSV + encoder mode combination.
    Cache is invalidated when the CSV file's mtime changes.
    """
    mtime = int(os.path.getmtime(csv_path))
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    return Path(CACHE_DIR) / f"{mode}_{mtime}.pkl"


def save_vector_cache(
    cache_path : Path,
    intents    : list[Intent],
    vectors    : dict[str, np.ndarray],
    mode       : str,
) -> None:
    """
    Persist intent vectors and metadata to disk.

    Cache format (pickle):
      {
        "mode"    : str,
        "slugs"   : list[str],
        "vectors" : dict[slug → np.ndarray],
        "intents" : dict[slug → Intent],
      }
    """
    payload = {
        "mode"    : mode,
        "slugs"   : [i.slug for i in intents],
        "vectors" : vectors,
        "intents" : {i.slug: i for i in intents},
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[Cache] Saved → {cache_path}")


def load_vector_cache(
    cache_path : Path,
) -> tuple[list[Intent], dict[str, np.ndarray]] | None:
    """
    Load intent vectors from disk cache.
    Returns (intents, vectors) or None if cache not found / corrupt.
    """
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        intents = [payload["intents"][s] for s in payload["slugs"]]
        vectors = payload["vectors"]
        print(f"[Cache] Loaded ← {cache_path}  ({len(intents)} intents)")
        return intents, vectors
    except Exception as e:
        print(f"[Cache] WARN: Could not load cache ({e}) — will recompute.")
        return None


def purge_stale_caches(csv_path: str, mode: str) -> None:
    """Remove cache files for the same mode that no longer match current mtime."""
    cache_dir = Path(CACHE_DIR)
    if not cache_dir.exists():
        return
    current = _cache_path(csv_path, mode).name
    for f in cache_dir.glob(f"{mode}_*.pkl"):
        if f.name != current:
            f.unlink()
            print(f"[Cache] Purged stale cache: {f.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  INTENT REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

class IntentRegistry:
    """
    Manages intent definitions and pre-computed bi-encoder vectors.

    Supports two modes:
      "single"   — all-mpnet-base-v2 only → 768-dim intent/query vectors
      "ensemble" — mpnet + multi-qa-mpnet → 1536-dim intent/query vectors

    Vectors are loaded from disk cache when available; computed and saved
    to disk on first run or when CSV changes.

    The mode MUST be the same for both intent vectors and query embedding.
    This is enforced by routing all query embedding through embed_query().
    """

    def __init__(
        self,
        biencoder_general : SentenceTransformer,
        biencoder_qa      : Optional[SentenceTransformer],
        mode              : str,
    ):
        if mode not in ("single", "ensemble"):
            raise ValueError(f"mode must be 'single' or 'ensemble', got '{mode}'")
        if mode == "ensemble" and biencoder_qa is None:
            raise ValueError("ensemble mode requires biencoder_qa.")

        self.biencoder_general = biencoder_general
        self.biencoder_qa      = biencoder_qa
        self.mode              = mode
        self._intents          : dict[str, Intent]     = {}
        self._vectors          : dict[str, np.ndarray] = {}

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts → single centroid vector.

        Single mode:
          embed all texts → average → normalise → 768-dim

        Ensemble mode:
          embed all texts with general model → average → normalise → 768-dim  (A)
          embed all texts with QA model      → average → normalise → 768-dim  (B)
          concatenate A + B → 1536-dim → normalise → final vector
        """
        def _mean_norm(encoder: SentenceTransformer) -> np.ndarray:
            vecs = encoder.encode(
                texts, normalize_embeddings=True,
                show_progress_bar=False, batch_size=128,
            )
            mean = vecs.mean(axis=0)
            norm = np.linalg.norm(mean)
            return mean / norm if norm > 0 else mean

        vec_general = _mean_norm(self.biencoder_general)

        if self.mode == "single":
            return vec_general   # already normalised

        # ensemble
        vec_qa   = _mean_norm(self.biencoder_qa)
        combined = np.concatenate([vec_general, vec_qa])
        norm     = np.linalg.norm(combined)
        return combined / norm if norm > 0 else combined

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query string.
        Uses the SAME process as intent vectors — always call this, never
        embed externally. Guarantees query and intent vectors are compatible.
        """
        return self._embed_texts([text])

    # ── Registration ──────────────────────────────────────────────────────────

    def _load_from_cache(
        self,
        intents    : list[Intent],
        vectors    : dict[str, np.ndarray],
    ) -> None:
        """Populate registry from pre-loaded cache data."""
        for intent in intents:
            self._intents[intent.slug] = intent
            self._vectors[intent.slug] = vectors[intent.slug]

    def build_from_intents(
        self,
        intents    : list[Intent],
        csv_path   : str,
        force      : bool = False,
    ) -> None:
        """
        Compute (or load from cache) vectors for all intents and register them.

        Args:
            intents   : list[Intent] from load_intents_from_csv()
            csv_path  : path to the CSV (used to compute cache key)
            force     : if True, skip cache and recompute from scratch
        """
        cache_path = _cache_path(csv_path, self.mode)
        purge_stale_caches(csv_path, self.mode)

        if not force:
            cached = load_vector_cache(cache_path)
            if cached is not None:
                cached_intents, cached_vectors = cached
                self._load_from_cache(cached_intents, cached_vectors)
                print(f"[Registry] Loaded {self.intent_count} intents from cache "
                      f"(mode={self.mode}).\n")
                return

        # Cache miss — compute vectors
        print(f"[Registry] Computing vectors for {len(intents)} intents "
              f"(mode={self.mode}, this may take a moment)...")
        for intent in intents:
            self._intents[intent.slug] = intent
            self._vectors[intent.slug] = self._embed_texts(
                [intent.description] + intent.examples
            )

        print(f"[Registry] ✓ {self.intent_count} intents computed.")
        save_vector_cache(cache_path, intents, self._vectors, self.mode)
        print(f"[Registry] Ready (mode={self.mode}).\n")

    def remove(self, slug: str) -> None:
        self._intents.pop(slug, None)
        self._vectors.pop(slug, None)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve_top_k(
        self,
        query_embedding : np.ndarray,
        top_k           : int,
    ) -> list[tuple[Intent, float]]:
        """
        Cosine similarity (= dot product since all vectors are normalised).
        Returns top-k (Intent, score) sorted descending.
        """
        if not self._intents:
            return []
        slugs  = list(self._vectors.keys())
        matrix = np.stack([self._vectors[s] for s in slugs])
        sims   = matrix @ query_embedding
        k      = min(top_k, len(slugs))
        idx    = np.argpartition(sims, -k)[-k:]
        idx    = idx[np.argsort(sims[idx])[::-1]]
        return [(self._intents[slugs[i]], float(sims[i])) for i in idx]

    @property
    def intent_count(self) -> int:
        return len(self._intents)


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 2 PLACEHOLDER
# ─────────────────────────────────────────────────────────────────────────────

def call_stage2(
    user_input : str,
    candidates : list[dict],
    route      : str,
) -> dict:
    """
    Stage 2 stub. Replace body with Qwen call when ready.

    Receives:
      user_input  : original text
      candidates  : list[dict] with slug, description, score, bucket
      route       : overall route string

    Must return: {"intent": "<slug or NO_MATCH>", "reason": "<sentence>"}
    """
    # ── REPLACE FROM HERE ────────────────────────────────────────────────────
    return {
        "intent"  : "STAGE2_NOT_IMPLEMENTED",
        "reason"  : "Stage 2 not yet wired up.",
        "error"   : None,
    }
    # ── TO HERE ──────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def classify(
    user_input      : str,
    registry        : IntentRegistry,
    retrieval_top_k : int  = RETRIEVAL_TOP_K,
    stage2_top_n    : int  = STAGE2_TOP_N,
    debug_retrieval : bool = DEBUG_RETRIEVAL,
    run_stage2      : bool = False,
) -> Stage1Result:
    """
    Classify user_input against all registered intents.

    Every candidate in top-K is independently bucketed by its own score:
      confirmed     — ALL intents with score ≥ THRESHOLD_CONFIRMED
      needs_review  — ALL intents with THRESHOLD_REVIEW ≤ score < CONFIRMED
      ambiguous     — ALL intents with THRESHOLD_AMBIGUOUS ≤ score < REVIEW
      weak_signal   — ALL intents below THRESHOLD_AMBIGUOUS

    A "win" is when the expected intent appears anywhere in `confirmed`.
    This is different from v1 where only top-1 was compared.
    """
    if not user_input or not user_input.strip():
        raise ValueError("user_input cannot be empty.")
    if registry.intent_count == 0:
        raise RuntimeError("No intents registered.")

    # ── Embed + retrieve ──────────────────────────────────────────────────────
    query_vec         = registry.embed_query(user_input)
    candidates_scored = registry.retrieve_top_k(query_vec, top_k=retrieval_top_k)

    if debug_retrieval:
        print(f"\n  [Stage 1 | mode={registry.mode} | "
              f"'{user_input[:60]}{'…' if len(user_input)>60 else ''}']")
        for intent, score in candidates_scored:
            bar    = "█" * int(score * 20)
            bucket = bucket_score(score)
            print(f"    {score:.4f}  [{bar:<20}]  {bucket:<12}  {intent.slug}")

    # ── Build candidate list with per-candidate bucket labels ─────────────────
    all_candidates: list[dict] = []
    for i, (intent, score) in enumerate(candidates_scored):
        all_candidates.append({
            "rank"        : i + 1,
            "slug"        : intent.slug,
            "description" : intent.description,
            "score"       : round(float(score), 4),
            "bucket"      : bucket_score(float(score)),
        })

    # ── Bucket into four groups ───────────────────────────────────────────────
    confirmed    = [c for c in all_candidates if c["bucket"] == Route.CONFIRMED]
    needs_review = [c for c in all_candidates if c["bucket"] == Route.NEEDS_REVIEW]
    ambiguous    = [c for c in all_candidates if c["bucket"] == Route.AMBIGUOUS]
    weak_signal  = [c for c in all_candidates if c["bucket"] == Route.WEAK_SIGNAL]

    # Overall route = bucket of the highest-scoring candidate
    top1_score    = all_candidates[0]["score"] if all_candidates else 0.0
    overall_route = all_candidates[0]["bucket"] if all_candidates else Route.WEAK_SIGNAL

    # Stage 2 candidates: all CONFIRMED first, then fill up to stage2_top_n
    # from lower buckets so Stage 2 always has context even when nothing is CONFIRMED
    stage2_candidates = confirmed[:]
    for c in needs_review + ambiguous + weak_signal:
        if len(stage2_candidates) >= stage2_top_n:
            break
        stage2_candidates.append(c)

    result = Stage1Result(
        overall_route     = overall_route,
        top1_score        = top1_score,
        all_candidates    = all_candidates,
        confirmed         = confirmed,
        needs_review      = needs_review,
        ambiguous         = ambiguous,
        weak_signal       = weak_signal,
        stage2_candidates = stage2_candidates,
        encoder_mode      = registry.mode,
    )

    # ── Stage 2 (stub) ────────────────────────────────────────────────────────
    if run_stage2 and overall_route != Route.CONFIRMED:
        result.stage2_output = call_stage2(  # type: ignore[attr-defined]
            user_input, stage2_candidates, overall_route
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  PRETTY PRINTER
# ─────────────────────────────────────────────────────────────────────────────

_BUCKET_COLOUR = {
    Route.CONFIRMED    : "\033[92m",   # green
    Route.NEEDS_REVIEW : "\033[93m",   # yellow
    Route.AMBIGUOUS    : "\033[33m",   # orange
    Route.WEAK_SIGNAL  : "\033[91m",   # red
}
_BOLD  = "\033[1m"
_RESET = "\033[0m"


def _colour(text: str, route: str) -> str:
    return f"{_BUCKET_COLOUR.get(route, '')}{text}{_RESET}"


def print_results(
    user_input    : str,
    result        : Stage1Result,
    expected_slug : Optional[str] = None,
) -> None:
    bar = "─" * 84

    # Correctness: win if expected appears anywhere in confirmed bucket
    correct_marker = ""
    if expected_slug:
        confirmed_slugs = [c["slug"] for c in result.confirmed]
        if expected_slug in confirmed_slugs:
            correct_marker = f"  {_BOLD}\033[92m✓ CONFIRMED MATCH\033[0m"
        elif any(c["slug"] == expected_slug for c in result.all_candidates):
            correct_marker = (
                f"  \033[93m◑ IN TOP-K but not CONFIRMED "
                f"(bucket={next(c['bucket'] for c in result.all_candidates if c['slug']==expected_slug)})"
                f"\033[0m"
            )
        else:
            correct_marker = f"  \033[91m✗ NOT IN TOP-K — expected: {expected_slug}\033[0m"

    print(f"\n{bar}")
    print(f"  INPUT   : {user_input[:115]}{'…' if len(user_input)>115 else ''}")
    print(
        f"  ROUTE   : {_colour(result.overall_route, result.overall_route)}"
        f"  (top-1: {result.top1_score:.4f})"
        f"  mode={result.encoder_mode}"
        f"{correct_marker}"
    )

    # Summary of how many intents fell into each bucket
    print(
        f"  BUCKETS : "
        f"{_colour(f'CONFIRMED={len(result.confirmed)}', Route.CONFIRMED)}  "
        f"{_colour(f'NEEDS_REVIEW={len(result.needs_review)}', Route.NEEDS_REVIEW)}  "
        f"{_colour(f'AMBIGUOUS={len(result.ambiguous)}', Route.AMBIGUOUS)}  "
        f"{_colour(f'WEAK_SIGNAL={len(result.weak_signal)}', Route.WEAK_SIGNAL)}"
    )

    print(f"\n  {'#':<4} {'SCORE':>6}  {'BUCKET':<13}  {'BAR':<22}  INTENT")
    print(f"  {'─'*4} {'─'*6}  {'─'*13}  {'─'*22}  {'─'*34}")

    for c in result.all_candidates:
        score    = c["score"]
        bucket   = c["bucket"]
        n_blocks = int(score * 20)
        fill     = "█" * n_blocks
        bold     = _BOLD if bucket == Route.CONFIRMED else ""
        col      = _BUCKET_COLOUR.get(bucket, "")
        print(
            f"  {bold}{col}#{c['rank']:<3} {score:.4f}  "
            f"{bucket:<13}  [{fill:<20}]  {c['slug']}{_RESET}"
        )

    if result.confirmed:
        print(f"\n  {_BOLD}CONFIRMED MATCHES ({len(result.confirmed)}):{_RESET}")
        for c in result.confirmed:
            print(f"    • {c['slug']}  ({c['score']:.4f})")

    if hasattr(result, "stage2_output"):
        s2 = result.stage2_output
        print(f"\n  STAGE 2: {s2.get('intent','—')}  |  {s2.get('reason','')}")

    print(bar + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_models(
    mode     : str           = ENCODER_MODE,
    hf_token : Optional[str] = None,
) -> tuple[SentenceTransformer, Optional[SentenceTransformer]]:
    """
    Load bi-encoder model(s) based on mode.

    single   → loads only all-mpnet-base-v2  (returns biencoder_general, None)
    ensemble → loads both mpnet models        (returns biencoder_general, biencoder_qa)
    """
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
        print("[INFO] Authenticated with HuggingFace.\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device : {device}  |  Mode : {mode}")

    print(f"[INFO] Loading bi-encoder (general): {BIENCODER_GENERAL_ID}")
    biencoder_general = SentenceTransformer(BIENCODER_GENERAL_ID, device=device)

    biencoder_qa = None
    if mode == "ensemble":
        print(f"[INFO] Loading bi-encoder (QA)     : {BIENCODER_QA_ID}")
        biencoder_qa = SentenceTransformer(BIENCODER_QA_ID, device=device)

    print("[INFO] Models ready.\n")
    return biencoder_general, biencoder_qa


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    current_mode = ENCODER_MODE

    # Load both models upfront so mode-toggle doesn't require a reload
    biencoder_general, biencoder_qa = load_models(mode="ensemble", hf_token=HF_TOKEN)

    # Load CSV
    intents, raw_df = load_intents_from_csv(CSV_PATH)

    # Build registry for initial mode
    def build_registry(mode: str) -> IntentRegistry:
        reg = IntentRegistry(biencoder_general, biencoder_qa, mode=mode)
        reg.build_from_intents(intents, CSV_PATH)
        return reg

    registry = build_registry(current_mode)
    print(f"[INFO] Registry ready — {registry.intent_count} intents | mode={current_mode}\n")

    # Sample test inputs
    positive_samples, negative_samples = sample_test_inputs(raw_df)

    print("=" * 84)
    print(f"  STAGE 1 ANALYSIS  |  mode={current_mode}")
    print(f"  Positives : {len(positive_samples)}  ({TEST_SAMPLES_PER_INTENT} per intent)")
    print(f"  Negatives : {len(negative_samples)}  (never True for any intent)")
    print(f"  Thresholds: CONFIRMED≥{THRESHOLD_CONFIRMED}  "
          f"NEEDS_REVIEW≥{THRESHOLD_REVIEW}  "
          f"AMBIGUOUS≥{THRESHOLD_AMBIGUOUS}  "
          f"WEAK_SIGNAL<{THRESHOLD_AMBIGUOUS}")
    print("=" * 84)

    print("\n── POSITIVE SAMPLES ──\n")
    for user_input, expected_slug in positive_samples:
        try:
            result = classify(user_input, registry)
            print_results(user_input, result, expected_slug=expected_slug)
        except Exception as e:
            print(f"[ERROR] {e}")

    print("\n── NEGATIVE SAMPLES (expect WEAK_SIGNAL / empty CONFIRMED) ──\n")
    for user_input in negative_samples:
        try:
            result = classify(user_input, registry)
            print_results(user_input, result)
        except Exception as e:
            print(f"[ERROR] {e}")

    # ── Interactive mode ──────────────────────────────────────────────────────
    print("=" * 84)
    print("  INTERACTIVE MODE")
    print("  Commands:")
    print("    quit            — exit")
    print("    mode            — toggle between 'single' and 'ensemble'")
    print("    debug           — toggle Stage 1 debug ranking output")
    print("    thresholds      — print current threshold values")
    print("=" * 84)

    debug_mode = DEBUG_RETRIEVAL

    while True:
        try:
            user_input = input("\nEnter user input: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Exiting.")
            break

        if not user_input:
            print("[WARN] Empty input.")
            continue

        low = user_input.lower()

        if low in ("quit", "exit", "q"):
            print("[INFO] Exiting.")
            break

        if low == "mode":
            current_mode = "single" if current_mode == "ensemble" else "ensemble"
            print(f"[INFO] Switching to mode='{current_mode}' ...")
            registry = build_registry(current_mode)
            print(f"[INFO] Registry rebuilt — {registry.intent_count} intents | mode={current_mode}")
            continue

        if low == "debug":
            debug_mode = not debug_mode
            print(f"[INFO] Debug retrieval: {'ON' if debug_mode else 'OFF'}")
            continue

        if low == "thresholds":
            print(
                f"  CONFIRMED≥{THRESHOLD_CONFIRMED}  "
                f"NEEDS_REVIEW≥{THRESHOLD_REVIEW}  "
                f"AMBIGUOUS≥{THRESHOLD_AMBIGUOUS}  "
                f"WEAK_SIGNAL<{THRESHOLD_AMBIGUOUS}"
            )
            continue

        try:
            result = classify(user_input, registry, debug_retrieval=debug_mode)
            print_results(user_input, result)
        except Exception as e:
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()