"""
Reproducibility pipeline for LMSYS-Chat-1M advice-scenario construction.

IMPORTANT
---------
- LMSYS-Chat-1M is gated. Do not redistribute source conversation text.
- These scripts preserve the historical study output basenames used downstream.
"""

import os
import re
import hashlib
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm

# ---------------- Configuration ----------------
PROJECT_DIR = Path(os.environ.get("FAACT_PROJECT_DIR", ".")).resolve()
DATA_DIR = PROJECT_DIR / "data"
RESTRICTED_DIR = DATA_DIR / "restricted"
RESTRICTED_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ID = "lmsys/lmsys-chat-1m"
HF_TOKEN = os.environ.get("HF_TOKEN")

TARGET_LANGUAGE = "English"
MIN_WORDS = 25
MAX_WORDS = 800
RANDOM_SEED = 20260916

OUTPUT_PATH = RESTRICTED_DIR / "lmsys_personal_advice_candidates.parquet"

if not HF_TOKEN:
    raise RuntimeError("Set HF_TOKEN in your environment before running.")

# ---------------- Helpers ----------------
def first_user_text(record):
    """Return the first non-empty user-authored turn, or None."""
    conversation = record.get("conversation") or []
    for turn in conversation:
        role = str(turn.get("role", "")).lower()
        content = turn.get("content")
        if role in {"user", "human"} and isinstance(content, str) and content.strip():
            return content.strip()
    return None

def normalize_text(text):
    return re.sub(r"\s+", " ", text.strip().lower())

def sha256_text(text):
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()

def deterministic_priority(text_hash):
    payload = f"{RANDOM_SEED}:{text_hash}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)

# High-recall signals. These are not final eligibility rules.
RELATIONSHIP_RE = re.compile(
    r"\b(boyfriend|girlfriend|partner|husband|wife|spouse|dating|relationship|"
    r"breakup|break up|friend|friendship|family|mother|father|mom|dad|sister|"
    r"brother|jealous|jealousy|trust|boundary|boundaries|conflict|argument)\b",
    re.I,
)
CAREER_RE = re.compile(
    r"\b(job|career|work|workplace|boss|manager|supervisor|coworker|colleague|"
    r"quit|resign|promotion|salary|interview|offer|employment|degree|college|"
    r"university|training|graduate school)\b",
    re.I,
)
WELLBEING_RE = re.compile(
    r"\b(stress|stressed|sad|sadness|lonely|loneliness|grief|grieving|"
    r"overwhelmed|motivation|unmotivated|self[- ]esteem|anxious|anxiety|"
    r"coping|emotional|feel|feeling)\b",
    re.I,
)
PERSONAL_RE = re.compile(
    r"\b(i|i'm|i’ve|i've|me|my|mine|we|our|us)\b", re.I
)
DECISION_RE = re.compile(
    r"\b(should i|what should|what do i do|advice|help me|how should i|"
    r"would you|do you think|is it worth|how do i deal|how can i cope|"
    r"what would you do|unsure|uncertain|decide|decision)\b",
    re.I,
)

# Obvious non-personal/task-oriented requests only. Keep conservative for recall.
NONPERSONAL_RE = re.compile(
    r"\b(write (?:me )?(?:a|an|the)? ?(?:poem|story|essay|email|code)|"
    r"translate (?:this|the following)|summari[sz]e (?:this|the following)|"
    r"roleplay as|act as|pretend (?:you are|to be)|"
    r"python code|javascript code|sql query|regular expression)\b",
    re.I,
)

def is_candidate(text):
    wc = len(text.split())
    if not (MIN_WORDS <= wc <= MAX_WORDS):
        return False, {}

    relationship = bool(RELATIONSHIP_RE.search(text))
    career = bool(CAREER_RE.search(text))
    wellbeing = bool(WELLBEING_RE.search(text))
    personal = bool(PERSONAL_RE.search(text))
    decision = bool(DECISION_RE.search(text))
    nonpersonal = bool(NONPERSONAL_RE.search(text))

    keep = personal and (relationship or career or wellbeing) and not nonpersonal
    signals = {
        "word_count": wc,
        "relationship_signal": relationship,
        "career_signal": career,
        "wellbeing_signal": wellbeing,
        "decision_support_signal": decision,
    }
    return keep, signals

# ---------------- Stream and prefilter ----------------
dataset = load_dataset(
    DATASET_ID,
    split="train",
    streaming=True,
    token=HF_TOKEN,
)

rows = []
n_total = 0
n_english_first_turn = 0

for record in tqdm(dataset, desc="Streaming LMSYS-Chat-1M"):
    n_total += 1

    if record.get("language") != TARGET_LANGUAGE:
        continue

    text = first_user_text(record)
    if not text:
        continue

    n_english_first_turn += 1
    keep, signals = is_candidate(text)
    if not keep:
        continue

    text_hash = sha256_text(text)
    rows.append({
        "source_text_hash": text_hash,
        "source_text": text,
        "random_priority": deterministic_priority(text_hash),
        **signals,
    })

candidates = pd.DataFrame(rows)
n_prefilter = len(candidates)

# Exact deduplication after whitespace/case normalization.
candidates = (
    candidates
    .sort_values("random_priority")
    .drop_duplicates(subset="source_text_hash", keep="first")
    .reset_index(drop=True)
)

candidates.to_parquet(OUTPUT_PATH, index=False)

print(f"Records inspected: {n_total:,}")
print(f"English records with usable first user turn: {n_english_first_turn:,}")
print(f"Prefilter candidates: {n_prefilter:,}")
print(f"Unique candidates after exact deduplication: {len(candidates):,}")
print(f"Saved: {OUTPUT_PATH}")

# Historical study-run checkpoints. Warnings preserve portability across future dataset versions.
expected = {
    "records_inspected": 1_000_000,
    "english_first_turns": 777_453,
    "prefilter_candidates": 7_508,
    "unique_candidates": 4_640,
}
observed = {
    "records_inspected": n_total,
    "english_first_turns": n_english_first_turn,
    "prefilter_candidates": n_prefilter,
    "unique_candidates": len(candidates),
}
for key, exp in expected.items():
    if observed[key] != exp:
        print(f"WARNING: {key}={observed[key]:,}; historical study run={exp:,}.")
