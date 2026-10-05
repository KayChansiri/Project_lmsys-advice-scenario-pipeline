from pathlib import Path
import os
import re
import hashlib

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm

DATASET_ID = "lmsys/lmsys-chat-1m"
TARGET_LANGUAGE = "English"
MIN_WORDS = 25
MAX_WORDS = 800
RANDOM_SEED = 20260916

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESTRICTED_DIR = PROJECT_DIR / "data" / "restricted"
RESTRICTED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = RESTRICTED_DIR / "lmsys_personal_advice_candidates.parquet"

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Set HF_TOKEN in your environment before running this script.")


def word_count(text):
    return len(text.split()) if isinstance(text, str) else 0


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def text_hash(text):
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def deterministic_priority(source_text_hash, seed=RANDOM_SEED):
    return hashlib.sha256(
        f"{seed}|{source_text_hash}".encode("utf-8")
    ).hexdigest()


def first_user_turn(conversation):
    if not conversation:
        return None

    for message in conversation:
        if not isinstance(message, dict):
            continue

        role = str(message.get("role", "")).lower()
        content = message.get("content")

        if role == "user" and isinstance(content, str):
            content = content.strip()
            if content:
                return content

    return None


def matches_any(text, patterns):
    text = str(text)
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


RELATIONSHIP_PATTERNS = [
    r"\bboyfriend\b", r"\bgirlfriend\b", r"\bpartner\b", r"\bspouse\b",
    r"\bhusband\b", r"\bwife\b", r"\bfianc[eé]e?\b", r"\brelationship\b",
    r"\bromantic\b", r"\bdating\b", r"\bdate\b", r"\bseeing someone\b",
    r"\bexclusive\b", r"\bcrush\b", r"\blove interest\b",
    r"\bsignificant other\b", r"\bex\b", r"\bbreakup\b", r"\bbroke up\b",
    r"\bseparation\b", r"\bdivorce\b", r"\bmarriage\b", r"\bcheating\b",
    r"\btrust\b", r"\bjealous\b", r"\bjealousy\b", r"\bcontrolling\b",
    r"\btoxic\b", r"\babusive\b", r"\bmanipulat", r"\bgaslight",
    r"\bghost", r"\bno contact\b", r"\blong.distance\b", r"\bargument\b",
    r"\barguing\b", r"\bconflict\b", r"\bboundar", r"\bfriend\b",
    r"\bfriends\b", r"\bbest friend\b", r"\broommate\b", r"\bfamily\b",
    r"\bparents?\b", r"\bmom\b", r"\bdad\b", r"\bmother\b",
    r"\bfather\b", r"\bsibling\b", r"\bsister\b", r"\bbrother\b",
    r"\bin.laws?\b",
]

CAREER_PATTERNS = [
    r"\bjob\b", r"\bcareer\b", r"\bprofession", r"\bwork\b",
    r"\bworking\b", r"\bworkplace\b", r"\bemployment\b", r"\bemployer\b",
    r"\bemployee\b", r"\boffice\b", r"\bboss\b", r"\bsupervisor\b",
    r"\bmanager\b", r"\bcoworker\b", r"\bco.worker\b", r"\bcolleague\b",
    r"\bteam lead\b", r"\bdirector\b", r"\bjob search\b", r"\bjob hunt\b",
    r"\bjob application\b", r"\bapply\b", r"\binterview\b", r"\bresume\b",
    r"\bcv\b", r"\bcover letter\b", r"\brecruiter\b", r"\bhiring\b",
    r"\bjob offer\b", r"\bquit\b", r"\bresign\b", r"\bchange jobs?\b",
    r"\bcareer switch\b", r"\bcareer path\b", r"\bnew job\b",
    r"\bposition\b", r"\bsalary\b", r"\bpay\b", r"\bcompensation\b",
    r"\braise\b", r"\bpromotion\b", r"\bdemotion\b", r"\bbonus\b",
    r"\bbenefits\b", r"\bnegotiate\b", r"\bfired\b", r"\bterminated\b",
    r"\blayoff\b", r"\blaid off\b", r"\bPIP\b", r"\btoxic workplace\b",
    r"\bwork conflict\b", r"\bharassment\b", r"\bdiscrimination\b",
    r"\bmicromanag", r"\bburnout\b", r"\binternship\b",
    r"\bgraduate school\b", r"\bcollege\b", r"\buniversity\b",
    r"\bdegree\b", r"\bmajor\b", r"\bPhD\b", r"\bmaster'?s\b",
    r"\bMBA\b", r"\bmed school\b", r"\blaw school\b", r"\bpostdoc\b",
    r"\bfellowship\b",
]

WELLBEING_PATTERNS = [
    r"\bmental health\b", r"\bwellbeing\b", r"\bwell-being\b",
    r"\bemotional health\b", r"\bdepress", r"\bsad\b", r"\bfeeling down\b",
    r"\bhopeless\b", r"\bempty\b", r"\bnumb\b", r"\bmiserable\b",
    r"\bdistress", r"\bemotional\b", r"\bupset\b", r"\bcrying\b",
    r"\banxi", r"\bpanic\b", r"\bworr", r"\bfear\b", r"\boverwhelm",
    r"\boverthinking\b", r"\bruminat", r"\bstress", r"\bburnout\b",
    r"\bexhaust", r"\bdrained\b", r"\bfatig", r"\btired\b",
    r"\bcan't cope\b", r"\bcannot cope\b", r"\bstruggling\b",
    r"\blonely\b", r"\bloneliness\b", r"\bisolat", r"\brejection\b",
    r"\babandon", r"\bgrief\b", r"\bgrieving\b", r"\bloss\b",
    r"\bself-esteem\b", r"\bself esteem\b", r"\bself-worth\b",
    r"\bconfidence\b", r"\binsecure\b", r"\binsecurity\b",
    r"\bmotivation\b", r"\bprocrastinat", r"\bpurpose\b", r"\bcoping\b",
    r"\btherapy\b", r"\btherapist\b", r"\bcounselor\b",
    r"\bpsychologist\b", r"\bpsychiatrist\b", r"\bprofessional help\b",
    r"\bseek help\b", r"\bget help\b", r"\bneed help\b", r"\binsomnia\b",
    r"\bsleep\b", r"\bfocus\b", r"\bconcentration\b", r"\bsuicid",
    r"\bself.harm\b", r"\bhurt myself\b", r"\bkill myself\b",
    r"\bdon't want to live\b", r"\bwant to die\b",
]

PERSONAL_NARRATIVE_PATTERNS = [
    r"\bmy (?:husband|wife|boyfriend|girlfriend|partner|spouse)\b",
    r"\bmy (?:friend|friends|best friend|roommate)\b",
    r"\bmy (?:mother|father|mom|dad|parents|family|sister|brother)\b",
    r"\bmy (?:job|career|boss|manager|coworker|colleague|workplace|team)\b",
    r"\bi work\b", r"\bi'm working\b", r"\bi am working\b",
    r"\bi got (?:a|an|the) job\b", r"\bi have (?:a|an|the) job\b",
    r"\bi was (?:fired|laid off|promoted|hired)\b",
    r"\bi got (?:fired|laid off|promoted|hired)\b",
    r"\bi graduated\b", r"\bi'm graduating\b", r"\bi am graduating\b",
    r"\bwe have been\b", r"\bwe've been\b", r"\bwe are dating\b",
    r"\bwe're dating\b", r"\bi'm dating\b", r"\bi am dating\b",
    r"\bwe broke up\b", r"\bwe've broken up\b", r"\bwe are arguing\b",
    r"\bwe've been arguing\b", r"\bi feel\b", r"\bi'm feeling\b",
    r"\bi am feeling\b", r"\bi've been feeling\b",
    r"\bi have been feeling\b", r"\bi struggle\b", r"\bi'm struggling\b",
    r"\bi am struggling\b", r"\bi'm worried\b", r"\bi am worried\b",
    r"\bi'm anxious\b", r"\bi am anxious\b", r"\bi feel stuck\b",
    r"\bi feel lost\b", r"\bi'm thinking about\b",
    r"\bi am thinking about\b", r"\bi'm considering\b",
    r"\bi am considering\b", r"\bi want to\b", r"\bi need to\b",
]

DECISION_SUPPORT_PATTERNS = [
    r"\bwhat should i do\b", r"\bwhat do i do\b", r"\bwhat would you do\b",
    r"\bany advice\b", r"\bneed advice\b", r"\blooking for advice\b",
    r"\badvice please\b", r"\bcan you advise\b", r"\bhelp me decide\b",
    r"\bhelp me figure out\b", r"\bshould i\b", r"\bshould we\b",
    r"\bhow should i\b", r"\bhow do i handle\b", r"\bhow do i deal with\b",
    r"\bhow do i respond\b", r"\bhow can i cope\b", r"\bhow can i handle\b",
    r"\bwhat can i do\b", r"\btrying to decide\b", r"\bcan't decide\b",
    r"\bcannot decide\b", r"\bnot sure whether\b",
    r"\bi'm not sure whether\b", r"\bi am not sure whether\b",
    r"\bi don't know whether\b", r"\bwhether to\b",
    r"\bthinking about leaving\b", r"\bthinking about quitting\b",
    r"\bthinking about moving\b", r"\bconsidering leaving\b",
    r"\bconsidering quitting\b", r"\bconsidering changing\b",
    r"\bis it worth\b",
]

NONPERSONAL_TASK_PATTERNS = [
    r"\bact as\b", r"\bplay the role\b", r"\brole[\s-]?play\b",
    r"\bpretend (?:to be|you are)\b", r"\bassume the role\b",
    r"\bstay in character\b",
    r"\bwrite (?:me )?(?:a |an |the )?(?:story|novel|fiction|scene|screenplay|play|poem)\b",
    r"\btell (?:me )?(?:a |the )?story\b",
    r"\bcreate (?:a |the )?(?:story|novel|fiction|character|dialogue)\b",
    r"\bsummarize\b", r"\bsummarise\b", r"\btranslate\b", r"\brewrite\b",
    r"\brephrase\b", r"\bproofread\b", r"\bparaphrase\b", r"\bextract\b",
    r"\bgiven the document\b", r"\bgiven the text\b", r"\bgiven the passage\b",
    r"\bdetermine if\b", r"\bclassify\b", r"\blabel each\b",
    r"\bsentiment analysis\b", r"\bfactually consistent\b",
    r"\bgenerate (?:a |an |the )?(?:script|article|essay|story|conversation)\b",
    r"\brespond in the first person\b", r"\bwrite code\b", r"\bdebug\b",
    r"\bSQL query\b", r"\bGDScript\b", r"\bPHP\b", r"\byou are a chatbot\b",
    r"\byou are an? (?:AI|assistant|agent|model)\b",
    r"\bignore (?:all )?previous instructions\b",
]


def candidate_features(text):
    n_words = word_count(text)
    relationship = matches_any(text, RELATIONSHIP_PATTERNS)
    career = matches_any(text, CAREER_PATTERNS)
    wellbeing = matches_any(text, WELLBEING_PATTERNS)
    personal = matches_any(text, PERSONAL_NARRATIVE_PATTERNS)
    decision = matches_any(text, DECISION_SUPPORT_PATTERNS)
    obvious_task = matches_any(text, NONPERSONAL_TASK_PATTERNS)

    keep = (
        MIN_WORDS <= n_words <= MAX_WORDS
        and personal
        and (relationship or career or wellbeing)
        and not obvious_task
    )

    return {
        "word_count": n_words,
        "personal_narrative_signal": personal,
        "decision_support_signal": decision,
        "relationship_signal": relationship,
        "career_signal": career,
        "wellbeing_signal": wellbeing,
        "obvious_task_signal": obvious_task,
        "prefilter_keep": keep,
    }


stream = load_dataset(
    DATASET_ID,
    split="train",
    streaming=True,
    token=HF_TOKEN,
)

candidate_rows = []
records_seen = 0
english_records = 0
usable_first_turns = 0
prefilter_candidates = 0

for record in tqdm(stream, desc="Streaming LMSYS-Chat-1M"):
    records_seen += 1

    if record.get("language") != TARGET_LANGUAGE:
        continue

    english_records += 1
    text = first_user_turn(record.get("conversation", []))

    if not text:
        continue

    usable_first_turns += 1
    features = candidate_features(text)

    if not features["prefilter_keep"]:
        continue

    prefilter_candidates += 1

    candidate_rows.append(
        {
            "source_text_hash": text_hash(text),
            "source_text": text,
            **features,
        }
    )

candidates = pd.DataFrame(candidate_rows)

before_dedup = len(candidates)

candidates = (
    candidates
    .drop_duplicates(subset=["source_text_hash"], keep="first")
    .copy()
)

candidates["random_priority"] = (
    candidates["source_text_hash"]
    .apply(deterministic_priority)
)

candidates = (
    candidates
    .sort_values("random_priority")
    .reset_index(drop=True)
)

candidates.to_parquet(OUTPUT_PATH, index=False)

print("Records seen:", records_seen)
print("English records:", english_records)
print("Usable first-user turns:", usable_first_turns)
print("Prefilter candidates before dedup:", before_dedup)
print("Unique candidates:", len(candidates))
print("Saved:", OUTPUT_PATH)
