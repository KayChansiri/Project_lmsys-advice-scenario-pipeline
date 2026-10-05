# LMSYS Advice Scenario Sampling Pipeline

This repository contains the reproducible sampling and screening pipeline used to identify candidate advice-seeking scenarios from **LMSYS-Chat-1M** for a study of LLM-generated decision support across everyday domains.

The public repository covers the workflow from source-data access through creation of a human-review batch. It intentionally does **not** publish raw LMSYS conversation text, API credentials, or completed human-review data.

## Study sampling target

The downstream study uses three domains:

- relationship
- career
- emotional wellbeing

The final study target is 50 human-confirmed scenarios per domain.

## Pipeline

1. Access LMSYS-Chat-1M through Hugging Face.
2. Restrict to English conversations.
3. Use only the **first user turn** from each conversation.
4. Restrict message length to 25–800 words.
5. Apply a high-recall deterministic prefilter for:
   - personal-situation signals;
   - target-domain signals;
   - obvious non-personal task exclusions.
6. Compute a SHA-256 hash of normalized source text.
7. Remove exact normalized duplicates.
8. Assign deterministic random priority using a fixed seed.
9. Apply a fixed LLM screening rubric for:
   - personal situation;
   - advice/decision-support intent;
   - primary domain;
   - sufficient context;
   - clinical diagnosis/treatment exclusion.
10. Prepare a deterministic human-review batch.

The deterministic prefilter and LLM screen are **candidate-screening tools**, not ground-truth labels. Final inclusion is based on human review.

## Important data-access note

This repository does not redistribute LMSYS-Chat-1M source text.

To reproduce the pipeline, obtain access to the dataset through its official Hugging Face distribution and provide your own Hugging Face token locally.

Do not commit raw conversation text or restricted derivatives to GitHub.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── .gitignore
├── src/
│   ├── 01_build_candidates.py
│   ├── 02_screen_candidates.py
│   └── 03_prepare_human_review.py
└── data/
    ├── restricted/
    ├── human_review/
    └── processed/
```

## Environment

Recommended: Python 3.12.

### Conda

```bash
conda env create -f environment.yml
conda activate facct-lmsys
```

Or:

```bash
python -m pip install -r requirements.txt
```

## Credentials

Set credentials locally. Do not hard-code them into scripts or notebooks.

```bash
export HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

Alternatively, enter them interactively when prompted by the screening script.

## Step 1: Build candidate pool

```bash
python src/01_build_candidates.py
```

Output:

```text
data/restricted/lmsys_personal_advice_candidates.parquet
```

## Step 2: LLM screening

```bash
python src/02_screen_candidates.py
```

Output:

```text
data/restricted/lmsys_llm_screening_checkpoint.parquet
```

The script is resumable. It skips source hashes already present in the checkpoint.

## Step 3: Prepare human-review batch

```bash
python src/03_prepare_human_review.py
```

Output:

```text
data/human_review/lmsys_human_review_batch1.csv
```

The public repository should **not** include the generated CSV because it contains LMSYS source text. The script is provided so authorized users can regenerate it locally.

## Human-review fields

The review file contains the following annotation columns:

- `human_eligible`
- `human_primary_domain`
- `human_sufficient_context`
- `human_notes`

Recommended coding values:

- `human_eligible`: `yes` / `no`
- `human_primary_domain`: `relationship` / `career` / `emotional_wellbeing`
- `human_sufficient_context`: `yes` / `no`

## Current screening checkpoint used in the study

At the point human review began, the working checkpoint contained:

- career: 257 LLM-eligible cases
- relationship: 256 LLM-eligible cases
- emotional wellbeing: 111 LLM-eligible cases

The initial human-review batch sampled 75 cases per domain in deterministic random order.

These counts are reported for provenance only. The raw checkpoint is not included in the public repository because it contains source text.

## Reproducibility notes

- Random seed: `20260916`
- Only the first user message is used.
- Exact duplicate detection is based on normalized lowercase whitespace-collapsed text.
- The original assistant response is not used.
- The fixed LLM-screening rubric is embedded verbatim in `src/02_screen_candidates.py`.
- Model-generated eligibility is recomputed programmatically from the component screening fields to prevent internally inconsistent `eligible=True` outputs.

## Public vs. restricted materials

### Appropriate for GitHub

- source code
- screening rubric
- regex patterns
- fixed random seed
- dependency files
- README / methods documentation
- codebook / annotation instructions
- public-safe aggregate counts
- hashed identifiers that cannot reasonably reconstruct the original text

### Keep out of GitHub

- raw LMSYS prompts/conversations
- `source_text`
- restricted parquet files
- completed human-review CSVs containing source text
- API keys or Hugging Face tokens
- old notebooks containing credentials
- logs that expose credentials

## Archiving for publication

For a paper submission, use the GitHub repository as the living code repository and create a frozen archival release of the exact version used for the paper.

A practical workflow is:

1. Finalize the repository.
2. Create a tagged release, for example `v1.0.0-paper`.
3. Archive that release with a long-term research repository such as Zenodo.
4. Cite the archival DOI in the paper's reproducibility/materials statement.
5. Also provide the GitHub URL for ongoing updates.

If the venue uses anonymous review, follow its current anonymity policy before exposing identifying repository metadata.

## Security

Never commit credentials. If a secret is accidentally committed, revoke it immediately. Removing it from the visible file is not enough if it remains in Git history.
