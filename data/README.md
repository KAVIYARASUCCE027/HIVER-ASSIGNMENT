# Dataset Setup

## Primary dataset

**Kaggle: `thoughtvector/customer-support-on-twitter`**
https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

A single CSV (`twcs.csv`, ~1.3GB, ~2.8M tweets) of real customer-support
interactions on Twitter across ~108 brand accounts (Apple, Amazon, Spotify,
Uber, Delta, etc.). Columns:

| column | meaning |
|---|---|
| `tweet_id` | unique tweet id |
| `author_id` | screen name for brand accounts (e.g. `sprintcare`), anonymized numeric id for customers |
| `inbound` | `True` if the tweet was sent *to* a brand (i.e. authored by a customer) |
| `created_at` | tweet timestamp |
| `text` | tweet text |
| `response_tweet_id` | comma-separated ids of tweets that replied to this one |
| `in_response_to_tweet_id` | id of the tweet this one replied to |

This repo does **not** commit the raw file. It is git-ignored under
`data/raw/`.

## How to obtain it

**Option A — official Kaggle source (recommended if you have a Kaggle account):**

```bash
pip install kaggle
# place your API token at ~/.kaggle/kaggle.json (from kaggle.com/settings/account)
kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw --unzip
```

This should produce `data/raw/twcs.csv`.

**Option B — verified public mirror (no login required):**

A community re-upload of the same dataset is hosted on the Hugging Face Hub
at [`SunidhiSriram/twcs`](https://huggingface.co/datasets/SunidhiSriram/twcs).
We did not compute a cryptographic hash against the official Kaggle file, so
we cannot claim it is byte-identical. Before using it we verified it against
the known Kaggle schema and sample records: the header row matches exactly
(`tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id`),
and the leading rows match the well-known `sprintcare`/`Ask_Spectrum` opening
records from the original dataset. If exact provenance matters for your use
case, prefer Option A (the official Kaggle download) below.

```bash
curl -L -o data/raw/twcs.csv \
  https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv
```

Either option must produce the same file at `data/raw/twcs.csv`.

## Optional secondary dataset

**Hugging Face: `PolyAI/banking77`** — used only, if at all, as a reference
point while designing the intent-classification methodology. It is not
mixed into the selected brand's final intent taxonomy (see
`reports/decision_log.md`).

```python
from datasets import load_dataset
banking77 = load_dataset("PolyAI/banking77")
```

## Processed data

`src/analysis/brand_analysis.py`, `src/ingestion/reconstruct_threads.py`,
and related scripts read `data/raw/twcs.csv` and write derived artifacts to
`data/processed/` (also git-ignored). See the root `README.md` for the
end-to-end reproduction steps.
