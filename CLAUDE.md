# CUSTOMER SUPPORT AI AGENT — CLAUDE CODE MASTER INSTRUCTION

You are the lead ML/AI engineer responsible for building this entire project.

Work directly inside the current repository and build a complete, runnable, professional-quality project.

The project is an AI customer-support agent trained/evaluated using:

Kaggle dataset:
`thoughtvector/customer-support-on-twitter`

Optional secondary dataset:
Hugging Face `PolyAI/banking77`

The final system must select ONE brand from the Twitter dataset and build an AI support agent for that brand.

The agent must:

1. Classify incoming customer messages into a small set of brand-specific intents.
2. Draft a reply grounded in historically similar support conversations from that brand.
3. Decide whether to auto-handle or escalate to a human, with an explicit reason.
4. Provide rigorous evaluation proving whether the system is trustworthy.

This is an evaluation-focused ML engineering project, NOT a generic chatbot.

---

## 0. MOST IMPORTANT WORKING PRINCIPLE

---

Do not immediately start building the final AI agent.

First understand the data.

You must inspect the dataset, analyze candidate brands, reconstruct conversations, and select the brand based on evidence.

Do not make important architectural assumptions without inspecting the actual data.

Do not fabricate results.

Do not claim a metric until it has actually been calculated by the evaluation pipeline.

When uncertain, inspect the repository/data and make a reasoned engineering decision.

---

## 1. FIRST ACTION — INSPECT THE REPOSITORY

---

Before modifying anything:

1. Inspect the current directory.
2. Identify existing files.
3. Identify whether the dataset is already downloaded.
4. Identify available Python/Node environments.
5. Check installed packages.
6. Check whether an existing project structure exists.
7. Read any existing README or documentation.

Do not delete existing useful work.

If the repository is empty, initialize a clean Python project.

Use Python unless there is a strong reason otherwise.

---

## 2. DATASET DISCOVERY

---

Locate the Customer Support on Twitter dataset.

Expected source:

`thoughtvector/customer-support-on-twitter`

If the dataset is not present locally, create a clear data-download/setup process.

Do NOT commit the raw multi-million-row dataset to Git.

The README must explain exactly how to obtain it.

Create:

`data/README.md`

with dataset setup instructions.

---

## 3. BRAND SELECTION

---

Before building the final agent, analyze the dataset.

Calculate at minimum:

* tweets per brand
* customer tweets per brand
* agent tweets per brand
* conversation/thread count
* average conversation length
* number of conversations containing customer + agent messages
* number of usable/resolved conversations
* common support topics

Create:

`src/analysis/brand_analysis.py`

and optionally:

`notebooks/01_brand_analysis.ipynb`

Generate a ranked table of candidate brands.

Example:

Brand | Conversations | Customer Tweets | Agent Tweets | Avg Turns | Usable Cases

Do NOT simply choose the most famous brand.

Choose the brand that gives the strongest experimental setting:

* enough historical data
* many customer-agent interactions
* recurring support patterns
* enough resolved cases
* reasonable intent diversity

Save the selected brand in:

`config/config.yaml`

Example:

brand:
name: "..."

Do not hard-code the brand elsewhere.

Document the reasoning in:

`reports/decision_log.md`

---

## 4. CONVERSATION RECONSTRUCTION

---

The Twitter dataset contains individual tweets and multi-turn interactions.

Build a robust preprocessing pipeline.

Create:

`src/ingestion/reconstruct_threads.py`

Normalize conversations into:

{
"conversation_id": "...",
"brand": "...",
"messages": [
{
"speaker": "customer",
"text": "..."
},
{
"speaker": "agent",
"text": "..."
}
]
}

Handle:

* ordering
* missing messages
* duplicate records
* malformed records
* empty tweets
* customer/brand identification
* conversation boundaries

Save processed data in an efficient format such as JSONL or Parquet.

Do not load millions of rows into memory unnecessarily.

Use streaming/chunked processing where appropriate.

---

## 5. DATA QUALITY

---

Implement basic cleaning:

* normalize whitespace
* remove obvious duplicated records
* identify malformed records
* preserve useful conversation context
* avoid destroying important support language
* remove/mask PII where appropriate

Do NOT aggressively normalize text because customer-support language is noisy and that noise may be useful.

---

## 6. INTENT DISCOVERY

---

Derive the intent taxonomy from the selected brand's actual conversations.

Do NOT simply use Banking77.

Do NOT automatically create 50–77 categories.

Target approximately:

8–15 intents.

Use:

* frequency analysis
* keyword analysis
* semantic similarity
* clustering where useful
* manual inspection
* LLM-assisted analysis if useful

The final taxonomy must be operationally meaningful.

Create:

`config/intents.yaml`

Each intent must contain:

* name
* description
* examples
* confusing/nearby intents
* escalation guidance

Example:

intents:

* name: order_tracking
  description: Customer asks about the status/location of an order
  escalation: false

The examples above are only illustrative.

Derive the actual categories from the selected brand.

---

## 7. GOLDEN EVALUATION SET

---

Create a golden evaluation set containing 150–250 manually labelled examples.

Target:

200 examples.

This must NOT simply be 200 random examples.

Use stratified sampling.

Include:

* frequent intents
* medium-frequency intents
* rare intents
* ambiguous messages
* short messages
* multi-turn context
* frustrated customers
* multiple-intent messages
* auto-handle cases
* escalation cases

Create:

`data/golden/golden_200.jsonl`

Schema:

{
"id": "...",
"conversation_id": "...",
"conversation": [...],
"customer_message": "...",
"intent": "...",
"expected_action": "auto_handle|escalate",
"expected_reason": "...",
"label_notes": "..."
}

Create:

`data/golden/README.md`

explaining:

* sampling methodology
* labeling methodology
* intent definitions
* handling of ambiguous examples
* escalation-label methodology

IMPORTANT:

Evaluation conversations must never leak into the retrieval corpus.

Split at conversation level.

A conversation_id must belong to only one split.

Prefer a temporal evaluation split when possible.

---

## 8. BASELINE 1 — MAJORITY CLASS

---

Implement a trivial baseline.

Every input receives the most frequent intent.

Create:

`src/classification/majority.py`

Evaluate it using the same golden evaluation set.

---

## 9. BASELINE 2 — TF-IDF + LOGISTIC REGRESSION

---

Implement:

TF-IDF
+
Logistic Regression

Create:

`src/classification/tfidf_baseline.py`

Use reasonable parameters.

Evaluate:

* accuracy
* macro F1
* weighted F1
* per-class precision
* per-class recall
* per-class F1

Macro F1 should be the primary classification metric.

---

## 10. LLM INTENT CLASSIFIER

---

Build an LLM-based intent classifier.

The classifier must receive:

* customer message
* relevant previous conversation context
* intent definitions

It must return structured JSON:

{
"intent": "...",
"confidence": 0.0,
"reason": "..."
}

Validate the returned schema.

Do not rely on unstructured LLM output.

Make the LLM provider configurable through `.env`.

Never hard-code API keys.

---

## 11. HISTORICAL RETRIEVAL / RAG

---

This is a core part of the project.

Build a retrieval system over historical support conversations.

Each retrievable case should contain:

* conversation_id
* customer issue
* agent response
* intent
* resolution information
* metadata

Use an embedding model and local vector index/vector database.

Keep the system simple and reproducible.

Suitable options include:

* FAISS
* Chroma
* another lightweight local vector store

Do not introduce unnecessary infrastructure.

The retrieval pipeline must:

1. Embed historical cases.
2. Store embeddings.
3. Search using the incoming customer issue.
4. Return top-K similar cases.
5. Prefer usable/resolved cases.
6. Avoid evaluation conversations.
7. Remove/mask PII.
8. Return similarity scores.

Create:

`src/retrieval/`

with modular components.

---

## 12. REPLY GENERATION

---

Build:

`src/generation/reply_generator.py`

The LLM should generate responses using retrieved historical evidence.

Prompt requirements:

* behave as support agent for selected brand
* use historical cases as evidence
* do not blindly copy historical responses
* do not expose PII
* do not invent policies
* do not invent refunds/compensation
* do not invent deadlines
* do not claim actions were taken unless supported
* be concise
* provide useful next steps
* escalate when evidence is insufficient

The model should receive:

Customer conversation
+
Detected intent
+
Retrieved historical cases

and produce:

{
"reply": "...",
"grounding_cases": ["...", "..."],
"confidence": 0.0
}

---

## 13. ESCALATION ENGINE

---

Do not allow the LLM to arbitrarily decide escalation.

Create a dedicated:

`src/escalation/policy.py`

Use explicit signals:

* intent confidence
* retrieval similarity
* evidence availability
* sensitive issue indicators
* customer frustration
* repeated unresolved issue
* multiple intents
* contradictory historical resolutions
* unsupported requested actions

Example policy:

AUTO-HANDLE if:

* high intent confidence
* strong retrieval evidence
* routine issue
* no sensitive action
* response is grounded

ESCALATE if:

* low intent confidence
* weak retrieval evidence
* contradictory historical evidence
* complex unresolved complaint
* sensitive account/security issue
* unsupported action
* system uncertainty

Thresholds must be configurable.

---

## 14. FINAL AGENT API

---

Create a central agent:

`src/agent.py`

Input:

{
"conversation": [...],
"customer_message": "..."
}

Output:

{
"intent": "...",
"intent_confidence": 0.0,
"retrieved_cases": [...],
"action": "auto_handle|escalate",
"reason": "...",
"reply": "..."
}

If escalated, reply may be null or a safe acknowledgement.

The system must be deterministic where possible.

---

## 15. EVALUATION HARNESS

---

Create:

`evaluation/run_eval.py`

It must evaluate:

1. Majority baseline
2. TF-IDF + Logistic Regression
3. LLM without RAG
4. LLM + RAG
5. LLM + RAG + escalation

Metrics:

### Classification

* accuracy
* macro F1
* weighted F1
* precision
* recall
* per-intent F1
* confusion matrix

### Escalation

* precision
* recall
* F1
* false-auto-handle rate
* false-escalation rate
* automation rate

### Reply

* correctness
* groundedness
* helpfulness
* brand consistency
* safety
* hallucination

Save machine-readable results.

For example:

`evaluation/results/results.json`

Also generate human-readable output.

---

## 16. LLM-AS-JUDGE

---

Create:

`evaluation/llm_judge.py`

Use a separate LLM evaluation prompt.

For every response, evaluate:

Correctness: 1–5
Groundedness: 1–5
Helpfulness: 1–5
Brand consistency: 1–5
Safety: 1–5

Also identify:

* hallucination
* unsupported claim
* missing information
* incorrect resolution
* inappropriate automation

Return strict JSON.

Example:

{
"correctness": 5,
"groundedness": 4,
"helpfulness": 5,
"brand_consistency": 4,
"safety": 5,
"hallucination": false,
"overall": 4.6,
"reason": "..."
}

---

## 17. HUMAN VALIDATION OF THE LLM JUDGE

---

This is mandatory.

Select approximately 50 generated replies.

Create a human-labeling file.

Humans must score the same dimensions as the LLM judge.

Then compare:

Human scores
vs
LLM judge scores

Calculate:

* Spearman correlation
* exact agreement
* agreement within one point
* binary acceptable/unacceptable agreement

Document disagreements.

Do NOT claim the LLM judge is reliable without this calibration.

---

## 18. ABLATION STUDY

---

Compare:

A. Majority baseline
B. TF-IDF + Logistic Regression
C. LLM without RAG
D. LLM + RAG
E. LLM + RAG + escalation

The key question:

Does historical retrieval actually improve support quality?

And:

Does escalation reduce unsafe automation?

Show this clearly.

---

## 19. FAILURE ANALYSIS

---

Identify the five most important failure modes from REAL evaluation results.

For every failure mode show:

1. Customer example
2. Expected behavior
3. Actual behavior
4. Why it failed
5. Hypothesis
6. Possible fix

Possible failures:

* ambiguous messages
* multiple intents
* missing context
* poor retrieval
* contradictory historical resolutions
* hallucinated policies
* over-escalation
* under-escalation
* incorrect intent
* inappropriate response tone

Do not invent examples.

---

## 20. MANDATORY SECTION

## "WHAT IS MISLEADING ABOUT MY HEADLINE NUMBER?"

---

Create this section in the report.

Be critical.

Discuss:

* class imbalance
* small golden set
* label noise
* sampling bias
* conversation leakage
* near-duplicates
* historical inconsistencies
* LLM judge bias
* retrieval leakage
* accuracy vs operational value
* benchmark vs production performance

Do not oversell the system.

---

## 21. AUTOMATION RATE

---

Report:

Total evaluation examples
Auto-handled
Escalated
Automation rate
Correct auto-handled cases
Incorrect auto-handled cases

A useful headline should ideally describe both quality and automation.

For example:

"X% of requests were automatically handled while Y% of automated responses were judged acceptable."

Use actual measured values.

---

## 22. REPORT

---

Create:

`reports/report.md`

Maximum six pages when rendered.

Sections:

1. Problem framing
2. Selected brand
3. Dataset
4. Intent taxonomy
5. Architecture
6. Retrieval
7. Escalation
8. Evaluation methodology
9. Baselines
10. Results
11. LLM judge validation
12. Failure analysis
13. Misleading headline number
14. Limitations
15. One-more-week plan

Clearly answer:

What does "good" mean for this brand?

Also explain what we deliberately chose NOT to build.

---

## 23. DECISION LOG

---

Create:

`reports/decision_log.md`

Include 10–15 non-obvious decisions.

Each decision must explain:

* decision
* alternatives
* reasoning
* tradeoff

Examples:

* brand selection
* number of intents
* conversation-level split
* temporal split
* embedding approach
* retrieval strategy
* escalation threshold
* macro F1 selection
* LLM judge calibration
* PII handling
* exclusion of sensitive actions
* why RAG is used
* why historical responses are evidence rather than truth

---

## 24. OPTIONAL STREAMLIT DEMO

---

If the core evaluation is complete and stable, build:

`app.py`

using Streamlit.

The demo should show:

Customer message
↓
Intent
↓
Confidence
↓
Retrieved historical cases
↓
Generated response
↓
Auto-handle / Escalate
↓
Reason

Do NOT prioritize the UI over evaluation.

If time is limited, skip the UI.

---

## 25. TESTING

---

Create tests for:

* preprocessing
* conversation reconstruction
* intent classification
* retrieval
* escalation
* response schema
* leakage prevention

Run:

`pytest`

before considering the project complete.

---

## 26. CONFIGURATION

---

Use:

`config/config.yaml`

for:

* selected brand
* intent settings
* retrieval top-K
* thresholds
* evaluation settings
* model names

Use:

`.env`

for:

* API keys
* LLM provider configuration

Provide:

`.env.example`

Never commit secrets.

---

## 27. REPRODUCIBILITY

---

The README must reproduce headline results in less than 15 minutes.

Optimize the evaluation path.

Do not require rebuilding millions of embeddings just to run evaluation.

Provide:

* cached processed data where legally/appropriately possible
* cached embeddings where possible
* deterministic seeds
* lightweight evaluation mode
* clear setup instructions

The evaluator should be able to run something similar to:

python -m src.analysis.brand_analysis

python -m src.ingestion.reconstruct_threads

python -m src.retrieval.index

python -m evaluation.run_eval

---

## 28. README

---

Create an excellent README.

It must contain:

# Project title

## Problem

## Selected brand

## Why this brand?

## Architecture

## Dataset

## Setup

## Environment variables

## Data preparation

## Intent taxonomy

## Running the agent

## Running evaluation

## Baselines

## Results

## LLM judge validation

## Failure analysis

## Limitations

## Reproduction

## What I would do next

Include an architecture diagram using Mermaid if useful.

---

## 29. REPOSITORY STRUCTURE

---

Target:

customer-support-agent/

├── README.md
├── CLAUDE.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── config/
│   ├── config.yaml
│   └── intents.yaml
│
├── data/
│   ├── README.md
│   ├── raw/
│   ├── processed/
│   └── golden/
│       ├── golden_200.jsonl
│       └── README.md
│
├── src/
│   ├── ingestion/
│   ├── analysis/
│   ├── classification/
│   ├── retrieval/
│   ├── generation/
│   ├── escalation/
│   └── agent.py
│
├── evaluation/
│   ├── run_eval.py
│   ├── metrics.py
│   ├── llm_judge.py
│   ├── human_judge.py
│   ├── ablation.py
│   └── results/
│
├── notebooks/
│   ├── 01_brand_analysis.ipynb
│   ├── 02_intent_analysis.ipynb
│   └── 03_failure_analysis.ipynb
│
├── reports/
│   ├── report.md
│   └── decision_log.md
│
├── tests/
│
└── app.py

---

## 30. ENGINEERING QUALITY

---

Write production-quality code.

Requirements:

* Python type hints
* docstrings for important components
* structured logging
* error handling
* configuration-driven behavior
* modular architecture
* no unnecessary abstraction
* no hard-coded secrets
* no hard-coded brand
* no fake results
* no unexplained magic numbers
* unit tests
* reproducible experiments

Avoid overengineering.

A simple working system with strong evaluation is better than a complicated system with weak evaluation.

---

## 31. IMPORTANT DATA LEAKAGE RULE

---

This is one of the most important requirements.

The golden evaluation set must not contaminate:

* training
* intent discovery
* retrieval index
* threshold tuning
* prompt development

At minimum, enforce conversation-level separation.

If possible, use:

EARLIER CONVERSATIONS
↓
TRAIN / RETRIEVAL

LATER CONVERSATIONS
↓
EVALUATION

Document exactly how leakage was prevented.

---

## 32. BANKING77

---

Banking77 is optional.

Use it ONLY if it provides useful support for intent-method experimentation.

Do not mix Banking77 labels with the selected brand's final taxonomy.

The final evaluation must represent the selected Twitter brand.

Explain whether Banking77 was used and why.

---

## 33. FINAL ACCEPTANCE CRITERIA

---

Do not consider the project complete until:

[ ] Brand selected using dataset evidence
[ ] Conversation reconstruction works
[ ] Intent taxonomy documented
[ ] Golden set contains 150–250 examples
[ ] Majority baseline works
[ ] TF-IDF baseline works
[ ] LLM classifier works
[ ] Historical RAG works
[ ] Reply generation works
[ ] Escalation layer works
[ ] Evaluation harness works
[ ] LLM judge works
[ ] Human-vs-LLM judge validation exists
[ ] Ablation study exists
[ ] Five real failure modes documented
[ ] "What is misleading about my headline number?" exists
[ ] Decision log contains 10–15 decisions
[ ] Report is <=6 pages
[ ] README is reproducible
[ ] Tests pass
[ ] No secrets committed
[ ] No evaluation leakage
[ ] No fabricated metrics

---

## 34. DEVELOPMENT WORKFLOW

---

Work in phases.

PHASE A:
Inspect repository and dataset.

PHASE B:
Analyze brands and select one.

PHASE C:
Reconstruct conversations.

PHASE D:
Define intents.

PHASE E:
Create golden set.

PHASE F:
Implement baselines.

PHASE G:
Implement RAG agent.

PHASE H:
Implement escalation.

PHASE I:
Implement evaluation.

PHASE J:
Implement LLM judge + human calibration.

PHASE K:
Failure analysis.

PHASE L:
Report + decision log + README.

PHASE M:
Testing + final cleanup.

After each major phase:

1. Run the relevant code.
2. Inspect results.
3. Fix errors.
4. Update documentation.
5. Only then continue.

Do not generate large amounts of code without running it.

---

## 35. FIRST RESPONSE / FIRST ACTION

---

Your first task is NOT to build the entire system.

Start by:

1. Inspecting the repository.
2. Finding the dataset.
3. Determining its schema.
4. Running a lightweight data analysis.
5. Ranking candidate brands.

Then report:

* dataset structure
* available columns
* number of records
* top candidate brands
* recommended brand
* why that brand is the best choice
* any data-quality problems discovered
* proposed next implementation step

Do not proceed to the final RAG implementation until the dataset and brand selection are understood.

After presenting the analysis, continue implementing the project unless a critical ambiguity blocks progress.

The final objective is a technically credible, reproducible, evaluation-first AI customer-support system—not merely a chatbot demo.
