# Golden Evaluation Set — Methodology

## Status

`golden_candidates.jsonl` (200 rows) exists now, produced by
`src/evaluation/sample_golden.py`. It is a **sampled, unlabeled** candidate
pool -- `intent`, `expected_action`, `expected_reason`, and `label_notes` are
all `null`. **`golden_200.jsonl` does not exist yet.** It will be created by a
human labeling pass over these candidates, once the taxonomy in
`config/intents.yaml` has been reviewed and confirmed stable. This file
documents the sampling methodology now and the labeling methodology that
pass must follow.

## Leakage prevention

Every candidate is drawn from `eval_pool` conversations only (the later ~15%
of AppleSupport conversations by root-tweet timestamp, cutoff
2017-11-21T07:55:44+00:00 -- see `reports/conversation_reconstruction.md`).
`retrieval_pool` conversations (the earlier ~85%) are what the intent
taxonomy, retrieval corpus, prompts, and thresholds are and will be built
from. A `conversation_id` belongs to exactly one split by construction, so no
golden example can also appear in the retrieval corpus.

The one deliberate exception: `sample_golden.py` reuses the Phase D keyword
tagger (`config/intents.yaml`'s `keywords` field, derived entirely from
`retrieval_pool` clustering) to compute a `suggested_intent` for stratifying
the sample and to help a human labeler start faster. This is a **fixed,
already-frozen** tool applied *to* eval_pool for bucketing purposes -- it is
not adjusted based on what's in eval_pool, so it does not leak eval_pool
information back into taxonomy design, retrieval, or prompts.

## Sampling methodology

Target: 200 examples (within the 150-250 range), stratified sampling with a
fixed seed (42), no conversation used twice, in this order (later strata
skip any conversation already used by an earlier one):

1. **Per-intent floor** (10 per intent x 14 intents = up to 140 slots): every
   intent in the taxonomy gets a guaranteed minimum presence, including the
   rare ones (`apple_watch_issues`, `app_store_purchase_issues`) that pure
   proportional sampling would round down to almost nothing.
2. **Ambiguous / multi-intent** (target 15): messages whose keyword tags hit
   more than one specific intent, or fall in the residual bucket -- the
   hardest classification cases.
3. **Short messages** (target 15): single-turn conversations, where there's
   the least context to work with.
4. **High frustration** (target 12): messages ranked by a simple lexicon +
   ALL-CAPS + repeated-punctuation score, taking the most frustrated.
5. **Unanswered / customer-only** (target 8): conversations with no historical
   agent reply -- there is no "what actually happened" to fall back on, so
   these specifically test the auto-handle-vs-escalate judgment without
   retrieval evidence.
6. **Escalate-by-default intents, oversampled** (target 10 beyond their
   per-intent floor): `account_apple_id_security` and
   `order_purchase_retail_support` are the taxonomy's designated
   always-escalate intents; the golden set needs enough of them to actually
   measure escalation precision/recall, not just classification accuracy.
7. **Context-dependent** (target 30, ~15% of 200): the target
   `customer_message` is deliberately the **last** customer turn, not the
   root, in conversations with multiple customer turns and at least one
   agent reply -- `conversation` holds everything before it. Classifying
   these requires reading the conversation, not just the one message (e.g.
   "Yes, this shows up" only makes sense after the agent's question).
8. **Random fill**: whatever's left to reach exactly 200, drawn from
   remaining eval_pool conversations, so long-tail patterns not explicitly
   targeted above still have a chance to appear.

Every candidate row also carries `sampling_stratum` (which step selected it)
and `suggested_intent` / `suggested_intent_all_matches` (the keyword tagger's
guess) so the human labeler can audit *why* an example was included and use
the suggestion as a starting point they're free to override.

## Labeling methodology (to be executed, not yet done)

For each candidate, a human labeler must fill in:

- **`intent`**: one name from `config/intents.yaml`. The `suggested_intent`
  is a starting point, not an answer -- Phase D found the keyword tagger has
  real recall gaps (see "Discovered during validation" in
  `reports/intent_discovery.md`), so every suggestion must be verified
  against the actual message and conversation context, not accepted as-is.
- **`expected_action`**: `auto_handle` or `escalate`. Follow the intent's
  `escalation_guidance` in `config/intents.yaml` as the default, but use
  judgment for the specific message -- e.g. an otherwise auto-handle intent
  should still be labeled `escalate` if the customer reports a safety issue,
  has already tried the standard fix, or the message shows repeated
  unresolved frustration.
- **`expected_reason`**: one sentence, concrete and specific to this
  message -- not a restatement of the intent's generic escalation_guidance.
- **`label_notes`**: anything a future reader needs to know -- especially
  *why*, if the example was hard to label, what made it ambiguous, or if the
  labeler disagreed with `suggested_intent`.

### Handling ambiguous examples

If a message genuinely fits more than one intent, pick the intent that
determines the **correct next action** (not necessarily the first-mentioned
symptom), and record the alternative(s) in `label_notes`. If a message truly
has no actionable content (see `vague_frustration_needs_clarification`),
label the intent as that, and set `expected_action` based on whether a
clarifying question is itself "auto_handle" (yes, in this taxonomy --
asking a clarifying question is the correct automated behavior, not a
failure to automate) or whether the surrounding context already makes
escalation obvious.

### Context-dependent examples

For the `context_dependent` stratum, label based on the full `conversation`
array plus the target `customer_message` together -- the intent and action
must reflect what the customer is asking *at that point in the thread*, which
may differ from the intent of their first message.

## Intent definitions

See `config/intents.yaml` for the full taxonomy (14 intents) and
`reports/intent_discovery.md` for frequency, rationale, real examples, and
the confusion-risk matrix each intent was checked against.
