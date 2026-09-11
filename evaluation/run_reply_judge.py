"""Phase J: LLM-as-judge reply-quality evaluation, human-validated.

Run:
    python -m evaluation.run_reply_judge

Stages (in order):
  1. Load & verify the 5 genuine human ratings (data/evaluation/human_reply_ratings.jsonl).
  2. Fill the other 20/25 validation-subset examples with Claude-simulated
     ratings (data/evaluation/claude_simulated_ratings.jsonl) -- explicitly
     NOT human data, kept in a separate file (see
     src/evaluation/simulated_rater_prompt.py's module docstring).
  3. Run the frozen LLM judge on all 25 validation-subset examples, both
     No-RAG and RAG replies (evaluation/results/judge_validation_subset.jsonl).
  4. Compute judge-vs-human (n=5) and judge-vs-simulated (n=20) agreement
     (evaluation/results/judge_agreement.json).
  5. Run the frozen judge on the full frozen 200 No-RAG + 200 RAG replies
     (evaluation/results/judge_no_rag.jsonl, judge_rag.jsonl).
  6. Paired RAG vs No-RAG comparison + subgroup + statistical tests
     (evaluation/results/rag_vs_no_rag.json).

Reads only frozen Phase G/H/I artifacts and the human-rating subset built
in Phase J's earlier step; never modifies them. All LLM calls are cached
(data/processed/llm_judge/, data/processed/claude_simulated_ratings/) so
re-running only retries examples that previously failed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from src.classification.llm_providers import LLMConfig
from src.evaluation.judge_runner import judge_reply
from src.evaluation.simulated_rater_runner import simulated_rate_reply

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GOLDEN_PATH = Path("data/golden/golden_200.jsonl")
NO_RAG_PATH = Path("evaluation/results/replies_no_rag.jsonl")
RAG_PATH = Path("evaluation/results/replies_rag.jsonl")
ESCALATION_PATH = Path("evaluation/results/escalation_results.json")
SUBSET_PATH = Path("data/evaluation/human_rating_subset.jsonl")
HUMAN_RATINGS_PATH = Path("data/evaluation/human_reply_ratings.jsonl")
SIMULATED_RATINGS_PATH = Path("data/evaluation/claude_simulated_ratings.jsonl")

JUDGE_SUBSET_PATH = Path("evaluation/results/judge_validation_subset.jsonl")
AGREEMENT_PATH = Path("evaluation/results/judge_agreement.json")
JUDGE_NO_RAG_PATH = Path("evaluation/results/judge_no_rag.jsonl")
JUDGE_RAG_PATH = Path("evaluation/results/judge_rag.jsonl")
RAG_VS_NORAG_PATH = Path("evaluation/results/rag_vs_no_rag.json")

DIMS = ["relevance_score", "helpfulness_score", "grounding_score", "factual_safety_score",
        "resolution_score", "uncertainty_score", "professionalism_score"]
EQUIVALENCE_MARGIN = 0.30  # predefined BEFORE seeing any full-dataset results (see decision log)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def load_frozen_data():
    golden = {g["id"]: g for g in _load_jsonl(GOLDEN_PATH)}
    no_rag = {r["example_id"]: r for r in _load_jsonl(NO_RAG_PATH)}
    rag = {r["example_id"]: r for r in _load_jsonl(RAG_PATH)}
    escalation = {r["example_id"]: r for r in json.loads(ESCALATION_PATH.read_text(encoding="utf-8"))["per_example"]}
    return golden, no_rag, rag, escalation


# ---------------------------------------------------------------------------
# Stage 1-2: human ratings + Claude-simulated fill-in for the rest of the 25
# ---------------------------------------------------------------------------

def verify_human_ratings() -> list[str]:
    if not HUMAN_RATINGS_PATH.exists():
        raise RuntimeError(f"{HUMAN_RATINGS_PATH} not found -- run the human-rating extraction step first")
    records = _load_jsonl(HUMAN_RATINGS_PATH)
    ids = [r["example_id"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate example_id in human ratings"
    for r in records:
        assert r["rater_type"] == "human"
        for prefix in ("no_rag", "rag"):
            dims = r[f"{prefix}_dimension_scores"]
            assert set(dims.keys()) == set(DIMS), f"{r['example_id']} {prefix} missing dimensions"
            for k, v in dims.items():
                assert isinstance(v, int) and 1 <= v <= 5, f"{r['example_id']} {prefix}.{k}={v} invalid"
    logger.info("Verified %d genuine human ratings: %s", len(ids), ids)
    return ids


def fill_simulated_ratings(human_ids: list[str], golden, no_rag, rag) -> None:
    subset = _load_jsonl(SUBSET_PATH)
    remaining = [r["example_id"] for r in subset if r["example_id"] not in human_ids]
    logger.info("Filling %d Claude-simulated ratings (NOT human data): %s", len(remaining), remaining)

    config = LLMConfig.from_env()
    records = []
    for gid in remaining:
        g, nr, rr = golden[gid], no_rag[gid], rag[gid]
        nr_result = simulated_rate_reply(
            gid, g["customer_message"], g["conversation"], nr["predicted_intent"],
            nr["reply"], None, "no_rag", config=config,
        )
        rr_result = simulated_rate_reply(
            gid, g["customer_message"], g["conversation"], rr["predicted_intent"],
            rr["reply"], rr["retrieved_evidence"], "rag", config=config,
        )
        records.append({
            "example_id": gid, "rater_type": "claude_simulated",
            "no_rag_dimension_scores": {k: getattr_dict(nr_result, k) for k in DIMS},
            "no_rag_overall_score": nr_result.overall_score,
            "no_rag_audit_notes": nr_result.audit_notes,
            "rag_dimension_scores": {k: getattr_dict(rr_result, k) for k in DIMS},
            "rag_overall_score": rr_result.overall_score,
            "rag_audit_notes": rr_result.audit_notes,
            "no_rag_error_kind": nr_result.error_kind, "rag_error_kind": rr_result.error_kind,
        })
        logger.info("simulated %s: no_rag=%s rag=%s", gid, nr_result.overall_score, rr_result.overall_score)

    with SIMULATED_RATINGS_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %s", SIMULATED_RATINGS_PATH)


def getattr_dict(obj, key):
    return getattr(obj, key)


# ---------------------------------------------------------------------------
# Stage 3: run the frozen judge on all 25 validation-subset examples
# ---------------------------------------------------------------------------

def run_judge_on_validation_subset(golden, no_rag, rag) -> None:
    subset = _load_jsonl(SUBSET_PATH)
    config = LLMConfig.from_env()
    records = []
    for r in subset:
        gid = r["example_id"]
        g, nr, rr = golden[gid], no_rag[gid], rag[gid]
        nr_j = judge_reply(gid, g["customer_message"], g["conversation"], nr["predicted_intent"],
                            nr["reply"], None, "no_rag", config=config)
        rr_j = judge_reply(gid, g["customer_message"], g["conversation"], rr["predicted_intent"],
                            rr["reply"], rr["retrieved_evidence"], "rag", config=config)
        records.append(nr_j.to_dict())
        records.append(rr_j.to_dict())
        logger.info("judge(validation subset) %s: no_rag=%s rag=%s", gid, nr_j.overall_score, rr_j.overall_score)

    with JUDGE_SUBSET_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %s (%d records)", JUDGE_SUBSET_PATH, len(records))


# ---------------------------------------------------------------------------
# Stage 4: agreement metrics
# ---------------------------------------------------------------------------

def _agreement_metrics(human_vals: list[float], judge_vals: list[float]) -> dict:
    h, j = np.array(human_vals, dtype=float), np.array(judge_vals, dtype=float)
    n = len(h)
    mae = float(np.mean(np.abs(h - j)))
    pearson_r, pearson_p = (stats.pearsonr(h, j) if n >= 3 and np.std(h) > 0 and np.std(j) > 0 else (None, None))
    spearman_r, spearman_p = (stats.spearmanr(h, j) if n >= 3 else (None, None))
    exact = float(np.mean(h == j))
    adjacent = float(np.mean(np.abs(h - j) <= 1))
    return {
        "n": n, "mae": mae,
        "pearson_r": float(pearson_r) if pearson_r is not None else None,
        "pearson_p": float(pearson_p) if pearson_p is not None else None,
        "spearman_r": float(spearman_r) if spearman_r is not None else None,
        "spearman_p": float(spearman_p) if spearman_p is not None else None,
        "exact_agreement_rate": exact, "adjacent_agreement_rate": adjacent,
        "mean_human": float(np.mean(h)), "mean_judge": float(np.mean(j)),
        "mean_bias_judge_minus_human": float(np.mean(j - h)),
    }


def compute_agreement() -> dict:
    human = {r["example_id"]: r for r in _load_jsonl(HUMAN_RATINGS_PATH)}
    simulated = {r["example_id"]: r for r in _load_jsonl(SIMULATED_RATINGS_PATH)}
    judge_records = _load_jsonl(JUDGE_SUBSET_PATH)
    judge = {}
    for r in judge_records:
        judge.setdefault(r["example_id"], {})[r["version"]] = r

    def build_pairs(rater_dict, rater_dim_key_fmt, rater_overall_key_fmt, ids):
        overall_h, overall_j = [], []
        dim_pairs = {d: ([], []) for d in DIMS}
        for gid in ids:
            rr = rater_dict[gid]
            for prefix, version in (("no_rag", "no_rag"), ("rag", "rag")):
                jr = judge[gid][version]
                if jr["overall_score"] is None:
                    continue  # judge call failed -- exclude, never impute
                overall_h.append(rr[f"{prefix}_overall_score"])
                overall_j.append(jr["overall_score"])
                for d in DIMS:
                    dim_pairs[d][0].append(rr[f"{prefix}_dimension_scores"][d])
                    dim_pairs[d][1].append(jr[d])
        return overall_h, overall_j, dim_pairs

    human_ids = list(human.keys())
    sim_ids = list(simulated.keys())

    h_overall, j_overall_h, h_dims = build_pairs(human, None, None, human_ids)
    s_overall, j_overall_s, s_dims = build_pairs(simulated, None, None, sim_ids)

    result = {
        "note": (
            "human_vs_judge uses the 5 GENUINE human-rated examples only -- this is the "
            "primary (and only real) human-validation evidence, with n=5 explicitly a very "
            "small sample. claude_simulated_vs_judge (n=20) is NOT human validation -- it is "
            "a supplementary self-consistency check between two differently-prompted LLM "
            "evaluation passes and must never be cited as evidence of human agreement."
        ),
        "human_vs_judge": {
            "overall": _agreement_metrics(h_overall, j_overall_h),
            "by_dimension": {d: _agreement_metrics(h_dims[d][0], h_dims[d][1]) for d in DIMS},
        },
        "claude_simulated_vs_judge": {
            "overall": _agreement_metrics(s_overall, j_overall_s),
            "by_dimension": {d: _agreement_metrics(s_dims[d][0], s_dims[d][1]) for d in DIMS},
        },
    }

    # weighted Cohen's kappa on the overall score, rounded to nearest int 1-5
    if len(h_overall) >= 2:
        h_int = np.clip(np.round(h_overall), 1, 5).astype(int)
        j_int = np.clip(np.round(j_overall_h), 1, 5).astype(int)
        result["human_vs_judge"]["overall"]["weighted_cohen_kappa"] = float(
            cohen_kappa_score(h_int, j_int, weights="linear", labels=[1, 2, 3, 4, 5])
        )
    if len(s_overall) >= 2:
        s_int = np.clip(np.round(s_overall), 1, 5).astype(int)
        js_int = np.clip(np.round(j_overall_s), 1, 5).astype(int)
        result["claude_simulated_vs_judge"]["overall"]["weighted_cohen_kappa"] = float(
            cohen_kappa_score(s_int, js_int, weights="linear", labels=[1, 2, 3, 4, 5])
        )

    AGREEMENT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s", AGREEMENT_PATH)
    return result


# ---------------------------------------------------------------------------
# Stage 5: full 400-reply judge evaluation
# ---------------------------------------------------------------------------

def run_full_judge(golden, no_rag, rag) -> None:
    config = LLMConfig.from_env()
    for version, replies, out_path in (("no_rag", no_rag, JUDGE_NO_RAG_PATH), ("rag", rag, JUDGE_RAG_PATH)):
        records = []
        for i, gid in enumerate(golden, 1):
            g, rep = golden[gid], replies[gid]
            evidence = rep["retrieved_evidence"] if version == "rag" else None
            jr = judge_reply(gid, g["customer_message"], g["conversation"], rep["predicted_intent"],
                              rep["reply"], evidence, version, config=config)
            records.append(jr.to_dict())
            if i % 25 == 0 or i == len(golden):
                logger.info("judge(%s): %d/%d", version, i, len(golden))
        with out_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Wrote %s (%d records)", out_path, len(records))


# ---------------------------------------------------------------------------
# Stage 6: paired RAG vs No-RAG comparison
# ---------------------------------------------------------------------------

def compute_rag_vs_no_rag(golden, escalation) -> dict:
    judge_no_rag = {r["example_id"]: r for r in _load_jsonl(JUDGE_NO_RAG_PATH)}
    judge_rag = {r["example_id"]: r for r in _load_jsonl(JUDGE_RAG_PATH)}
    no_rag_replies = {r["example_id"]: r for r in _load_jsonl(NO_RAG_PATH)}
    rag_replies = {r["example_id"]: r for r in _load_jsonl(RAG_PATH)}

    pairs = []
    for gid in golden:
        nr, rr = judge_no_rag[gid], judge_rag[gid]
        if nr["overall_score"] is None or rr["overall_score"] is None:
            continue
        diff = rr["overall_score"] - nr["overall_score"]
        if diff > EQUIVALENCE_MARGIN:
            pref = "rag_better"
        elif diff < -EQUIVALENCE_MARGIN:
            pref = "no_rag_better"
        else:
            pref = "equivalent"
        pairs.append({
            "example_id": gid, "no_rag_score": nr["overall_score"], "rag_score": rr["overall_score"],
            "diff": diff, "preference": pref,
            "has_evidence": bool(rag_replies[gid]["retrieved_evidence"]),
            "escalation_action": escalation[gid]["policy_action"],
            "true_intent": golden[gid]["intent"],
        })

    def summarize(subset):
        if not subset:
            return None
        no_rag_scores = [p["no_rag_score"] for p in subset]
        rag_scores = [p["rag_score"] for p in subset]
        diffs = [p["diff"] for p in subset]
        n = len(subset)
        rag_better = sum(1 for p in subset if p["preference"] == "rag_better")
        no_rag_better = sum(1 for p in subset if p["preference"] == "no_rag_better")
        equivalent = sum(1 for p in subset if p["preference"] == "equivalent")
        return {
            "n": n,
            "mean_no_rag": float(np.mean(no_rag_scores)), "mean_rag": float(np.mean(rag_scores)),
            "median_no_rag": float(np.median(no_rag_scores)), "median_rag": float(np.median(rag_scores)),
            "mean_diff_rag_minus_no_rag": float(np.mean(diffs)),
            "rag_better_n": rag_better, "rag_better_pct": rag_better / n,
            "no_rag_better_n": no_rag_better, "no_rag_better_pct": no_rag_better / n,
            "equivalent_n": equivalent, "equivalent_pct": equivalent / n,
        }

    overall = summarize(pairs)

    # Statistical test: Wilcoxon signed-rank (paired, ordinal-appropriate,
    # no normality assumption) + paired bootstrap CI on the mean difference.
    diffs_arr = np.array([p["diff"] for p in pairs])
    nonzero = diffs_arr[diffs_arr != 0]
    if len(nonzero) >= 10:
        wstat, wp = stats.wilcoxon(diffs_arr)
    else:
        wstat, wp = None, None

    rng = np.random.default_rng(20260910)
    boot_means = [np.mean(rng.choice(diffs_arr, size=len(diffs_arr), replace=True)) for _ in range(10000)]
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    stat_result = {
        "test": "Wilcoxon signed-rank on paired (RAG - No-RAG) overall-score differences",
        "wilcoxon_statistic": float(wstat) if wstat is not None else None,
        "wilcoxon_p_value": float(wp) if wp is not None else None,
        "n_nonzero_diffs": int(len(nonzero)),
        "bootstrap_mean_diff_95ci": {
            "method": "paired bootstrap, 10000 resamples, seed=20260910",
            "point_estimate": float(np.mean(diffs_arr)),
            "ci_low": float(ci_low), "ci_high": float(ci_high),
        },
    }

    subgroups = {
        "sufficient_evidence": summarize([p for p in pairs if p["has_evidence"]]),
        "insufficient_evidence": summarize([p for p in pairs if not p["has_evidence"]]),
        "escalate": summarize([p for p in pairs if p["escalation_action"] == "ESCALATE"]),
        "auto_handle": summarize([p for p in pairs if p["escalation_action"] == "AUTO_HANDLE"]),
    }
    by_intent = {}
    for intent in sorted(set(p["true_intent"] for p in pairs)):
        by_intent[intent] = summarize([p for p in pairs if p["true_intent"] == intent])

    result = {
        "equivalence_margin": EQUIVALENCE_MARGIN,
        "equivalence_margin_note": "Predefined before any full-dataset judge results were seen (see reports/decision_log.md).",
        "overall": overall,
        "statistical_test": stat_result,
        "subgroups": subgroups,
        "by_intent": by_intent,
        "pairs": pairs,
    }
    RAG_VS_NORAG_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s", RAG_VS_NORAG_PATH)
    return result


def main() -> None:
    golden, no_rag, rag, escalation = load_frozen_data()

    logger.info("Stage 1: verify human ratings")
    human_ids = verify_human_ratings()

    logger.info("Stage 2: fill Claude-simulated ratings for remaining %d examples", 25 - len(human_ids))
    fill_simulated_ratings(human_ids, golden, no_rag, rag)

    logger.info("Stage 3: run frozen judge on all 25 validation-subset examples")
    run_judge_on_validation_subset(golden, no_rag, rag)

    logger.info("Stage 4: compute judge-vs-human and judge-vs-simulated agreement")
    compute_agreement()

    logger.info("Stage 5: run frozen judge on full 200 No-RAG + 200 RAG")
    run_full_judge(golden, no_rag, rag)

    logger.info("Stage 6: paired RAG vs No-RAG comparison")
    compute_rag_vs_no_rag(golden, escalation)

    logger.info("Phase J evaluation complete.")


if __name__ == "__main__":
    main()
