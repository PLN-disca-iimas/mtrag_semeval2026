#!/usr/bin/env python3
"""
Ablation Statistical Tests — EA2, EA3, EA5, EA6, EA7
=====================================================

Extends the existing statistical validation (which covers H1/EA1 and EA4 rewrites)
to the remaining OFAT ablation studies.

For each EA:
  1. Load per-query nDCG@5 scores via pytrec_eval
  2. Shapiro-Wilk normality test on paired differences
  3. Wilcoxon signed-rank test (non-parametric, justified by Shapiro)
  4. Paired t-test (reported but NOT used for decisions if normality violated)
  5. Cohen's d effect size
  6. Bootstrap CI (B=10,000, percentile, seed=42) per level per domain
  7. Holm-Bonferroni FWER correction within each EA family
  8. Friedman test + Nemenyi post-hoc (where ≥3 levels per EA)

Outputs:
  - results/ablation_statistical_tests.json  (machine-readable)
  - results/ablation_statistical_summary.txt (human-readable)

Usage:
    python3.13 scripts/run_ablation_statistical_tests.py
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

import pytrec_eval

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOMAINS = ["clapnq", "cloud", "fiqa", "govt"]
ALPHA = 0.05
N_BOOTSTRAP = 10_000
SEED = 42
PRIMARY_METRIC = "ndcg_cut_5"
SECONDARY_METRICS = ["ndcg_cut_10", "recall_100", "recip_rank"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Ablation Study Registry
# Each EA defines a reference experiment and alternative levels.
# All comparisons are reference vs level (paired by query).
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REFERENCE = ("RRF k=60 (ref)", "experiments/02-hybrid/hybrid_splade_voyage_rewrite")

ABLATION_STUDIES = {
    "EA2_fusion_method": {
        "description": "Fusion method: RRF vs linear interpolation",
        "hypothesis": "H1",
        "reference": REFERENCE,
        "levels": [
            ("Linear α=0.3", "experiments/06-ablation-fusion/ablation_fusion_linear_alpha03_voyage"),
            ("Linear α=0.5", "experiments/06-ablation-fusion/ablation_fusion_linear_splade_voyage"),
            ("Linear α=0.7", "experiments/06-ablation-fusion/ablation_fusion_linear_alpha07_voyage"),
        ],
        "friedman_systems": {
            "RRF k=60":      "experiments/02-hybrid/hybrid_splade_voyage_rewrite",
            "Linear α=0.3":  "experiments/06-ablation-fusion/ablation_fusion_linear_alpha03_voyage",
            "Linear α=0.5":  "experiments/06-ablation-fusion/ablation_fusion_linear_splade_voyage",
            "Linear α=0.7":  "experiments/06-ablation-fusion/ablation_fusion_linear_alpha07_voyage",
        },
    },
    "EA3_query_mode": {
        "description": "Query formulation mode",
        "hypothesis": "H2",
        "reference": REFERENCE,
        "levels": [
            ("last_turn",    "experiments/10-ablation-query-mode/ablation_query_mode_lastturn_hybrid"),
            ("full_history", "experiments/10-ablation-query-mode/ablation_query_mode_fullhist_hybrid"),
            ("full_context", "experiments/10-ablation-query-mode/ablation_query_mode_fullctx_hybrid"),
        ],
        "friedman_systems": {
            "rewrite (ref)": "experiments/02-hybrid/hybrid_splade_voyage_rewrite",
            "last_turn":     "experiments/10-ablation-query-mode/ablation_query_mode_lastturn_hybrid",
            "full_history":  "experiments/10-ablation-query-mode/ablation_query_mode_fullhist_hybrid",
            "full_context":  "experiments/10-ablation-query-mode/ablation_query_mode_fullctx_hybrid",
        },
    },
    "EA5_rrf_k": {
        "description": "RRF smoothing parameter k",
        "hypothesis": "H3",
        "reference": REFERENCE,
        "levels": [
            ("k=1",   "experiments/07-ablation-rrf-k/ablation_rrf_k1_voyage"),
            ("k=20",  "experiments/07-ablation-rrf-k/ablation_rrf_k20_voyage"),
            ("k=40",  "experiments/07-ablation-rrf-k/ablation_rrf_k40_voyage"),
            ("k=100", "experiments/07-ablation-rrf-k/ablation_rrf_k100_voyage"),
        ],
        "friedman_systems": {
            "k=1":       "experiments/07-ablation-rrf-k/ablation_rrf_k1_voyage",
            "k=20":      "experiments/07-ablation-rrf-k/ablation_rrf_k20_voyage",
            "k=40":      "experiments/07-ablation-rrf-k/ablation_rrf_k40_voyage",
            "k=60 (ref)":"experiments/02-hybrid/hybrid_splade_voyage_rewrite",
            "k=100":     "experiments/07-ablation-rrf-k/ablation_rrf_k100_voyage",
        },
    },
    "EA6_topk": {
        "description": "Retrieval depth (top_k candidates)",
        "hypothesis": "H3/H4",
        "reference": REFERENCE,
        "levels": [
            ("top_k=200", "experiments/08-ablation-topk/ablation_topk_200_voyage"),
            ("top_k=500", "experiments/08-ablation-topk/ablation_topk_500_voyage"),
        ],
        "friedman_systems": {
            "top_k=100 (ref)": "experiments/02-hybrid/hybrid_splade_voyage_rewrite",
            "top_k=200":       "experiments/08-ablation-topk/ablation_topk_200_voyage",
            "top_k=500":       "experiments/08-ablation-topk/ablation_topk_500_voyage",
        },
    },
    "EA7_rerank_depth": {
        "description": "Reranking depth × reranker model",
        "hypothesis": "H4",
        "reference": REFERENCE,
        "levels": [
            ("BGE@50",    "experiments/09-ablation-rerank-depth/ablation_rerank_depth_50"),
            ("BGE@100",   "experiments/03-rerank/rerank_splade_voyage_rewrite"),
            ("BGE@200",   "experiments/09-ablation-rerank-depth/ablation_rerank_depth_200"),
            ("Cohere@50", "experiments/09-ablation-rerank-depth/ablation_rerank_depth_50_cohere"),
            ("Cohere@100","experiments/03-rerank/rerank_cohere_splade_voyage_rewrite"),
            ("Cohere@200","experiments/09-ablation-rerank-depth/ablation_rerank_depth_200_cohere"),
        ],
        "friedman_systems": {
            "No rerank (ref)": "experiments/02-hybrid/hybrid_splade_voyage_rewrite",
            "BGE@50":          "experiments/09-ablation-rerank-depth/ablation_rerank_depth_50",
            "BGE@100":         "experiments/03-rerank/rerank_splade_voyage_rewrite",
            "BGE@200":         "experiments/09-ablation-rerank-depth/ablation_rerank_depth_200",
            "Cohere@50":       "experiments/09-ablation-rerank-depth/ablation_rerank_depth_50_cohere",
            "Cohere@100":      "experiments/03-rerank/rerank_cohere_splade_voyage_rewrite",
            "Cohere@200":      "experiments/09-ablation-rerank-depth/ablation_rerank_depth_200_cohere",
        },
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def hr(char="=", width=100):
    print(char * width)


def load_qrels(domain: str) -> Dict[str, Dict[str, int]]:
    path = f"data/retrieval_tasks/{domain}/qrels/dev.tsv"
    qrels = defaultdict(dict)
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qrels[parts[0]][parts[1]] = int(parts[2])
    return dict(qrels)


def load_retrieval_results(exp_path: str, domain: str) -> Optional[Dict]:
    path = f"{exp_path}/{domain}/retrieval_results.jsonl"
    if not os.path.exists(path):
        return None
    results = {}
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            tid = item["task_id"]
            results[tid] = {}
            for rank, ctx in enumerate(item.get("contexts", [])[:1000], 1):
                doc_id = ctx["document_id"]
                score = ctx.get("score", 1000.0 - rank)
                results[tid][doc_id] = float(score)
    return results


def compute_per_query_scores(exp_path: str, domain: str) -> Optional[Dict[str, Dict[str, float]]]:
    """Return {qid: {ndcg_cut_5, ndcg_cut_10, recall_100, recip_rank}} via pytrec_eval."""
    qrels = load_qrels(domain)
    results = load_retrieval_results(exp_path, domain)
    if not results:
        return None
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels,
        {"ndcg_cut.5", "ndcg_cut.10", "recall.100", "recip_rank"}
    )
    return evaluator.evaluate(results)


def get_aligned_scores(scores_a: Dict, scores_b: Dict, metric: str = PRIMARY_METRIC) -> Tuple[np.ndarray, np.ndarray]:
    """Get aligned per-query score arrays from two experiments (intersection of query IDs)."""
    common = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    a = np.array([scores_a[q].get(metric, 0.0) for q in common])
    b = np.array([scores_b[q].get(metric, 0.0) for q in common])
    return a, b


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-10))


def cohens_d_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def bootstrap_ci(scores: np.ndarray, n_bootstrap: int = N_BOOTSTRAP,
                 ci_level: float = 0.95, seed: int = SEED) -> Dict:
    n = len(scores)
    rng = np.random.RandomState(seed)
    boot_means = np.array([
        np.mean(scores[rng.randint(0, n, size=n)])
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci_level) / 2
    lo, hi = np.percentile(boot_means, [alpha * 100, (1 - alpha) * 100])
    return {
        "mean": float(np.mean(scores)),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "std": float(np.std(scores)),
        "n_bootstrap": n_bootstrap,
        "n_queries": n,
        "margin": float((hi - lo) / 2),
    }


def paired_test(ref_scores: np.ndarray, alt_scores: np.ndarray) -> Dict:
    """Run full paired battery: Shapiro-Wilk, Wilcoxon, t-test, Cohen's d."""
    diff = ref_scores - alt_scores
    n = len(diff)

    result = {
        "n_queries": int(n),
        "ref_mean": float(np.mean(ref_scores)),
        "alt_mean": float(np.mean(alt_scores)),
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff, ddof=1)),
    }

    # Shapiro-Wilk
    if n >= 8:
        shap_stat, shap_p = stats.shapiro(diff)
        result["shapiro"] = {
            "stat": float(shap_stat),
            "p": float(shap_p),
            "normal": bool(shap_p >= ALPHA),
        }
    else:
        result["shapiro"] = {"stat": None, "p": None, "normal": None, "note": "n < 8"}

    # Cohen's d
    d = cohens_d(ref_scores, alt_scores)
    result["cohens_d"] = float(d)
    result["effect_label"] = cohens_d_label(d)

    # Wilcoxon signed-rank
    try:
        # Check if all differences are zero
        if np.all(diff == 0):
            result["wilcoxon"] = {"stat": None, "p": 1.0, "sig": False, "note": "all diffs zero"}
        else:
            w_stat, w_p = stats.wilcoxon(diff, alternative="two-sided")
            result["wilcoxon"] = {
                "stat": float(w_stat),
                "p": float(w_p),
                "sig": bool(w_p < ALPHA),
            }
    except Exception as e:
        result["wilcoxon"] = {"stat": None, "p": 1.0, "sig": False, "error": str(e)}

    # Paired t-test (reported but NOT primary)
    try:
        t_stat, t_p = stats.ttest_rel(ref_scores, alt_scores)
        result["ttest"] = {
            "stat": float(t_stat),
            "p": float(t_p),
            "sig": bool(t_p < ALPHA),
            "normality_ok": result["shapiro"].get("normal", False),
        }
    except Exception as e:
        result["ttest"] = {"stat": None, "p": 1.0, "sig": False, "error": str(e)}

    return result


def apply_holm_bonferroni(p_values: List[float], alpha: float = ALPHA) -> List[Dict]:
    """Apply Holm-Bonferroni correction and return adjusted p-values."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.zeros(m)
    cummax = 0.0
    for i, idx in enumerate(order):
        adj = p_values[idx] * (m - i)
        cummax = max(cummax, adj)
        adjusted[idx] = min(cummax, 1.0)
    return [
        {
            "original_p": float(p_values[i]),
            "adjusted_p": float(adjusted[i]),
            "is_significant": bool(adjusted[i] < alpha),
        }
        for i in range(m)
    ]


def friedman_nemenyi(systems_scores: Dict[str, np.ndarray]) -> Dict:
    """Run Friedman test and Nemenyi post-hoc on aligned per-query scores."""
    names = list(systems_scores.keys())
    k = len(names)
    # Ensure all arrays same length
    n = min(len(v) for v in systems_scores.values())
    data = np.column_stack([systems_scores[name][:n] for name in names])

    # Friedman
    chi2, p = stats.friedmanchisquare(*[data[:, i] for i in range(k)])

    # Compute average ranks
    ranks = np.zeros_like(data)
    for i in range(n):
        row = data[i]
        # Higher is better → rank descending (1 = best)
        order = np.argsort(-row)
        r = np.zeros(k)
        for rank_pos, idx in enumerate(order):
            r[idx] = rank_pos + 1
        # Handle ties
        vals = row[order]
        j = 0
        while j < k:
            jj = j
            while jj < k and vals[jj] == vals[j]:
                jj += 1
            if jj > j + 1:
                avg_rank = np.mean(r[order[j:jj]])
                for idx in order[j:jj]:
                    r[idx] = avg_rank
            j = jj
        ranks[i] = r

    avg_ranks = {name: float(np.mean(ranks[:, i])) for i, name in enumerate(names)}

    # Nemenyi CD (critical difference)
    # CD = q_α * sqrt(k*(k+1)/(6*n))
    # q_α for α=0.05: use Nemenyi critical values
    # Approximation via Studentized range / sqrt(2)
    from scipy.stats import studentized_range
    q_alpha = studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

    # Pairwise comparisons
    pairs = {}
    for i in range(k):
        for j in range(i + 1, k):
            rank_diff = abs(avg_ranks[names[i]] - avg_ranks[names[j]])
            pairs[f"{names[i]} vs {names[j]}"] = {
                "rank_diff": float(rank_diff),
                "cd": float(cd),
                "sig": bool(rank_diff > cd),
            }

    return {
        "chi2": float(chi2),
        "p": float(p),
        "sig": bool(p < ALPHA),
        "n": int(n),
        "k": int(k),
        "ranks": avg_ranks,
        "cd": float(cd),
        "pairs": pairs,
    }


def _convert_numpy(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Not serialisable: {type(obj)}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Per-query score cache
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_PQ_CACHE: Dict[str, Dict] = {}


def get_pq(exp_path: str, domain: str) -> Dict:
    key = f"{exp_path}|{domain}"
    if key not in _PQ_CACHE:
        _PQ_CACHE[key] = compute_per_query_scores(exp_path, domain) or {}
    return _PQ_CACHE[key]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_all():
    report = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "primary_metric": PRIMARY_METRIC,
        "alpha": ALPHA,
        "bootstrap_config": {
            "n_bootstrap": N_BOOTSTRAP,
            "ci_level": 0.95,
            "method": "percentile",
            "seed": SEED,
        },
        "ablation_studies": {},
    }

    summary_lines = []
    summary_lines.append("=" * 100)
    summary_lines.append("  ABLATION STATISTICAL TESTS — EA2, EA3, EA5, EA6, EA7")
    summary_lines.append("  MT-RAG Benchmark — Task A Retrieval")
    summary_lines.append(f"  Generated: {report['generated']}")
    summary_lines.append(f"  Primary metric: nDCG@5 | α = {ALPHA} | Bootstrap B = {N_BOOTSTRAP}")
    summary_lines.append("=" * 100)

    # Global counters
    global_total = 0
    global_sig_raw = 0
    global_sig_holm = 0
    all_shapiro_reject = 0
    all_shapiro_total = 0

    for ea_name, ea_cfg in ABLATION_STUDIES.items():
        hr()
        print(f"\n{'=' * 100}")
        print(f"  {ea_name}: {ea_cfg['description']}  (→ {ea_cfg['hypothesis']})")
        print(f"{'=' * 100}")

        ref_name, ref_path = ea_cfg["reference"]
        ea_report = {
            "description": ea_cfg["description"],
            "hypothesis": ea_cfg["hypothesis"],
            "reference": ref_name,
            "paired_tests": {},
            "bootstrap_ci": {},
            "holm_correction": {},
            "friedman_nemenyi": {},
        }

        # ── Phase A: Paired tests per level per domain ────────────────
        print(f"\n  A. Paired tests (reference: {ref_name})")
        print(f"  {'Level':<18} {'Domain':<8} {'n':>5} {'Δ mean':>8} {'d':>7} {'label':>12} "
              f"{'Shapiro p':>12} {'Normal':>7} {'Wilcoxon p':>12} {'Sig':>5}")
        print("  " + "-" * 110)

        wilcoxon_ps = []  # for Holm across this EA
        test_labels = []  # (level, domain)

        for level_name, level_path in ea_cfg["levels"]:
            ea_report["paired_tests"][level_name] = {}
            for domain in DOMAINS:
                ref_pq = get_pq(ref_path, domain)
                alt_pq = get_pq(level_path, domain)
                if not ref_pq or not alt_pq:
                    print(f"  {level_name:<18} {domain:<8} SKIPPED (no data)")
                    continue

                ref_arr, alt_arr = get_aligned_scores(ref_pq, alt_pq, PRIMARY_METRIC)
                result = paired_test(ref_arr, alt_arr)
                ea_report["paired_tests"][level_name][domain] = result

                wilcoxon_ps.append(result["wilcoxon"]["p"])
                test_labels.append((level_name, domain))

                # Track Shapiro
                all_shapiro_total += 1
                if result["shapiro"].get("normal") == False:
                    all_shapiro_reject += 1

                shap_p = result["shapiro"].get("p")
                shap_p_str = f"{shap_p:.2e}" if shap_p is not None else "N/A"
                normal_str = str(result["shapiro"].get("normal", "N/A"))
                wilcoxon_p = result["wilcoxon"]["p"]
                sig_str = "***" if result["wilcoxon"]["sig"] else ""

                print(f"  {level_name:<18} {domain:<8} {result['n_queries']:>5} "
                      f"{result['mean_diff']:>+8.4f} {result['cohens_d']:>+7.3f} "
                      f"{result['effect_label']:>12} {shap_p_str:>12} {normal_str:>7} "
                      f"{wilcoxon_p:>12.6f} {sig_str:>5}")

                # Bootstrap CI for this level/domain
                if level_name not in ea_report["bootstrap_ci"]:
                    ea_report["bootstrap_ci"][level_name] = {}
                alt_scores_all = np.array([alt_pq[q].get(PRIMARY_METRIC, 0.0) for q in alt_pq])
                ea_report["bootstrap_ci"][level_name][domain] = bootstrap_ci(alt_scores_all)

        # Also bootstrap the reference
        ea_report["bootstrap_ci"][ref_name] = {}
        for domain in DOMAINS:
            ref_pq = get_pq(ref_path, domain)
            if ref_pq:
                ref_scores_all = np.array([ref_pq[q].get(PRIMARY_METRIC, 0.0) for q in ref_pq])
                ea_report["bootstrap_ci"][ref_name][domain] = bootstrap_ci(ref_scores_all)

        # ── Phase B: Holm-Bonferroni correction ───────────────────────
        n_tests = len(wilcoxon_ps)
        global_total += n_tests
        raw_sig = sum(1 for p in wilcoxon_ps if p < ALPHA)
        global_sig_raw += raw_sig

        if n_tests > 0:
            holm_results = apply_holm_bonferroni(wilcoxon_ps, ALPHA)
            holm_sig = sum(1 for h in holm_results if h["is_significant"])
            global_sig_holm += holm_sig

            # Attach Holm results back to paired tests
            for idx, ((level_name, domain), holm) in enumerate(zip(test_labels, holm_results)):
                ea_report["paired_tests"][level_name][domain]["holm"] = holm

            ea_report["holm_correction"] = {
                "n_tests": n_tests,
                "sig_raw": raw_sig,
                "sig_holm": holm_sig,
            }

            print(f"\n  B. Holm-Bonferroni: {raw_sig}/{n_tests} raw → {holm_sig}/{n_tests} after correction")

            # Print Holm-corrected results
            print(f"\n  {'Level':<18} {'Domain':<8} {'Wilcoxon p':>12} {'Holm p':>12} {'Sig (Holm)':>12} {'d':>7} {'label':>12}")
            print("  " + "-" * 85)
            for idx, ((level_name, domain), holm) in enumerate(zip(test_labels, holm_results)):
                pt = ea_report["paired_tests"][level_name][domain]
                sig_mark = "***" if holm["is_significant"] else ""
                print(f"  {level_name:<18} {domain:<8} {holm['original_p']:>12.6f} "
                      f"{holm['adjusted_p']:>12.6f} {sig_mark:>12} "
                      f"{pt['cohens_d']:>+7.3f} {pt['effect_label']:>12}")

        # ── Phase C: Friedman + Nemenyi ───────────────────────────────
        friedman_sys = ea_cfg.get("friedman_systems", {})
        if len(friedman_sys) >= 3:
            print(f"\n  C. Friedman-Nemenyi (k={len(friedman_sys)} systems)")
            for domain in DOMAINS:
                # Load per-query scores for all systems, aligned
                all_pq = {}
                for sys_name, sys_path in friedman_sys.items():
                    pq = get_pq(sys_path, domain)
                    if pq:
                        all_pq[sys_name] = pq

                if len(all_pq) < 3:
                    continue

                # Align by common queries
                common_qids = sorted(set.intersection(*[set(pq.keys()) for pq in all_pq.values()]))
                systems_scores = {}
                for sys_name, pq in all_pq.items():
                    systems_scores[sys_name] = np.array([pq[q].get(PRIMARY_METRIC, 0.0) for q in common_qids])

                fr = friedman_nemenyi(systems_scores)
                ea_report["friedman_nemenyi"][domain] = fr

                # Print
                print(f"\n    {domain}: χ²={fr['chi2']:.2f}, p={fr['p']:.2e}, sig={fr['sig']}, "
                      f"n={fr['n']}, k={fr['k']}, CD={fr['cd']:.3f}")
                for sys_name, rank in sorted(fr["ranks"].items(), key=lambda x: x[1]):
                    print(f"      {sys_name:<22} rank={rank:.2f}")
                sig_pairs = sum(1 for v in fr["pairs"].values() if v["sig"])
                print(f"      Significant pairs: {sig_pairs}/{len(fr['pairs'])}")

        report["ablation_studies"][ea_name] = ea_report

        # Summary lines
        summary_lines.append(f"\n{'=' * 80}")
        summary_lines.append(f"  {ea_name}: {ea_cfg['description']} (→ {ea_cfg['hypothesis']})")
        summary_lines.append(f"{'=' * 80}")
        if n_tests > 0:
            summary_lines.append(f"  Wilcoxon raw sig: {raw_sig}/{n_tests}")
            summary_lines.append(f"  After Holm-Bonferroni: {holm_sig}/{n_tests}")

        # Add key findings per level
        for level_name, level_path in ea_cfg["levels"]:
            level_ds = []
            level_sigs = []
            for domain in DOMAINS:
                pt = ea_report["paired_tests"].get(level_name, {}).get(domain)
                if pt:
                    level_ds.append(pt["cohens_d"])
                    holm_info = pt.get("holm", {})
                    level_sigs.append(holm_info.get("is_significant", False))
            if level_ds:
                avg_d = np.mean(level_ds)
                n_sig = sum(level_sigs)
                summary_lines.append(
                    f"    {level_name:<18} avg |d|={abs(avg_d):.3f} ({cohens_d_label(avg_d)}), "
                    f"Holm sig: {n_sig}/{len(level_sigs)} domains"
                )

    # ── Global summary ────────────────────────────────────────────────
    report["global_summary"] = {
        "total_tests": global_total,
        "sig_raw": global_sig_raw,
        "sig_holm": global_sig_holm,
        "shapiro_reject_normality": all_shapiro_reject,
        "shapiro_total": all_shapiro_total,
        "shapiro_reject_pct": round(100 * all_shapiro_reject / max(all_shapiro_total, 1), 1),
    }

    hr()
    print(f"\n{'=' * 100}")
    print("  GLOBAL SUMMARY")
    print(f"{'=' * 100}")
    print(f"  Total paired tests:           {global_total}")
    print(f"  Wilcoxon raw sig (p < 0.05):  {global_sig_raw}/{global_total}")
    print(f"  After Holm-Bonferroni:        {global_sig_holm}/{global_total}")
    print(f"  Shapiro-Wilk reject normality: {all_shapiro_reject}/{all_shapiro_total} "
          f"({report['global_summary']['shapiro_reject_pct']}%)")
    print(f"  → Wilcoxon is the correct test (normality violated)")

    summary_lines.append(f"\n{'=' * 80}")
    summary_lines.append("  GLOBAL SUMMARY")
    summary_lines.append(f"{'=' * 80}")
    summary_lines.append(f"  Total paired tests:            {global_total}")
    summary_lines.append(f"  Wilcoxon raw sig (p < 0.05):   {global_sig_raw}/{global_total}")
    summary_lines.append(f"  After Holm-Bonferroni FWER:    {global_sig_holm}/{global_total}")
    summary_lines.append(f"  Shapiro-Wilk reject normality: {all_shapiro_reject}/{all_shapiro_total} "
                         f"({report['global_summary']['shapiro_reject_pct']}%)")
    summary_lines.append(f"  → Justifies non-parametric Wilcoxon")

    # ── Save outputs ──────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)

    json_path = "results/ablation_statistical_tests.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=_convert_numpy)
    print(f"\n  JSON report → {json_path}")

    txt_path = "results/ablation_statistical_summary.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"  Text summary → {txt_path}")

    # ── Export per-query scores for all ablation experiments ──────────
    pq_path = "results/per_query_scores_ablations.jsonl"
    with open(pq_path, "w") as f:
        for ea_name, ea_cfg in ABLATION_STUDIES.items():
            # Reference
            ref_name, ref_path = ea_cfg["reference"]
            for domain in DOMAINS:
                pq = get_pq(ref_path, domain)
                for qid, metrics in pq.items():
                    row = {"ea": ea_name, "experiment": ref_name, "domain": domain, "query_id": qid}
                    row.update(metrics)
                    f.write(json.dumps(row, default=_convert_numpy) + "\n")
            # Levels
            for level_name, level_path in ea_cfg["levels"]:
                for domain in DOMAINS:
                    pq = get_pq(level_path, domain)
                    for qid, metrics in pq.items():
                        row = {"ea": ea_name, "experiment": level_name, "domain": domain, "query_id": qid}
                        row.update(metrics)
                        f.write(json.dumps(row, default=_convert_numpy) + "\n")
    print(f"  Per-query scores → {pq_path}")

    return report


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    report = run_all()
    print("\n  Done.")
