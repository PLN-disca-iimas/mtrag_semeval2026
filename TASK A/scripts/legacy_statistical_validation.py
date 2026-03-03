#!/usr/bin/env python3
"""
Validación Estadística Completa de Experimentos MT-RAG
Incluye:
- Bootstrap Confidence Intervals (ya calculados, validación)
- Paired t-tests y Wilcoxon signed-rank tests
- Análisis de significancia estadística
- Reporte completo de validez
"""

import json
import os
import numpy as np
from scipy import stats
from collections import defaultdict
import pytrec_eval

domains = ['clapnq', 'cloud', 'fiqa', 'govt']

def load_qrels(domain):
    """Load qrels in TREC format"""
    qrels_path = f"data/retrieval_tasks/{domain}/qrels/dev.tsv"
    qrels_dict = defaultdict(dict)
    
    with open(qrels_path) as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                qid, docid, rel = parts[0], parts[1], int(parts[2])
                qrels_dict[qid][docid] = rel
    
    return dict(qrels_dict)

def load_retrieval_results(exp_path, domain):
    """Load retrieval results and convert to TREC format"""
    results_path = f"{exp_path}/{domain}/retrieval_results.jsonl"
    if not os.path.exists(results_path):
        return None
    
    results_dict = {}
    with open(results_path) as f:
        for line in f:
            item = json.loads(line)
            task_id = item['task_id']
            contexts = item.get('contexts', [])
            
            results_dict[task_id] = {}
            for rank, ctx in enumerate(contexts[:100], 1):
                doc_id = ctx['document_id']
                score = ctx.get('score', 100.0 - rank)  # Use rank as score if not present
                results_dict[task_id][doc_id] = float(score)
    
    return results_dict

def compute_per_query_metrics(exp_path, domain):
    """Compute nDCG@10 and Recall@100 for each query"""
    qrels = load_qrels(domain)
    results = load_retrieval_results(exp_path, domain)
    
    if not results:
        return None
    
    # Evaluate
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {'ndcg_cut.5', 'ndcg_cut.10', 'recall.100'}
    )
    scores = evaluator.evaluate(results)
    
    # Extract metrics per query
    per_query = {}
    for qid, metrics in scores.items():
        per_query[qid] = {
            'ndcg_5': metrics.get('ndcg_cut_5', 0.0),
            'ndcg_10': metrics.get('ndcg_cut_10', 0.0),
            'recall_100': metrics.get('recall_100', 0.0)
        }
    
    return per_query

def paired_test(scores_a, scores_b, test='both'):
    """Perform paired statistical tests with normality check (Gap 3)."""
    # Align scores by query ID
    common_qids = set(scores_a.keys()) & set(scores_b.keys())
    
    if len(common_qids) == 0:
        return None
    
    pairs_a = [scores_a[qid] for qid in sorted(common_qids)]
    pairs_b = [scores_b[qid] for qid in sorted(common_qids)]
    
    results = {'n_queries': len(common_qids)}

    # Normality check on the difference vector (Gap 3)
    diff = np.array(pairs_a) - np.array(pairs_b)
    if len(diff) >= 8:  # Shapiro-Wilk requires n >= 3, but 8+ is practical
        try:
            sw_stat, sw_pval = stats.shapiro(diff)
            normality_ok = bool(sw_pval >= 0.05)
            results['shapiro_wilk'] = {
                'statistic': float(sw_stat),
                'p_value': float(sw_pval),
                'normal': normality_ok,
            }
        except Exception:
            normality_ok = False
            results['shapiro_wilk'] = {'error': 'test_failed'}
    else:
        normality_ok = False
        results['shapiro_wilk'] = {'error': 'n_too_small'}
    
    # Paired t-test — only report if normality holds
    if test in ['both', 't-test']:
        t_stat, t_pval = stats.ttest_rel(pairs_a, pairs_b)
        results['t_test'] = {
            'statistic': float(t_stat),
            'p_value': float(t_pval),
            'significant': bool(t_pval < 0.05),
            'normality_ok': normality_ok,
            'note': ('Assumption satisfied' if normality_ok
                     else 'Normality NOT satisfied (Shapiro p<0.05); '
                          'prefer Wilcoxon result'),
        }
    
    # Wilcoxon signed-rank test (non-parametric)
    if test in ['both', 'wilcoxon']:
        try:
            w_stat, w_pval = stats.wilcoxon(pairs_a, pairs_b, alternative='two-sided')
            results['wilcoxon'] = {
                'statistic': float(w_stat),
                'p_value': float(w_pval),
                'significant': bool(w_pval < 0.05)
            }
        except ValueError as e:
            results['wilcoxon'] = {'error': str(e)}
    
    # Effect size (Cohen's d)
    cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
    results['cohens_d'] = float(cohens_d)
    results['mean_diff'] = float(np.mean(diff))
    results['std_diff'] = float(np.std(diff, ddof=1))
    
    return results

def apply_holm_bonferroni(p_values, alpha=0.05):
    """Apply Holm-Bonferroni step-down correction (Gap 1).

    Returns list of dicts in the same order as input.
    """
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.zeros(m)
    cummax = 0.0
    for i, idx in enumerate(order):
        adj = p_values[idx] * (m - i)
        cummax = max(cummax, adj)
        adjusted[idx] = min(cummax, 1.0)
    return [
        {'original_p': float(p_values[i]),
         'adjusted_p': float(adjusted[i]),
         'rank': int(np.where(order == i)[0][0] + 1),
         'is_significant': bool(adjusted[i] < alpha)}
        for i in range(m)
    ]


def main():
    print("=" * 100)
    print("VALIDACIÓN ESTADÍSTICA COMPLETA - MT-RAG BENCHMARK")
    print("=" * 100)
    
    PRIMARY_METRIC = 'ndcg_5'        # Gap 5: consistent with pipeline bootstrap
    METRIC_LABEL   = 'nDCG@5'

    # Experiments to compare
    experiments = [
        ('No-Rewrite', 'experiments/hybrid_splade_bge15_norewrite'),
        ('GT-Rewrite', 'experiments/hybrid_splade_bge15_rewrite'),
        ('Cohere-OWN', 'experiments/hybrid_splade_bge15_rewrite_own'),
        ('Cohere-V2', 'experiments/hybrid_splade_bge15_rewrite_v2'),
        ('Cohere-V3', 'experiments/hybrid_splade_bge15_rewrite_v3'),
    ]
    
    # ====== PART 1: BOOTSTRAP CI VALIDATION ======
    print("\n" + "=" * 100)
    print(f"1. BOOTSTRAP CONFIDENCE INTERVALS (95%) — {METRIC_LABEL}")
    print("=" * 100)
    print(f"\n{'Experiment':<20} {'Domain':<10} {'Mean':>8} {'CI Lower':>10} {'CI Upper':>10} {'Margin':>10}")
    print("-" * 100)
    
    bootstrap_summary = {}
    for exp_name, exp_path in experiments:
        bootstrap_summary[exp_name] = {}
        for domain in domains:
            report_path = f"{exp_path}/{domain}/analysis_report.json"
            if os.path.exists(report_path):
                with open(report_path) as f:
                    report = json.load(f)
                    ci_key = 'bootstrap_ci_ndcg_at_5'
                    if ci_key in report:
                        ci = report[ci_key]
                        lo = ci.get('ci_lower', ci.get('lower', 0.0))
                        hi = ci.get('ci_upper', ci.get('upper', 0.0))
                        margin = (hi - lo) / 2
                        bootstrap_summary[exp_name][domain] = ci
                        print(f"{exp_name:<20} {domain:<10} {ci['mean']:>8.4f} {lo:>10.4f} {hi:>10.4f} ±{margin:>9.4f}")
    
    # ====== PART 2: COMPUTE PER-QUERY METRICS ======
    print("\n" + "=" * 100)
    print(f"2. EXTRACCIÓN DE MÉTRICAS POR QUERY ({METRIC_LABEL})")
    print("=" * 100)
    
    per_query_data = {}
    for exp_name, exp_path in experiments:
        per_query_data[exp_name] = {}
        print(f"\nProcesando {exp_name}...")
        for domain in domains:
            try:
                scores = compute_per_query_metrics(exp_path, domain)
                if scores:
                    per_query_data[exp_name][domain] = scores
                    metric_scores = [v[PRIMARY_METRIC] for v in scores.values()]
                    mean_val = np.mean(metric_scores)
                    print(f"  {domain:<10}: {len(scores)} queries, Mean {METRIC_LABEL} = {mean_val:.4f}")
            except Exception as e:
                print(f"  {domain:<10}: Error - {e}")
    
    # ── Gap 6: export per-query scores to JSONL ──────────────────────────
    per_query_export_path = 'per_query_scores_all_experiments.jsonl'
    with open(per_query_export_path, 'w') as pqf:
        for exp_name, dom_data in per_query_data.items():
            for domain, qscores in dom_data.items():
                for qid, metrics in qscores.items():
                    row = {'experiment': exp_name, 'domain': domain, 'query_id': qid}
                    row.update(metrics)
                    pqf.write(json.dumps(row) + '\n')
    print(f"\n✓ Per-query scores exported to {per_query_export_path}")

    # ====== PART 3: PAIRED STATISTICAL TESTS ======
    print("\n" + "=" * 100)
    print(f"3. PRUEBAS DE SIGNIFICANCIA ESTADÍSTICA ({METRIC_LABEL})")
    print("=" * 100)
    
    # Critical comparisons
    comparisons = [
        ('No-Rewrite', 'GT-Rewrite', 'Impacto de Rewrite (GT)'),
        ('GT-Rewrite', 'Cohere-OWN', 'Cohere vs GT'),
        ('GT-Rewrite', 'Cohere-V2', 'Cohere V2 vs GT'),
        ('GT-Rewrite', 'Cohere-V3', 'Cohere V3 vs GT'),
        ('Cohere-OWN', 'Cohere-V2', 'V2 vs OWN'),
        ('Cohere-OWN', 'Cohere-V3', 'V3 vs OWN'),
        ('Cohere-V2', 'Cohere-V3', 'V3 vs V2'),
    ]
    
    statistical_results = {}
    # Collect ALL raw p-values for Holm-Bonferroni later (Gap 1)
    all_raw_pvalues_wilcoxon = []
    all_raw_pvalues_ttest   = []
    pvalue_keys = []           # (comparison_desc, domain) for look-up
    
    for exp_a, exp_b, description in comparisons:
        print(f"\n{'=' * 100}")
        print(f"Comparación: {description} ({exp_a} vs {exp_b})")
        print(f"{'=' * 100}")
        
        statistical_results[description] = {}
        
        for domain in domains:
            if exp_a not in per_query_data or exp_b not in per_query_data:
                continue
            if domain not in per_query_data[exp_a] or domain not in per_query_data[exp_b]:
                continue
            
            # Use PRIMARY_METRIC (nDCG@5) consistently (Gap 5)
            scores_a = {qid: v[PRIMARY_METRIC] for qid, v in per_query_data[exp_a][domain].items()}
            scores_b = {qid: v[PRIMARY_METRIC] for qid, v in per_query_data[exp_b][domain].items()}
            
            # Run tests (now includes Shapiro-Wilk, Gap 3)
            test_results = paired_test(scores_a, scores_b)
            
            if test_results:
                statistical_results[description][domain] = test_results
                pvalue_keys.append((description, domain))
                
                # Collect raw p-values
                if 'wilcoxon' in test_results and 'error' not in test_results['wilcoxon']:
                    all_raw_pvalues_wilcoxon.append(test_results['wilcoxon']['p_value'])
                else:
                    all_raw_pvalues_wilcoxon.append(1.0)
                if 't_test' in test_results:
                    all_raw_pvalues_ttest.append(test_results['t_test']['p_value'])
                else:
                    all_raw_pvalues_ttest.append(1.0)
                
                print(f"\nDominio: {domain}")
                print(f"  N queries: {test_results['n_queries']}")
                print(f"  Mean Difference: {test_results['mean_diff']:.4f} (±{test_results['std_diff']:.4f})")
                print(f"  Cohen's d (effect size): {test_results['cohens_d']:.3f}")
                
                # Normality info (Gap 3)
                sw = test_results.get('shapiro_wilk', {})
                if 'p_value' in sw:
                    norm_marker = "✓ Normal" if sw['normal'] else "✗ Non-normal"
                    print(f"  Shapiro-Wilk: p={sw['p_value']:.4f} → {norm_marker}")
                
                if 't_test' in test_results:
                    sig = "✓ SIGNIFICATIVO" if test_results['t_test']['significant'] else "✗ No significativo"
                    note = test_results['t_test'].get('note', '')
                    print(f"  Paired t-test: p={test_results['t_test']['p_value']:.4f} {sig}")
                    if not test_results['t_test'].get('normality_ok', True):
                        print(f"    ⚠  {note}")
                
                if 'wilcoxon' in test_results and 'error' not in test_results['wilcoxon']:
                    sig = "✓ SIGNIFICATIVO" if test_results['wilcoxon']['significant'] else "✗ No significativo"
                    print(f"  Wilcoxon test: p={test_results['wilcoxon']['p_value']:.4f} {sig}")
    
    # ====== PART 3b: HOLM-BONFERRONI CORRECTION (Gap 1) ======
    print("\n" + "=" * 100)
    print("3b. CORRECCIÓN POR PRUEBAS MÚLTIPLES (Holm-Bonferroni)")
    print("=" * 100)

    n_tests = len(pvalue_keys)
    holm_wilcoxon = apply_holm_bonferroni(all_raw_pvalues_wilcoxon) if n_tests else []
    holm_ttest    = apply_holm_bonferroni(all_raw_pvalues_ttest)    if n_tests else []

    print(f"\nTotal tests corregidos: {n_tests}")
    print(f"\n{'Comparison':<30} {'Domain':<10} {'raw_p_W':>10} {'adj_p_W':>10} {'Sig_W':>6}  {'raw_p_t':>10} {'adj_p_t':>10} {'Sig_t':>6}")
    print("-" * 110)

    sig_after_correction = 0
    for i, (desc, dom) in enumerate(pvalue_keys):
        hw = holm_wilcoxon[i]
        ht = holm_ttest[i]
        w_sig = "✓" if hw['is_significant'] else "✗"
        t_sig = "✓" if ht['is_significant'] else "✗"
        if hw['is_significant']:
            sig_after_correction += 1
        print(f"  {desc:<28} {dom:<10} {hw['original_p']:>10.4f} {hw['adjusted_p']:>10.4f} {w_sig:>6}  {ht['original_p']:>10.4f} {ht['adjusted_p']:>10.4f} {t_sig:>6}")

        # Attach Holm results back to statistical_results
        statistical_results[desc][dom]['holm_bonferroni_wilcoxon'] = hw
        statistical_results[desc][dom]['holm_bonferroni_ttest']    = ht

    # ====== PART 4: SUMMARY REPORT ======
    print("\n" + "=" * 100)
    print("4. REPORTE RESUMEN DE VALIDEZ ESTADÍSTICA")
    print("=" * 100)
    
    print("\n### Técnicas Estadísticas Aplicadas:")
    print("  1. Bootstrap CI (1000 iter, seed=42) — reproducible")
    print("  2. Shapiro-Wilk normality test on paired differences")
    print("  3. Paired t-test (conditional on normality)")
    print("  4. Wilcoxon signed-rank (non-parametric, primary test)")
    print("  5. Cohen's d effect size (sample-corrected, ddof=1)")
    print("  6. Holm-Bonferroni step-down correction for FWER control")
    print(f"  7. Per-query scores exported to {per_query_export_path}")
    
    # Count significant results (raw vs corrected)
    raw_sig_count = 0
    for comp_name, domains_data in statistical_results.items():
        for domain, results in domains_data.items():
            if 'wilcoxon' in results and results['wilcoxon'].get('significant'):
                raw_sig_count += 1
    
    print(f"\n  Significativos (Wilcoxon, raw α=0.05):       {raw_sig_count}/{n_tests}")
    print(f"  Significativos (Wilcoxon, Holm-Bonferroni):  {sig_after_correction}/{n_tests}")
    
    # Normality summary
    norm_fail = 0
    for comp_name, domains_data in statistical_results.items():
        for domain, results in domains_data.items():
            sw = results.get('shapiro_wilk', {})
            if sw.get('normal') is False:
                norm_fail += 1
    print(f"  Normality failures (Shapiro p<0.05):         {norm_fail}/{n_tests}")
    if norm_fail > 0:
        print("  → Paired t-test unreliable for those comparisons; use Wilcoxon")

    # Domain-specific insights
    print("\n### Insights por Dominio:")
    for domain in domains:
        print(f"\n{domain.upper()}:")
        best_exp = None
        best_score = -1
        for exp_name in ['No-Rewrite', 'GT-Rewrite', 'Cohere-OWN', 'Cohere-V2', 'Cohere-V3']:
            if exp_name in per_query_data and domain in per_query_data[exp_name]:
                scores = [v[PRIMARY_METRIC] for v in per_query_data[exp_name][domain].values()]
                mean = np.mean(scores)
                if mean > best_score:
                    best_score = mean
                    best_exp = exp_name
        
        print(f"  Mejor configuración: {best_exp} ({METRIC_LABEL} = {best_score:.4f})")
    
    # Save results
    output = {
        'primary_metric': METRIC_LABEL,
        'bootstrap_ci': bootstrap_summary,
        'statistical_tests': statistical_results,
        'holm_bonferroni_applied': True,
        'total_tests': n_tests,
        'summary': {
            'total_comparisons': n_tests,
            'significant_raw_wilcoxon': raw_sig_count,
            'significant_holm_wilcoxon': sig_after_correction,
            'normality_failures': norm_fail,
        }
    }
    
    with open('statistical_validation_report.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Reporte completo guardado en: statistical_validation_report.json")
    print(f"✓ Per-query scores guardados en: {per_query_export_path}")
    print("\n" + "=" * 100)
    print("VALIDACIÓN ESTADÍSTICA COMPLETADA")
    print("=" * 100)

if __name__ == "__main__":
    main()
