#!/usr/bin/env python3
"""
Main experiment runner script.

Usage:
    # Run single experiment on one domain
    python scripts/run_experiment.py --experiment A6_hybrid_rerank --domain fiqa
    
    # Run single experiment on all domains
    python scripts/run_experiment.py --experiment A6_hybrid_rerank --domain all
    
    # Run all experiments on one domain
    python scripts/run_experiment.py --experiment all --domain clapnq
    
    # Dry run (validate configs without execution)
    python scripts/run_experiment.py --experiment A6_hybrid_rerank --domain fiqa --dry-run

    # Parallel execution on 2 GPUs (domains run in parallel)
    python scripts/run_experiment.py --experiment all --domain all --parallel 2
"""

import argparse
import concurrent.futures
import logging
import os
from pathlib import Path
import sys
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.run import run_pipeline
from utils.config_loader import load_config, merge_configs
from utils.logger import setup_logger
from utils.hf_manager import HFManager


DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]

# List of all available experiments (UPDATED: Match actual config file names)
EXPERIMENTS = [
    # Baselines - Replications (last turn)
    "replication_bm25",
    "replication_bge15",
    "replication_bgem3",
    "replication_splade",
    "replication_voyage",
    # Baselines - Full History
    "A0_baseline_bm25_fullhist",
    "A0_baseline_splade_fullhist",
    "A1_baseline_bgem3_fullhist",
    "A1_baseline_voyage_fullhist",
    # Baselines - Ground Truth Rewrites (Oracle)
    "A2_baseline_bm25_rewrite",
    "A2_baseline_splade_rewrite",
    "A2_baseline_bge15_rewrite",
    "A2_baseline_bgem3_rewrite",
    "A2_baseline_voyage_rewrite",
    # Hybrid
    "hybrid_splade_voyage_norewrite",
    "hybrid_splade_voyage_rewrite",
    "hybrid_splade_bge15_norewrite",
    "hybrid_splade_bge15_rewrite",
    # Reranking
    "rerank_splade_voyage_rewrite",
    "rerank_splade_bge15_rewrite",
    "rerank_cohere_splade_bge15_rewrite",
    "rerank_cohere_splade_voyage_rewrite",
    # Ablation: Fusion method
    "ablation_fusion_linear_splade_voyage",
    "ablation_fusion_linear_splade_bge15",
    "ablation_fusion_linear_alpha03_voyage",
    "ablation_fusion_linear_alpha07_voyage",
    # Ablation: RRF k
    "ablation_rrf_k1_voyage",
    "ablation_rrf_k20_voyage",
    "ablation_rrf_k40_voyage",
    "ablation_rrf_k100_voyage",
    # Ablation: Retrieval top-k
    "ablation_topk_100_voyage",
    "ablation_topk_200_voyage",
    "ablation_topk_500_voyage",
    "ablation_topk_100_bge15",
    # Ablation: Rerank depth
    "ablation_rerank_depth_50",
    "ablation_rerank_depth_200",
    "ablation_rerank_depth_50_cohere",
    "ablation_rerank_depth_200_cohere",
    # Ablation: Query mode on hybrid
    "ablation_query_mode_lastturn_hybrid",
    "ablation_query_mode_fullhist_hybrid",
    "ablation_query_mode_fullctx_hybrid",
    # Ablation: Component isolation
    "ablation_component_splade_cohere_v3",
    "ablation_component_voyage_cohere_v3",
    "ablation_component_bge15_cohere_v3",
    # Ablation: Rewrite strategy
    "hybrid_splade_voyage_rewrite_own_replica",
    "hybrid_splade_voyage_rewrite_own_improved",
    "hybrid_splade_voyage_rewrite_own_local",
]

# Mapping of experiments to subdirectories (FIXED: Match actual directory names)
EXPERIMENT_DIRS = {
    # Baselines - Replications
    "replication_bm25": "0-baselines",
    "replication_bge15": "0-baselines",
    "replication_bgem3": "0-baselines",
    "replication_splade": "0-baselines",
    "replication_voyage": "0-baselines",
    # Baselines - Full History
    "A0_baseline_bm25_fullhist": "0-baselines",
    "A0_baseline_splade_fullhist": "0-baselines",
    "A1_baseline_bgem3_fullhist": "0-baselines",
    "A1_baseline_voyage_fullhist": "0-baselines",
    # Baselines - Ground Truth Rewrites
    "A2_baseline_bm25_rewrite": "0-baselines",
    "A2_baseline_splade_rewrite": "0-baselines",
    "A2_baseline_bge15_rewrite": "0-baselines",
    "A2_baseline_bgem3_rewrite": "0-baselines",
    "A2_baseline_voyage_rewrite": "0-baselines",
    # Hybrid
    "hybrid_splade_voyage_norewrite": "02-hybrid",
    "hybrid_splade_voyage_rewrite": "02-hybrid",
    "hybrid_splade_bge15_norewrite": "02-hybrid",
    "hybrid_splade_bge15_rewrite": "02-hybrid",
    # Reranking
    "rerank_splade_voyage_rewrite": "03-rerank",
    "rerank_splade_bge15_rewrite": "03-rerank",
    "rerank_cohere_splade_bge15_rewrite": "03-rerank",
    "rerank_cohere_splade_voyage_rewrite": "03-rerank",

    # Ablation: Fusion method
    "ablation_fusion_linear_splade_voyage": "06-ablation-fusion",
    "ablation_fusion_linear_splade_bge15": "06-ablation-fusion",
    "ablation_fusion_linear_alpha03_voyage": "06-ablation-fusion",
    "ablation_fusion_linear_alpha07_voyage": "06-ablation-fusion",
    # Ablation: RRF k
    "ablation_rrf_k1_voyage": "07-ablation-rrf-k",
    "ablation_rrf_k20_voyage": "07-ablation-rrf-k",
    "ablation_rrf_k40_voyage": "07-ablation-rrf-k",
    "ablation_rrf_k100_voyage": "07-ablation-rrf-k",
    # Ablation: Retrieval top-k
    "ablation_topk_100_voyage": "08-ablation-topk",
    "ablation_topk_200_voyage": "08-ablation-topk",
    "ablation_topk_500_voyage": "08-ablation-topk",
    "ablation_topk_100_bge15": "08-ablation-topk",
    # Ablation: Rerank depth
    "ablation_rerank_depth_50": "09-ablation-rerank-depth",
    "ablation_rerank_depth_200": "09-ablation-rerank-depth",
    "ablation_rerank_depth_50_cohere": "09-ablation-rerank-depth",
    "ablation_rerank_depth_200_cohere": "09-ablation-rerank-depth",
    # Ablation: Query mode on hybrid
    "ablation_query_mode_lastturn_hybrid": "10-ablation-query-mode",
    "ablation_query_mode_fullhist_hybrid": "10-ablation-query-mode",
    "ablation_query_mode_fullctx_hybrid": "10-ablation-query-mode",
    # Ablation: Component isolation
    "ablation_component_splade_cohere_v3": "11-ablation-components",
    "ablation_component_voyage_cohere_v3": "11-ablation-components",
    "ablation_component_bge15_cohere_v3": "11-ablation-components",
    # Ablation: Rewrite strategy
    "hybrid_splade_voyage_rewrite_own_replica": "12-rewrite-ablation",
    "hybrid_splade_voyage_rewrite_own_improved": "12-rewrite-ablation",
    "hybrid_splade_voyage_rewrite_own_local": "12-rewrite-ablation",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RAG benchmark experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--experiment", "-e",
        required=True,
        help=f"Experiment name or 'all'. Options: {', '.join(EXPERIMENTS)}"
    )
    
    parser.add_argument(
        "--domain", "-d",
        required=True,
        help=f"Domain name or 'all'. Options: {', '.join(DOMAINS)}"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running experiment"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments"),
        help="Output directory for results (default: experiments/)"
    )
    
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Configuration directory (default: configs/)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing results"
    )

    parser.add_argument(
        "--baseline-path",
        type=str,
        help="Path to baseline results file for significance testing"
    )

    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel workers (e.g. 2 for dual-GPU). Each worker is "
             "pinned to a different GPU via CUDA_VISIBLE_DEVICES. (default: 1)"
    )
    
    return parser.parse_args()


def resolve_experiments(experiment_name):
    """Resolve experiment name to list of experiments."""
    if experiment_name == "all":
        return EXPERIMENTS
    elif experiment_name in EXPERIMENTS:
        return [experiment_name]
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")


def resolve_domains(domain_name):
    """Resolve domain name to list of domains."""
    if domain_name == "all":
        return DOMAINS
    elif domain_name in DOMAINS:
        return [domain_name]
    else:
        raise ValueError(f"Unknown domain: {domain_name}")


def _substitute_domain(obj, domain_name):
    """Recursively substitute {domain} in strings."""
    if isinstance(obj, dict):
        return {k: _substitute_domain(v, domain_name) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_domain(item, domain_name) for item in obj]
    elif isinstance(obj, str):
        return obj.replace("{domain}", domain_name)
    else:
        return obj


def _run_single(experiment, domain, config_dir, output_dir_root, force,
                dry_run, baseline_path, verbose, gpu_id=None, run_label=""):
    """Run a single experiment/domain pair.

    This is the workhorse called both in sequential mode and inside each
    parallel worker.  When *gpu_id* is not None the worker pins itself to
    that GPU via ``CUDA_VISIBLE_DEVICES`` **before** any CUDA library is
    touched.

    Returns
    -------
    tuple[str, bool, str]
        ``(run_name, success, error_message)``
    """
    import yaml  # local import for subprocess safety

    run_name = f"{experiment}/{domain}"

    # Pin GPU if requested (must happen before torch/CUDA init)
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_level = logging.DEBUG if verbose else logging.INFO
    logger = setup_logger(f"run_{experiment}_{domain}", level=log_level)
    logger.info(f"\n{'='*80}")
    logger.info(f"{run_label}{experiment} on {domain}  [GPU {gpu_id}]")
    logger.info(f"{'='*80}\n")

    try:
        # Find experiment config in subdirectory
        experiment_subdir = EXPERIMENT_DIRS.get(experiment, "")
        experiment_config_path = Path(config_dir) / "experiments" / experiment_subdir / f"{experiment}.yaml"

        # Load and merge configs
        config = merge_configs(
            base_config=Path(config_dir) / "base.yaml",
            domain_config=Path(config_dir) / "domains" / f"{domain}.yaml",
            experiment_config=experiment_config_path,
        )

        config = _substitute_domain(config, domain)

        # Handle query_file_suffix override for replication experiments
        if "query_file_suffix" in config.get("data", {}):
            suffix = config["data"]["query_file_suffix"]
            original_path = config["data"]["query_file"]
            config["data"]["query_file"] = original_path.replace("questions", suffix)
            logger.info(f"Using query file: {config['data']['query_file']}")

        # FIX: Dynamically adjust index_path if hardcoded to clapnq
        if "retrieval" in config and "index_path" in config["retrieval"]:
            index_path = config["retrieval"]["index_path"]
            if "clapnq" in index_path and domain != "clapnq":
                new_index_path = index_path.replace("clapnq", domain)
                config["retrieval"]["index_path"] = new_index_path
                logger.info(f"Fixed index_path for {domain}: {new_index_path}")

        # Setup output directory
        experiment_subdir = EXPERIMENT_DIRS.get(experiment, "")
        output_dir = Path(output_dir_root) / experiment_subdir / experiment / domain
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save resolved config
        config_path = output_dir / "config_resolved.yaml"
        if not config_path.exists() or force:
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)

        if dry_run:
            logger.info(f"✓ Configuration validated for {run_name}")
            logger.info(f"  Output would be written to: {output_dir}")
        else:
            logger.info("Starting pipeline execution...")
            start_time = datetime.now()

            # Resolve baseline_path: if it's a directory, look for
            # {baseline_path}/{domain}/retrieval_results.jsonl
            resolved_baseline = None
            if baseline_path:
                bp = Path(baseline_path)
                if bp.is_dir():
                    candidate = bp / domain / "retrieval_results.jsonl"
                    if candidate.exists():
                        resolved_baseline = candidate
                        logger.info(f"Resolved baseline to: {resolved_baseline}")
                    else:
                        logger.warning(
                            f"Baseline dir exists but no results for {domain}: {candidate}"
                        )
                elif bp.exists():
                    resolved_baseline = bp

            run_pipeline(
                config=config,
                domain=domain,
                output_dir=output_dir,
                force=force,
                baseline_path=resolved_baseline,
            )

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✓ Completed {run_name} in {duration:.2f}s")

        return (run_name, True, "")

    except Exception as e:
        logger.error(f"✗ Failed: {run_name}")
        logger.error(f"  Error: {e}", exc_info=verbose)
        return (run_name, False, str(e))


def main():
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger("run_experiment", level=log_level)

    # Resolve experiments and domains
    experiments = resolve_experiments(args.experiment)
    domains = resolve_domains(args.domain)
    n_parallel = max(1, args.parallel)

    logger.info(f"Running {len(experiments)} experiment(s) on {len(domains)} domain(s)")
    logger.info(f"Experiments: {', '.join(experiments)}")
    logger.info(f"Domains: {', '.join(domains)}")
    if n_parallel > 1:
        logger.info(f"Parallel workers: {n_parallel} (one GPU each)")

    # Initialize HF Manager (kept in main process)
    hf_manager = HFManager()

    total_runs = len(experiments) * len(domains)
    results = []  # list of (run_name, success, error)

    if n_parallel <= 1:
        # ── Sequential mode (original behaviour) ────────────────────────
        for idx, experiment in enumerate(experiments):
            for jdx, domain in enumerate(domains):
                run_num = idx * len(domains) + jdx + 1
                label = f"Run {run_num}/{total_runs}: "
                run_name, ok, err = _run_single(
                    experiment=experiment,
                    domain=domain,
                    config_dir=str(args.config_dir),
                    output_dir_root=str(args.output_dir),
                    force=args.force,
                    dry_run=args.dry_run,
                    baseline_path=args.baseline_path,
                    verbose=args.verbose,
                    gpu_id=None,
                    run_label=label,
                )
                results.append((run_name, ok, err))

                # Upload results in main process
                if ok and not args.dry_run and hf_manager.enabled:
                    experiment_subdir = EXPERIMENT_DIRS.get(experiment, "")
                    out = args.output_dir / experiment_subdir / experiment / domain
                    hf_manager.upload_directory(out)
    else:
        # ── Parallel mode: dispatch across N GPUs ───────────────────────
        # Build flat list of (experiment, domain) jobs
        jobs = [(exp, dom) for exp in experiments for dom in domains]

        logger.info(f"Dispatching {len(jobs)} jobs across {n_parallel} GPUs …")

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_parallel) as pool:
            futures = {}
            for job_idx, (experiment, domain) in enumerate(jobs):
                gpu_id = job_idx % n_parallel
                label = f"Job {job_idx + 1}/{total_runs}: "
                fut = pool.submit(
                    _run_single,
                    experiment=experiment,
                    domain=domain,
                    config_dir=str(args.config_dir),
                    output_dir_root=str(args.output_dir),
                    force=args.force,
                    dry_run=args.dry_run,
                    baseline_path=args.baseline_path,
                    verbose=args.verbose,
                    gpu_id=gpu_id,
                    run_label=label,
                )
                futures[fut] = (experiment, domain)

            for fut in concurrent.futures.as_completed(futures):
                experiment, domain = futures[fut]
                try:
                    run_name, ok, err = fut.result()
                except Exception as exc:
                    run_name = f"{experiment}/{domain}"
                    ok, err = False, str(exc)
                results.append((run_name, ok, err))

                if ok:
                    logger.info(f"  ✓ {run_name}")
                    if not args.dry_run and hf_manager.enabled:
                        experiment_subdir = EXPERIMENT_DIRS.get(experiment, "")
                        out = args.output_dir / experiment_subdir / experiment / domain
                        hf_manager.upload_directory(out)
                else:
                    logger.error(f"  ✗ {run_name}: {err}")

    # ── Summary ─────────────────────────────────────────────────────────
    failed_runs = [name for name, ok, _ in results if not ok]

    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total runs: {total_runs}")
    logger.info(f"Successful: {total_runs - len(failed_runs)}")
    logger.info(f"Failed: {len(failed_runs)}")

    if failed_runs:
        logger.error("\nFailed runs:")
        for run in failed_runs:
            logger.error(f"  - {run}")
        sys.exit(1)
    else:
        logger.info("\n✓ All runs completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
