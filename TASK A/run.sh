#!/usr/bin/env bash
# =============================================================================
#  MT-RAG Benchmark – Unified Experiment Runner
# =============================================================================
#  Single entry-point script for running any combination of experiments and
#  domains.  Wraps scripts/run_experiment.py (config merging) and
#  scripts/build_indices.py (index building).
#
#  Usage examples:
#    ./run.sh --help
#    ./run.sh --list                          # list all experiments & domains
#    ./run.sh -e replication_bm25 -d clapnq   # single experiment, single domain
#    ./run.sh -e replication_bm25 -d all       # single experiment, all domains
#    ./run.sh -e all -d all                    # full sweep
#    ./run.sh -e "replication_bm25,replication_splade" -d "clapnq,fiqa"
#    ./run.sh --category baselines -d all      # run all baseline experiments
#    ./run.sh --build-indices -d all -m all    # build all indices
#    ./run.sh -e replication_bm25 -d clapnq --dry-run
# =============================================================================
set -euo pipefail

# ─── Resolve project root (directory containing this script) ─────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── PYTHONPATH (required for src/ imports) ───────────────────────────────────────
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/src:${PYTHONPATH:-}"

# ─── Constants ───────────────────────────────────────────────────────────────
DOMAINS=(clapnq fiqa govt cloud)

# Experiments grouped by category
declare -A CATEGORY_MAP
CATEGORY_MAP=(
  [baselines]="replication_bm25 replication_bge15 replication_bgem3 replication_splade replication_voyage A0_baseline_bm25_fullhist A0_baseline_splade_fullhist A1_baseline_bgem3_fullhist A1_baseline_voyage_fullhist A2_baseline_bm25_rewrite A2_baseline_splade_rewrite A2_baseline_bge15_rewrite A2_baseline_bgem3_rewrite A2_baseline_voyage_rewrite"
  [hybrid]="hybrid_splade_voyage_norewrite hybrid_splade_voyage_rewrite hybrid_splade_bge15_norewrite hybrid_splade_bge15_rewrite"
  [rerank]="rerank_splade_voyage_rewrite rerank_splade_bge15_rewrite rerank_cohere_splade_bge15_rewrite rerank_cohere_splade_voyage_rewrite"
  [ablation-fusion]="ablation_fusion_linear_splade_voyage ablation_fusion_linear_splade_bge15 ablation_fusion_linear_alpha03_voyage ablation_fusion_linear_alpha07_voyage"
  [ablation-rrf-k]="ablation_rrf_k1_voyage ablation_rrf_k20_voyage ablation_rrf_k40_voyage ablation_rrf_k100_voyage"
  [ablation-topk]="ablation_topk_100_voyage ablation_topk_200_voyage ablation_topk_500_voyage ablation_topk_100_bge15"
  [ablation-rerank]="ablation_rerank_depth_50 ablation_rerank_depth_200 ablation_rerank_depth_50_cohere ablation_rerank_depth_200_cohere"
  [ablation-query]="ablation_query_mode_lastturn_hybrid ablation_query_mode_fullhist_hybrid ablation_query_mode_fullctx_hybrid"
  [ablation-components]="ablation_component_splade_cohere_v3 ablation_component_voyage_cohere_v3 ablation_component_bge15_cohere_v3"
  [ablation-all]="ablation_fusion_linear_splade_voyage ablation_fusion_linear_splade_bge15 ablation_fusion_linear_alpha03_voyage ablation_fusion_linear_alpha07_voyage ablation_rrf_k1_voyage ablation_rrf_k20_voyage ablation_rrf_k40_voyage ablation_rrf_k100_voyage ablation_topk_100_voyage ablation_topk_200_voyage ablation_topk_500_voyage ablation_topk_100_bge15 ablation_rerank_depth_50 ablation_rerank_depth_200 ablation_rerank_depth_50_cohere ablation_rerank_depth_200_cohere ablation_query_mode_lastturn_hybrid ablation_query_mode_fullhist_hybrid ablation_query_mode_fullctx_hybrid ablation_component_splade_cohere_v3 ablation_component_voyage_cohere_v3 ablation_component_bge15_cohere_v3"
  [rewrite-ablation]="hybrid_splade_voyage_rewrite_own_replica hybrid_splade_voyage_rewrite_own_improved hybrid_splade_voyage_rewrite_own_local"
)

ALL_EXPERIMENTS=""
for cat in baselines hybrid rerank ablation-fusion ablation-rrf-k ablation-topk ablation-rerank ablation-query ablation-components rewrite-ablation; do
  ALL_EXPERIMENTS+="${CATEGORY_MAP[$cat]} "
done
ALL_EXPERIMENTS=$(echo "$ALL_EXPERIMENTS" | xargs)

INDEX_MODELS=(bm25 splade bge-base-1.5 bge-m3 bgem3_dense bgem3_sparse bgem3_colbert bgem3_all)

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ─── Helper functions ────────────────────────────────────────────────────────
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

show_help() {
  cat <<EOF
${BOLD}MT-RAG Benchmark – Unified Experiment Runner${NC}

${BOLD}USAGE${NC}
  ./run.sh [OPTIONS]

${BOLD}EXPERIMENT OPTIONS${NC}
  -e, --experiment NAMES   Experiment name(s). Comma-separated or "all".
  -d, --domain NAMES       Domain name(s). Comma-separated or "all".
                            Domains: ${DOMAINS[*]}
  -c, --category CAT       Run all experiments in a category.
                            Categories: baselines, query, hybrid, rerank, finetune,
                            ablation-fusion, ablation-rrf-k, ablation-topk,
                            ablation-rerank, ablation-query, ablation-components,
                            ablation-all
      --baseline-path PATH  Baseline results path for significance testing.

${BOLD}INDEX OPTIONS${NC}
      --build-indices       Build retrieval indices instead of running experiments.
  -m, --model NAMES        Index model(s). Comma-separated or "all".
                            Models: ${INDEX_MODELS[*]}
      --batch-size N        Indexing batch size (default: 100).
      --corpus-dir DIR      Corpus directory (default: data/processed/).

${BOLD}EXECUTION OPTIONS${NC}
      --dry-run             Validate configs without executing.
      --force               Overwrite existing results / indices.
  -v, --verbose            Enable debug logging.
  -o, --output-dir DIR     Output root directory (default: experiments/).
      --config-dir DIR      Config directory (default: configs/).

${BOLD}INFO${NC}
  -l, --list               List all available experiments, domains, and categories.
  -h, --help               Show this help message.

${BOLD}EXAMPLES${NC}
  # Run one experiment on one domain
  ./run.sh -e replication_bm25 -d clapnq

  # Run all baselines on all domains
  ./run.sh --category baselines -d all

  # Run specific experiments on specific domains
  ./run.sh -e "replication_bm25,replication_splade" -d "clapnq,fiqa"

  # Full experiment sweep with significance testing
  ./run.sh -e all -d all --baseline-path experiments/0-baselines/replication_bm25

  # Build all indices
  ./run.sh --build-indices -d all -m all

  # Dry run to validate configuration
  ./run.sh -e replication_bm25 -d clapnq --dry-run
EOF
}

show_list() {
  echo -e "\n${BOLD}Available Domains:${NC}"
  for d in "${DOMAINS[@]}"; do
    echo "  - $d"
  done

  echo -e "\n${BOLD}Available Experiment Categories:${NC}"
  for cat in baselines query hybrid rerank finetune ablation-fusion ablation-rrf-k ablation-topk ablation-rerank ablation-query ablation-components; do
    echo -e "\n  ${CYAN}[$cat]${NC}"
    for exp in ${CATEGORY_MAP[$cat]}; do
      echo "    - $exp"
    done
  done

  echo -e "\n${BOLD}Available Index Models:${NC}"
  for m in "${INDEX_MODELS[@]}"; do
    echo "  - $m"
  done
  echo ""
}

# ─── Parse arguments ────────────────────────────────────────────────────────
EXPERIMENT=""
DOMAIN=""
CATEGORY=""
MODEL=""
BUILD_INDICES=false
DRY_RUN=false
FORCE=false
VERBOSE=false
OUTPUT_DIR="experiments"
CONFIG_DIR="configs"
BASELINE_PATH=""
BATCH_SIZE=100
CORPUS_DIR="data/processed"
LIST=false

while [[ $# -gt 0 ]]; do
  case $1 in
    -e|--experiment)   EXPERIMENT="$2";      shift 2;;
    -d|--domain)       DOMAIN="$2";          shift 2;;
    -c|--category)     CATEGORY="$2";        shift 2;;
    -m|--model)        MODEL="$2";           shift 2;;
    --build-indices)   BUILD_INDICES=true;   shift;;
    --dry-run)         DRY_RUN=true;         shift;;
    --force)           FORCE=true;           shift;;
    -v|--verbose)      VERBOSE=true;         shift;;
    -o|--output-dir)   OUTPUT_DIR="$2";      shift 2;;
    --config-dir)      CONFIG_DIR="$2";      shift 2;;
    --baseline-path)   BASELINE_PATH="$2";   shift 2;;
    --batch-size)      BATCH_SIZE="$2";      shift 2;;
    --corpus-dir)      CORPUS_DIR="$2";      shift 2;;
    -l|--list)         LIST=true;            shift;;
    -h|--help)         show_help;            exit 0;;
    *)                 error "Unknown option: $1"; show_help; exit 1;;
  esac
done

# ─── List mode ───────────────────────────────────────────────────────────────
if $LIST; then
  show_list
  exit 0
fi

# ─── Activate virtual environment if available ───────────────────────────────
if [[ -d "venv" ]] && [[ -f "venv/bin/activate" ]]; then
  source venv/bin/activate
  info "Activated virtual environment: venv/"
fi

# ─── Load .env if present ───────────────────────────────────────────────────
if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
  info "Loaded environment variables from .env"
fi

# ─── Validate Python ────────────────────────────────────────────────────────
if ! command -v python &>/dev/null; then
  error "Python not found. Please install Python 3.10+ or activate your virtualenv."
  exit 1
fi

# ─── Build Indices Mode ─────────────────────────────────────────────────────
if $BUILD_INDICES; then
  if [[ -z "$DOMAIN" ]]; then
    error "Must specify --domain (-d) for --build-indices."
    exit 1
  fi
  if [[ -z "$MODEL" ]]; then
    error "Must specify --model (-m) for --build-indices."
    exit 1
  fi

  # Resolve domains
  if [[ "$DOMAIN" == "all" ]]; then
    domains=("${DOMAINS[@]}")
  else
    IFS=',' read -ra domains <<< "$DOMAIN"
  fi

  # Resolve models
  if [[ "$MODEL" == "all" ]]; then
    models=("${INDEX_MODELS[@]}")
  else
    IFS=',' read -ra models <<< "$MODEL"
  fi

  info "Building indices for ${#domains[@]} domain(s) x ${#models[@]} model(s)..."

  total=0; ok=0; fail=0
  for d in "${domains[@]}"; do
    for m in "${models[@]}"; do
      total=$((total + 1))
      info "[$total] Building index: domain=${d}  model=${m}"

      cmd="python scripts/build_indices.py --domain $d --model $m --batch-size $BATCH_SIZE --corpus-dir $CORPUS_DIR"
      $FORCE   && cmd+=" --force"
      $VERBOSE && cmd+=" --verbose"

      if $DRY_RUN; then
        echo "  [DRY RUN] $cmd"
        ok=$((ok + 1))
      else
        if eval "$cmd"; then
          success "Index built: ${d}/${m}"
          ok=$((ok + 1))
        else
          warn "Index FAILED: ${d}/${m}"
          fail=$((fail + 1))
        fi
      fi
    done
  done

  echo ""
  echo -e "${BOLD}Index Build Summary${NC}"
  echo "  Total:   $total"
  echo -e "  ${GREEN}Success: $ok${NC}"
  [[ $fail -gt 0 ]] && echo -e "  ${RED}Failed:  $fail${NC}"
  exit $([[ $fail -eq 0 ]] && echo 0 || echo 1)
fi

# ─── Experiment Mode ────────────────────────────────────────────────────────
# Resolve experiments list
if [[ -n "$CATEGORY" ]]; then
  if [[ -z "${CATEGORY_MAP[$CATEGORY]+x}" ]]; then
    error "Unknown category: $CATEGORY"
    error "Valid categories: baselines, query, hybrid, rerank, finetune"
    exit 1
  fi
  IFS=' ' read -ra experiments <<< "${CATEGORY_MAP[$CATEGORY]}"
  info "Category '$CATEGORY' → ${#experiments[@]} experiment(s)"
elif [[ "$EXPERIMENT" == "all" ]]; then
  IFS=' ' read -ra experiments <<< "$ALL_EXPERIMENTS"
elif [[ -n "$EXPERIMENT" ]]; then
  IFS=',' read -ra experiments <<< "$EXPERIMENT"
else
  error "Must specify --experiment (-e), --category (-c), or --build-indices."
  show_help
  exit 1
fi

# Resolve domains list
if [[ -z "$DOMAIN" ]]; then
  error "Must specify --domain (-d)."
  exit 1
fi

if [[ "$DOMAIN" == "all" ]]; then
  domains=("${DOMAINS[@]}")
else
  IFS=',' read -ra domains <<< "$DOMAIN"
fi

# Validate experiment names
for exp in "${experiments[@]}"; do
  if ! echo "$ALL_EXPERIMENTS" | grep -qw "$exp"; then
    warn "Unknown experiment: '$exp' (not in registry, will pass through to run_experiment.py)"
  fi
done

# Validate domain names
for d in "${domains[@]}"; do
  found=false
  for vd in "${DOMAINS[@]}"; do
    [[ "$d" == "$vd" ]] && found=true && break
  done
  if ! $found; then
    error "Unknown domain: '$d'. Valid domains: ${DOMAINS[*]}"
    exit 1
  fi
done

# ─── Run Experiments ─────────────────────────────────────────────────────────
total_runs=$(( ${#experiments[@]} * ${#domains[@]} ))
info "Running ${#experiments[@]} experiment(s) x ${#domains[@]} domain(s) = ${BOLD}${total_runs} run(s)${NC}"
echo ""

run_num=0; ok=0; fail=0; skip=0
start_time=$(date +%s)

for exp in "${experiments[@]}"; do
  for d in "${domains[@]}"; do
    run_num=$((run_num + 1))

    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  Run ${run_num}/${total_runs}:  experiment=${CYAN}${exp}${NC}  domain=${CYAN}${d}${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    cmd="python scripts/run_experiment.py --experiment $exp --domain $d --output-dir $OUTPUT_DIR --config-dir $CONFIG_DIR"
    $DRY_RUN && cmd+=" --dry-run"
    $FORCE   && cmd+=" --force"
    $VERBOSE && cmd+=" --verbose"
    [[ -n "$BASELINE_PATH" ]] && cmd+=" --baseline-path $BASELINE_PATH"

    if $DRY_RUN; then
      echo "  [DRY RUN] $cmd"
      ok=$((ok + 1))
    else
      run_start=$(date +%s)
      if eval "$cmd"; then
        run_end=$(date +%s)
        elapsed=$(( run_end - run_start ))
        success "${exp}/${d} completed in ${elapsed}s"
        ok=$((ok + 1))
      else
        run_end=$(date +%s)
        elapsed=$(( run_end - run_start ))
        warn "${exp}/${d} FAILED after ${elapsed}s"
        fail=$((fail + 1))
      fi
    fi
    echo ""
  done
done

end_time=$(date +%s)
total_elapsed=$(( end_time - start_time ))

# ─── Summary ─────────────────────────────────────────────────────────────────
echo -e "${BOLD}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Experiment Run Summary${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════════════════${NC}"
echo "  Total runs:  $total_runs"
echo -e "  ${GREEN}Succeeded:   $ok${NC}"
[[ $fail -gt 0 ]] && echo -e "  ${RED}Failed:      $fail${NC}"
echo "  Duration:    ${total_elapsed}s ($(( total_elapsed / 60 ))m $(( total_elapsed % 60 ))s)"
echo ""

exit $([[ $fail -eq 0 ]] && echo 0 || echo 1)
