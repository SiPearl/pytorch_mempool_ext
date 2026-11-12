#!/usr/bin/env bash
#
# NUMA allocator validation campaign: placement, bandwidth, socket locality,
# the F.linear / FFN benchmark and the report.
#
# Run it from a shell where `python3` is the .venv that holds the PyTorch build
# and pytorch_numa_mem. A failing stage does not stop the campaign; only a
# fundamental problem does (torch cannot be imported, a node cannot be used).
# run_hw.sh calls this script after its own checks.
#
# It uses N threads, pinned to N physical cores of the DDR nodes, and records
# the machine load before every stage.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
EXT_ROOT="$(dirname "${SCRIPT_DIR}")"
# The stage logs can then be followed with tail -f while a stage runs.
export PYTHONUNBUFFERED=1

usage() {
  cat <<EOF
Usage: run_campaign.sh [options]

Node sets. The defaults are the Rhea layout; pass them on any other machine.
  --ddr-nodes LIST              DDR nodes. Their physical cores run the stages. (default 0-3)
  --hbm-nodes LIST              HBM nodes, on the socket of the DDR nodes (default 4-7)
  --allow-remote                accept HBM nodes on another socket. Only for the
                                socket-locality control run.

Threads
  --threads N                   OpenMP and MKL threads (default 20). The stages are pinned
                                with numactl --physcpubind to N physical cores, spread
                                evenly over the DDR nodes. SMT siblings are not used.

Paths
  --out-dir DIR                 write the results into DIR and make no archive
                                (default: results/<timestamp> next to this script,
                                plus a .tar.gz of it)
  --lib PATH                    libnuma_allocator.so
                                (default: ../libnuma_allocator/build/libnuma_allocator.so)
  --pytorch-root DIR            PyTorch checkout (default: where the imported torch lives)

Sizes
  --tensor-size-mb N            placement and bandwidth tensors (default 512)
  --warmups N                   bandwidth warm-up calls (default 3)
  --repetitions N               bandwidth timed calls (default 10)
  --dtype NAME                  linear benchmark dtype: bfloat16, float16, float32 (default bfloat16)
  --linear-iterations N         timed decode rounds, a multiple of 3 (default 21)
  --linear-prefill-iterations N timed prefill rounds, a multiple of 3 (default 6)
  --param-tier TIER             where the FFN parameters live: hbm, ddr, default (default hbm)
  --cache-flush-mib N           stream N MiB before every timed linear call (default 0)
  --max-minutes N               stop the linear benchmark before it measures if it is
                                estimated to take longer (default 60)
  --skip-linear                 do not run the linear benchmark
  --dry-run                     32 MiB tensors, matrix sizes divided by 16, few repetitions.
                                Checks that every stage runs; the timings mean nothing.
  -h, --help                    show this text

Examples
  run_campaign.sh --ddr-nodes 4-7 --hbm-nodes 12-15 --threads 20
  run_campaign.sh --ddr-nodes 4-7 --hbm-nodes 8-11 --threads 20 --allow-remote --skip-linear
  run_campaign.sh --ddr-nodes 0 --hbm-nodes 0 --threads 4 --dry-run
EOF
}

log() { echo "[run_campaign] $*"; }
warn() { echo "[run_campaign][WARNING] $*" >&2; }
die() {
  echo "[run_campaign][ERROR] $*" >&2
  exit 1
}

DDR_NODES="0-3"
HBM_NODES="4-7"
ALLOW_REMOTE=0
THREADS=20
OUT_DIR=""
LIB="${EXT_ROOT}/libnuma_allocator/build/libnuma_allocator.so"
PYTORCH_ROOT=""
TENSOR_SIZE_MB=512
WARMUPS=3
REPETITIONS=10
DTYPE="bfloat16"
LINEAR_ITERATIONS=21
LINEAR_PREFILL_ITERATIONS=6
PARAM_TIER="hbm"
CACHE_FLUSH_MIB=0
MAX_MINUTES=60
SKIP_LINEAR=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    --allow-remote) ALLOW_REMOTE=1 ;;
    --skip-linear) SKIP_LINEAR=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --*)
      [[ $# -ge 2 ]] || die "$1 needs a value (see --help)"
      case "$1" in
        --ddr-nodes) DDR_NODES="$2" ;;
        --hbm-nodes) HBM_NODES="$2" ;;
        --threads) THREADS="$2" ;;
        --out-dir) OUT_DIR="$2" ;;
        --lib) LIB="$2" ;;
        --pytorch-root) PYTORCH_ROOT="$2" ;;
        --tensor-size-mb) TENSOR_SIZE_MB="$2" ;;
        --warmups) WARMUPS="$2" ;;
        --repetitions) REPETITIONS="$2" ;;
        --dtype) DTYPE="$2" ;;
        --linear-iterations) LINEAR_ITERATIONS="$2" ;;
        --linear-prefill-iterations) LINEAR_PREFILL_ITERATIONS="$2" ;;
        --param-tier) PARAM_TIER="$2" ;;
        --cache-flush-mib) CACHE_FLUSH_MIB="$2" ;;
        --max-minutes) MAX_MINUTES="$2" ;;
        *) die "unknown option $1 (see --help)" ;;
      esac
      shift
      ;;
    *) die "unexpected argument $1 (see --help)" ;;
  esac
  shift
done

LINEAR_ARGS=()
if [[ "${DRY_RUN}" == "1" ]]; then
  TENSOR_SIZE_MB=32
  WARMUPS=1
  REPETITIONS=3
  LINEAR_ITERATIONS=6
  LINEAR_PREFILL_ITERATIONS=3
  LINEAR_ARGS+=(--shape-divisor 16 --warmup 3)
fi
# "${array[@]+...}" below keeps an empty array from tripping `set -u` on old bash.
REMOTE_ARGS=()
[[ "${ALLOW_REMOTE}" == "1" ]] && REMOTE_ARGS+=(--allow-remote)

command -v numactl >/dev/null 2>&1 || die "numactl is not on PATH. It pins the stages to their cores."
python3 -c 'import torch' 2>/dev/null \
  || die "python3 cannot import torch. Activate the .venv of the PyTorch checkout first."
[[ -f "${LIB}" ]] || die "${LIB} not found. Run make in ${EXT_ROOT}/libnuma_allocator or pass --lib."
if [[ -z "${PYTORCH_ROOT}" ]]; then
  PYTORCH_ROOT="$(python3 -c 'import os, torch; print(os.path.dirname(os.path.dirname(os.path.realpath(torch.__file__))))')"
fi

STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
MAKE_ARCHIVE=0
if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${SCRIPT_DIR}/results/${TIMESTAMP}"
  MAKE_ARCHIVE=1
fi
mkdir -p "${OUT_DIR}" || die "cannot create ${OUT_DIR}"
RESULTS_DIR="$(cd "${OUT_DIR}" && pwd -P)"
log "results directory: ${RESULTS_DIR}"
[[ "${DRY_RUN}" == "1" ]] && log "DRY RUN: small sizes, the timings mean nothing"

# The node sets must be usable before anything is measured: they exist, are
# inside Mems_allowed, are on one socket (unless --allow-remote), have enough
# physical cores for the threads, and every node has room for five tensors. A
# bound allocation that does not fit gets the process killed, not an error.
if ! python3 "${SCRIPT_DIR}/hbm_test_utils.py" --ddr-nodes "${DDR_NODES}" --hbm-nodes "${HBM_NODES}" \
  --threads "${THREADS}" --min-free-mb "$((5 * TENSOR_SIZE_MB))" "${REMOTE_ARGS[@]+"${REMOTE_ARGS[@]}"}" \
  --out "${RESULTS_DIR}/topology.json" --pin-out "${RESULTS_DIR}/pinned_cpus.txt" \
  > "${RESULTS_DIR}/topology.txt" 2>&1; then
  cat "${RESULTS_DIR}/topology.txt" >&2
  die "the node selection cannot be used on this machine, see the [ERROR] lines above"
fi
cat "${RESULTS_DIR}/topology.txt"
PIN_CPUS="$(cat "${RESULTS_DIR}/pinned_cpus.txt")"

export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
# --membind: the default allocator follows the process policy, the DDR nodes.
PIN=(numactl "--physcpubind=${PIN_CPUS}" "--membind=${DDR_NODES}")
log "every stage runs under: ${PIN[*]} with OMP_NUM_THREADS=MKL_NUM_THREADS=${THREADS}"

SKIPPED_STAGES=""
[[ "${SKIP_LINEAR}" == "1" ]] && SKIPPED_STAGES="linear_benchmark"
{
  echo "started_utc: ${STARTED_UTC}"
  echo "dry_run: ${DRY_RUN}"
  echo "ddr_nodes: ${DDR_NODES}"
  echo "hbm_nodes: ${HBM_NODES}"
  echo "allow_remote: ${ALLOW_REMOTE}"
  echo "threads: ${THREADS}"
  echo "pinned_cpus: ${PIN_CPUS}"
  echo "stage_binding: ${PIN[*]}"
  echo "lib: ${LIB}"
  echo "tensor_size_mb: ${TENSOR_SIZE_MB}"
  echo "warmups: ${WARMUPS}"
  echo "repetitions: ${REPETITIONS}"
  echo "dtype: ${DTYPE}"
  echo "linear_iterations: ${LINEAR_ITERATIONS}"
  echo "linear_prefill_iterations: ${LINEAR_PREFILL_ITERATIONS}"
  echo "param_tier: ${PARAM_TIER}"
  echo "cache_flush_mib: ${CACHE_FLUSH_MIB}"
  echo "max_minutes: ${MAX_MINUTES}"
  echo "skipped_stages: ${SKIPPED_STAGES}"
} > "${RESULTS_DIR}/campaign_config.txt"

python3 "${SCRIPT_DIR}/hbm_test_utils.py" --mode source --pytorch-root "${PYTORCH_ROOT}" \
  --files-out "${RESULTS_DIR}/ext_files.sha256" > "${RESULTS_DIR}/git_info.txt" 2>&1 \
  || warn "could not record the source revisions, see ${RESULTS_DIR}/git_info.txt"

{
  echo "=== hostname ==="
  hostname
  echo "=== date ==="
  date -u
  echo "=== uname -a ==="
  uname -a
  echo "=== lscpu ==="
  lscpu 2>&1 || echo "(lscpu unavailable)"
  echo "=== numactl -H ==="
  numactl -H 2>&1
  echo "=== Mems_allowed and Cpus_allowed of this shell ==="
  grep -E '^(Mems|Cpus)_allowed' /proc/self/status
  echo "=== numastat ==="
  numastat 2>&1 || echo "(numastat unavailable)"
} > "${RESULTS_DIR}/system_info.txt"

# Pinned like the stages, so that the kernel it reports for F.linear is the
# one the stages get.
if ! "${PIN[@]}" python3 "${SCRIPT_DIR}/hbm_test_utils.py" --mode provenance --pytorch-root "${PYTORCH_ROOT}" \
  --lib "${LIB}" --config-out "${RESULTS_DIR}/torch_config.txt" > "${RESULTS_DIR}/build_info.txt" 2>&1; then
  cat "${RESULTS_DIR}/build_info.txt" >&2
  die "torch, pytorch_numa_mem or libnuma_allocator cannot be loaded, see above"
fi
cat "${RESULTS_DIR}/build_info.txt"

FAILED_STAGES=""
run_stage() {
  local name="$1"
  shift
  log "=== stage: ${name} ==="
  # Other processes can disturb the timings; this records the load before the stage.
  {
    uptime
    top -bn1 | head -20
  } > "${RESULTS_DIR}/load_before_${name}.txt" 2>&1
  if "${PIN[@]}" python3 "$@" > "${RESULTS_DIR}/${name}.log" 2>&1; then
    log "stage ${name} completed"
  else
    warn "stage ${name} exited with status $?, see ${RESULTS_DIR}/${name}.log. Continuing."
    FAILED_STAGES="${FAILED_STAGES} ${name}"
  fi
}

COMMON_ARGS=(--lib "${LIB}" --ddr-nodes "${DDR_NODES}" --hbm-nodes "${HBM_NODES}")

run_stage placement "${SCRIPT_DIR}/test_placement.py" --out-dir "${RESULTS_DIR}" "${COMMON_ARGS[@]}" \
  --tensor-size-mb "${TENSOR_SIZE_MB}"

run_stage bandwidth "${SCRIPT_DIR}/test_bandwidth.py" --out-dir "${RESULTS_DIR}" "${COMMON_ARGS[@]}" \
  --tensor-size-mb "${TENSOR_SIZE_MB}" --warmups "${WARMUPS}" --repetitions "${REPETITIONS}"

run_stage socket_locality "${SCRIPT_DIR}/test_socket_locality.py" --out-dir "${RESULTS_DIR}" "${COMMON_ARGS[@]}" \
  --tensor-size-mb "${TENSOR_SIZE_MB}" "${REMOTE_ARGS[@]+"${REMOTE_ARGS[@]}"}"

if [[ "${SKIP_LINEAR}" == "1" ]]; then
  log "=== stage: linear_benchmark (skipped, --skip-linear) ==="
else
  run_stage linear_benchmark "${SCRIPT_DIR}/benchmark_hbm_linear.py" "${COMMON_ARGS[@]}" --dtype "${DTYPE}" \
    --iterations "${LINEAR_ITERATIONS}" --prefill-iterations "${LINEAR_PREFILL_ITERATIONS}" \
    --param-tier "${PARAM_TIER}" --cache-flush-mib "${CACHE_FLUSH_MIB}" --max-minutes "${MAX_MINUTES}" \
    "${LINEAR_ARGS[@]+"${LINEAR_ARGS[@]}"}" --json-output "${RESULTS_DIR}/linear_benchmark.json"
fi

log "=== generating the report ==="
python3 "${SCRIPT_DIR}/generate_report.py" --results-dir "${RESULTS_DIR}" || FAILED_STAGES="${FAILED_STAGES} report"

echo "============================================================"
echo "Report:  ${RESULTS_DIR}/summary.md"
if [[ "${MAKE_ARCHIVE}" == "1" ]]; then
  ARCHIVE="${SCRIPT_DIR}/results/numa_allocator_validation_${TIMESTAMP}.tar.gz"
  tar -C "${SCRIPT_DIR}/results" -czf "${ARCHIVE}" "${TIMESTAMP}"
  echo "Archive: ${ARCHIVE}"
fi
if [[ -n "${FAILED_STAGES}" ]]; then
  echo "Stages that exited with an error:${FAILED_STAGES}"
  echo "============================================================"
  exit 1
fi
echo "============================================================"
