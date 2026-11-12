#!/usr/bin/env bash
#
# One non-interactive run of everything on a target machine:
#
#   1. check every prerequisite, and stop with a message if one is missing
#   2. record the machine, the toolchain and the revisions of both source trees
#   3. libnuma_allocator smoke test on the two node sets
#   4. pytorch_numa_mem pytest suite, with the residency checks on those node sets
#   5. the campaign (run_campaign.sh)
#   6. pack everything into results_<host>_<timestamp>.tar.gz and print its path
#
# The run uses N threads (default 20) and is pinned with numactl --physcpubind
# to N physical cores of the DDR nodes. It records the machine load before
# every benchmark and checks the free memory of every target node before it
# allocates.
#
# It writes only below --out. It needs no network and no sudo. PyTorch with the
# allocator override, libnuma_allocator and pytorch_numa_mem must be built first.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
EXT_ROOT="$(dirname "${SCRIPT_DIR}")"
LIB="${EXT_ROOT}/libnuma_allocator/build/libnuma_allocator.so"
SMOKE_TEST="${EXT_ROOT}/libnuma_allocator/build/test_numa_allocator"

usage() {
  cat <<EOF
Usage: run_hw.sh --ddr-nodes LIST --hbm-nodes LIST --out DIR [options] [-- run_campaign.sh options]

Required. The node IDs are never guessed; read them from \`numactl -H\`.
  --ddr-nodes LIST    DDR NUMA nodes of one socket, for example 4-7. Their physical
                      cores run everything.
  --hbm-nodes LIST    HBM NUMA nodes of the same socket, for example 12-15
  --out DIR           directory for the results. Each run adds
                      DIR/results_<host>_<timestamp>/ and the .tar.gz of it.

Options
  --threads N         threads and cores to use (default 20). OMP_NUM_THREADS and
                      MKL_NUM_THREADS are set to N, and the run is pinned to N
                      physical cores, spread evenly over the DDR nodes. SMT siblings
                      are read from sysfs and never used.
  --allow-remote      accept HBM nodes on another socket than the DDR nodes. This is
                      the socket-locality control: the same run with remote HBM.
                      Without it, such a node selection is refused.
  --dtype NAME        dtype of the linear benchmark: bfloat16 (default), float16, float32
  --dry-run           small sizes, about a minute. Checks that every step runs on
                      this machine; the timings mean nothing. The only mode in
                      which the DDR and HBM node sets may overlap, so it also
                      runs on a machine with one NUMA node and no HBM (pass the
                      same node for both sets).
  -h, --help          show this text
  --                  everything after it goes to run_campaign.sh unchanged
                      (run_campaign.sh --help lists the options)

Run it from a shell where the .venv of the PyTorch checkout is active.

Examples
  run_hw.sh --ddr-nodes 4-7 --hbm-nodes 12-15 --threads 20 --out \$HOME/poc_results
  run_hw.sh --ddr-nodes 0-3 --hbm-nodes 8-11 --threads 20 --out \$HOME/poc_results
  run_hw.sh --ddr-nodes 4-7 --hbm-nodes 8-11 --threads 20 --allow-remote --out \$HOME/poc_results -- --skip-linear
  run_hw.sh --ddr-nodes 0 --hbm-nodes 0 --threads 4 --out /tmp/poc_dry --dry-run
EOF
}

log() { echo "[run_hw] $*"; }
die() {
  echo "[run_hw][ERROR] $*" >&2
  exit 1
}

DDR_NODES=""
HBM_NODES=""
OUT=""
THREADS=20
ALLOW_REMOTE=0
DRY_RUN=0
CAMPAIGN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    --dry-run) DRY_RUN=1 ;;
    --allow-remote) ALLOW_REMOTE=1 ;;
    --)
      shift
      CAMPAIGN_ARGS+=("$@")
      break
      ;;
    --ddr-nodes | --hbm-nodes | --out | --threads | --dtype)
      [[ $# -ge 2 ]] || die "$1 needs a value (see --help)"
      case "$1" in
        --ddr-nodes) DDR_NODES="$2" ;;
        --hbm-nodes) HBM_NODES="$2" ;;
        --out) OUT="$2" ;;
        --threads) THREADS="$2" ;;
        *) CAMPAIGN_ARGS+=("$1" "$2") ;;
      esac
      shift
      ;;
    *) die "unknown argument $1 (see --help)" ;;
  esac
  shift
done

[[ -n "${DDR_NODES}" ]] || die "--ddr-nodes is required. Read the node IDs from \`numactl -H\`; this script never guesses them."
[[ -n "${HBM_NODES}" ]] || die "--hbm-nodes is required. Read the node IDs from \`numactl -H\`; this script never guesses them."
[[ -n "${OUT}" ]] || die "--out DIR is required"
[[ "${THREADS}" =~ ^[1-9][0-9]*$ ]] || die "--threads needs a positive number, got '${THREADS}'"
# "${array[@]+...}" below keeps an empty array from tripping `set -u` on old bash.
REMOTE_ARGS=()
[[ "${ALLOW_REMOTE}" == "1" ]] && REMOTE_ARGS+=(--allow-remote)

# 1. Prerequisites.

[[ "$(uname -s)" == "Linux" ]] || die "this harness needs Linux (NUMA syscalls, /proc, /sys)"
for tool in numactl lscpu git tar python3 uptime top; do
  command -v "${tool}" >/dev/null 2>&1 || die "\`${tool}\` is not on PATH. It is required; ask the site how to load it (no sudo is needed to use it)."
done
python3 -c 'import sys; sys.exit(sys.prefix == sys.base_prefix)' \
  || die "python3 is not a virtual environment. Activate the .venv of the PyTorch checkout: source \"\$PYTORCH_DIR/.venv/bin/activate\""
python3 -c 'import torch' >/dev/null 2>&1 \
  || die "python3 cannot import torch. Build PyTorch in this .venv first."
[[ -f "${LIB}" && -x "${SMOKE_TEST}" ]] \
  || die "${LIB} or its smoke test is missing. Build them: make -C ${EXT_ROOT}/libnuma_allocator"
if ! IMPORT_ERROR="$(python3 -c 'import pytorch_numa_mem' 2>&1)"; then
  echo "${IMPORT_ERROR}" | tail -3 >&2
  die "python3 cannot import pytorch_numa_mem. Install it into this .venv: uv pip install -e ${EXT_ROOT}/pytorch_numa_mem --no-build-isolation. An undefined symbol means that this PyTorch build does not have the allocator override: check the branch."
fi
python3 -c 'import pytest' >/dev/null 2>&1 \
  || die "pytest is not installed in this .venv. Install it: uv pip install pytest"

mkdir -p "${OUT}" 2>/dev/null || die "cannot create ${OUT}"
OUT="$(cd "${OUT}" && pwd -P)"
[[ -w "${OUT}" ]] || die "${OUT} is not writable"

HOST="$(hostname -s 2>/dev/null || hostname)"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
NAME="results_${HOST}_${TIMESTAMP}"
RUN_DIR="${OUT}/${NAME}"
ENV_DIR="${RUN_DIR}/environment"
SCRATCH="${OUT}/tmp_${TIMESTAMP}"
mkdir "${RUN_DIR}" "${ENV_DIR}" "${SCRATCH}" || die "cannot create ${RUN_DIR}"
trap 'rm -rf "${SCRATCH}"' EXIT

# Everything a child process would put in /tmp or in a cache goes below --out.
# No bytecode files, so the source trees stay as they were.
export TMPDIR="${SCRATCH}"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH}/torchinductor"
export PYTHONDONTWRITEBYTECODE=1

{
  echo "command: $0 --ddr-nodes ${DDR_NODES} --hbm-nodes ${HBM_NODES} --threads ${THREADS} --out ${OUT}$([[ "${ALLOW_REMOTE}" == "1" ]] && echo " --allow-remote")$([[ "${DRY_RUN}" == "1" ]] && echo " --dry-run") ${CAMPAIGN_ARGS[*]+"${CAMPAIGN_ARGS[*]}"}"
  echo "started_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(hostname)"
  echo "threads: ${THREADS}"
  echo "allow_remote: ${ALLOW_REMOTE}"
  echo "dry_run: ${DRY_RUN}"
} > "${RUN_DIR}/command.txt"

# The node sets: they must exist, be inside Mems_allowed, be on one socket
# (unless --allow-remote), have N physical cores, and every node needs room for
# the campaign: five tensors of its default size, 512 MiB (32 MiB in a dry run).
# The same call chooses the cores to pin to.

MIN_FREE_MB=2560
[[ "${DRY_RUN}" == "1" ]] && MIN_FREE_MB=160
if ! python3 "${SCRIPT_DIR}/hbm_test_utils.py" --ddr-nodes "${DDR_NODES}" --hbm-nodes "${HBM_NODES}" \
  --threads "${THREADS}" --min-free-mb "${MIN_FREE_MB}" "${REMOTE_ARGS[@]+"${REMOTE_ARGS[@]}"}" \
  --out "${ENV_DIR}/topology.json" --pin-out "${ENV_DIR}/pinned_cpus.txt" > "${ENV_DIR}/topology.txt" 2>&1; then
  cat "${ENV_DIR}/topology.txt" >&2
  die "the node sets cannot be used on this machine, see the [ERROR] lines above and \`numactl -H\`"
fi
cat "${ENV_DIR}/topology.txt"
if [[ "${DRY_RUN}" != "1" ]] && grep -q "node sets overlap" "${ENV_DIR}/topology.txt"; then
  die "the DDR and HBM node sets overlap. A real run needs two different sets; --dry-run accepts the same set twice."
fi

PIN_CPUS="$(cat "${ENV_DIR}/pinned_cpus.txt")"
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
PIN=(numactl "--physcpubind=${PIN_CPUS}" "--membind=${DDR_NODES}")
log "pinned to CPUs ${PIN_CPUS} with OMP_NUM_THREADS=MKL_NUM_THREADS=${THREADS}"

# 2. Record the machine before anything runs.

log "recording the environment in ${ENV_DIR}"
numactl -H > "${ENV_DIR}/numactl_H.txt" 2>&1
grep -E '^(Mems|Cpus)_allowed' /proc/self/status > "${ENV_DIR}/mems_allowed.txt"
lscpu > "${ENV_DIR}/lscpu.txt" 2>&1
{
  uptime
  top -bn1 | head -20
} > "${ENV_DIR}/load_at_start.txt" 2>&1
{
  uname -a
  cat /etc/os-release 2>/dev/null
  echo "page size: $(getconf PAGESIZE)"
  echo "transparent_hugepage: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo unknown)"
  echo "numa_balancing: $(cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo unknown)"
} > "${ENV_DIR}/kernel.txt"
{
  for compiler in cc c++ gcc g++; do
    if command -v "${compiler}" >/dev/null 2>&1; then
      echo "${compiler}: $(command -v "${compiler}"): $("${compiler}" --version | head -1)"
    else
      echo "${compiler}: not on PATH"
    fi
  done
  echo "The compiler that built PyTorch is in python_torch.txt (build_compiler)."
} > "${ENV_DIR}/compiler.txt"
# Pinned like the steps, so that the F.linear kernel and its timing in this
# record are the ones the benchmarks get. torch_config.txt receives
# torch.__config__.show() and torch.__config__.parallel_info().
"${PIN[@]}" python3 "${SCRIPT_DIR}/hbm_test_utils.py" --mode provenance --lib "${LIB}" \
  --config-out "${ENV_DIR}/torch_config.txt" > "${ENV_DIR}/python_torch.txt" 2>&1 \
  || die "torch, pytorch_numa_mem or libnuma_allocator cannot be loaded: $(cat "${ENV_DIR}/python_torch.txt")"
python3 "${SCRIPT_DIR}/hbm_test_utils.py" --mode source \
  --files-out "${ENV_DIR}/ext_files.sha256" > "${ENV_DIR}/repos.txt" 2>&1 \
  || die "cannot record the source revisions: $(cat "${ENV_DIR}/repos.txt")"
env | grep -E '^(OMP_|GOMP_|KMP_|MKL_|SLURM_JOB_ID=|SLURM_CPUS|SLURM_MEM)' | sort > "${ENV_DIR}/env_vars.txt"
grep -E '^(pytorch|ext)_(branch|sha|dirty)' "${ENV_DIR}/repos.txt"
if grep -q '_sha: not a git repository' "${ENV_DIR}/repos.txt"; then
  log "WARNING: a source tree is not a git repository, so it has no revision. Its content hash is in ${ENV_DIR}/repos.txt."
fi
grep -E '^(torch_version|build_compiler|bfloat16_linear_rows1|float32_linear_rows1):' "${ENV_DIR}/python_torch.txt"

# 3 to 5. The steps. A failing step does not stop the next one: the run is
# started by hand, so one run should bring back everything it can.

FAILED=""
run_step() {
  local name="$1"
  shift
  log "=== ${name} ==="
  local start=${SECONDS}
  "$@" > "${RUN_DIR}/${name}.log" 2>&1
  local rc=$?
  if [[ ${rc} -eq 0 ]]; then
    echo "${name}: PASS ($((SECONDS - start)) s)" | tee -a "${RUN_DIR}/status.txt"
  else
    echo "${name}: FAIL (exit status ${rc}, $((SECONDS - start)) s), see ${RUN_DIR}/${name}.log" | tee -a "${RUN_DIR}/status.txt"
    tail -5 "${RUN_DIR}/${name}.log" | sed 's/^/    /'
    FAILED="${FAILED} ${name}"
  fi
}

run_step 01_smoke_test "${PIN[@]}" "${SMOKE_TEST}" "${DDR_NODES}" "${HBM_NODES}"

# A subshell, so that the cd does not leak into the next step.
run_pytest() (
  cd "${EXT_ROOT}/pytorch_numa_mem" || exit 1
  NUMA_ALLOCATOR_LIB="${LIB}" NUMA_MEM_NODES_A="${DDR_NODES}" NUMA_MEM_NODES_B="${HBM_NODES}" \
    "${PIN[@]}" python3 -m pytest -q -rs -p no:cacheprovider --basetemp="${SCRATCH}/pytest"
)
run_step 02_pytest run_pytest

# The campaign pins its stages itself, to the same cores, and records the
# machine load before each of them.
[[ "${DRY_RUN}" == "1" ]] && CAMPAIGN_ARGS+=(--dry-run)
run_step 03_campaign bash "${SCRIPT_DIR}/run_campaign.sh" --ddr-nodes "${DDR_NODES}" --hbm-nodes "${HBM_NODES}" \
  --threads "${THREADS}" "${REMOTE_ARGS[@]+"${REMOTE_ARGS[@]}"}" --lib "${LIB}" --out-dir "${RUN_DIR}/campaign" \
  "${CAMPAIGN_ARGS[@]+"${CAMPAIGN_ARGS[@]}"}"

# 6. Pack.

echo "finished_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${RUN_DIR}/command.txt"
ARCHIVE="${OUT}/${NAME}.tar.gz"
tar -C "${OUT}" -czf "${ARCHIVE}" "${NAME}" || die "could not write ${ARCHIVE}; the results are in ${RUN_DIR}"

echo "============================================================"
cat "${RUN_DIR}/status.txt"
grep -h '^\*\*Verdict' "${RUN_DIR}/campaign/summary.md" 2>/dev/null
echo
echo "Archive to copy back:"
echo "${ARCHIVE}"
echo "============================================================"
[[ -z "${FAILED}" ]]
