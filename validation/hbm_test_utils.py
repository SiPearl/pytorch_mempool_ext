"""Shared helpers for the NUMA allocator validation campaign.

Placement is always measured from the operating system and never taken from
what the allocator reports about itself:

1. move_pages(2) in query mode (nodes == NULL, so nothing migrates) gives the
   node of sampled pages of the tensor itself.
2. /proc/self/numa_maps gives the policy and the per-node page counts of the
   mapping that holds the tensor. The kernel merges adjacent mappings that have
   the same policy, so one line can cover several tensors of one allocator.

The NUMA syscalls go through syscall(2) with ctypes, as libnuma_allocator does.
libnuma is not used.

Standalone:
    python3 hbm_test_utils.py --ddr-nodes 0-3 --hbm-nodes 4-7 --threads 20      # check a node selection
    python3 hbm_test_utils.py --mode provenance      # the imported torch, the extension, the library
    python3 hbm_test_utils.py --mode source          # revisions of the PyTorch checkout and of this tree
"""

import argparse
import ctypes
import glob
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

VALIDATION_DIR = Path(__file__).resolve().parent
EXT_ROOT = VALIDATION_DIR.parent
DEFAULT_LIB = EXT_ROOT / "libnuma_allocator" / "build" / "libnuma_allocator.so"

# origin/viable/strict of the fork when the PoC branch was created.
POC_BASE_SHA = "0729d86c88f5af969d570a1af7a3ef22b018c8ed"

# Rhea. run_hw.sh never relies on these: it requires both lists.
DEFAULT_DDR_NODES = "0-3"
DEFAULT_HBM_NODES = "4-7"

# Pass criteria, applied to residency measured after the memory was written.
BIND_MIN_PERCENT = 99.0
INTERLEAVE_TOLERANCE_POINTS = 5.0

TIERS = ("hbm", "ddr", "default")


def parse_node_list(spec):
    """ "0-3", "8,9", "0-1,8-9" -> sorted node IDs."""
    nodes = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        first, _, last = part.partition("-")
        nodes.update(range(int(first), int(last or first) + 1))
    return sorted(nodes)


def format_node_list(nodes):
    return ",".join(str(node) for node in sorted(nodes))


def add_allocator_args(parser):
    parser.add_argument("--lib", default=str(DEFAULT_LIB), help="path to libnuma_allocator.so")
    parser.add_argument("--ddr-nodes", default=DEFAULT_DDR_NODES, help="DDR node list (default: Rhea, %(default)s)")
    parser.add_argument("--hbm-nodes", default=DEFAULT_HBM_NODES, help="HBM node list (default: Rhea, %(default)s)")


def run_cmd(argv, timeout=60):
    """(returncode, stdout + stderr). A missing tool is a nonzero return code, not an exception."""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout + proc.stderr
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, f"<failed to run {argv[0]}: {exc!r}>"


def _read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


# NUMA syscalls

_SYSCALL_NUMBERS = {
    "x86_64": {"get_mempolicy": 239, "move_pages": 279},
    "aarch64": {"get_mempolicy": 236, "move_pages": 239},
}
MPOL_F_ADDR = 1 << 1
MPOL_F_MEMS_ALLOWED = 1 << 2
MPOL_MODE_FLAGS = (1 << 15) | (1 << 14) | (1 << 13)
MPOL_MODE_NAMES = {
    0: "default",
    1: "preferred",
    2: "bind",
    3: "interleave",
    4: "local",
    5: "preferred_many",
    6: "weighted_interleave",
}
MAX_NODES = 1024
_ULONG_BITS = 8 * ctypes.sizeof(ctypes.c_ulong)

_libc = None


def _syscall(name, *args):
    """syscall(2). Returns (return value, errno). Only architectures with a known number are supported."""
    global _libc
    numbers = _SYSCALL_NUMBERS.get(platform.machine())
    if numbers is None:
        raise OSError(f"no {name} syscall number known for {platform.machine()!r}")
    if _libc is None:
        _libc = ctypes.CDLL(None, use_errno=True)
        _libc.syscall.restype = ctypes.c_long
    ctypes.set_errno(0)
    rc = _libc.syscall(ctypes.c_long(numbers[name]), *args)
    return rc, ctypes.get_errno()


def _get_mempolicy(address, flags):
    mode = ctypes.c_int(-1)
    mask = (ctypes.c_ulong * (MAX_NODES // _ULONG_BITS))()
    rc, err = _syscall(
        "get_mempolicy",
        ctypes.byref(mode),
        mask,
        ctypes.c_ulong(MAX_NODES + 1),
        ctypes.c_void_p(address),
        ctypes.c_ulong(flags),
    )
    if rc != 0:
        raise OSError(err, f"get_mempolicy failed: {os.strerror(err)}")
    nodes = [w * _ULONG_BITS + b for w, word in enumerate(mask) for b in range(_ULONG_BITS) if word >> b & 1]
    return mode.value, nodes


def _policy_dict(mode, nodes):
    return {"mode": MPOL_MODE_NAMES.get(mode & ~MPOL_MODE_FLAGS, str(mode)), "nodes": nodes}


def mems_allowed():
    """Nodes this process may allocate from. The same call libnuma_allocator validates against."""
    return _get_mempolicy(None, MPOL_F_MEMS_ALLOWED)[1]


def address_mempolicy(address):
    """Policy of the mapping that holds address: {"mode": "bind", "nodes": [...]}."""
    return _policy_dict(*_get_mempolicy(address, MPOL_F_ADDR))


def query_page_nodes(addresses):
    """Node of each page, or a negative errno per page (-ENOENT: not resident).

    Returns (status list, None) or (None, error). nodes is NULL, which makes
    move_pages a pure query: no page can be migrated from here.
    """
    if not addresses:
        return [], None
    count = len(addresses)
    pages = (ctypes.c_void_p * count)(*addresses)
    status = (ctypes.c_int * count)(*([-1] * count))
    try:
        rc, err = _syscall(
            "move_pages",
            ctypes.c_long(0),
            ctypes.c_ulong(count),
            pages,
            ctypes.c_void_p(None),
            status,
            ctypes.c_long(0),
        )
    except OSError as exc:
        return None, str(exc)
    if rc != 0:
        return None, f"move_pages returned {rc}: {os.strerror(err)}"
    return list(status), None


def collect_environment():
    """What every run records: numactl -H, Mems_allowed and the settings that affect placement."""
    status = {}
    for line in (_read_text("/proc/self/status") or "").splitlines():
        key, _, value = line.partition(":")
        if key in ("Cpus_allowed_list", "Mems_allowed", "Mems_allowed_list"):
            status[key] = value.strip()
    env = {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "numactl_h": run_cmd(["numactl", "-H"])[1],
        "mems_allowed_list": status.get("Mems_allowed_list"),
        "mems_allowed_mask": status.get("Mems_allowed"),
        "cpus_allowed_list": status.get("Cpus_allowed_list"),
        "page_size": os.sysconf("SC_PAGESIZE"),
        "transparent_hugepage": _read_text("/sys/kernel/mm/transparent_hugepage/enabled"),
        "numa_balancing": _read_text("/proc/sys/kernel/numa_balancing"),
    }
    try:
        env["mems_allowed_from_get_mempolicy"] = mems_allowed()
        env["process_mempolicy"] = _policy_dict(*_get_mempolicy(None, 0))
    except OSError as exc:
        env["get_mempolicy_error"] = str(exc)
    return env


# Result recorder shared by the test scripts

VALID_STATUSES = ("PASS", "FAIL", "WARN", "INFO", "SKIP")


def machine_load():
    """`uptime` and the first 20 lines of `top -bn1`."""
    return {"uptime": run_cmd(["uptime"])[1].strip(), "top": "\n".join(run_cmd(["top", "-bn1"])[1].splitlines()[:20])}


class Campaign:
    def __init__(self, name, config=None):
        self.name = name
        self.config = config or {}
        self.started_at = time.time()
        self.environment = collect_environment()
        self.checks = []

    def check(self, name, status, **detail):
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status {status!r}")
        self.checks.append({"name": name, "status": status, **detail})
        summary = detail.get("summary", "")
        print(f"[{status:4s}] {name}" + (f": {summary}" if summary else ""), flush=True)

    def record_load(self, name):
        """Machine load before a benchmark, so that a noisy run can be identified later."""
        load = machine_load()
        self.check(f"load[{name}]", "INFO", summary=load["uptime"], **load)

    def require_memory(self, name, node_sets, planned_mb):
        """Check every target node before a benchmark allocates. False means: do not run it."""
        record, problems = check_node_memory(node_sets, planned_mb)
        summary = "; ".join(problems) or ", ".join(
            f"node {node}: {info['usable_mb']} MiB usable, {info['planned_mb']} MiB planned" for node, info in record.items()
        )
        self.check(f"memory[{name}]", "FAIL" if problems else "PASS", summary=summary, nodes=record)
        return not problems

    def counts(self):
        counts = dict.fromkeys(VALID_STATUSES, 0)
        for check in self.checks:
            counts[check["status"]] += 1
        return counts

    def save(self, path):
        result = {
            "name": self.name,
            "config": self.config,
            "started_at": self.started_at,
            "duration_s": time.time() - self.started_at,
            "environment": self.environment,
            "counts": self.counts(),
            "checks": self.checks,
        }
        Path(path).write_text(json.dumps(result, indent=2, default=str))
        print(f"[INFO] wrote {path}")

    def had_failures(self):
        return self.counts()["FAIL"] > 0


def guarded(campaign, name, fn):
    """Run fn(). An exception becomes a FAIL check, so one broken case does not stop the stage."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        campaign.check(name, "FAIL", summary=f"{type(exc).__name__}: {exc}")
        return None


# Topology, read from sysfs. Nothing here chooses NUMA nodes: it describes the
# machine and checks the nodes it is given.

SYSFS_NODE_ROOT = "/sys/devices/system/node"
SYSFS_CPU_ROOT = "/sys/devices/system/cpu"


def node_memory_mb(node_id):
    """Memory of one node in MiB, from its meminfo.

    "usable" is MemFree plus the page cache, which the kernel reclaims for a
    bound allocation. MemFree alone is far too low on a machine that has been
    up for a while.
    """
    text = _read_text(f"{SYSFS_NODE_ROOT}/node{node_id}/meminfo") or ""
    meminfo = {key: int(value) for key, value in re.findall(r"^Node \d+ (\S+):\s+(\d+)", text, re.M)}
    free = meminfo.get("MemFree", 0)
    usable = free + meminfo.get("Active(file)", 0) + meminfo.get("Inactive(file)", 0)
    return {"total_mb": meminfo.get("MemTotal", 0) // 1024, "free_mb": free // 1024, "usable_mb": usable // 1024}


def check_node_memory(node_sets, planned_mb):
    """Compare what a benchmark plans to allocate with the memory of every target node.

    node_sets and planned_mb map a tier ("ddr", "hbm") to its nodes and to MiB.
    Returns (per-node record, problems). Every node of a tier must have room
    for the whole amount. A bound allocation goes to the node nearest to the
    thread that first touches it, so it can all land on one node, and when the
    node set is full the kernel kills the process instead of returning an error.
    """
    planned = {}
    for tier, nodes in node_sets.items():
        for node in nodes:
            planned[node] = planned.get(node, 0) + planned_mb.get(tier, 0)
    record, problems = {}, []
    for node, need in sorted(planned.items()):
        record[node] = {"planned_mb": need, **node_memory_mb(node)}
        if record[node]["usable_mb"] < need:
            problems.append(
                f"node {node} has {record[node]['usable_mb']} MiB free or in page cache "
                f"(MemFree {record[node]['free_mb']} MiB), the benchmark plans up to {need} MiB"
            )
    return record, problems


def get_numa_topology():
    """Every NUMA node: CPUs, physical cores, sockets, distances and memory."""
    node_ids = sorted(int(re.search(r"node(\d+)$", path)[1]) for path in glob.glob(f"{SYSFS_NODE_ROOT}/node[0-9]*"))
    nodes = {}
    for node_id in node_ids:
        cpulist = _read_text(f"{SYSFS_NODE_ROOT}/node{node_id}/cpulist") or ""
        cores, sockets = set(), set()
        for cpu in parse_node_list(cpulist):
            base = f"{SYSFS_CPU_ROOT}/cpu{cpu}/topology"
            # The SMT siblings of a core are read, never assumed: "40,120" or "0-1".
            cores.add(tuple(parse_node_list(_read_text(f"{base}/thread_siblings_list") or str(cpu))))
            sockets.add(int(_read_text(f"{base}/physical_package_id") or -1))
        distances = (_read_text(f"{SYSFS_NODE_ROOT}/node{node_id}/distance") or "").split()
        nodes[node_id] = {
            "cpulist": cpulist,
            "has_cpus": bool(cpulist),
            "physical_cores": sorted(cores),
            "sockets": sorted(sockets),
            # One value per online node, in node order.
            "distances": dict(zip(node_ids, map(int, distances))) if len(distances) == len(node_ids) else {},
            **node_memory_mb(node_id),
        }
    # A node without CPUs belongs to the socket of the CPU nodes it is closest to.
    cpu_nodes = [node_id for node_id in node_ids if nodes[node_id]["has_cpus"]]
    for info in nodes.values():
        if not info["has_cpus"] and info["distances"] and cpu_nodes:
            nearest = min(info["distances"][node] for node in cpu_nodes)
            closest = [node for node in cpu_nodes if info["distances"][node] == nearest]
            info["sockets"] = sorted({socket for node in closest for socket in nodes[node]["sockets"]})
    return {"nodes": nodes}


def node_cpus(topology, node_ids):
    cpus = set()
    for node_id in node_ids:
        info = topology["nodes"].get(node_id)
        if info and info["cpulist"]:
            cpus.update(parse_node_list(info["cpulist"]))
    return sorted(cpus)


def sockets_of(topology, node_ids):
    return sorted({socket for node in node_ids for socket in topology["nodes"][node]["sockets"]})


def distance_range(topology, from_nodes, to_nodes):
    """NUMA distance between two node sets as text: "13", "21 to 23", or "unknown"."""
    values = [topology["nodes"][a]["distances"].get(b) for a in from_nodes for b in to_nodes]
    if not values or None in values:
        return "unknown"
    return str(min(values)) if min(values) == max(values) else f"{min(values)} to {max(values)}"


def format_cpu_list(cpus):
    """[40, 41, 42, 50] -> "40-42,50"."""
    ranges = []
    for cpu in sorted(cpus):
        if ranges and cpu == ranges[-1][1] + 1:
            ranges[-1][1] = cpu
        else:
            ranges.append([cpu, cpu])
    return ",".join(str(first) if first == last else f"{first}-{last}" for first, last in ranges)


def choose_pin_cpus(topology, ddr_nodes, threads):
    """{node: [cpu, ...]}: `threads` physical cores, spread evenly over the DDR nodes.

    One CPU per physical core, the lowest sibling that this process may use, so
    SMT siblings are never used. The nodes are filled round-robin with their
    lowest cores first: 20 threads on four nodes of ten cores take the first
    five cores of each node. Raises ValueError if there are not enough cores.
    """
    allowed = os.sched_getaffinity(0)
    available = {}
    for node in ddr_nodes:
        usable = ([cpu for cpu in core if cpu in allowed] for core in topology["nodes"][node]["physical_cores"])
        available[node] = [cpus[0] for cpus in usable if cpus]
    total = sum(len(cpus) for cpus in available.values())
    if threads < 1 or threads > total:
        nodes = format_node_list(ddr_nodes)
        raise ValueError(f"--threads {threads} does not fit: nodes {nodes} have {total} physical cores that this process may use")
    chosen = {node: [] for node in ddr_nodes}
    while sum(len(cpus) for cpus in chosen.values()) < threads:
        for node in ddr_nodes:
            if len(chosen[node]) < len(available[node]) and sum(len(cpus) for cpus in chosen.values()) < threads:
                chosen[node].append(available[node][len(chosen[node])])
    return chosen


def validate_node_selection(topology, ddr, hbm, allowed, min_free_mb=0, allow_remote=False):
    """(issues, notes, locality). An issue stops the run; a note is only unusual.

    locality is "local" when the HBM nodes are on the socket of the DDR nodes,
    whose cores run the benchmarks, and "remote" otherwise. A remote set is an
    issue unless allow_remote is set, which only the socket-locality control uses.
    """
    nodes = topology["nodes"]
    issues, notes = [], []
    for role, selected in (("ddr", ddr), ("hbm", hbm)):
        for node_id in selected:
            info = nodes.get(node_id)
            if info is None:
                issues.append({"node": node_id, "role": role, "problem": "node does not exist"})
            elif info["total_mb"] == 0:
                issues.append({"node": node_id, "role": role, "problem": "node has no memory"})
            elif allowed is not None and node_id not in allowed:
                issues.append({"node": node_id, "role": role, "problem": "node is outside Mems_allowed"})
            elif role == "ddr" and not info["has_cpus"]:
                issues.append({"node": node_id, "role": role, "problem": "node has no CPUs, and the DDR nodes supply the cores"})
            elif role == "hbm" and info["has_cpus"]:
                notes.append(f"HBM node {node_id} has CPUs; HBM in flat mode normally shows up as CPU-less nodes")
    if issues:
        return issues, notes, "unknown"

    _, problems = check_node_memory({"ddr": ddr, "hbm": hbm}, {"ddr": min_free_mb, "hbm": min_free_mb})
    issues += [{"node": "-", "role": "memory", "problem": problem} for problem in problems]

    ddr_sockets, hbm_sockets = sockets_of(topology, ddr), sockets_of(topology, hbm)
    locality = "local" if len(ddr_sockets) == 1 and hbm_sockets == ddr_sockets else "remote"
    if len(ddr_sockets) != 1:
        problem = f"the DDR nodes are on sockets {format_node_list(ddr_sockets)}; a run pins its threads to the cores of one socket"
        issues.append({"node": format_node_list(ddr), "role": "ddr", "problem": problem})
    elif locality == "remote" and not allow_remote:
        problem = (
            f"the HBM nodes are on socket {format_node_list(hbm_sockets)}, the DDR nodes and their cores on socket "
            f"{format_node_list(ddr_sockets)}. A run measures one socket. Only the socket-locality control uses a "
            "remote HBM set: add --allow-remote for it"
        )
        issues.append({"node": format_node_list(hbm), "role": "hbm", "problem": problem})
    elif locality == "remote":
        sockets = f"the HBM nodes are on socket {format_node_list(hbm_sockets)}, the cores on socket {format_node_list(ddr_sockets)}"
        notes.append(f"remote control run: {sockets}")
    elif allow_remote:
        notes.append("--allow-remote was given, but the HBM nodes are on the socket of the DDR nodes: this is not a remote control")
    if set(ddr) & set(hbm):
        notes.append("the DDR and HBM node sets overlap, so this run cannot tell the two tiers apart")
    return issues, notes, locality


# Threads. The OpenMP runtime reads its settings once, so they must be in the
# environment before torch is imported. run_campaign.sh exports the count and
# pins the process; a script started on its own uses what it finds.


def thread_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    help_text = "OpenMP and MKL threads (default: OMP_NUM_THREADS, else one per physical core this process is bound to)"
    parser.add_argument("--threads", type=int, default=None, help=help_text)
    return parser


def configure_threads():
    """Set OMP_NUM_THREADS and MKL_NUM_THREADS. Call before importing torch. Returns what was chosen."""
    args, _ = thread_args_parser().parse_known_args()
    affinity = sorted(os.sched_getaffinity(0))
    cores = set()
    for cpu in affinity:
        siblings = _read_text(f"{SYSFS_CPU_ROOT}/cpu{cpu}/topology/thread_siblings_list") or str(cpu)
        cores.add(tuple(parse_node_list(siblings)))
    exported = os.environ.get("OMP_NUM_THREADS")
    if args.threads:
        threads, source = args.threads, "--threads"
        if exported is not None and exported.strip() != str(threads):
            raise SystemExit(f"ERROR: OMP_NUM_THREADS={exported} is exported but --threads {threads} was requested")
    elif exported is not None:
        threads, source = int(exported), "OMP_NUM_THREADS from the environment"
    else:
        threads, source = len(cores), "one per physical core in the CPU affinity"
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")
    return {
        "threads": threads,
        "threads_source": source,
        "cpu_affinity": format_cpu_list(affinity),
        "cpus_in_affinity": len(affinity),
        "physical_cores_in_affinity": len(cores),
        "omp_proc_bind": os.environ["OMP_PROC_BIND"],
        "omp_places": os.environ["OMP_PLACES"],
    }


# Allocators and what they report


def open_allocators(lib, ddr_nodes, hbm_nodes, mode="bind", prefault=False):
    """One NumaAllocator per tier. "default" maps to None: the process-wide CPU allocator."""
    from pytorch_numa_mem import NumaAllocator

    return {
        "hbm": NumaAllocator(lib, hbm_nodes, mode=mode, prefault=prefault),
        "ddr": NumaAllocator(lib, ddr_nodes, mode=mode, prefault=prefault),
        "default": None,
    }


COUNTER_FIELDS = ("create_calls", "destroy_calls", "alloc_calls", "free_calls", "live_allocations", "live_bytes")


def counters_snapshot(allocator):
    return {field: getattr(allocator.counters, field) for field in COUNTER_FIELDS}


def allocations_under_scope(fn, allocator):
    """How many tensors fn() allocates on this thread: run it once in a use_allocator scope and count."""
    from pytorch_numa_mem import use_allocator

    before = allocator.counters.alloc_calls
    with use_allocator(allocator):
        fn()
    return allocator.counters.alloc_calls - before


def tensor_record(tensor, allocators):
    """What the allocators say about a tensor. Placement is checked against the OS, not against this."""
    owners = [name for name, allocator in allocators.items() if allocator is not None and allocator.owns(tensor)]
    return {
        "data_ptr": hex(tensor.data_ptr()),
        "nbytes": tensor.nbytes,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "owned_by": owners[0] if owners else "default",
    }


# Physical placement

MAX_SAMPLE_PAGES = 4096
SAMPLE_BLOCK_PAGES = 64


def _numa_maps_entry(address):
    """Policy and per-node page counts of the mapping that holds address."""
    vma = None
    with open("/proc/self/maps") as maps:
        for line in maps:
            start, _, end = line.split()[0].partition("-")
            if int(start, 16) <= address < int(end, 16):
                vma = (int(start, 16), int(end, 16))
                break
    if vma is None:
        return {"found": False, "reason": "no mapping holds the address"}
    key = format(vma[0], "x")
    with open("/proc/self/numa_maps") as numa_maps:
        fields = next((line.split() for line in numa_maps if line.split(None, 1)[0] == key), None)
    if fields is None:
        return {"found": False, "reason": "the mapping has no numa_maps entry"}
    nodes_pages = {m[1]: int(m[2]) for m in (re.fullmatch(r"N(\d+)=(\d+)", field) for field in fields[2:]) if m}
    return {
        "found": bool(nodes_pages),
        "reason": None if nodes_pages else "numa_maps lists no resident pages",
        "placement_method": "numa_maps",
        "vma_start": hex(vma[0]),
        "vma_bytes": vma[1] - vma[0],
        "policy": fields[1],
        "nodes_pages": nodes_pages,
        "total_pages": sum(nodes_pages.values()),
    }


def _sample_page_offsets(n_pages):
    """Page indices to query: all of them, or contiguous blocks at seeded random offsets.

    Evenly spaced pages or blocks can alias with MPOL_INTERLEAVE, which places
    pages (or huge pages) round-robin: a stride that is a multiple of the node
    count reports every sampled page on one node. One random block per stratum
    covers the whole tensor without a fixed stride.
    """
    if n_pages <= MAX_SAMPLE_PAGES:
        return list(range(n_pages)), "all_pages"
    n_blocks = MAX_SAMPLE_PAGES // SAMPLE_BLOCK_PAGES
    rng = random.Random(n_pages)
    offsets = []
    for block in range(n_blocks):
        lo = block * n_pages // n_blocks
        hi = (block + 1) * n_pages // n_blocks - SAMPLE_BLOCK_PAGES
        start = rng.randint(lo, max(lo, hi))
        offsets.extend(range(start, start + SAMPLE_BLOCK_PAGES))
    return offsets, f"{n_blocks}_blocks_of_{SAMPLE_BLOCK_PAGES}_contiguous_pages_at_seeded_random_offsets"


def _sampled_pages(tensor):
    """Node of sampled pages of the tensor's own byte range."""
    address, nbytes = tensor.data_ptr(), tensor.nbytes
    result = {"placement_method": "move_pages", "found": False}
    if nbytes == 0:
        return {**result, "reason": "tensor has zero bytes"}
    page_size = os.sysconf("SC_PAGESIZE")
    first_page = address // page_size
    n_pages = (address + nbytes - 1) // page_size - first_page + 1
    offsets, strategy = _sample_page_offsets(n_pages)
    result.update(pages_in_tensor=n_pages, pages_sampled=len(offsets), sampling_strategy=strategy)

    status, error = query_page_nodes([(first_page + offset) * page_size for offset in offsets])
    if status is None:
        return {**result, "reason": error}
    nodes_pages, errno_counts = {}, {}
    for value in status:
        counts = nodes_pages if value >= 0 else errno_counts
        counts[str(value)] = counts.get(str(value), 0) + 1
    total = sum(nodes_pages.values())
    result.update(
        nodes_pages=nodes_pages,
        total_pages=total,
        pages_not_resident=len(offsets) - total,
        page_errno_counts=errno_counts,
    )
    if total == 0:
        return {**result, "reason": f"no sampled page is resident (per-page errno counts: {errno_counts})"}
    return {**result, "found": True}


def get_tensor_node_distribution(tensor):
    """Where the pages of a tensor are. Read-only: write to the tensor first.

    The sampled pages of the tensor itself decide. The numa_maps entry of the
    enclosing mapping is attached as "vma" and used alone only when move_pages
    cannot run. "placement_method" names the source, or is "unavailable" with a
    reason: placement is never made up.
    """
    address = tensor.data_ptr()
    pages = _sampled_pages(tensor)
    vma = _numa_maps_entry(address)
    if pages["found"]:
        result = pages
    elif vma["found"]:
        result = {**vma, "placement_method_reason": f"move_pages: {pages['reason']}"}
    else:
        result = {
            **pages,
            "placement_method": "unavailable",
            "reason": f"move_pages: {pages['reason']}; numa_maps: {vma['reason']}",
        }
    result["address"] = hex(address)
    result["vma"] = vma
    try:
        result["vma_policy"] = address_mempolicy(address)
    except OSError as exc:
        result["vma_policy"] = {"error": str(exc)}
    return result


def touch_tensor(tensor):
    """Write every element, so that every page is resident and placed."""
    return tensor.fill_(1)


def percentage_on_nodes(distribution, nodes):
    """Percentage of the resolved pages that are on one of nodes, or None without placement data."""
    if not distribution.get("found") or not distribution.get("total_pages"):
        return None
    wanted = {str(node) for node in nodes}
    on_nodes = sum(count for node, count in distribution["nodes_pages"].items() if node in wanted)
    return 100.0 * on_nodes / distribution["total_pages"]


def evaluate_placement(distribution, nodes, mode="bind", tolerance=INTERLEAVE_TOLERANCE_POINTS):
    """Apply the pass criteria to a distribution measured after the tensor was written.

    Every mode: at least BIND_MIN_PERCENT of the sampled pages on the node set.
    interleave: also every node within `tolerance` percentage points of an even
    share. The spread uses the numa_maps counts when there are any, because they
    cover every page of the mapping; all of a mapping has one policy.
    """
    if not distribution.get("found"):
        reason = distribution.get("reason")
        return {"status": "FAIL", "summary": f"could not determine physical placement: {reason}"}
    percent = percentage_on_nodes(distribution, nodes)
    label = format_node_list(nodes)
    result = {
        "percent_on_expected_nodes": percent,
        "criterion": f">= {BIND_MIN_PERCENT}% of sampled pages on nodes {label}",
    }
    ok = percent >= BIND_MIN_PERCENT
    summary = f"{percent:.2f}% of {distribution['total_pages']} pages on nodes {label} ({distribution['placement_method']})"
    if mode == "interleave":
        source = distribution["vma"] if distribution["vma"]["found"] else distribution
        on_nodes = {node: source["nodes_pages"].get(str(node), 0) for node in nodes}
        total = sum(on_nodes.values()) or 1
        shares = {node: 100.0 * count / total for node, count in on_nodes.items()}
        even = 100.0 / len(nodes)
        deviation = max(abs(share - even) for share in shares.values())
        ok = ok and deviation <= tolerance
        result.update(
            per_node_percent=shares,
            spread_source=source["placement_method"],
            max_deviation_points=deviation,
            criterion=f"{result['criterion']}, and every node within {tolerance} points of {even:.1f}%",
        )
        summary += "; spread " + " ".join(f"N{node}={share:.1f}%" for node, share in shares.items())
    return {**result, "status": "PASS" if ok else "FAIL", "summary": summary}


# Provenance. The compiler that built PyTorch, the compiler on PATH and the
# checkout are different things and are reported separately. Unknowns stay
# UNKNOWN.


def detect_build_compiler(pytorch_root, torch_config):
    """(compiler, where it was found) for the imported PyTorch."""
    if torch_config:
        name = re.search(r"^\s*-\s*(GCC|Clang|clang)\s+(\S+)\s*$", torch_config, re.M)
        path = re.search(r"CXX_COMPILER=([^\s,]+)", torch_config)
        parts = ([f"{name[1]} {name[2]}"] if name else []) + ([f"CXX_COMPILER={path[1]}"] if path else [])
        if parts:
            return " ".join(parts), "torch.__config__.show()"
    if pytorch_root:
        cache = _read_text(os.path.join(pytorch_root, "build", "CMakeCache.txt")) or ""
        compiler = re.search(r"^CMAKE_CXX_COMPILER:[^=]*=(.+)$", cache, re.M)
        if compiler:
            return compiler[1].strip(), "build/CMakeCache.txt"
    return "UNKNOWN", "none"


def onednn_primitives(fn):
    """Run fn() with oneDNN verbose output on. Returns the primitives it executed, as "primitive:implementation".

    None means that this build has no oneDNN. oneDNN prints from C, so file
    descriptor 1 points at a temporary file during the call.
    """
    import torch

    if not torch.backends.mkldnn.is_available():
        fn()
        return None
    libc = ctypes.CDLL(None)
    sys.stdout.flush()
    saved = os.dup(1)
    with tempfile.TemporaryFile() as capture:
        os.dup2(capture.fileno(), 1)
        try:
            with torch.backends.mkldnn.verbose(torch.backends.mkldnn.VERBOSE_ON):
                fn()
        finally:
            libc.fflush(None)
            os.dup2(saved, 1)
            os.close(saved)
        capture.seek(0)
        lines = capture.read().decode(errors="replace").splitlines()
    executed = (line.split(",") for line in lines if line.startswith("onednn_verbose") and ",exec," in line)
    return sorted({f"{fields[5]}:{fields[6]}" for fields in executed if len(fields) > 6})


def linear_path(dtype, in_features=5120, out_features=20480, rows=(1, 8)):
    """{rows: text}: the kernel F.linear uses for dtype, and the time of one call.

    The default shape is OPT-13B ffn_fc1. One row is the decode case; a few
    rows stand for prefill, which can take a different kernel.
    """
    import torch
    import torch.nn.functional as F

    weight = torch.randn(out_features, in_features).to(dtype)
    bias = torch.randn(out_features).to(dtype)
    result = {}
    for count in rows:
        x = torch.randn(count, in_features).to(dtype)
        F.linear(x, weight, bias)
        start = time.perf_counter()
        primitives = onednn_primitives(lambda: F.linear(x, weight, bias))
        elapsed_ms = (time.perf_counter() - start) * 1e3
        if primitives is None:
            path = "this build has no oneDNN, so PyTorch's own addmm kernel"
        elif not primitives:
            path = "no oneDNN primitive ran, so PyTorch's own addmm kernel (BLAS for float32)"
        else:
            path = "oneDNN " + " ".join(primitives)
        result[count] = f"{path}; one call took {elapsed_ms:.2f} ms with verbose output on"
    return result


CPU_FLAGS_OF_INTEREST = (
    "avx2 avx512f avx512_bf16 avx512_fp16 avx512_vnni avx_vnni avx_ne_convert amx_tile amx_bf16 amx_int8 "
    "sve sve2 svebf16 bf16 i8mm asimdhp"
).split()


def collect_build_provenance(pytorch_root=None, lib=None):
    """(info, raw torch config). A failed import is reported under "import_error", not raised."""
    info = {"python_executable": sys.executable, "python_version": platform.python_version()}
    # Before torch is imported, as in the benchmarks: same OpenMP settings for the F.linear record below.
    threads = configure_threads()
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        info["import_error"] = repr(exc)
        return info, None
    info["torch_version"] = torch.__version__
    info["torch_file"] = torch.__file__
    info["torch_git_version"] = getattr(torch.version, "git_version", None) or "UNKNOWN"
    raw_config = torch.__config__.show()
    info["build_compiler"], info["build_compiler_source"] = detect_build_compiler(pytorch_root, raw_config)
    raw_config = f"torch.__config__.show():\n{raw_config}\n\ntorch.__config__.parallel_info():\n{torch.__config__.parallel_info()}"
    info["mkldnn_available"] = torch.backends.mkldnn.is_available()
    info["cpu_capability"] = torch.backends.cpu.get_cpu_capability()
    cpu_flags = re.search(r"^(?:flags|Features)\s*:(.*)$", _read_text("/proc/cpuinfo") or "", re.M)
    present = set(cpu_flags[1].split()) if cpu_flags else set()
    info["cpu_flags"] = " ".join(flag for flag in CPU_FLAGS_OF_INTEREST if flag in present) or "none of the listed ones"
    info["omp_num_threads"] = os.environ["OMP_NUM_THREADS"]
    info["mkl_num_threads"] = os.environ["MKL_NUM_THREADS"]
    info["torch_num_threads"] = torch.get_num_threads()
    info["cpu_affinity"] = threads["cpu_affinity"]
    # The matmul path in use: BF16 is the benchmark default, float32 the fallback.
    for name in ("bfloat16", "float32"):
        for rows, text in linear_path(getattr(torch, name)).items():
            info[f"{name}_linear_rows{rows}"] = text
    rc, out = run_cmd(["cc", "--version"])
    info["shell_compiler"] = out.splitlines()[0].strip() if rc == 0 and out.strip() else "(no cc on PATH)"
    try:
        import pytorch_numa_mem

        info["pytorch_numa_mem_file"] = pytorch_numa_mem.__file__
    except Exception as exc:  # noqa: BLE001
        info["import_error"] = f"pytorch_numa_mem: {exc!r}"
    if lib:
        info["libnuma_allocator"] = os.path.realpath(lib)
        try:
            info["libnuma_allocator_abi_version"] = ctypes.CDLL(lib).numa_allocator_abi_version()
        except (OSError, AttributeError) as exc:
            info["import_error"] = f"libnuma_allocator: {exc!r}"
    return info, raw_config


def _git(root, *args):
    """Output of a read-only git command, or None. GIT_OPTIONAL_LOCKS=0 keeps `git status` from touching the index."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def repo_info(prefix, root):
    root = os.path.realpath(root)
    info = {f"{prefix}_root": root}
    top = _git(root, "rev-parse", "--show-toplevel")
    if top is None or os.path.realpath(top) != root:
        info.update({f"{prefix}_branch": "none", f"{prefix}_sha": "not a git repository", f"{prefix}_dirty": "unknown"})
        return info
    info[f"{prefix}_branch"] = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    info[f"{prefix}_sha"] = _git(root, "rev-parse", "HEAD")
    info[f"{prefix}_dirty"] = "yes" if _git(root, "status", "--porcelain") else "no"
    return info


SOURCE_SUFFIXES = (".c", ".h", ".cpp", ".py", ".sh", ".toml", ".md")


def ext_source_hashes():
    """{relative path: sha256} of the source files of this tree, without build output and results."""
    hashes = {}
    for top in ("libnuma_allocator", "pytorch_numa_mem", "validation"):
        for directory, subdirs, files in os.walk(EXT_ROOT / top):
            subdirs[:] = [d for d in subdirs if not d.startswith((".", "build", "__pycache__")) and d != "results"]
            for name in files:
                if name.endswith(SOURCE_SUFFIXES) or name == "Makefile":
                    path = Path(directory) / name
                    hashes[str(path.relative_to(EXT_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(hashes.items()))


def _main_source(args):
    """Print "key: value" lines for the two source trees. Works without git and without a git checkout."""
    pytorch_root = args.pytorch_root
    if pytorch_root is None:
        import torch

        pytorch_root = os.path.dirname(os.path.dirname(os.path.realpath(torch.__file__)))
    info = repo_info("pytorch", pytorch_root)
    info["pytorch_base_sha"] = POC_BASE_SHA
    changed = _git(pytorch_root, "diff", "--name-only", POC_BASE_SHA, "HEAD")
    info["pytorch_changed_files_vs_base"] = "unknown" if changed is None else " ".join(changed.split()) or "none"
    info.update(repo_info("ext", EXT_ROOT))
    hashes = ext_source_hashes()
    listing = "".join(f"{digest}  {path}\n" for path, digest in hashes.items())
    info["ext_content_sha256"] = hashlib.sha256(listing.encode()).hexdigest()
    for key, value in info.items():
        print(f"{key}: {value}")
    print()
    print("pytorch git status --short:")
    print(_git(pytorch_root, "status", "--short") or "")
    if args.files_out:
        Path(args.files_out).write_text(listing)
    return 0


def _main_provenance(args):
    """Print "key: value" lines. Exit status 1 if torch, the extension or the library cannot be loaded."""
    info, raw_config = collect_build_provenance(args.pytorch_root, args.lib)
    for key, value in info.items():
        print(f"{key}: {value}")
    if raw_config and args.config_out:
        Path(args.config_out).write_text(raw_config)
    return 1 if "import_error" in info else 0


def _main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["topology", "provenance", "source"], default="topology")
    parser.add_argument("--pytorch-root", default=None, help="PyTorch checkout (default: where the imported torch lives)")
    parser.add_argument("--config-out", default=None, help="write the torch configuration here (provenance)")
    parser.add_argument("--files-out", default=None, help="write the sha256 of every source file of this tree here (source)")
    parser.add_argument("--threads", type=int, default=None, help="choose this many physical cores on the DDR nodes (topology)")
    parser.add_argument("--pin-out", default=None, help="write the chosen CPU list here, for numactl --physcpubind (topology)")
    parser.add_argument("--min-free-mb", type=int, default=0, help="memory that every selected node must have, in MiB (topology)")
    parser.add_argument("--allow-remote", action="store_true", help="accept HBM nodes on another socket than the DDR nodes (topology)")
    parser.add_argument("--out", default=None, help="write the topology JSON here")
    add_allocator_args(parser)
    args = parser.parse_args()
    if args.mode == "provenance":
        return _main_provenance(args)
    if args.mode == "source":
        return _main_source(args)

    ddr = parse_node_list(args.ddr_nodes)
    hbm = parse_node_list(args.hbm_nodes)
    topology = get_numa_topology()
    try:
        allowed = mems_allowed()
    except OSError:
        allowed = None
    issues, notes, locality = validate_node_selection(topology, ddr, hbm, allowed, args.min_free_mb, args.allow_remote)
    pinned = {}
    if args.threads is not None and not issues:
        try:
            pinned = choose_pin_cpus(topology, ddr, args.threads)
        except ValueError as exc:
            issues.append({"node": format_node_list(ddr), "role": "ddr", "problem": str(exc)})

    for node_id, info in topology["nodes"].items():
        cpus = f"cpus={info['cpulist'] or '-'} ({len(info['physical_cores'])} cores)" if info["has_cpus"] else "no CPUs"
        memory = f"{info['total_mb'] / 1024:.1f} GiB, {info['usable_mb'] / 1024:.1f} GiB free or in page cache"
        print(f"node{node_id}: socket {','.join(map(str, info['sockets'])) or '?'}, {cpus}, {memory}")
    print(f"DDR nodes:    {format_node_list(ddr)}")
    print(f"HBM nodes:    {format_node_list(hbm)} ({locality} to the DDR nodes)")
    if not issues:
        print(f"distances:    DDR to HBM {distance_range(topology, ddr, hbm)}; DDR to DDR {distance_range(topology, ddr, ddr)}")
    print(f"Mems_allowed: {format_node_list(allowed) if allowed is not None else 'unknown'}")
    if pinned:
        per_node = "; ".join(f"node {node}: {format_cpu_list(cpus)}" for node, cpus in pinned.items())
        print(f"pinned CPUs:  {args.threads} physical cores ({per_node})")
    for issue in issues:
        print(f"[ERROR] node {issue['node']} ({issue['role']}): {issue['problem']}")
    for note in notes:
        print(f"[NOTE] {note}")

    result = {
        "selected_ddr_nodes": ddr,
        "selected_hbm_nodes": hbm,
        "locality": locality,
        "allow_remote": args.allow_remote,
        "mems_allowed": allowed,
        "threads": args.threads,
        "pinned_cpus_per_node": pinned,
        "detected": topology,
        "selection_issues": issues,
        "selection_notes": notes,
        "selection_valid": not issues,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")
    if args.pin_out and pinned:
        Path(args.pin_out).write_text(format_cpu_list(cpu for cpus in pinned.values() for cpu in cpus) + "\n")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(_main())
