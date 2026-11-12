"""Socket locality of the HBM node set, seen from the cores the run is pinned to.

run_campaign.sh pins every stage to physical cores of the DDR nodes. This stage
records how far the HBM nodes are from those cores (sockets and NUMA
distances), checks that an HBM tensor lands on its node set whatever the CPU
binding is, and times fill and copy on HBM and on DDR in the same process.

A normal run has the HBM nodes on the socket of the DDR nodes ("local").
`--allow-remote` marks the control run, where the HBM nodes are on another
socket ("remote"): the same commands with the other socket's HBM nodes. The
HBM/DDR ratio of the control, next to the ratio of the normal run, is the cost
of remote HBM. DDR is local in both runs, so each run carries its own baseline.

Usage:
    python3 test_socket_locality.py --out-dir results --ddr-nodes 4-7 --hbm-nodes 12-15
    python3 test_socket_locality.py --out-dir results --ddr-nodes 4-7 --hbm-nodes 8-11 --allow-remote
"""

import argparse
import os
import statistics
import sys
import time

from hbm_test_utils import (
    Campaign,
    add_allocator_args,
    configure_threads,
    counters_snapshot,
    distance_range,
    evaluate_placement,
    format_node_list,
    get_numa_topology,
    get_tensor_node_distribution,
    guarded,
    node_cpus,
    open_allocators,
    parse_node_list,
    sockets_of,
    tensor_record,
    thread_args_parser,
    touch_tensor,
)

THREAD_INFO = configure_threads()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[thread_args_parser()]
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--allow-remote", action="store_true", help="this is the control run: the HBM nodes are expected on another socket")
    parser.add_argument("--tensor-size-mb", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=5)
    add_allocator_args(parser)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "socket_locality.json")

    c = Campaign("socket_locality", vars(args))
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        c.check("import_torch", "FAIL", summary=repr(exc))
        c.save(out_path)
        return 1
    c.check("import_torch", "PASS", summary=torch.__file__)

    node_sets = {"ddr": parse_node_list(args.ddr_nodes), "hbm": parse_node_list(args.hbm_nodes)}
    topology = get_numa_topology()

    # The affinity from before torch was imported: OpenMP then binds the main thread to one core.
    affinity = parse_node_list(THREAD_INFO["cpu_affinity"])
    on_ddr_nodes = set(affinity) <= set(node_cpus(topology, node_sets["ddr"]))
    c.check(
        "process_cpu_affinity",
        "PASS" if on_ddr_nodes else "WARN",
        summary=f"CPUs {THREAD_INFO['cpu_affinity']}, {'all' if on_ddr_nodes else 'not all'} on the DDR nodes {args.ddr_nodes}",
        **THREAD_INFO,
    )

    cpu_sockets, hbm_sockets = sockets_of(topology, node_sets["ddr"]), sockets_of(topology, node_sets["hbm"])
    locality = "local" if len(cpu_sockets) == 1 and hbm_sockets == cpu_sockets else "remote"
    expected = "remote" if args.allow_remote else "local"
    c.check(
        "locality",
        "PASS" if locality == expected else "FAIL",
        summary=(
            f"the HBM nodes are {locality}: socket {format_node_list(hbm_sockets)}, cores on socket "
            f"{format_node_list(cpu_sockets)}; expected {expected}. NUMA distance DDR to HBM "
            f"{distance_range(topology, node_sets['ddr'], node_sets['hbm'])}; DDR to DDR "
            f"{distance_range(topology, node_sets['ddr'], node_sets['ddr'])}"
        ),
        locality=locality,
        expected=expected,
        cpu_sockets=cpu_sockets,
        hbm_sockets=hbm_sockets,
        distances={node: topology["nodes"][node]["distances"] for node in node_sets["ddr"]},
    )

    c.record_load("fill and copy")
    megabytes = args.tensor_size_mb
    if not c.require_memory("fill and copy", node_sets, {"ddr": 2 * megabytes, "hbm": 2 * megabytes}):
        c.save(out_path)
        return 1

    allocators = open_allocators(args.lib, node_sets["ddr"], node_sets["hbm"])
    numel = megabytes * 2**20 // 4
    gbps = {}

    def run_tier(tier):
        alloc = allocators[tier]
        a, out = touch_tensor(alloc.empty(numel)), touch_tensor(alloc.empty(numel))
        distribution = get_tensor_node_distribution(out)
        c.check(
            f"placement[{tier}]",
            metadata=tensor_record(out, allocators),
            counters=counters_snapshot(alloc),
            distribution=distribution,
            expected_nodes=node_sets[tier],
            mode="bind",
            **evaluate_placement(distribution, node_sets[tier]),
        )
        for op, fn, factor in (("fill", lambda: out.fill_(3.0), 1), ("copy", lambda: out.copy_(a), 2)):
            fn()
            samples = []
            for _ in range(args.repetitions):
                start = time.perf_counter()
                fn()
                samples.append(time.perf_counter() - start)
            gbps[op, tier] = numel * 4 * factor / statistics.median(samples) / 1e9

    for tier in ("hbm", "ddr"):
        guarded(c, f"placement[{tier}]", lambda: run_tier(tier))

    for op in ("fill", "copy"):
        if (op, "hbm") in gbps and (op, "ddr") in gbps:
            ratio = gbps[op, "hbm"] / gbps[op, "ddr"]
            c.check(
                f"ratio[{op}]",
                "INFO",
                summary=f"{locality} HBM {gbps[op, 'hbm']:.1f} GB/s, local DDR {gbps[op, 'ddr']:.1f} GB/s, HBM/DDR {ratio:.2f}x",
                locality=locality,
                hbm_gbps=gbps[op, "hbm"],
                ddr_gbps=gbps[op, "ddr"],
                hbm_over_ddr=ratio,
            )

    c.save(out_path)
    return 1 if c.had_failures() else 0


if __name__ == "__main__":
    sys.exit(main())
