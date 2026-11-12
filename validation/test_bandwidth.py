"""Memory bandwidth sanity comparison: HBM nodes, DDR nodes, default allocator.

Not a formal benchmark. It shows whether memory on the HBM nodes behaves
differently from memory on the DDR nodes for three streaming operations.

Three columns. "hbm" and "ddr" go through the same NumaAllocator code, bound to
different nodes, so their ratio is a memory effect. "default" is the
process-wide CPU allocator.

Two variants of every operation:

    preallocated  every tensor exists before the timed loop (factory, out=).
                  No allocation is timed. This is the memory effect.
    scoped        the operation allocates its result inside
                  `with use_allocator(...)` and the result is dropped, so
                  mmap + mbind + page faults + munmap are timed. This is the
                  end-to-end cost. The default column allocates without a scope.

bytes moved per call: fill 1x, copy 2x, add 3x the tensor size.

Before each column it records the machine load and checks that every target
node has room for the four tensors the column holds at once.

Usage:
    python3 test_bandwidth.py --out-dir results --ddr-nodes 0-3 --hbm-nodes 4-7 --threads 20
"""

import argparse
import os
import statistics
import sys
import time

from hbm_test_utils import (
    TIERS,
    Campaign,
    add_allocator_args,
    allocations_under_scope,
    configure_threads,
    evaluate_placement,
    get_tensor_node_distribution,
    guarded,
    open_allocators,
    parse_node_list,
    thread_args_parser,
    touch_tensor,
)

THREAD_INFO = configure_threads()

VARIANTS = ("preallocated", "scoped")
OPS = {"fill": 1, "copy": 2, "add": 3}


def time_op(fn, warmups, repetitions):
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return samples


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[thread_args_parser()]
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tensor-size-mb", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    add_allocator_args(parser)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "bandwidth.json")

    c = Campaign("bandwidth", vars(args))
    try:
        import torch
        from pytorch_numa_mem import use_allocator
    except Exception as exc:  # noqa: BLE001
        c.check("import_torch", "FAIL", summary=repr(exc))
        c.save(out_path)
        return 1
    c.check("import_torch", "PASS", summary=torch.__file__)
    c.check(
        "threading_config",
        "INFO",
        summary=f"{THREAD_INFO['threads']} threads ({THREAD_INFO['threads_source']}) on CPUs {THREAD_INFO['cpu_affinity']}, torch.get_num_threads()={torch.get_num_threads()}",
        torch_num_threads=torch.get_num_threads(),
        **THREAD_INFO,
    )

    node_sets = {"ddr": parse_node_list(args.ddr_nodes), "hbm": parse_node_list(args.hbm_nodes)}
    allocators = open_allocators(args.lib, node_sets["ddr"], node_sets["hbm"])
    numel = args.tensor_size_mb * 2**20 // 4
    nbytes = numel * 4
    gbps = {}

    def run_tier(tier):
        alloc = allocators[tier]
        c.record_load(tier)
        # The default allocator follows the process policy, which run_campaign.sh sets to the DDR nodes.
        target = tier if alloc else "ddr"
        if not c.require_memory(tier, {target: node_sets[target]}, {target: 4 * args.tensor_size_mb}):
            return
        a, b, out = (touch_tensor(alloc.empty(numel) if alloc else torch.empty(numel)) for _ in range(3))

        distribution = get_tensor_node_distribution(out)
        verdict = evaluate_placement(distribution, node_sets.get(tier, node_sets["ddr"]))
        if alloc is None:
            verdict["status"] = "INFO"
        c.check(f"setup[{tier}]", distribution=distribution, **verdict)

        def in_scope(fn):
            if alloc is None:
                return fn

            def scoped():
                with use_allocator(alloc):
                    fn()

            return scoped

        cases = {
            ("fill", "preallocated"): lambda: a.fill_(3.0),
            ("copy", "preallocated"): lambda: out.copy_(a),
            ("add", "preallocated"): lambda: torch.add(a, b, out=out),
            ("fill", "scoped"): in_scope(lambda: torch.full((numel,), 3.0)),
            ("copy", "scoped"): in_scope(lambda: a.clone()),
            ("add", "scoped"): in_scope(lambda: a + b),
        }
        for (op, variant), fn in cases.items():
            if variant == "preallocated":
                # Any allocator works as the probe: it only counts.
                allocs_per_call = allocations_under_scope(fn, alloc or allocators["ddr"])
                samples = time_op(fn, args.warmups, args.repetitions)
            elif alloc is None:
                allocs_per_call = None
                samples = time_op(fn, args.warmups, args.repetitions)
            else:
                for _ in range(args.warmups):
                    fn()
                before = alloc.counters.alloc_calls
                samples = time_op(fn, 0, args.repetitions)
                allocs_per_call = (alloc.counters.alloc_calls - before) / args.repetitions
            median = statistics.median(samples)
            rate = nbytes * OPS[op] / median / 1e9
            gbps[op, variant, tier] = rate
            unexpected_allocation = variant == "preallocated" and allocs_per_call != 0
            c.check(
                f"bandwidth[{op},{variant},{tier}]",
                "WARN" if unexpected_allocation else "PASS",
                summary=f"median {median * 1e3:.3f} ms, {rate:.1f} GB/s, alloc_calls per call: {allocs_per_call}",
                min_s=min(samples),
                median_s=median,
                max_s=max(samples),
                n=len(samples),
                approx_gbps=rate,
                bytes_moved_per_call=nbytes * OPS[op],
                alloc_calls_per_call=allocs_per_call,
            )

    for tier in TIERS:
        guarded(c, f"bandwidth[{tier}]", lambda: run_tier(tier))

    for variant in VARIANTS:
        for op in OPS:
            rates = {tier: gbps.get((op, variant, tier)) for tier in TIERS}
            if None in rates.values():
                continue
            c.check(
                f"ratio[{op},{variant}]",
                "INFO",
                summary=f"HBM/DDR {rates['hbm'] / rates['ddr']:.2f}x, HBM/default {rates['hbm'] / rates['default']:.2f}x",
                gbps=rates,
                hbm_over_ddr=rates["hbm"] / rates["ddr"],
                hbm_over_default=rates["hbm"] / rates["default"],
                ddr_over_default=rates["ddr"] / rates["default"],
            )

    c.save(out_path)
    return 1 if c.had_failures() else 0


if __name__ == "__main__":
    sys.exit(main())
