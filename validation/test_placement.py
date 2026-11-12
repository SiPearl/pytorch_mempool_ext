"""Physical NUMA placement of tensors allocated through pytorch_numa_mem.

Tensors come from the factories (alloc.empty) and from eager ops inside
`with use_allocator(alloc)`. Each one is written, and then the kernel is asked
where its pages are. The allocator counters are recorded next to each result,
but the verdict comes only from the measured residency:

    bind        >= 99% of the sampled pages on the node set
    interleave  the same, and every node within --interleave-tolerance
                percentage points of an even share

Usage:
    python3 test_placement.py --out-dir results --ddr-nodes 0-3 --hbm-nodes 4-7
"""

import argparse
import os
import sys

from hbm_test_utils import (
    INTERLEAVE_TOLERANCE_POINTS,
    Campaign,
    add_allocator_args,
    counters_snapshot,
    evaluate_placement,
    get_tensor_node_distribution,
    guarded,
    open_allocators,
    parse_node_list,
    run_cmd,
    tensor_record,
    touch_tensor,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tensor-size-mb", type=int, default=512)
    parser.add_argument("--interleave-tolerance", type=float, default=INTERLEAVE_TOLERANCE_POINTS)
    add_allocator_args(parser)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "placement.json")

    c = Campaign("placement", vars(args))
    try:
        import torch
        from pytorch_numa_mem import use_allocator
    except Exception as exc:  # noqa: BLE001
        c.check("import_torch", "FAIL", summary=repr(exc))
        c.save(out_path)
        return 1
    c.check("import_torch", "PASS", summary=torch.__file__)

    node_sets = {"ddr": parse_node_list(args.ddr_nodes), "hbm": parse_node_list(args.hbm_nodes)}
    numel = args.tensor_size_mb * 2**20 // 4
    # The resize case holds the old and the new storage at once: 2.5 tensors per tier.
    if not c.require_memory("placement", node_sets, dict.fromkeys(node_sets, 3 * args.tensor_size_mb)):
        c.save(out_path)
        return 1

    def record(name, tensor, tier, allocators, mode="bind", must_be_resident=False):
        """Judge a tensor that was already written."""
        distribution = get_tensor_node_distribution(tensor)
        verdict = evaluate_placement(distribution, node_sets[tier], mode, args.interleave_tolerance)
        meta = tensor_record(tensor, allocators)
        if meta["owned_by"] != tier:
            verdict["status"] = "FAIL"
            verdict["summary"] += f"; storage is owned by the {meta['owned_by']} allocator, expected {tier}"
        if must_be_resident and distribution.get("pages_not_resident"):
            verdict["status"] = "FAIL"
            verdict["summary"] += f"; {distribution['pages_not_resident']} sampled pages are not resident"
        c.check(
            name,
            metadata=meta,
            counters=counters_snapshot(allocators[tier]),
            distribution=distribution,
            expected_nodes=node_sets[tier],
            mode=mode,
            **verdict,
        )

    def bind_cases(tier):
        allocators = open_allocators(args.lib, node_sets["ddr"], node_sets["hbm"])
        alloc = allocators[tier]

        factory = touch_tensor(alloc.empty(numel))
        record(f"placement[{tier},bind,factory]", factory, tier, allocators)
        del factory

        with use_allocator(alloc):
            ones = torch.ones(numel)
            op_output = ones + ones
        record(f"placement[{tier},bind,scope]", op_output, tier, allocators)
        del ones

        # Outside the scope: the storage keeps the allocator that created it.
        touch_tensor(op_output.resize_(numel + numel // 2))
        record(f"placement[{tier},bind,resize_after_scope]", op_output, tier, allocators)
        del op_output

        counters = counters_snapshot(alloc)
        balanced = counters["live_allocations"] == 0 and counters["alloc_calls"] == counters["free_calls"]
        c.check(
            f"counters_balanced[{tier}]",
            "PASS" if balanced else "FAIL",
            summary=f"{counters['alloc_calls']} alloc calls, {counters['free_calls']} free calls, {counters['live_allocations']} live",
            counters=counters,
        )

    def prefault_case(tier):
        allocators = open_allocators(args.lib, node_sets["ddr"], node_sets["hbm"], prefault=True)
        # Not written from Python: the allocator itself must have placed every page.
        tensor = allocators[tier].empty(numel)
        record(f"placement[{tier},bind,prefault]", tensor, tier, allocators, must_be_resident=True)

    def interleave_case(tier):
        allocators = open_allocators(args.lib, node_sets["ddr"], node_sets["hbm"], mode="interleave")
        tensor = touch_tensor(allocators[tier].empty(numel))
        record(f"placement[{tier},interleave,factory]", tensor, tier, allocators, mode="interleave")

    for tier in ("ddr", "hbm"):
        guarded(c, f"placement[{tier},bind]", lambda: bind_cases(tier))
        guarded(c, f"placement[{tier},bind,prefault]", lambda: prefault_case(tier))
        guarded(c, f"placement[{tier},interleave,factory]", lambda: interleave_case(tier))

    def default_case():
        # No pass criterion: the default allocator follows the process policy, which run_campaign.sh sets to the DDR nodes.
        tensor = touch_tensor(torch.empty(numel))
        distribution = get_tensor_node_distribution(tensor)
        verdict = evaluate_placement(distribution, node_sets["ddr"])
        c.check(
            "placement[default]",
            "INFO",
            summary=verdict["summary"],
            metadata=tensor_record(tensor, {}),
            distribution=distribution,
            expected_nodes=node_sets["ddr"],
            mode="process policy",
            percent_on_expected_nodes=verdict.get("percent_on_expected_nodes"),
        )

    guarded(c, "placement[default]", default_case)

    rc, out = run_cmd(["numastat", "-p", str(os.getpid())])
    c.check("numastat", "INFO", summary="captured" if rc == 0 else "numastat is not available", output=out)

    c.save(out_path)
    return 1 if c.had_failures() else 0


if __name__ == "__main__":
    sys.exit(main())
