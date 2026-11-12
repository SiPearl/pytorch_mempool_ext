#!/usr/bin/env python3
"""F.linear and an FFN block with the OPT-13B matrix shapes on HBM, DDR and default memory.

Three columns everywhere. "hbm" and "ddr" are two NumaAllocator instances bound
to the HBM and the DDR nodes, so the same allocator code runs in both and their
ratio is a memory effect. "default" is the process-wide CPU allocator.

Two comparisons:

    weight-tier      Only the weight of one F.linear moves between the columns.
                     The input and the bias stay on the default allocator.
    activation-tier  Only the activations of one FFN block move:
                     fc1 -> activation -> fc2 -> add the input.
                     The parameters stay on --param-tier in every column, and
                     the block input is a copy on the column's tier.

Two variants of each:

    preallocated  Every tensor exists before the timed loop and the operations
                  write with out=. Nothing is allocated while timing, which a
                  probe run verifies. This isolates the memory effect.
    scoped        The operations allocate their results inside
                  `with use_allocator(...)`, as a model would, and the results
                  are freed inside the timed interval. This is the end-to-end
                  cost, including mmap, mbind, page faults and munmap. The
                  default column runs the same operations without a scope. In
                  weight-tier the output therefore moves together with the weight.

The script only measures. Placement is read back from the kernel and is part of
the contract: a tensor that is not on its node set is reported as a failure.
Before each case it records the machine load and checks that every target node
has room for what the case allocates; a case that does not fit is not run.

Usage:
    python3 benchmark_hbm_linear.py --ddr-nodes 0-3 --hbm-nodes 4-7 --threads 20 --json-output linear.json
    python3 benchmark_hbm_linear.py --ddr-nodes 0 --hbm-nodes 0 --shape-divisor 16 --dtype float32
"""

import argparse
import json
import os
import statistics
import sys
import time

from hbm_test_utils import (
    TIERS,
    add_allocator_args,
    allocations_under_scope,
    check_node_memory,
    collect_environment,
    configure_threads,
    evaluate_placement,
    get_tensor_node_distribution,
    linear_path,
    machine_load,
    open_allocators,
    parse_node_list,
    tensor_record,
    thread_args_parser,
)

THREAD_INFO = configure_threads()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from pytorch_numa_mem import use_allocator  # noqa: E402

# OPT-13B: hidden size 5120, FFN size 20480. F.linear weights are (out_features, in_features).
HIDDEN, FFN = 5120, 20480
SHAPES = (("attn_proj", HIDDEN, HIDDEN), ("ffn_fc1", HIDDEN, FFN), ("ffn_fc2", FFN, HIDDEN))
MODE_ROWS = {"decode": 1, "prefill": 512}
COMPARISONS = ("weight-tier", "activation-tier")
VARIANTS = ("preallocated", "scoped")

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
# The same kernel on the same values should agree bitwise. The tolerances only
# allow for a different vector path when the alignment of two allocations differs.
TOLERANCES = {
    torch.bfloat16: {"rtol": 1.6e-2, "atol": 1.0e-2},
    torch.float16: {"rtol": 1.0e-3, "atol": 1.0e-3},
    torch.float32: {"rtol": 1.0e-5, "atol": 1.0e-5},
}
ACTIVATIONS = {
    "relu": lambda h, out=None: torch.clamp(h, min=0, out=out),
    "gelu": lambda h, out=None: torch._C._nn.gelu(h, out=out),
}


def linear_into(out, x, weight, bias):
    """F.linear for a 2-D input, written into out. F.linear itself computes addmm(bias, x, weight.t())."""
    if bias is None:
        return torch.mm(x, weight.t(), out=out)
    return torch.addmm(bias, x, weight.t(), out=out)


def ffn(x, params, activation):
    w1, b1, w2, b2 = params
    return F.linear(activation(F.linear(x, w1, b1)), w2, b2) + x


def ffn_into(buffers, x, params, activation):
    w1, b1, w2, b2 = params
    h1, a1, h2, y = buffers
    linear_into(h1, x, w1, b1)
    activation(h1, out=a1)
    linear_into(h2, a1, w2, b2)
    return torch.add(h2, x, out=y)


class Bench:
    def __init__(self, args):
        self.args = args
        self.dtype = DTYPES[args.dtype]
        self.node_sets = {"ddr": parse_node_list(args.ddr_nodes), "hbm": parse_node_list(args.hbm_nodes)}
        self.allocators = open_allocators(args.lib, self.node_sets["ddr"], self.node_sets["hbm"])
        self.flush_buf = None
        if args.cache_flush_mib > 0:
            self.flush_buf = torch.zeros(args.cache_flush_mib * 2**20, dtype=torch.uint8)
        self.failures = []
        self.warnings = []

    def place(self, tier, tensor):
        """A copy of tensor on tier. None stays None."""
        if tensor is None:
            return None
        allocator = self.allocators[tier]
        return allocator.clone_from(tensor) if allocator else tensor.clone()

    def empty(self, tier, *size):
        allocator = self.allocators[tier]
        tensor = allocator.empty(*size, dtype=self.dtype) if allocator else torch.empty(*size, dtype=self.dtype)
        return tensor.zero_()

    def scoped(self, tier, fn):
        allocator = self.allocators[tier]
        if allocator is None:
            return fn

        def run():
            with use_allocator(allocator):
                return fn()

        return run

    def randn(self, *size, scale=1.0):
        return (torch.randn(*size) * scale).to(self.dtype)

    def check_tensor(self, case, name, tensor, tier):
        """Ownership and measured placement of a tensor that was already written. Returns the evidence."""
        record = tensor_record(tensor, self.allocators)
        evidence = {"tier": tier, **record}
        if record["owned_by"] != tier:
            self.fail(case, f"{name} is owned by the {record['owned_by']} allocator, expected {tier}")
        if tier != "default":
            distribution = get_tensor_node_distribution(tensor)
            verdict = evaluate_placement(distribution, self.node_sets[tier])
            evidence.update(placement=verdict, distribution=distribution)
            if verdict["status"] != "PASS":
                self.fail(case, f"{name} is not on the {tier} nodes: {verdict['summary']}")
        return evidence

    def fail(self, case, message):
        message = f"[{case}] {message}"
        print(f"  ERROR {message}", file=sys.stderr)
        self.failures.append(message)

    def before_case(self, case, tier_elements):
        """Record the machine load and check the memory of every target node. None means: do not run the case.

        tier_elements maps a tier to the number of elements the case keeps on
        it. The default allocator follows the process policy, which
        run_campaign.sh sets to the DDR nodes. It also holds the float32
        originals of the weights for a moment: two of them, of 4 bytes per
        element, while a weight is created.
        """
        itemsize = torch.empty(0, dtype=self.dtype).element_size()
        megabytes = {tier: count * itemsize // 2**20 + 64 for tier, count in tier_elements.items()}
        transient = 2 * max(tier_elements.values()) * 4 // 2**20
        planned = {"hbm": megabytes["hbm"], "ddr": megabytes["ddr"] + megabytes["default"] + transient}
        memory, problems = check_node_memory(self.node_sets, planned)
        for problem in problems:
            self.fail(case, f"not run, it would not fit: {problem}")
        load = machine_load()
        print(f"  load: {load['uptime']}")
        return None if problems else {"load_before": load, "memory_before": memory}

    def compare_outputs(self, case, outputs):
        """Every column must compute the same values. Outside every timed interval."""
        tolerance = TOLERANCES[self.dtype]
        reference = outputs["default"]
        result = {"finite": bool(torch.isfinite(reference).all()), **tolerance}
        if not result["finite"]:
            self.fail(case, "the default output is not finite")
        for tier in ("hbm", "ddr"):
            result[f"{tier}_equal_to_default"] = bool(torch.equal(outputs[tier], reference))
            close = bool(torch.allclose(outputs[tier], reference, **tolerance))
            result[f"{tier}_allclose_to_default"] = close
            if not close:
                self.fail(case, f"the {tier} output differs from the default output")
        return result

    def measure(self, cases, iterations):
        """Interleaved timing of {tier: fn}. Returns {tier: [ns]} and the alloc_calls of each allocator.

        The order rotates every round, so no column always runs first or always
        runs after the same neighbour. Only fn() is timed, and whatever it
        returns is dropped inside the interval.
        """
        tiers = list(cases)
        samples = {tier: [] for tier in tiers}
        alloc_calls = {}
        with torch.inference_mode():
            for step in range(self.args.warmup + iterations):
                if step == self.args.warmup:
                    alloc_calls = {t: a.counters.alloc_calls for t, a in self.allocators.items() if a}
                shift = step % len(tiers)
                for tier in tiers[shift:] + tiers[:shift]:
                    fn = cases[tier]
                    if self.flush_buf is not None:
                        self.flush_buf.add_(1)
                    start = time.perf_counter_ns()
                    fn()
                    elapsed = time.perf_counter_ns() - start
                    if step >= self.args.warmup:
                        samples[tier].append(elapsed)
        alloc_calls = {t: self.allocators[t].counters.alloc_calls - before for t, before in alloc_calls.items()}
        return samples, alloc_calls

    def run_case(self, case, variant, bodies, iterations, flops):
        """Time one case. bodies is {tier: fn} without scopes; the scoped variant wraps them here."""
        probe = self.allocators["ddr"]
        with torch.inference_mode():
            allocs_per_step = {tier: allocations_under_scope(fn, probe) for tier, fn in bodies.items()}
        source = "one probe run inside a scope"
        cases = bodies
        if variant == "scoped":
            cases = {tier: self.scoped(tier, fn) for tier, fn in bodies.items()}
        elif any(allocs_per_step.values()):
            message = f"[{case}] the preallocated variant allocates inside the timed loop: {allocs_per_step}"
            print(f"  WARNING {message}")
            self.warnings.append(message)

        if self.args.hold_seconds > 0:
            print(f"  HOLD pid={os.getpid()} for {self.args.hold_seconds} s (numastat -p {os.getpid()})")
            time.sleep(self.args.hold_seconds)

        samples, counted = self.measure(cases, iterations)
        if variant == "scoped":
            allocs_per_step.update({tier: calls / iterations for tier, calls in counted.items()})
            source = "allocator counters over the timed loop (default: one probe run inside a scope)"

        stats = {}
        for tier in TIERS:
            seconds = [ns / 1e9 for ns in samples[tier]]
            median, mean = statistics.median(seconds), statistics.fmean(seconds)
            stats[tier] = {
                "samples_ns": samples[tier],
                "min_seconds": min(seconds),
                "median_seconds": median,
                "mean_seconds": mean,
                "coefficient_of_variation": statistics.pstdev(seconds) / mean,
                "gflop_per_second": flops / median / 1e9,
                "alloc_calls_per_step": allocs_per_step[tier],
            }
        medians = {tier: stats[tier]["median_seconds"] for tier in TIERS}
        ratios = {
            "hbm_over_ddr_time_ratio": medians["hbm"] / medians["ddr"],
            "hbm_over_default_time_ratio": medians["hbm"] / medians["default"],
            "ddr_over_default_time_ratio": medians["ddr"] / medians["default"],
        }
        print(
            "  "
            + "  ".join(f"{t}={medians[t] * 1e3:.4f}ms (cv {stats[t]['coefficient_of_variation']:.3f})" for t in TIERS)
            + f"  hbm/ddr={ratios['hbm_over_ddr_time_ratio']:.3f}"
            + f"  alloc_calls/step={ {t: allocs_per_step[t] for t in TIERS} }"
        )
        return {"statistics": stats, "ratios": ratios, "alloc_calls_per_step_source": source}

    def iterations(self, mode):
        count = self.args.iterations if mode == "decode" else self.args.prefill_iterations
        if count % len(TIERS):
            print(f"  WARNING iterations={count} is not a multiple of {len(TIERS)}: the run order is not fully balanced")
        return count

    def weight_tier(self, shape, mode, variant):
        label, in_features, out_features = shape
        rows = self.rows(mode)
        case = f"weight-tier/{variant}/{label}/{mode}"
        print(f"\n== {case}: M={rows} K={in_features} N={out_features} {self.args.dtype}")

        # Per tier: one weight and, in the scoped variant, one output. The default allocator also holds the source.
        elements = out_features * in_features + rows * out_features
        before = self.before_case(case, {"hbm": elements, "ddr": elements, "default": 2 * elements})
        if before is None:
            return None

        torch.manual_seed(self.args.seed)
        x = self.randn(rows, in_features)
        source = self.randn(out_features, in_features, scale=in_features**-0.5)
        bias = self.randn(out_features) if self.args.bias else None
        weights = {tier: self.place(tier, source) for tier in TIERS}

        evidence = {f"weight[{tier}]": self.check_tensor(case, f"weight[{tier}]", w, tier) for tier, w in weights.items()}
        for tier, weight in weights.items():
            if not torch.equal(weight, source):
                self.fail(case, f"weight[{tier}] is not a byte copy of the source weight")
        if len({w.data_ptr() for w in weights.values()} | {source.data_ptr()}) != len(TIERS) + 1:
            self.fail(case, "two weights share a data pointer")

        if variant == "preallocated":
            out = torch.zeros(rows, out_features, dtype=self.dtype)
            bodies = {tier: (lambda w=w: linear_into(out, x, w, bias)) for tier, w in weights.items()}
            with torch.inference_mode():
                outputs = {tier: fn().clone() for tier, fn in bodies.items()}
        else:
            bodies = {tier: (lambda w=w: F.linear(x, w, bias)) for tier, w in weights.items()}
            with torch.inference_mode():
                outputs = {tier: self.scoped(tier, fn)() for tier, fn in bodies.items()}
            for tier, output in outputs.items():
                evidence[f"output[{tier}]"] = self.check_tensor(case, f"output[{tier}]", output, tier)
        correctness = self.compare_outputs(case, outputs)
        del outputs

        timing = self.run_case(case, variant, bodies, self.iterations(mode), 2.0 * rows * out_features * in_features)
        return {
            "comparison": "weight-tier",
            "variant": variant,
            "shape": label,
            "mode": mode,
            "M": rows,
            "K": in_features,
            "N": out_features,
            "tensors": evidence,
            "correctness": correctness,
            **before,
            **timing,
        }

    def activation_tier(self, mode, variant):
        hidden, ffn_size = HIDDEN // self.args.shape_divisor, FFN // self.args.shape_divisor
        rows = self.rows(mode)
        param_tier = self.args.param_tier
        case = f"activation-tier/{variant}/ffn_block/{mode}"
        print(f"\n== {case}: M={rows} hidden={hidden} ffn={ffn_size} {self.args.dtype} parameters on {param_tier}")

        # Per tier: the input and four activation buffers. The parameter tier adds both weights, and the
        # default allocator holds one source weight while the other is created.
        elements = dict.fromkeys(TIERS, rows * (3 * hidden + 2 * ffn_size))
        elements[param_tier] += 2 * hidden * ffn_size
        elements["default"] += hidden * ffn_size
        before = self.before_case(case, elements)
        if before is None:
            return None

        torch.manual_seed(self.args.seed)
        x_source = self.randn(rows, hidden)
        sources = (
            self.randn(ffn_size, hidden, scale=hidden**-0.5),
            self.randn(ffn_size) if self.args.bias else None,
            self.randn(hidden, ffn_size, scale=ffn_size**-0.5),
            self.randn(hidden) if self.args.bias else None,
        )
        params = tuple(self.place(param_tier, p) for p in sources)
        del sources
        inputs = {tier: self.place(tier, x_source) for tier in TIERS}
        activation = ACTIVATIONS[self.args.activation]

        evidence = {"fc1.weight": self.check_tensor(case, "fc1.weight", params[0], param_tier)}
        evidence["fc2.weight"] = self.check_tensor(case, "fc2.weight", params[2], param_tier)
        for tier, x in inputs.items():
            evidence[f"input[{tier}]"] = self.check_tensor(case, f"input[{tier}]", x, tier)
            if not torch.equal(x, x_source):
                self.fail(case, f"input[{tier}] is not a byte copy of the source input")

        if variant == "preallocated":
            sizes = ((rows, ffn_size), (rows, ffn_size), (rows, hidden), (rows, hidden))
            buffers = {tier: [self.empty(tier, *size) for size in sizes] for tier in TIERS}
            bodies = {tier: (lambda t=tier: ffn_into(buffers[t], inputs[t], params, activation)) for tier in TIERS}
            with torch.inference_mode():
                outputs = {tier: fn().clone() for tier, fn in bodies.items()}
            for tier in TIERS:
                # After the run above, so that every page of the buffers was written by the block itself.
                evidence[f"fc1_output[{tier}]"] = self.check_tensor(case, f"fc1_output[{tier}]", buffers[tier][0], tier)
                evidence[f"block_output[{tier}]"] = self.check_tensor(case, f"block_output[{tier}]", buffers[tier][3], tier)
        else:
            bodies = {tier: (lambda t=tier: ffn(inputs[t], params, activation)) for tier in TIERS}
            with torch.inference_mode():
                outputs = {tier: self.scoped(tier, fn)() for tier, fn in bodies.items()}
            for tier, output in outputs.items():
                evidence[f"block_output[{tier}]"] = self.check_tensor(case, f"block_output[{tier}]", output, tier)
        correctness = self.compare_outputs(case, outputs)
        del outputs

        timing = self.run_case(case, variant, bodies, self.iterations(mode), 4.0 * rows * hidden * ffn_size)
        return {
            "comparison": "activation-tier",
            "variant": variant,
            "shape": "ffn_block",
            "mode": mode,
            "M": rows,
            "K": hidden,
            "N": ffn_size,
            "param_tier": param_tier,
            "activation": self.args.activation,
            "tensors": evidence,
            "correctness": correctness,
            **before,
            **timing,
        }

    def rows(self, mode):
        return 1 if mode == "decode" else max(2, MODE_ROWS[mode] // self.args.shape_divisor)

    def estimate_minutes(self, shapes, modes, variants, comparisons):
        """Time one F.linear per mode at the ffn_fc1 shape and scale by the flops of the planned run.

        A build without a fast kernel for the dtype can be orders of magnitude
        slower, and this run is started by hand: better to stop here than to
        run for hours.
        """
        _, hidden, ffn_size = shapes[1]
        weight = self.randn(ffn_size, hidden)
        rounds = {"decode": self.args.warmup + self.args.iterations, "prefill": self.args.warmup + self.args.prefill_iterations}
        seconds = 0.0
        per_call = {}
        with torch.inference_mode():
            for mode in modes:
                rows = self.rows(mode)
                x = self.randn(rows, hidden)
                F.linear(x[:1], weight)
                start = time.perf_counter()
                F.linear(x, weight)
                per_call[mode] = time.perf_counter() - start
                flops = 0.0
                if "weight-tier" in comparisons:
                    flops += sum(2.0 * rows * k * n for _, k, n in shapes)
                if "activation-tier" in comparisons:
                    flops += 4.0 * rows * hidden * ffn_size
                # Every round times three columns, and each case runs a few extra calls for its checks.
                calls = len(variants) * len(TIERS) * (rounds[mode] + 2)
                seconds += calls * flops * per_call[mode] / (2.0 * rows * hidden * ffn_size)
        return {"minutes": seconds / 60, "seconds_per_ffn_fc1_call": per_call}


def subset(text, allowed, flag):
    chosen = [item.strip() for item in text.split(",") if item.strip()]
    if not chosen or len(set(chosen)) != len(chosen) or not set(chosen) <= set(allowed):
        raise SystemExit(f"ERROR: {flag} must be a comma separated subset of {','.join(allowed)}, got {text!r}")
    return chosen


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[thread_args_parser()]
    )
    add_allocator_args(parser)
    parser.add_argument("--comparisons", default=",".join(COMPARISONS), help="subset of %(default)s")
    parser.add_argument("--variants", default=",".join(VARIANTS), help="subset of %(default)s")
    parser.add_argument("--modes", default="decode,prefill", help="subset of %(default)s (M=1 and M=512 rows)")
    parser.add_argument("--param-tier", choices=TIERS, default="hbm", help="where the FFN parameters live in activation-tier")
    parser.add_argument("--activation", choices=sorted(ACTIVATIONS), default="relu", help="FFN activation (OPT uses relu)")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=6, help="untimed rounds before the timed ones")
    parser.add_argument("--iterations", type=int, default=21, help="timed rounds for decode; use a multiple of 3")
    parser.add_argument("--prefill-iterations", type=int, default=6, help="timed rounds for prefill; use a multiple of 3")
    parser.add_argument("--shape-divisor", type=int, default=1, help="divide the matrix sizes and the prefill rows, to check the plumbing on a small machine")
    parser.add_argument("--cache-flush-mib", type=int, default=0, help="stream a buffer of this size before every timed call, outside the timed interval")
    parser.add_argument("--hold-seconds", type=int, default=0, help="pause before timing each case, to inspect the process with numastat")
    parser.add_argument("--max-minutes", type=float, default=60, help="stop before measuring if the run is estimated to take longer")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-bias", dest="bias", action="store_false", help="OPT linear layers have a bias, so it is on by default")
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    comparisons = subset(args.comparisons, COMPARISONS, "--comparisons")
    variants = subset(args.variants, VARIANTS, "--variants")
    modes = subset(args.modes, tuple(MODE_ROWS), "--modes")
    torch.set_num_threads(THREAD_INFO["threads"])

    provenance = {
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_file": torch.__file__,
        "torch_git_version": getattr(torch.version, "git_version", None) or "UNKNOWN",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "mkldnn_available": torch.backends.mkldnn.is_available(),
        **THREAD_INFO,
    }
    print("F.linear and FFN block: hbm, ddr and default memory")
    for key, value in provenance.items():
        print(f"  {key}: {value}")

    bench = Bench(args)
    shapes = [(label, k // args.shape_divisor, n // args.shape_divisor) for label, k, n in SHAPES]
    # Which kernel the benchmark is about to time, at the ffn_fc1 shape it uses.
    kernels = linear_path(bench.dtype, shapes[1][1], shapes[1][2], rows=sorted({min(bench.rows(mode), 8) for mode in modes}))
    for rows, text in kernels.items():
        print(f"  F.linear {args.dtype}, {rows} row(s): {text}")
    provenance["linear_path"] = kernels
    estimate = bench.estimate_minutes(shapes, modes, variants, comparisons)
    timings = ", ".join(f"{mode} {t * 1e3:.1f} ms" for mode, t in estimate["seconds_per_ffn_fc1_call"].items())
    print(f"  one ffn_fc1 call: {timings}; estimated timed work: {estimate['minutes']:.1f} min, plus setup")
    if estimate["minutes"] > args.max_minutes:
        raise SystemExit(
            f"ERROR: the timed work is estimated at {estimate['minutes']:.1f} min, above --max-minutes {args.max_minutes:g} "
            f"(one ffn_fc1 call: {timings}). If these times are far above what this machine should reach, the build "
            f"has no fast kernel for {args.dtype}: use --dtype float32. Otherwise reduce --iterations and "
            "--prefill-iterations, use --modes decode, or raise --max-minutes."
        )
    results = []
    for mode in modes:
        for variant in variants:
            if "weight-tier" in comparisons:
                results.extend(bench.weight_tier(shape, mode, variant) for shape in shapes)
            if "activation-tier" in comparisons:
                results.append(bench.activation_tier(mode, variant))
    results = [entry for entry in results if entry is not None]

    print("\nSUMMARY (median ms; a time ratio below 1 means the first tier is faster)")
    header = f"  {'comparison':<16}{'variant':<13}{'shape':<11}{'mode':<8}{'hbm':>10}{'ddr':>10}{'default':>10}{'hbm/ddr':>9}{'hbm/def':>9}  alloc_calls/step (hbm)"
    print(header)
    for entry in results:
        stats = entry["statistics"]
        print(
            f"  {entry['comparison']:<16}{entry['variant']:<13}{entry['shape']:<11}{entry['mode']:<8}"
            + "".join(f"{stats[tier]['median_seconds'] * 1e3:>10.4f}" for tier in TIERS)
            + f"{entry['ratios']['hbm_over_ddr_time_ratio']:>9.3f}{entry['ratios']['hbm_over_default_time_ratio']:>9.3f}"
            + f"  {stats['hbm']['alloc_calls_per_step']}"
        )

    if args.json_output:
        payload = {
            "provenance": provenance,
            "environment": collect_environment(),
            "duration_estimate": estimate,
            "configuration": {**vars(args), "comparisons": comparisons, "variants": variants, "modes": modes},
            "definitions": {
                "tiers": "hbm and ddr: NumaAllocator bound to --hbm-nodes and --ddr-nodes; default: the process-wide CPU allocator",
                "time_ratio": "median seconds of the first tier divided by the second; below 1 means the first is faster",
                "weight-tier": "x and bias on the default allocator; only the weight tier differs (scoped: the output too)",
                "activation-tier": "FFN block fc1 -> activation -> fc2 -> add input; parameters on param_tier; input and activations on the column's tier",
                "alloc_calls_per_step": "allocations made by one timed call",
            },
            "results": results,
            "failures": bench.failures,
            "warnings": bench.warnings,
        }
        # Serialize fully first, so that an error cannot leave a truncated file.
        encoded = json.dumps(payload, indent=2, allow_nan=False, default=str)
        with open(args.json_output, "w") as handle:
            handle.write(encoded)
        print(f"\nwrote {args.json_output}")

    if bench.failures:
        raise SystemExit(f"\nFAILED: {len(bench.failures)} problem(s), see above")
    return 0


if __name__ == "__main__":
    sys.exit(main())
