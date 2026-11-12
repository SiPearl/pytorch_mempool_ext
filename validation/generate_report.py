"""Write summary.md and summary.json for a campaign result directory.

Reads whatever the stages left behind. A missing or failed stage is reported as
such; it does not stop the report.

Usage:
    python3 generate_report.py --results-dir results/<timestamp>
"""

import argparse
import json
import os
import sys

from hbm_test_utils import TIERS, format_cpu_list

REQUIRED_STAGES = ("placement", "bandwidth", "socket_locality", "linear_benchmark")

# A FAIL or WARN whose summary contains one of these means that the harness
# could not observe something, not that the allocator misbehaved.
HARNESS_ISSUE_PATTERNS = (
    "could not determine physical placement",
    "numastat is not available",
    "stage did not produce a result file",
    # A node without room for a benchmark: the machine was busy, the benchmark did not run.
    "free or in page cache",
)


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def load_key_values(path):
    """ "key: value" lines as a dict. Lines without a colon are ignored."""
    values = {}
    try:
        with open(path) as f:
            for line in f:
                key, sep, value = line.partition(":")
                if sep and key.strip() and " " not in key.strip():
                    values.setdefault(key.strip(), value.strip())
    except OSError:
        pass
    return values


def md_table(headers, rows):
    if not rows:
        return "_no data_"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def fmt(value, spec, missing="n/a"):
    return format(value, spec) if value is not None else missing


def get(values, key):
    return values.get(key, "unknown")


def policy_text(policy):
    """{"mode": "bind", "nodes": [0, 1]} -> "bind:0,1"."""
    if not isinstance(policy, dict) or "mode" not in policy:
        return "unknown"
    nodes = ",".join(map(str, policy.get("nodes", [])))
    return f"{policy['mode']}:{nodes}" if nodes else policy["mode"]


def checks(stage, prefix):
    return [check for check in (stage or {}).get("checks", []) if check["name"].startswith(prefix)]


def collect_findings(stages, topology, skipped):
    """Everything that is not plain expected behaviour, each with a category."""
    findings = []

    def add(stage, check, status, summary, category=None):
        if category is None:
            harness = any(pattern in summary for pattern in HARNESS_ISSUE_PATTERNS)
            category = "TEST_HARNESS_ISSUE" if harness else "FAILURE" if status == "FAIL" else "OBSERVATION"
        findings.append({"stage": stage, "check": check, "status": status, "summary": summary, "category": category})

    for name, stage in stages.items():
        if stage is None:
            if name in REQUIRED_STAGES and name not in skipped:
                add(name, None, "MISSING", "stage did not produce a result file")
        elif name == "linear_benchmark":
            for message in stage.get("failures", []):
                add(name, None, "FAIL", message)
            for message in stage.get("warnings", []):
                add(name, None, "WARN", message)
        else:
            for check in stage.get("checks", []):
                if check["status"] in ("FAIL", "WARN"):
                    add(name, check["name"], check["status"], check.get("summary", ""))
    for issue in (topology or {}).get("selection_issues", []):
        add("topology", f"node{issue['node']}", "FAIL", f"{issue['role']}: {issue['problem']}")
    for note in (topology or {}).get("selection_notes", []):
        add("topology", None, "NOTE", note, "OBSERVATION")
    return findings


def determine_verdict(stages, findings, skipped, config, topology):
    missing = [name for name in REQUIRED_STAGES if stages[name] is None and name not in skipped]
    import_failed = any(check["status"] == "FAIL" for stage in stages.values() for check in checks(stage, "import_torch"))
    if missing or import_failed:
        verdict = "NOT REPRODUCIBLE"
    elif any(finding["category"] == "FAILURE" for finding in findings):
        verdict = "FAILURES FOUND"
    elif any(finding["status"] == "FAIL" for finding in findings):
        verdict = "HARNESS PROBLEMS"
    else:
        verdict = "WORKING AS EXPECTED"
    if (topology or {}).get("locality") == "remote":
        verdict = f"REMOTE CONTROL RUN (--allow-remote): {verdict}"
    if config.get("dry_run") == "1":
        verdict = f"DRY RUN (plumbing check, no measurement): {verdict}"
    return verdict, {"missing_stages": missing, "import_failed": import_failed}


def placement_rows(stage):
    rows = []
    for check in checks(stage, "placement["):
        dist = check.get("distribution") or {}
        meta = check.get("metadata") or {}
        rows.append(
            (
                check["name"],
                check.get("mode", "-"),
                ",".join(map(str, check.get("expected_nodes", []))),
                ",".join(sorted(dist.get("nodes_pages", {}), key=int)),
                fmt(check.get("percent_on_expected_nodes"), ".2f") + "%",
                policy_text(dist.get("vma_policy")),
                dist.get("placement_method", "n/a"),
                f"{dist.get('pages_sampled', '-')} of {dist.get('pages_in_tensor', '-')}",
                meta.get("owned_by", "-"),
                check["status"],
            )
        )
    return rows


def build_markdown(results_dir, stages, topology, verdict, findings, config):
    git_info = load_key_values(os.path.join(results_dir, "git_info.txt"))
    build_info = load_key_values(os.path.join(results_dir, "build_info.txt"))
    environment = next((stage["environment"] for stage in stages.values() if stage and "environment" in stage), {})

    md = ["# NUMA allocator validation report\n", f"**Verdict: {verdict}**\n"]
    if config.get("dry_run") == "1":
        md.append(
            "This was a dry run: small tensors, matrix sizes divided by 16 and a few repetitions. "
            "It shows that every stage runs. The timings and ratios below mean nothing.\n"
        )

    if (topology or {}).get("locality") == "remote":
        md.append(
            "This is the socket-locality control: the HBM nodes are on another socket than the cores and the DDR nodes. "
            "Compare its HBM/DDR ratios with those of the run on the local HBM nodes; do not use its numbers as HBM results.\n"
        )

    md.append("## Run\n")
    md.append(f"- host: {environment.get('hostname', 'unknown')} ({environment.get('machine', 'unknown')}), kernel {environment.get('kernel', 'unknown')}")
    md.append(f"- started: {get(config, 'started_utc')}")
    md.append(f"- DDR nodes: {get(config, 'ddr_nodes')}; HBM nodes: {get(config, 'hbm_nodes')}; --allow-remote: {get(config, 'allow_remote')}")
    md.append(f"- threads: {get(config, 'threads')} (OMP_NUM_THREADS and MKL_NUM_THREADS), pinned to CPUs {get(config, 'pinned_cpus')}")
    md.append(f"- stage binding: {get(config, 'stage_binding')}")
    md.append(f"- tensor size: {get(config, 'tensor_size_mb')} MiB; dtype of the linear benchmark: {get(config, 'dtype')}\n")

    md.append("## Source provenance\n")
    md.append(f"- PyTorch checkout: {get(git_info, 'pytorch_root')}")
    md.append(f"- branch: {get(git_info, 'pytorch_branch')}; HEAD: {get(git_info, 'pytorch_sha')}")
    md.append(f"- PoC base: {get(git_info, 'pytorch_base_sha')}; files changed since the base: {get(git_info, 'pytorch_changed_files_vs_base')}")
    md.append(f"- uncommitted changes in the PyTorch checkout: {get(git_info, 'pytorch_dirty')}")
    md.append(f"- external tree: {get(git_info, 'ext_root')}; revision: {get(git_info, 'ext_sha')}; uncommitted changes: {get(git_info, 'ext_dirty')}")
    md.append(f"- external tree content hash (sha256 over the source files): {get(git_info, 'ext_content_sha256')}\n")

    md.append("## Compiled PyTorch\n")
    md.append(f"- Python: {get(build_info, 'python_version')} ({get(build_info, 'python_executable')})")
    md.append(f"- torch.__version__: {get(build_info, 'torch_version')}; torch.version.git_version: {get(build_info, 'torch_git_version')}")
    md.append(f"- torch.__file__: {get(build_info, 'torch_file')}")
    md.append(f"- compiler that built PyTorch: {get(build_info, 'build_compiler')} (from {get(build_info, 'build_compiler_source')})")
    md.append(f"- compiler on PATH now, for context only: {get(build_info, 'shell_compiler')}")
    md.append(f"- pytorch_numa_mem: {get(build_info, 'pytorch_numa_mem_file')}")
    md.append(f"- libnuma_allocator: {get(build_info, 'libnuma_allocator')} (ABI version {get(build_info, 'libnuma_allocator_abi_version')})")
    md.append(f"- oneDNN available: {get(build_info, 'mkldnn_available')}; CPU capability: {get(build_info, 'cpu_capability')}; CPU flags: {get(build_info, 'cpu_flags')}")
    md.append("- kernel that `F.linear` uses at the `ffn_fc1` shape (`torch_config.txt` has `torch.__config__.show()`):")
    for dtype in ("bfloat16", "float32"):
        for rows in (1, 8):
            md.append(f"  - {dtype}, {rows} row(s): {get(build_info, f'{dtype}_linear_rows{rows}')}")
    md.append("")

    md.append("## Environment\n")
    allowed = environment.get("mems_allowed_from_get_mempolicy")
    allowed = ",".join(map(str, allowed)) if allowed is not None else "unknown"
    md.append(f"- Mems_allowed: {environment.get('mems_allowed_list', 'unknown')} (get_mempolicy: {allowed})")
    md.append(f"- Cpus_allowed of the stages: {environment.get('cpus_allowed_list', 'unknown')}")
    md.append(f"- process memory policy of the stages: {policy_text(environment.get('process_mempolicy'))}")
    md.append(f"- page size: {environment.get('page_size', 'unknown')}; transparent huge pages: {environment.get('transparent_hugepage', 'unknown')}; numa_balancing: {environment.get('numa_balancing', 'unknown')}\n")
    md.append("`numactl -H`:\n")
    md.append("```\n" + (environment.get("numactl_h") or "(not recorded)").rstrip() + "\n```\n")

    md.append("## NUMA topology\n")
    if topology:
        rows = [
            (
                node,
                ",".join(map(str, info["sockets"])),
                info["cpulist"] or "none",
                len(info["physical_cores"]),
                f"{info['total_mb'] / 1024:.1f} GiB",
                f"{info['usable_mb'] / 1024:.1f} GiB",
            )
            for node, info in sorted(topology["detected"]["nodes"].items(), key=lambda item: int(item[0]))
        ]
        md.append(md_table(["node", "socket", "CPUs", "physical cores", "memory", "free or in page cache"], rows) + "\n")
        pinned = "; ".join(f"node {node}: {format_cpu_list(cpus)}" for node, cpus in topology.get("pinned_cpus_per_node", {}).items())
        md.append(f"The HBM nodes are {topology.get('locality')} to the DDR nodes. Pinned CPUs per DDR node: {pinned}.\n")
    else:
        md.append("_topology.json not found_\n")

    md.append("## Physical placement\n")
    md.append(
        "Measured after the tensors were written. Criterion: at least 99% of the sampled pages on the node set; "
        "for interleave also an even spread. `policy` is what get_mempolicy reports for the mapping.\n"
    )
    headers = ["check", "mode", "expected nodes", "observed nodes", "on expected", "policy", "method", "pages sampled", "owner", "result"]
    md.append(md_table(headers, placement_rows(stages["placement"])) + "\n")
    rows = [
        (
            check["name"],
            " ".join(f"N{node}={share:.1f}%" for node, share in check["per_node_percent"].items()),
            f"{check['max_deviation_points']:.2f}",
            check["spread_source"],
            check["status"],
        )
        for check in checks(stages["placement"], "placement[")
        if "per_node_percent" in check
    ]
    md.append("Interleave spread:\n")
    md.append(md_table(["check", "share per node", "largest deviation (points)", "source", "result"], rows) + "\n")
    rows = [(check["name"], check["status"], check.get("summary", "")) for check in checks(stages["placement"], "counters_balanced[")]
    md.append("Allocator counters after the tensors were dropped:\n")
    md.append(md_table(["check", "result", "counters"], rows) + "\n")

    md.append("## Machine load and memory before each benchmark\n")
    md.append(
        "`uptime` before each stage and each benchmark shows the machine load; the "
        "`load_before_<stage>.txt` files and the stage JSON files also hold the head of `top`.\n"
    )
    rows = []
    for name, stage in stages.items():
        try:
            with open(os.path.join(results_dir, f"load_before_{name}.txt")) as f:
                rows.append((f"stage {name}", f.readline().strip()))
        except OSError:
            pass
        rows += [(f"{name}: {check['name'][5:-1]}", check.get("uptime", "")) for check in checks(stage, "load[")]
        if name == "linear_benchmark" and stage:
            rows += [
                (f"{e['comparison']}/{e['variant']}/{e['shape']}/{e['mode']}", e["load_before"]["uptime"])
                for e in stage["results"]
            ]
    md.append(md_table(["before", "uptime"], rows) + "\n")
    rows = [
        (f"{name}: {check['name']}", check["status"], check.get("summary", ""))
        for name, stage in stages.items()
        for check in checks(stage, "memory[")
    ]
    rows += [
        (
            f"linear_benchmark: {e['comparison']}/{e['variant']}/{e['shape']}/{e['mode']}",
            "PASS",
            ", ".join(f"node {node}: {m['usable_mb']} MiB usable, {m['planned_mb']} MiB planned" for node, m in e["memory_before"].items()),
        )
        for e in (stages["linear_benchmark"] or {}).get("results", [])
    ]
    md.append("Memory of the target nodes, checked before the benchmarks allocate. A benchmark that does not fit is not run:\n")
    md.append(md_table(["check", "result", "nodes"], rows) + "\n")

    md.append("## Bandwidth sanity check\n")
    md.append(
        "Not a formal benchmark. `preallocated` times no allocation and shows the memory effect; `scoped` allocates "
        "the result inside the timed call and shows the end-to-end cost.\n"
    )
    thread_check = next(iter(checks(stages["bandwidth"], "threading_config")), None)
    if thread_check:
        md.append(f"Threads: {thread_check.get('summary')}\n")
    rows = []
    for check in checks(stages["bandwidth"], "ratio["):
        op, variant = check["name"][len("ratio[") : -1].split(",")
        per_call = next(iter(checks(stages["bandwidth"], f"bandwidth[{op},{variant},hbm]")), {}).get("alloc_calls_per_call")
        rows.append(
            (op, variant)
            + tuple(f"{check['gbps'][tier]:.1f}" for tier in TIERS)
            + (f"{check['hbm_over_ddr']:.2f}x", f"{check['hbm_over_default']:.2f}x", per_call)
        )
    headers = ["operation", "variant", "HBM GB/s", "DDR GB/s", "default GB/s", "HBM/DDR", "HBM/default", "alloc_calls per call (HBM)"]
    md.append(md_table(headers, rows) + "\n")

    md.append("## F.linear and FFN block (OPT-13B shapes)\n")
    linear = stages["linear_benchmark"]
    if linear is None:
        md.append("_not run_\n")
    else:
        md.append(
            "Median time per call in ms. A time ratio below 1 means that HBM is faster. `weight-tier` moves only "
            "the weight of one F.linear; `activation-tier` moves only the activations of one FFN block, with the "
            f"parameters on `{linear['configuration'].get('param_tier')}`. `alloc_calls per step` is the number of "
            "allocations in one timed call, for HBM / DDR / default.\n"
        )
        rows = []
        for entry in linear["results"]:
            stats = entry["statistics"]
            rows.append(
                (entry["comparison"], entry["variant"], entry["shape"], f"{entry['mode']} (M={entry['M']})")
                + tuple(f"{stats[tier]['median_seconds'] * 1e3:.4f}" for tier in TIERS)
                + (
                    f"{entry['ratios']['hbm_over_ddr_time_ratio']:.3f}",
                    f"{entry['ratios']['hbm_over_default_time_ratio']:.3f}",
                    " / ".join(f"{stats[tier]['alloc_calls_per_step']:g}" for tier in TIERS),
                )
            )
        headers = ["comparison", "variant", "shape", "mode", "HBM ms", "DDR ms", "default ms", "HBM/DDR time", "HBM/default time", "alloc_calls per step"]
        md.append(md_table(headers, rows) + "\n")
        provenance = linear["provenance"]
        md.append(f"Threads: {provenance.get('threads')} ({provenance.get('threads_source')}) on CPUs {provenance.get('cpu_affinity')}.\n")
        for rows, text in provenance.get("linear_path", {}).items():
            md.append(f"- `F.linear` in this benchmark, {rows} row(s): {text}")
        md.append("")

    md.append("## Socket locality\n")
    md.append(
        "Where the HBM nodes are, seen from the pinned cores. A normal run has them on the same socket. The control "
        "run (`--allow-remote`) has them on another socket; its HBM/DDR ratios, next to those of the normal run, "
        "are the cost of remote HBM.\n"
    )
    socket = stages["socket_locality"]
    rows = [
        (check["name"], check["status"], check.get("summary", ""))
        for check in (socket or {}).get("checks", [])
        if not check["name"].startswith(("load[", "memory[", "import_torch"))
    ]
    md.append(md_table(["check", "result", "summary"], rows) + "\n")

    md.append("## Findings\n")
    if findings:
        for category in ("FAILURE", "TEST_HARNESS_ISSUE", "OBSERVATION"):
            selected = [finding for finding in findings if finding["category"] == category]
            if selected:
                md.append(f"**{category}**\n")
                md += [f"- [{f['stage']}]" + (f" {f['check']}" if f["check"] else "") + f" ({f['status']}): {f['summary']}" for f in selected]
                md.append("")
    else:
        md.append("_none_\n")
    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    stages = {name: load_json(os.path.join(args.results_dir, f"{name}.json")) for name in REQUIRED_STAGES}
    topology = load_json(os.path.join(args.results_dir, "topology.json"))
    config = load_key_values(os.path.join(args.results_dir, "campaign_config.txt"))
    skipped = set(config.get("skipped_stages", "").split())

    findings = collect_findings(stages, topology, skipped)
    verdict, detail = determine_verdict(stages, findings, skipped, config, topology)
    markdown = build_markdown(args.results_dir, stages, topology, verdict, findings, config)

    md_path = os.path.join(args.results_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write(markdown)

    linear = stages["linear_benchmark"] or {}
    summary = {
        "verdict": verdict,
        "verdict_detail": detail,
        "config": config,
        "source_provenance": load_key_values(os.path.join(args.results_dir, "git_info.txt")),
        "build_provenance": load_key_values(os.path.join(args.results_dir, "build_info.txt")),
        "stage_counts": {name: stage.get("counts") if stage else None for name, stage in stages.items()},
        "bandwidth_ratios": {check["name"]: check["hbm_over_ddr"] for check in checks(stages["bandwidth"], "ratio[")},
        "socket_locality": {
            check["name"]: {key: check.get(key) for key in ("locality", "hbm_over_ddr")}
            for check in checks(stages["socket_locality"], "ratio[")
        },
        "linear_time_ratios_hbm_over_ddr": {
            f"{e['comparison']}/{e['variant']}/{e['shape']}/{e['mode']}": e["ratios"]["hbm_over_ddr_time_ratio"]
            for e in linear.get("results", [])
        },
        "findings": findings,
    }
    json_path = os.path.join(args.results_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"verdict: {verdict}")
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
