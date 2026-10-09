"""Render bench/results/*.json as the Markdown tables used in the README.

python bench/render_results.py > bench/results/RESULTS.md
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

LABELS = {
    "passthrough": "auth + rate limit + accounting only",
    "guard": "+ guardrails (injectguard rules + PII)",
    "router": "+ costroute router (`model: auto`)",
    "cache_miss": "+ exact & semantic cache, all misses",
    "all_on_miss": "guard + router + caches (misses)",
    "cache_hit": "exact-cache hits (upstream skipped)",
}


def loadtest() -> str:
    path = os.path.join(RES, "loadtest.json")
    if not os.path.exists(path):
        return "_loadtest.json not found_\n"
    d = json.load(open(path))
    c = d["conditions"]
    levels = list(d["direct"])
    out = [
        f"Mock upstream latency {c['mock_latency_ms']:.0f} ms; {c['cpus']} vCPU {c['machine']}, "
        f"Python {c['python']}; {c['note']}.\n"
    ]
    out.append("| configuration | " + " | ".join(f"c={lv} p50 / p95 (ms)" for lv in levels) + " |")
    out.append("|---|" + "---|" * len(levels))
    row = ["direct to mock (baseline)"]
    for lv in levels:
        r = d["direct"][lv]
        row.append(f"{r['p50_ms']:.1f} / {r['p95_ms']:.1f}")
    out.append("| " + " | ".join(row) + " |")
    for name, per in d["gateway"].items():
        row = [LABELS.get(name, name)]
        for lv in levels:
            r = per[lv]
            row.append(f"{r['p50_ms']:.1f} / {r['p95_ms']:.1f}")
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    out.append(
        "Gateway overhead = gateway percentile minus direct percentile at the same concurrency "
        "(negative = faster than calling the upstream):\n"
    )
    out.append(
        "| configuration | "
        + " | ".join(f"c={lv} p50 / p95 overhead (ms)" for lv in levels)
        + " | throughput at c="
        + levels[-1]
        + " (req/s) | server-side mean overhead (ms) |"
    )
    out.append("|---|" + "---|" * (len(levels) + 2))
    out.append(
        "| direct to mock | " + " | ".join("0 / 0" for _ in levels) + f" | {d['direct'][levels[-1]]['rps']} | n/a |"
    )
    for name, per in d["gateway"].items():
        row = [LABELS.get(name, name)]
        for lv in levels:
            r = per[lv]
            row.append(f"{r['overhead_p50_ms']:+.1f} / {r['overhead_p95_ms']:+.1f}")
        row.append(str(per[levels[-1]]["rps"]))
        so = per.get("server_side_overhead_mean_ms")
        row.append("n/a" if so is None else f"{so:.2f}")
        out.append("| " + " | ".join(row) + " |")
    errs = sum(per[lv]["errors"] for per in d["gateway"].values() for lv in levels)
    out.append(f"\nFailed requests across all gateway runs: {errs}.\n")
    return "\n".join(out)


def semantic() -> str:
    path = os.path.join(RES, "semantic_cache.json")
    if not os.path.exists(path):
        return "_semantic_cache.json not found_\n"
    d = json.load(open(path))
    out = [
        f"{d['dataset']}; dev n={d['n_dev']}, test n={d['n_test']} "
        f"({d['duplicate_share_test']:.1%} of test pairs are labelled duplicates). Threshold chosen on dev as "
        f"the lowest with hit precision >= {d['target_hit_precision_on_dev']:.0%}.\n"
    ]
    out.append(
        "| embedder | ms / text | threshold | hit rate | correct hits | false hits (of all lookups) | "
        "hit precision | duplicates caught |"
    )
    out.append("|---|---|---|---|---|---|---|---|")
    for name, r in d["embedders"].items():
        if "error" in r:
            out.append(f"| {name} | not run: {r['error']} | | | | | | |")
            continue
        m = r["test_at_chosen"]
        prec = "n/a" if m["hit_precision"] is None else f"{m['hit_precision']:.1%}"
        out.append(
            f"| {r['name']} | {r['embed_ms_per_text']:.2f} | {m['threshold']:.3f} | {m['hit_rate']:.1%} | "
            f"{m['correct_hit_rate']:.1%} | {m['false_hit_rate']:.1%} | {prec} | "
            f"{m['duplicate_recall']:.1%} |"
        )
    out.append("\nThreshold sweep on the test half:\n")
    out.append("| embedder | threshold | hit rate | false hits | hit precision | duplicates caught |")
    out.append("|---|---|---|---|---|---|")
    for name, r in d["embedders"].items():
        for m in r.get("test_sweep", []):
            prec = "n/a" if m["hit_precision"] is None else f"{m['hit_precision']:.1%}"
            out.append(
                f"| {r['name']} | {m['threshold']:.2f} | {m['hit_rate']:.1%} | {m['false_hit_rate']:.1%} | "
                f"{prec} | {m['duplicate_recall']:.1%} |"
            )
    pa = d["pricing_assumption"]
    out.append(
        f"\nSavings at the chosen threshold, assuming {pa['prompt_tokens']} prompt + "
        f"{pa['completion_tokens']} completion tokens per request and list prices "
        "(USD per 1M input / output tokens):\n"
    )
    out.append("| embedder | model (price) | cost / request | saved per 1M requests | wrong answers per 1M requests |")
    out.append("|---|---|---|---|---|")
    for name, r in d["embedders"].items():
        for model, v in r.get("dollars_at_chosen", {}).items():
            p = pa["prices_usd_per_1M_tokens"][model]
            out.append(
                f"| {r['name']} | {model} (${p[0]} / ${p[1]}) | ${v['cost_per_request_usd']:.6f} | "
                f"${v['saved_per_1M_requests_usd']:,.0f} | {v['wrong_answers_per_1M_requests']:,} |"
            )
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    print("## Load test\n")
    print(loadtest())
    print("## Semantic cache (QQP)\n")
    print(semantic())
