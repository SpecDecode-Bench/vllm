import argparse
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import os

# k (num_spec_tokens) for each method
ks = {
    "none":            -1,
    "ngram":            3,
    "ngram5":           5,
    "eagle":            3,
    "eagle3":           3,
    "draft_model":      3,
    "mtp":              3,
}

# Method name as used in bench_latency.py output filename.
# ngram5 is ngram run with k=5; mtp maps to deepseek_mtp in bench_latency.py.
FILENAME_METHOD = {
    "ngram5": "ngram",
    "mtp":    "deepseek_mtp",
}

# Model filename string: matches model.replace('/', '_') in bench_latency.py
MODEL_FILENAME_STR = {
    "llama3-8b":  "meta-llama_Llama-3.1-8B-Instruct",
    "llama3-70b": "meta-llama_Meta-Llama-3-70B-Instruct",
    "qwen3-8b":   "Qwen_Qwen3-8B",
}

# Methods to plot per model (ngram5 is added dynamically for instructcoder)
MODEL_METHODS = {
    "llama3-8b":  ["ngram", "eagle", "eagle3"],
    "llama3-70b": ["ngram", "eagle", "draft_model"],
    "qwen3-8b":   ["ngram", "eagle3", "mtp"],
}

markers = {
    "ngram":           "s",
    "ngram5":          "o",
    "eagle":           "D",
    "eagle3":          "^",
    "draft_model":     "v",
    "mtp":             "p",
}
colors = {
    "ngram":           "#1f77b4",
    "ngram5":          "#1f77b4",
    "eagle":           "#ff7f0e",
    "eagle3":          "#2ca02c",
    "draft_model":     "tab:purple",
    "mtp":             "#e377c2",
}
method_display = {
    "ngram":           "N-gram (k=3)",
    "ngram5":          "N-gram (k=5)",
    "eagle":           "EAGLE (k=3)",
    "eagle3":          "EAGLE-3 (k=3)",
    "draft_model":     "Draft Model",
    "mtp":             "MTP",
}


def get_file(results_dir: str, method: str, dataset: str, model_key: str) -> str:
    model_str = MODEL_FILENAME_STR[model_key]
    filename_method = FILENAME_METHOD.get(method, method)
    k = ks[method]
    return f"{results_dir}/latency_{dataset}_{filename_method}_num-spec-tokens-{k}_all_{model_str}.jsonl"


def load_method_data(method: str, dataset: str, model_key: str,
                     results_dir: str, max_reqs: int = 500):
    """Load benchmark data for a method.

    Returns {batch_size: {request_id: record}}, or None if file is missing.
    Only 'stop'-finished requests are kept, capped at max_reqs per batch size.
    """
    filename = get_file(results_dir, method, dataset, model_key)
    if not os.path.exists(filename):
        print(f"  File not found, skipping {method}: {filename}")
        return None

    raw_data = defaultdict(dict)
    skipped = defaultdict(int)
    with open(filename) as f:
        for line in f:
            d = json.loads(line)
            bs = d["batch_size"]
            if d["finished_reason"] != "stop":
                skipped[bs] += 1
                continue
            if len(raw_data[bs]) >= max_reqs:
                continue
            request_id = hash(d.get("prompt", ""))
            raw_data[bs][request_id] = d

    total = sum(len(v) for v in raw_data.values())
    skip_str = dict(skipped) if skipped else "none"
    print(f"  {method}: {total} reqs loaded, skipped={skip_str}")
    for bs, reqs in raw_data.items():
        if len(reqs) < max_reqs:
            print(f"    Warning: only {len(reqs)}/{max_reqs} reqs for batch_size={bs}")
    return dict(raw_data)


def compute_throughput(method_data):
    """Returns {batch_size: avg_tokens_per_sec}, or None."""
    if method_data is None:
        return None
    result = {}
    for bs, reqs in method_data.items():
        tpts = [sum(d["output_len_lists"]) / d["duration"] for d in reqs.values()]
        result[bs] = sum(tpts) / len(tpts)
    return result


def draw(results_dir: str, dataset: str, model_key: str, figures_dir: str):
    # ngram5 is only available for instructcoder
    methods_to_plot = ["ngram", "eagle", "eagle3"]
    if dataset == "instructcoder":
        methods_to_plot = ["ngram", "ngram5", "eagle", "eagle3"]

    print(f"\n{'='*50}")
    print(f"Dataset: {dataset}  Model: {model_key}")
    print(f"{'='*50}")

    # Load baseline and all methods
    baseline_tpt = compute_throughput(load_method_data("none", dataset, model_key, results_dir))
    if baseline_tpt is None:
        print(f"  No baseline data, skipping.")
        return

    method_tpt = {}
    for method in methods_to_plot:
        tpt = compute_throughput(load_method_data(method, dataset, model_key, results_dir))
        if tpt is not None:
            method_tpt[method] = tpt

    if not method_tpt:
        print(f"  No method data, skipping.")
        return

    # Common batch sizes across all methods and the baseline
    common_bs = set(baseline_tpt.keys())
    for tpt in method_tpt.values():
        common_bs &= set(tpt.keys())
    if not common_bs:
        print(f"  No common batch sizes, skipping.")
        return
    common_bs = sorted(common_bs)
    print(f"  Common batch sizes: {common_bs}")

    # Plot speedups
    plt.figure(figsize=(3.5, 2.5))
    ylim = 2.2
    for method in methods_to_plot:
        if method not in method_tpt:
            continue
        tpt = method_tpt[method]
        xs, speedups = [], []
        for bs in common_bs:
            if bs in baseline_tpt and bs in tpt:
                s = tpt[bs] / baseline_tpt[bs]
                if s > ylim:
                    ylim = s + 0.1
                xs.append(bs)
                speedups.append(s)
        if speedups:
            print(f"  Speedup {method}: {[f'{s:.3f}' for s in speedups]}")
            plt.plot(xs, speedups,
                     marker=markers[method],
                     color=colors[method],
                     label=method_display[method],
                     markersize=10)

    fontsize = 10
    plt.ylim(bottom=0.9, top=ylim)
    plt.xticks(common_bs, common_bs, size=fontsize)
    plt.xlabel("Batch Size", fontsize=fontsize)
    plt.ylabel("Speedup", fontsize=fontsize)
    plt.tight_layout()
    plt.grid(True)

    os.makedirs(figures_dir, exist_ok=True)
    out_path = f"{figures_dir}/speedup_{model_key}_{dataset}.pdf"
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def save_legend(figures_dir: str):
    from matplotlib.lines import Line2D
    methods = ["ngram", "ngram5", "eagle", "eagle3"]
    artists = [
        Line2D([0], [0], marker=markers[m], color=colors[m],
               linestyle="-", markersize=10, linewidth=2)
        for m in methods
    ]
    labels = [method_display[m] for m in methods]

    _, ax = plt.subplots(figsize=(0.1, 0.1))
    ax.legend(artists, labels, fontsize=12, ncol=len(methods), loc="center", frameon=True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    os.makedirs(figures_dir, exist_ok=True)
    out_path = f"{figures_dir}/legend.pdf"
    plt.savefig(out_path, format="pdf", bbox_inches="tight", transparent=True)
    plt.close()
    print(f"Saved legend: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot speedup from bench_latency.py results."
    )
    parser.add_argument("--results_dir", required=True,
                        help="Directory containing bench_latency.py output .jsonl files")
    parser.add_argument("--model_key", default="llama3-8b",
                        choices=list(MODEL_FILENAME_STR.keys()),
                        help="Model identifier key")
    parser.add_argument("--datasets", nargs="+",
                        default=["instructcoder", "cnndailymail", "sharegpt", "gsm8k"],
                        help="Datasets to plot")
    parser.add_argument("--figures_dir", default=None,
                        help="Output directory for figures (default: {results_dir}/figures)")
    args = parser.parse_args()

    figures_dir = args.figures_dir or f"{args.results_dir}/figures"

    for dataset in args.datasets:
        draw(args.results_dir, dataset, args.model_key, figures_dir)

    save_legend(figures_dir)
