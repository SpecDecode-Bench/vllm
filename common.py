import argparse
import os
from collections import defaultdict
import json
import sys
import numpy as np

# TODO: make it configurable
# Only for profiling
# VLLM_PATH="/data/jerry/jerry-vllm-0.10.1-bench/vllm-edit/vllm/benchmarks"
# VLLM_PATH="/data/jerry/jerry-vllm-0.10.1-bench/vllm-edit/vllm/benchmarks"
# VLLM_PATH="/data/lily/sd-benchmark-paper/vllm-benchmark/benchmarks/"
# sys.path.append(VLLM_PATH)
SEED=42

# Shared constants
FIELD_TO_LABELS = {
    "duration": "Request Latency (s)",
    "output_len": "Output Length (tokens)",
}

# Standardized colors for speculative decoding methods
# Used across all plotting scripts for consistency
METHOD_COLORS = {
    "none": "#808080",      # Gray - baseline (no SD)
    "org": "#808080",       # Gray - baseline (no SD)
    "N-gram": "#2E86AB",    # Blue - N-gram
    "ngram": "#2E86AB",     # Blue - N-gram (alternative key)
    "EAGLE": "#F18F01",     # Orange - EAGLE
    "eagle": "#F18F01",     # Orange - EAGLE (alternative key)
    "EAGLE3": "#A23B72",    # Purple - EAGLE3
    "eagle3": "#A23B72",    # Purple - EAGLE3 (alternative key)
    "MTP": "#C73E1D",       # Red - MTP
    "mtp": "#C73E1D",       # Red - MTP (alternative key)
    "Draft Model": "#06A77D", # Green - Draft Model
    "draft_model": "#06A77D", # Green - Draft Model (alternative key)
}

def get_method_color(method_name: str, default: str = "#666666") -> str:
    """
    Get standardized color for a method name.

    Args:
        method_name: Name of the method (e.g., "N-gram", "EAGLE", "ngram")
        default: Default color to return if method not found

    Returns:
        Hex color string
    """
    return METHOD_COLORS.get(method_name, default)

from benchmarks.benchmark_dataset import (AIMODataset,
                               ShareGPTDataset,
                               SonnetDataset,
                               MTBenchDataset,
                               InstructCoderDataset,
                               CNNDailyMailDataset,
                               BlazeditDataset,
                               GPQADataset,
                               GSM8KDataset)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name or path.",
    )
    parser.add_argument(
        "--dataset",
        # choices=["mtbench", "instructcoder", "sharegpt", "cnndailymail"],
        default="sharegpt",
        help="Dataset to use for generating requests.",
    )
    parser.add_argument(
        "--method",
        # choices=["none", "ngram", "eagle", "eagle3"], # TODO: add MTP, add eagle3-tree
        default="ngram",
        help="Speculative decoding method to use.",
    )
    parser.add_argument(
        "--num_spec_tokens",
        type=int,
        default=-1,
        help="Number of speculative tokens to use.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save the results.",
    )
    parser.add_argument(
        "--num_reqs",
        type=int,
        default=500,
        help="Number of requests to use for latency benchmarking.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=32,
        help="Maximum number of thousand (in units of 1024) tokens (used in sampling parameter).",
    )
    # batch size for time breakdown
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs='+',
        default=[1, 8, 16, 32, 64, 128],
        help="List of batch sizes for latency breakdown.",
    )

    # Warmup only
    parser.add_argument("--is_warmup", action="store_true", default=False)
    # Plotting only
    parser.add_argument("--method_spec_tokens", type=str, default='{"none": [-1], "ngram": [5], "eagle": [3], "eagle3": [3]}', help="Only for Plotting; JSON string mapping methods to list of num_spec_tokens.")
    parser.add_argument('--distance_json', type=str)
    parser.add_argument('--min_dist', type=float)
    parser.add_argument('--max_dist', type=float)
    # BLEU score options
    parser.add_argument('--bleu_n', type=int, default=4,
                        help="BLEU-N score to use (e.g., 4 for BLEU-4). Default: 4")
    parser.add_argument('--ngram_only', action='store_true', default=False,
                        help="If set, use only the specified n-gram (e.g., --bleu_n 3 --ngram_only gives (0,0,1) for 3-gram only)")
    args = parser.parse_args()
    return args

def get_bleu_weights(bleu_n, ngram_only=False):
    """
    Convert BLEU-N specification to weights for sentence_bleu.

    Args:
        bleu_n: BLEU-N score (1-8), e.g., 4 for BLEU-4
        ngram_only: If True, use only the specified n-gram (e.g., only 3-gram)

    Returns:
        tuple: Weights for sentence_bleu function (compact, no trailing zeros)

    Examples:
        BLEU-1: (1,)
        BLEU-2: (0.5, 0.5)
        BLEU-3: (1/3, 1/3, 1/3)
        BLEU-4: (0.25, 0.25, 0.25, 0.25)
        3-gram only: (0, 0, 1)
        5-gram only: (0, 0, 0, 0, 1)
    """

    if ngram_only:
        # Only the specified n-gram gets weight 1, all others get 0
        # We need weights up to position n (e.g., for 3-gram only: (0, 0, 1))
        weights = tuple([1.0 if i == bleu_n - 1 else 0.0 for i in range(bleu_n)])
    else:
        # Cumulative: all n-grams from 1 to n get equal weights
        # We only need weights up to position n
        weights = tuple([1/bleu_n for i in range(bleu_n)])

    return weights

def get_bleu_name(bleu_n, ngram_only=False):
    """
    Get a string name for the BLEU configuration.

    Args:
        bleu_n: BLEU-N score (1-8)
        ngram_only: If True, indicates only a single n-gram is used

    Returns:
        str: Name like "bleu4" or "3gram-only"
    """
    if ngram_only:
        return f"{bleu_n}gram-only"
    else:
        return f"bleu{bleu_n}"


def get_eagle_model(model, use_eagle3=False):
    if model == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":
        return "yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B"
    elif model == "meta-llama/Llama-3.1-8B-Instruct":
        if use_eagle3:
            return "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
        else:
            return "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
    elif model == "meta-llama/Meta-Llama-3-8B-Instruct":
        if use_eagle3:
            raise ValueError("EAGLE3 is not supported for Meta-Llama-3-8B-Instruct.")
        else:
            return "yuhuili/EAGLE-LLaMA3-Instruct-8B"
    elif model == "meta-llama/Meta-Llama-3-70B-Instruct":
        if use_eagle3:
            raise ValueError("EAGLE3 is not supported for meta-llama/Meta-Llama-3-70B-Instruct model.")
        else:
            return "yuhuili/EAGLE-LLaMA3-Instruct-70B"
    elif model == "Qwen/Qwen3-8B":
        if use_eagle3:
            return "AngelSlim/Qwen3-8B_eagle3"
        else:
            raise ValueError("Please use EAGLE3 for Qwen3.")
    elif model == "Qwen/Qwen3-32B":
        if use_eagle3:
            return "AngelSlim/Qwen3-32B_eagle3"
        else:
            raise ValueError("Please use EAGLE3 for Qwen3.")
    else:
        raise ValueError(f"Unsupported model for EAGLE/EAGLE3: {model}.")

def get_output_filename(args):
    return f"{args.results_dir}/latency_{args.dataset}_{args.method}_num-spec-tokens-{args.num_spec_tokens}_all_{args.model.replace('/', '_')}.jsonl"

def load_data(filename):
    data = defaultdict(dict)
    if os.path.exists(filename) == False:
        print(f"File {filename} does not exist.")
        return None

    with open(filename, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            record = json.loads(line)
            batch_size = record["batch_size"]
            prompt = record["prompt"]
            model = record["model"]
            request_id = hash(model+prompt)
            data[batch_size][request_id] = record
            record["id"] = len(data[batch_size]) - 1 # use index as id
    return data

def _get_token_counts(record):
    """Get total token count (prompt + output) for a record."""
    return record.get("prompt_len") + record.get("output_len_stats").get("mean")

def _get_output_len(record):
    return record.get("output_len")

def _get_avg_output_len(record):
    return record.get("output_len_stats").get("mean")

def _get_all_output_token_counts(record):
    return sum(record.get("output_len_lists"))

def compute_avg_throughput_first_n(data, field, n=None):
    """
    Compute average throughput considering only the first n requests.

    Args:
        data: Dictionary mapping request_id to records
        field: Field name to average (e.g., 'duration')
        n: Number of requests to consider. If None, use all requests.

    Returns:
        Average throughput (tokens per second) for the first n requests
    """

    return sum([_get_token_counts(record)/record[field] for _, record in data]) / len(data)

def compute_avg_latency_per_token_first_n(data, field, n=None):
    """
    Compute average latency per token considering only the first n requests.

    Args:
        data: Dictionary mapping request_id to records
        field: Field name to average (e.g., 'duration')
        n: Number of requests to consider. If None, use all requests.

    Returns:
        Average latency per token for the first n requests
    """

    return sum([record[field]/_get_token_counts(record) for _, record in data]) / len(data)

def compute_avg_latency_first_n(data, field, n=None):
    """
    Compute average latency considering only the first n requests.

    Args:
        data: Dictionary mapping request_id to records
        field: Field name to average (e.g., 'duration')
        n: Number of requests to consider. If None, use all requests.

    Returns:
        Average value of the specified field for the first n requests
    """
    items = get_first_n_items(data, n)

    if not items:
        return None

    return sum([record[field] for _, record in items]) / len(items)

def get_dataset(args):
    dataset = None
    if args.dataset == "aime":
        dataset_path = "AI-MO/aimo-validation-aime"
        dataset = AIMODataset(
            dataset_path=dataset_path,
            dataset_subset=None,
            dataset_split="train",
            random_seed=SEED,
        )
    elif args.dataset == "sonnet":
        dataset_path = "./vllm-benchmark/benchmarks/sonnet.txt"
        dataset = SonnetDataset(
            dataset_path=dataset_path,
            random_seed=SEED,
        )
    elif args.dataset == "mtbench":
        dataset_path = "philschmid/mt-bench"
        dataset = MTBenchDataset(
            dataset_path=dataset_path,
            dataset_subset=None,
            dataset_split="train",
            random_seed=SEED,
        )
    elif args.dataset == "instructcoder": # MINOR: it's actually called instructcoder instead of instructcode
        dataset_path = "likaixin/InstructCoder"
        dataset = InstructCoderDataset(
            dataset_path=dataset_path,
            dataset_subset=None,
            dataset_split="train",
            random_seed=SEED,
        )
    elif args.dataset == "sharegpt":
        dataset_path = "/data/lily/ShareGPT_V3_unfiltered_cleaned_split.json"
        dataset = ShareGPTDataset(
            dataset_path=dataset_path,
            random_seed=SEED,
        )
    elif args.dataset == "cnndailymail":
        dataset_path = "abisee/cnn_dailymail"
        dataset = CNNDailyMailDataset(
            dataset_path=dataset_path,
            dataset_subset="3.0.0",
            dataset_split="train",
            random_seed=SEED,
        )
    elif args.dataset == "blazedit5k":
        dataset_path = "vdaita/edit_5k_char"
        dataset = BlazeditDataset(
            dataset_path=dataset_path,
            dataset_split="train",
            random_seed=SEED,
        )
    elif args.dataset == "blazedit10k":
        dataset_path = "vdaita/edit_10k_char"
        dataset = BlazeditDataset(
            dataset_path=dataset_path,
            dataset_split="train",
            random_seed=SEED,
        )
    elif args.dataset == "gsm8k":
        dataset_path = "openai/gsm8k"
        dataset = GSM8KDataset(
            dataset_path=dataset_path,
            dataset_subset="main",
            dataset_split="train",
            random_seed=SEED,
        )
    elif args.dataset == "gpqa_main":
        dataset_path = "Idavidrein/gpqa"
        dataset = GPQADataset(
            dataset_path=dataset_path,
            dataset_subset="gpqa_main",
            dataset_split="train",
            random_seed=SEED,
        )
    elif args.dataset == "gpqa_extended":
        dataset_path = "Idavidrein/gpqa"
        dataset = GPQADataset(
            dataset_path=dataset_path,
            dataset_subset="gpqa_extended",
            dataset_split="train",
            random_seed=SEED,
        )
    elif args.dataset == "gpqa_diamond":
        dataset_path = "Idavidrein/gpqa"
        dataset = GPQADataset(
            dataset_path=dataset_path,
            dataset_subset="gpqa_diamond",
            dataset_split="train",
            random_seed=SEED,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    return dataset


# ============================================================================
# Generic Cross-Method Filtering Framework
# ============================================================================
#
# This framework provides two filtering heuristics for benchmark results:
#   1. Stop Reason Filter: Removes requests with abnormal termination
#   2. Length Variance Filter: Removes requests with inconsistent output lengths
#
# Key Design Principles:
#   - Cross-method consistency: If a request fails in ANY method, filter from ALL
#   - Extensible API: Use dict-based methods_data for easy addition of new methods
#   - Separation of concerns: Identification logic separate from filtering logic
#
# Usage:
#   methods = {'org': org_data, 'ngram': ngram_data, 'eagle': eagle_data, 'new_method': new_data}
#   filtered = filter_by_stop_reason(methods)
#   filtered = filter_by_length_variance(filtered, enable_datasets={'gsm8k'})
#
# ============================================================================

def apply_cross_method_filter(methods_data, should_filter_func, filter_name="filter"):
    """
    Generic framework for cross-method filtering with consistent behavior.

    Core principle: If a request has issues in ANY method, filter it from ALL methods.

    Args:
        methods_data: Dict mapping method names to their data
                     Format: {method_name: {batch_size: {request_id: record}}}
        should_filter_func: Function that determines which requests to filter.
                           Signature: (methods_data) -> set of keys to remove
                           Keys can be request_ids or (dataset, batch_size, request_id) tuples
        filter_name: Name of the filter for logging purposes

    Returns:
        Dict of filtered data with same structure as input
    """
    print(f"\nApplying {filter_name}...")

    # Get the set of keys (request_ids or tuples) to remove
    keys_to_remove = should_filter_func(methods_data)

    print(f"Total items to filter: {len(keys_to_remove)}")
    print(f"keys to remove sample: {list(keys_to_remove)[:5]}")

    # Filter all methods
    filtered_methods = {}
    for method_name, method_data in methods_data.items():
        if not method_data:
            filtered_methods[method_name] = None
            continue

        filtered_data = defaultdict(dict)
        for batch_size, requests in method_data.items():
            for request_id, record in requests.items():
                # Check if this request should be filtered
                # Support both simple request_id and (dataset, model, request_id) tuple keys
                simple_key = request_id
                complex_key = (record.get('dataset'), record.get('model'), request_id)

                if simple_key not in keys_to_remove and complex_key not in keys_to_remove:
                    filtered_data[batch_size][request_id] = record

        filtered_methods[method_name] = filtered_data

    print(f"{filter_name} complete.\n")
    return filtered_methods


# ============================================================================
# Filtering Heuristic #1: Stop Reason Filter
# ============================================================================

def _identify_abnormal_stop_requests(methods_data):
    """
    Identify requests with abnormal stop reasons across all methods.

    Returns:
        Set of request_ids that should be filtered
    """
    abnormal_stopped_requests = set()
    abnoraml_stopped_reasons = set()
    combined_output_length_map = defaultdict(set)

    print("Scanning for abnormal stop reasons...")
    for method_name, method_data in methods_data.items():
        if not method_data:
            print(f"  No data for method: {method_name}")
            continue

        for batch_size, requests in method_data.items():
            for request_id, record in requests.items():
                combined_output_length_map[request_id].add(
                    record.get("output_len_stats", {}).get("mean", 0)
                )
                if record.get("finished_reason") != "stop":
                    abnormal_stopped_requests.add(request_id)
                    abnoraml_stopped_reasons.add(record.get("finished_reason"))
                    print(f"  Method {method_name}, batch_size {batch_size}, request {request_id}: "
                          f"finished_reason={record.get('finished_reason')}, "
                          f"output_len={record.get('output_len_stats', {}).get('mean', 0)}")


    print(f"  Total unique requests: {len(combined_output_length_map)}")
    print(f"  Requests with abnormal stop: {len(abnormal_stopped_requests)}")
    print(f"  Abnormal stop reasons: {abnoraml_stopped_reasons}")
    return abnormal_stopped_requests


def filter_by_stop_reason(methods_data):
    """
    Filter requests based on stop reason (Filtering Heuristic #1).

    Removes requests where generation terminated due to reaching max output length.
    Specifically filters out requests where finished_reason != "stop".
    If a request has abnormal stop in ANY method, it's filtered from ALL methods.

    Args:
        methods_data: Dict mapping method names to their data
                     Format: {method_name: {batch_size: {request_id: record}}}

    Returns:
        Dict of filtered data with same structure as input

    Example:
        >>> methods = {'org': org_data, 'ngram': ngram_data, 'eagle': eagle_data}
        >>> filtered = filter_by_stop_reason(methods)
        >>> org_filtered = filtered['org']
    """
    return apply_cross_method_filter(
        methods_data,
        _identify_abnormal_stop_requests,
        filter_name="Stop Reason Filter"
    )


# ============================================================================
# Filtering Heuristic #2: Length Variance Filter
# ============================================================================

def _identify_high_variance_requests(methods_data, enable_datasets, length_threshold):
    """
    Identify requests with high output length variance across methods within the same batch size.

    Returns:
        Set of (dataset, batch_size, request_id) tuples that should be filtered
    """
    combined_length_map = defaultdict(list)

    # Collect output lengths per (dataset, batch_size, request_id)
    for method_name, method_data in methods_data.items():
        if not method_data:
            continue
        for batch_size, requests in method_data.items():
            for request_id, record in requests.items():
                dataset = record.get('dataset')
                model = record.get('model')
                if dataset in enable_datasets:
                    key = (dataset, model, batch_size, request_id)
                    output_len = record.get("output_len_stats", {}).get("mean", 0)
                    combined_length_map[key].append(output_len)

    # Identify high-variance requests
    removed_keys = set()
    variance_stats = defaultdict(list)

    for (dataset, model, batch_size, request_id), lengths in combined_length_map.items():
        if len(lengths) > 1:
            variance = max(lengths) - min(lengths)
            variance_stats[dataset, model].append(variance)
            if variance > length_threshold:
                # NOTE: if one request fails in ANY method and ANY bach size, we remove it from ALL methods and ALL batch sizes
                removed_keys.add((dataset, model, request_id))
                print(f"  Dataset {dataset}, model {model}, batch_size {batch_size}, request {request_id}: "
                      f"variance={variance:.1f} tokens (lengths: {sorted(lengths)})")

    # Print summary statistics
    if variance_stats:
        print("\n  Variance statistics by dataset and model:")
        for dataset, model in sorted(variance_stats.keys()):
            variances = variance_stats[dataset, model]
            print(f"    {dataset}, {model}: checked={len(variances)}, "
                  f"mean_var={np.mean(variances):.1f}, max_var={max(variances):.1f}")

    return removed_keys


def filter_by_length_variance(methods_data, enable_datasets=None, length_threshold=1000):
    """
    Filter requests based on output length variance across methods (Filtering Heuristic #2).

    For each (dataset, batch_size, request_id) tuple, compares output lengths across methods.
    Removes pairs where max(output_len) - min(output_len) > threshold.
    If variance is too high, the request is filtered from ALL methods.

    This is optional and can be selectively enabled per dataset, useful for identifying
    cases where different methods produce drastically different output lengths.

    Args:
        Signature 1 (new dict-based API):
            methods_data: Dict mapping method names to their data
                                Format: {method_name: {batch_size: {request_id: record}}}
            enable_datasets: Set/list of dataset names to enable filtering for.
            length_threshold: Maximum allowed variance in output length (default: 1000 tokens)

    Returns:
        Dict (new API)

    Example (new API):
        >>> methods = {'org': org_data, 'ngram': ngram_data, 'eagle': eagle_data}
        >>> filtered = filter_by_length_variance(
        ...     methods,
        ...     enable_datasets={'gsm8k', 'cnndailymail'},
        ...     length_threshold=500
        ... )

    """
    # New dict-based API
    is_legacy_api = False

    if not enable_datasets:
        return methods_data

    if isinstance(enable_datasets, list):
        enable_datasets = set(enable_datasets)

    print(f"  Enabled datasets: {enable_datasets}")
    print(f"  Length threshold: {length_threshold} tokens")

    filtered = apply_cross_method_filter(
        methods_data,
        lambda data: _identify_high_variance_requests(data, enable_datasets, length_threshold),
        filter_name="Length Variance Filter"
    )

    return filtered


# ============================================================================
# Backward Compatibility Wrappers
# ============================================================================

def filter_results_by_generation_length_and_stop_reason(org_data, ngram_data, eagle_data, eagle3_data, mtp_data, draft_model_data,
                                                        length_threshold=1000):
    """
    Legacy wrapper for stop reason filter. Use filter_by_stop_reason() for new code.

    Args:
        org_data, ngram_data, eagle_data, eagle3_data: Data dicts for each method

    Returns:
        Tuple of filtered data (org, ngram, eagle, eagle3)
    """
    methods_data = {
        "org": org_data,
        "ngram": ngram_data,
        "eagle": eagle_data,
        "eagle3": eagle3_data,
        "mtp": mtp_data,
        "draft_model": draft_model_data,
    }

    filtered = filter_by_stop_reason(methods_data)

    filtered = filter_by_length_variance(
        filtered,
        enable_datasets={'gsm8k', 'cnndailymail', 'gpqa_main', 'sharegpt', "instructcoder", "aime"},
        length_threshold=length_threshold
    )


    return (filtered.get("org"), filtered.get("ngram"),
            filtered.get("eagle"), filtered.get("eagle3"),
            filtered.get("mtp"), filtered.get("draft_model"))


# ============================================================================
# Plotting Utilities
# ============================================================================

def plot_batch_field(dataset,
                     model,
                     field,
                     org_data,
                     ngram_data,
                     eagle_data,
                     eagle3_data,
                     out_dir="figures/paper",
                     mtp_data=None,
                     draft_model_data=None,
                     first_n_requests=None,
                     use_per_token=True,
                     min_dist=None,
                     max_dist=None,
                     abs_min_dist=None,
                     abs_max_dist=None):
    """
    Unified plotting function for speedup comparison across different methods.

    Args:
        dataset: Dataset name for labeling
        model: Model name for labeling (will be sanitized)
        field: Field to analyze ('duration', 'output_len', etc.)
        org_data: Original method data
        ngram_data: N-gram speculative decoding data
        eagle_data: Eagle speculative decoding data
        eagle3_data: Eagle3 speculative decoding data
        mtp_data: MTP (Medusa-Tree-based Pruning) data (optional)
        draft_model_data: Draft model data (optional)
        first_n_requests: Number of requests to consider (None = all)
        use_per_token: If True, compute latency per token; otherwise use absolute latency
        min_dist: Minimum distance threshold for filtering (optional, for BLEU filtering)
        max_dist: Maximum distance threshold for filtering (optional, for BLEU filtering)
        abs_min_dist: Absolute minimum distance (for display in title)
        abs_max_dist: Absolute maximum distance (for display in title)

    Returns:
        None (saves figure to file)
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError('matplotlib is required to plot: ' + str(e))

    plt.figure(figsize=(5, 4.5))

    # Determine batch sizes based on available data
    if draft_model_data is not None:
        all_batch_sizes = sorted(list(draft_model_data.keys())) if draft_model_data else []
    elif mtp_data is not None:
        all_batch_sizes = sorted(list(mtp_data.keys())) if mtp_data else []
    else:
        all_batch_sizes = sorted(list(org_data.keys())) if org_data else []

    org_latencies = []
    ngram_latencies = []
    eagle_latencies = []
    eagle3_latencies = []
    mtp_latencies = []
    draft_model_latencies = []

    # Choose computation function based on use_per_token flag
    compute_func = compute_avg_latency_per_token_first_n if use_per_token else compute_avg_latency_first_n

    for batch_size in all_batch_sizes:
        # Calculate the average latency for each method
        org_latency = compute_func(org_data[batch_size], field, first_n_requests)
        ngram_latency = compute_func(ngram_data[batch_size], field, first_n_requests) if ngram_data else None
        eagle_latency = compute_func(eagle_data[batch_size], field, first_n_requests) if eagle_data else None
        eagle3_latency = compute_func(eagle3_data[batch_size], field, first_n_requests) if eagle3_data else None
        mtp_latency = compute_func(mtp_data[batch_size], field, first_n_requests) if mtp_data else None
        draft_model_latency = compute_func(draft_model_data[batch_size], field, first_n_requests) if draft_model_data else None

        # Compute speedup: Latency without SD divided by Latency with SD
        org_latencies.append(org_latency)
        ngram_latencies.append(org_latency / ngram_latency if ngram_latency else None)

        # Choose the better speedup between Eagle and Eagle3
        if eagle_latency and eagle3_latency:
            best_eagle_speedup = max(org_latency / eagle_latency, org_latency / eagle3_latency)
        elif eagle_latency:
            best_eagle_speedup = org_latency / eagle_latency
        elif eagle3_latency:
            best_eagle_speedup = org_latency / eagle3_latency
        else:
            best_eagle_speedup = None

        eagle_latencies.append(best_eagle_speedup)
        mtp_latencies.append(org_latency / mtp_latency if mtp_latency else None)
        draft_model_latencies.append(org_latency / draft_model_latency if draft_model_latency else None)

        print(org_latency, ngram_latency, eagle_latency, eagle3_latency, mtp_latency, draft_model_latency)

    # Plotting speedup with bigger markers and thicker lines
    plt.plot(all_batch_sizes, ngram_latencies, label="Ngram", marker='s', markersize=8, linewidth=2)
    if eagle_data or eagle3_data:
        plt.plot(all_batch_sizes, eagle_latencies, label="Eagle", marker='*', markersize=8, linewidth=2)
    if mtp_data:
        plt.plot(all_batch_sizes, mtp_latencies, label="MTP", marker='^', markersize=8, linewidth=2)
    if draft_model_data:
        plt.plot(all_batch_sizes, draft_model_latencies, label="Draft Model", marker='o', markersize=8, linewidth=2)

    sample_size = len(next(iter(org_data.values()))) if org_data else 0
    plt.legend()
    plt.ylim(bottom=0.9)
    plt.xlabel(f"Batch Size (n={sample_size})", fontsize=12)
    if dataset == "instructcoder":
        plt.ylabel("Speedup", fontsize=12)
    plt.tight_layout()
    plt.grid()

    # Handle title and output path based on distance filtering
    if min_dist is not None and max_dist is not None:
        plt.title(f"Speedup (dist_range in [{min_dist}, {max_dist}], abs_dist in [{abs_min_dist:.4f}, {abs_max_dist:.4f}])",
                  fontsize=10, y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        out_path = f"{out_dir}/{dataset}_{model}_{field}_speedup_comparison_dist-{min_dist}-{max_dist}.png"
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        file_ext = "pdf" if use_per_token else "png"
        out_path = f"{out_dir}/{dataset}_{model}_{field}_speedup_comparison.{file_ext}"

    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    plt.savefig(out_path)
    print(f"Saved figure to {out_path}")
    plt.close()