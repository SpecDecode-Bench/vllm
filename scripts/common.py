import argparse
import os
from collections import defaultdict
import json
import numpy as np

SEED=42

from vllm.benchmarks.datasets_bench import (ShareGPTDataset,
                                            CNNDailyMailDataset,
                                            GPQADataset,
                                            AIMODataset,
                                            InstructCoderDataset,
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
    args = parser.parse_args()
    return args

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
            request_id = hash(prompt)
            data[batch_size][request_id] = record
            record["id"] = len(data[batch_size]) - 1 # use index as id
    return data

def get_dataset(args):
    dataset = None
    # if args.dataset == "cnndailymail":
    #     dataset_path = "abisee/cnn_dailymail"
    #     dataset = CNNDailyMailDataset(
    #         dataset_path=dataset_path,
    #         dataset_subset="3.0.0",
    #         dataset_split="train",
    #         random_seed=SEED,
    #     )
    # elif args.dataset == "sharegpt":
    #     dataset_path = "/data/lily/ShareGPT_V3_unfiltered_cleaned_split.json"
    #     dataset = ShareGPTDataset(
    #         dataset_path=dataset_path,
    #         random_seed=SEED,
    #     )
    # elif args.dataset == "aime":
    #     dataset_path = "AI-MO/aimo-validation-aime"
    #     dataset = AIMODataset(
    #         dataset_path=dataset_path,
    #         dataset_subset=None,
    #         dataset_split="train",
    #         random_seed=SEED,
    #     )
    # elif args.dataset == "gpqa_main":
    #     dataset_path = "Idavidrein/gpqa"
    #     dataset = GPQADataset(
    #         dataset_path=dataset_path,
    #         dataset_subset="gpqa_main",
    #         dataset_split="train",
    #         random_seed=SEED,
    #     )
    # else:
    #     raise ValueError(f"Unknown dataset: {args.dataset}")
    # return dataset

    if args.dataset == "aime":
        dataset_path = "AI-MO/aimo-validation-aime"
        dataset = AIMODataset(
            dataset_path=dataset_path,
            dataset_subset=None,
            dataset_split="train",
            random_seed=SEED,
        )
    # elif args.dataset == "sonnet":
    #     dataset_path = "./vllm-benchmark/benchmarks/sonnet.txt"
    #     dataset = SonnetDataset(
    #         dataset_path=dataset_path,
    #         random_seed=SEED,
    #     )
    # elif args.dataset == "mtbench":
    #     dataset_path = "philschmid/mt-bench"
    #     dataset = MTBenchDataset(
    #         dataset_path=dataset_path,
    #         dataset_subset=None,
    #         dataset_split="train",
    #         random_seed=SEED,
    #     )
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
    # elif args.dataset == "blazedit5k":
    #     dataset_path = "vdaita/edit_5k_char"
    #     dataset = BlazeditDataset(
    #         dataset_path=dataset_path,
    #         dataset_split="train",
    #         random_seed=SEED,
    #     )
    # elif args.dataset == "blazedit10k":
    #     dataset_path = "vdaita/edit_10k_char"
    #     dataset = BlazeditDataset(
    #         dataset_path=dataset_path,
    #         dataset_split="train",
    #         random_seed=SEED,
    #     )
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
#   - Backward compatibility: Legacy 4-parameter API still supported
#
# Usage (New API - Recommended):
#   methods = {'org': org_data, 'ngram': ngram_data, 'eagle': eagle_data, 'new_method': new_data}
#   filtered = filter_by_stop_reason(methods)
#   filtered = filter_by_length_variance(filtered, enable_datasets={'gsm8k'})
#
# Usage (Legacy API - Backward Compatible):
#   org, ngram, eagle, eagle3 = filter_results_by_generation_length_and_stop_reason(
#       org_data, ngram_data, eagle_data, eagle3_data
#   )
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
                # Support both simple request_id and (dataset, batch_size, request_id) tuple keys
                simple_key = request_id
                complex_key = (record.get('dataset'), record.get('model'), batch_size, request_id)

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
    combined_length_map = defaultdict(set)

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
                    combined_length_map[key].add(output_len)

    # Identify high-variance requests
    removed_keys = set()
    variance_stats = defaultdict(list)

    for (dataset, model, batch_size, request_id), lengths in combined_length_map.items():
        if len(lengths) > 1:
            variance = max(lengths) - min(lengths)
            variance_stats[dataset, model].append(variance)
            if variance > length_threshold:
                removed_keys.add((dataset, model, batch_size, request_id))
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


def filter_by_length_variance(methods_data_or_org, ngram_data=None, eagle_data=None,
                              eagle3_data=None, enable_datasets=None, length_threshold=1000):
    """
    Filter requests based on output length variance across methods (Filtering Heuristic #2).

    For each (dataset, batch_size, request_id) tuple, compares output lengths across methods.
    Removes pairs where max(output_len) - min(output_len) > threshold.
    If variance is too high, the request is filtered from ALL methods.

    This is optional and can be selectively enabled per dataset, useful for identifying
    cases where different methods produce drastically different output lengths.

    Args:
        Signature 1 (new dict-based API):
            methods_data_or_org: Dict mapping method names to their data
                                Format: {method_name: {batch_size: {request_id: record}}}
            enable_datasets: Set/list of dataset names to enable filtering for.
            length_threshold: Maximum allowed variance in output length (default: 1000 tokens)

        Signature 2 (legacy 4-parameter API for backward compatibility):
            methods_data_or_org: org_data
            ngram_data, eagle_data, eagle3_data: Data dicts for other methods
            enable_datasets: Set/list of dataset names to enable filtering for.
            length_threshold: Maximum allowed variance in output length (default: 1000 tokens)

    Returns:
        Dict (new API) or tuple of 4 values (legacy API)

    Example (new API):
        >>> methods = {'org': org_data, 'ngram': ngram_data, 'eagle': eagle_data}
        >>> filtered = filter_by_length_variance(
        ...     methods,
        ...     enable_datasets={'gsm8k', 'cnndailymail'},
        ...     length_threshold=500
        ... )

    Example (legacy API):
        >>> org, ngram, eagle, eagle3 = filter_by_length_variance(
        ...     org_data, ngram_data, eagle_data, eagle3_data,
        ...     enable_datasets={'gsm8k'},
        ...     length_threshold=500
        ... )
    """
    # Detect which API is being used
    if ngram_data is not None or eagle_data is not None or eagle3_data is not None:
        # Legacy 4-parameter API
        methods_data = {
            "org": methods_data_or_org,
            "ngram": ngram_data,
            "eagle": eagle_data,
            "eagle3": eagle3_data,
        }
        is_legacy_api = True
    else:
        # New dict-based API
        methods_data = methods_data_or_org
        is_legacy_api = False

    if not enable_datasets:
        if is_legacy_api:
            return (methods_data.get("org"), methods_data.get("ngram"),
                    methods_data.get("eagle"), methods_data.get("eagle3"))
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

    if is_legacy_api:
        return (filtered.get("org"), filtered.get("ngram"),
                filtered.get("eagle"), filtered.get("eagle3"))
    return filtered


# ============================================================================
# Backward Compatibility Wrappers (Legacy 4-parameter API)
# ============================================================================

def filter_results_by_generation_length_and_stop_reason(org_data, ngram_data, eagle_data, eagle3_data):
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
    }

    filtered = filter_by_stop_reason(methods_data)

    return (filtered.get("org"), filtered.get("ngram"),
            filtered.get("eagle"), filtered.get("eagle3"))