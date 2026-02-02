import argparse
import os
from collections import defaultdict
import json
import sys

# Add parent directory to path to import benchmarks module
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

SEED=42

from benchmarks.benchmark_dataset import (ShareGPTDataset,
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
        default="sharegpt",
        help="Dataset to use for generating requests.",
    )
    parser.add_argument(
        "--method",
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
    if args.dataset == "aime":
        dataset_path = "AI-MO/aimo-validation-aime"
        dataset = AIMODataset(
            dataset_path=dataset_path,
            dataset_subset=None,
            dataset_split="train",
            random_seed=SEED,
        )
    elif args.dataset == "instructcoder":
        dataset_path = "likaixin/InstructCoder"
        dataset = InstructCoderDataset(
            dataset_path=dataset_path,
            dataset_subset=None,
            dataset_split="train",
            random_seed=SEED,
        )
    elif args.dataset == "sharegpt":
        dataset_path = "/data/lily/ShareGPT_V3_unfiltered_cleaned_split.json" # local path
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