import json
import argparse
from dataclasses import dataclass

MODEL_TO_NAMES = {
    "r1-distill-llama-8B" : "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "llama3-8B" : "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3.1-8B" : "meta-llama/Llama-3.1-8B-Instruct",
    "llama3-70B" : "meta-llama/Meta-Llama-3-70B",
    "qwen3-8B" : "Qwen/Qwen3-8B",
}

@dataclass
class AccStats:
    lens: list[int]
    probs: list[float] = None
    entropies: list[float] = None

    def __post_init__(self):
        if self.probs is not None:
            assert len(self.lens) == len(self.probs), "Length of lens and probs must match"
        if self.entropies is not None:
            assert len(self.lens) == len(self.entropies), "Length of lens and entropies must match"

        # remove the prefill accepted lens
        self.lens = self.lens[1:]

        # remove the last proposed tokens
        if self.probs:
            self.probs = self.probs[:-1]
        if self.entropies:
            self.entropies = self.entropies[:-1]

    @property
    def length(self):
        return len(self.lens)

# def cleanup(acc_stats: AccStats) ->
#     # Remove the prefill phase
#     data = data[1:]
#     # Cap the maximum value to 10
#     data = [min(x, 10) for x in data]
#     return data

def load_data(datapath, tokenizer, verbose=False):
    acceptance_stats = []
    with open(datapath, "r") as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            probs = data['acc'].get('acc_prob', None)
            entropies = data['acc'].get('acc_entropy', None)
            if probs and probs[0] == []: # we return empty list for ngram
                probs = None
            if entropies and entropies[0] == []: # we return empty list for ngram
                entropies = None
            stat = AccStats(
                lens=data['acc']['acc_len'],
                probs=probs,
                entropies=probs
            )
            acceptance_stats.append(stat)
            if verbose:
                print("Input:", tokenizer.decode(data['prompt_token_ids']))
                print("Output:", tokenizer.decode(data['generated_token_ids']))
                print("=============================================")

    max_length = max(stats.length for stats in acceptance_stats)

    print(f"Load {len(acceptance_stats)} with max length {max_length}")
    return acceptance_stats

def parse_args():
    """Parse command-line arguments for acceptance statistics visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize acceptance statistics for speculative decoding"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_TO_NAMES.keys()),
        help="Model name to analyze"
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["ngram", "eagle", "eagle3", "draft_model"],
        help="Decoding method (e.g., eagle, ngram)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["gsm8k", "cnndailymail", "instructcoder", "sharegpt", "aime", "gqpa-main"],
        help="Dataset name (e.g., gsm8k, mtbench, cnndailymail)"
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default=None,
        required=True,
        help="Path to the acceptance stats JSONL file. If not provided, uses default path: /data/lily/batch-sd/data/{model}/{method}_{dataset}_acceptance_stats.jsonl"
    )
    return parser.parse_args()