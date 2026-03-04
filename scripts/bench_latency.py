import sys
import os
import json
import time

# IMPORTANT: Set environment variables BEFORE importing vllm
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V1"] = "1"
print(f"VLLM_ENABLE_V1_MULTIPROCESSING: {os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING')}"
      , flush=True)
print(f"VLLM_USE_V1: {os.environ.get('VLLM_USE_V1')}"
      , flush=True)

from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Gauge, Histogram, Vector

from common import (parse_args,
                    get_eagle_model,
                    get_output_filename,
                    get_dataset,
                    SEED)
SEED = 42

def get_speculative_token_tree(num_spec_tokens):
    if num_spec_tokens == 4:
        return "[(0,), (1,), (0,0), (1,0)]"
    # elif num_spec_tokens == 6:
    #     return "[(0,), (1,), (0,0), (1,0), (0,1), (1,1)]"
    elif num_spec_tokens == 6:
        # ROOT
        # ├── 0
        # │  ├── 0
        # │      └── 0
        # └── 1
        #    ├── 0
        #        └── 0
        return "[(0,), (1,), (0, 0), (1, 0), (0, 0, 0), (1, 0, 0)]"
    elif num_spec_tokens == 10:
        # ROOT
        # ├── 0
        # │  ├── 0
        # │  │   └── 0
        # │  └── 1
        # │      └── 0
        # └── 1
        #    ├── 0
        #    │  └── 0
        #    └── 1
        #       └── 0
        return "[(0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1), (0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]"
    elif num_spec_tokens == 14:
        # ROOT
        # ├── 0
        # │  ├── 0
        # │  │   └── 0
        # │  │   └── 1
        # │  └── 1
        # │      └── 0
        # │      └── 1
        # └── 1
        #    ├── 0
        #    │  └── 0
        #    │  └── 1
        #    └── 1
        #       └── 0
        #       └── 1
        return "[(0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1), (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]"
    else:
        raise ValueError(f"Unsupported number of speculative tokens: {num_spec_tokens}")

def get_llm(args):
    if args.method == "none":
        speculative_config = None
    elif args.method == "ngram":
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": args.num_spec_tokens,
            "prompt_lookup_max": 7,
            "prompt_lookup_min": 3,
        }
    elif args.method in ["eagle", "eagle3"]:
        speculative_config={
            "method": args.method,
            "model": get_eagle_model(args.model, args.method == "eagle3"),
            "num_speculative_tokens": args.num_spec_tokens,
            "speculative_token_tree": get_speculative_token_tree(args.num_spec_tokens),
        }
    elif args.method == "draft_model":
        assert args.draft_model is not None and args.draft_model != ""
        speculative_config = {
            "method": args.method,
            "model": args.draft_model,
            "num_speculative_tokens": args.num_spec_tokens,
            "disable_padded_drafter_batch": True,
            "max_model_len": args.max_model_len,
        }
    elif args.method == "deepseek_mtp" and args.model == "zai-org/GLM-4.5-Air":
        speculative_config = {
            "method": args.method,
            "model": args.model,
            "num_speculative_tokens": args.num_spec_tokens,
        }
    elif args.method == "qwen3_next_mtp":
        speculative_config = {
            "method": args.method,
            "num_speculative_tokens": args.num_spec_tokens,
        }
    elif args.method == "suffix":
        speculative_config = {
            "method": "suffix",
            "num_speculative_tokens": args.num_spec_tokens if args.num_spec_tokens > 0 else 32,
        }
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    def _get_tp_size(model):
        if any(string in model for string in ["70", "80", "GLM", "Next"]):
            return 4
        elif "32" in model:
            return 2
        else:
            return 1

    llm = LLM(
            model=args.model,
            tensor_parallel_size=_get_tp_size(args.model),
            speculative_config=speculative_config,
            disable_log_stats=False,
            enable_prefix_caching=False,
            seed=SEED,
            dtype="bfloat16",
            trust_remote_code=True,
            enforce_eager=False, # NOTE: we default to use Cuda Graph
        )
    return llm

if __name__ == "__main__":
    args = parse_args()
    tokenizer = get_tokenizer(args.model,
                            tokenizer_mode="auto",
                            trust_remote_code=False)
    llm = get_llm(args)
    dataset = get_dataset(args)

    NUM_REQUESTS = args.num_reqs
    BATCH_SIZES = args.batch_sizes
    sampling_params = SamplingParams(temperature=0.0,
                                 max_tokens=args.max_tokens*1024,
                                 ignore_eos=False)

    if args.is_warmup:
        all_requests = dataset.sample(
            num_requests=1,
            tokenizer=tokenizer,
            output_len=None,
        )
        input_request = all_requests[0]
        prompts = [input_request.prompt] * 5
        _ = llm.generate(prompts, sampling_params, use_tqdm=False)
        print("Warmup done.")
        exit(0)

    # Actual run
    all_requests = dataset.sample(
            num_requests=NUM_REQUESTS,
            tokenizer=tokenizer,
            output_len=None,
        )

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    for batch_size in BATCH_SIZES:
        latencies = []
        for i in range(NUM_REQUESTS):
            input_request = all_requests[i]
            prompts = [input_request.prompt] * batch_size
            start_time = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
            end_time = time.perf_counter()
            duration = end_time - start_time

            if args.method == "none":
                # No spec decoding, set to None
                num_drafts = None
                num_draft_tokens = None
                num_accepted_tokens = None
                acceptance_counts = None
                avg_acceptance_rate = None
                acceptance_length = None
                acceptance_rate_per_pos = None
            else:
                try:
                    metrics = llm.get_metrics()
                except AssertionError as e:
                    print(f"ERROR: Failed to get metrics: {e}")
                    print(f"log_stats value: {llm.llm_engine.log_stats}")
                    print(f"Skipping this request...")
                    continue

                total_num_output_tokens = sum(
                    len(output.outputs[0].token_ids) for output in outputs
                )
                num_drafts = 0
                num_draft_tokens = 0
                num_accepted_tokens = 0
                effective_num_spec_tokens = args.num_spec_tokens if args.num_spec_tokens > 0 else 32
                acceptance_counts = [0] * effective_num_spec_tokens
                for metric in metrics:
                    if metric.name == "vllm:spec_decode_num_drafts":
                        assert isinstance(metric, Counter)
                        num_drafts += metric.value
                    elif metric.name == "vllm:spec_decode_num_draft_tokens":
                        assert isinstance(metric, Counter)
                        num_draft_tokens += metric.value
                    elif metric.name == "vllm:spec_decode_num_accepted_tokens":
                        assert isinstance(metric, Counter)
                        num_accepted_tokens += metric.value
                    elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
                        assert isinstance(metric, Vector)
                        for pos in range(len(metric.values)):
                            acceptance_counts[pos] += metric.values[pos]

                acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1
                avg_acceptance_rate = num_accepted_tokens / num_draft_tokens if num_draft_tokens > 0 else 0.0

                # print("-" * 50)
                # print(f"total_num_output_tokens: {total_num_output_tokens}")
                # print(f"num_drafts: {num_drafts}")
                # print(f"num_draft_tokens: {num_draft_tokens}")
                # print(f"num_accepted_tokens: {num_accepted_tokens}")
                # print(f"mean acceptance length: {acceptance_length:.2f}")
                # print("-" * 50)

                # print acceptance at each token position
                acceptance_rate_per_pos = [0.0] * effective_num_spec_tokens
                for i in range(len(acceptance_counts)):
                    acceptance_rate_per_pos[i] = acceptance_counts[i] / num_drafts if num_drafts > 0 else 0
                    # print(f"acceptance at token {i}: {acceptance_rate:.2f}")

            # Basic result
            result = {
                "duration": duration,
                "batch_size": batch_size,
                "model": args.model,
                "method": args.method,
                "dataset": args.dataset,
                "prompt_len": input_request.prompt_len,
                "output_len": len(outputs[0].outputs[0].token_ids),
                "output_len_stats": {
                    "mean": sum([len(o.outputs[0].token_ids) for o in outputs]) / len(outputs),
                    "min": min([len(o.outputs[0].token_ids) for o in outputs]),
                    "max": max([len(o.outputs[0].token_ids) for o in outputs]),
                },
                "mean_acceptance_length": acceptance_length,
                "acceptance_rate": avg_acceptance_rate,
                "num_draft_tokens": num_draft_tokens,
                "num_accepted_tokens": num_accepted_tokens,
                "num_accepted_tokens_per_pos": acceptance_counts,
                "acceptance_rate_per_pos": acceptance_rate_per_pos,
                "num_drafts": num_drafts,
                "output_len_lists": [len(o.outputs[0].token_ids) for o in outputs],
                "finished_reason": outputs[0].outputs[0].finish_reason,
                "prompt": input_request.prompt,
                "generated_output": outputs[0].outputs[0].text,
            }

            # Save basic result
            with open(get_output_filename(args), "a") as f:
                f.write(json.dumps(result))
                f.write("\n")

            latencies.append(duration)
        print(f"Average latency: {sum(latencies) / len(latencies)} seconds")

    # wait for a short while to ensure all logs are flushed
    time.sleep(3)