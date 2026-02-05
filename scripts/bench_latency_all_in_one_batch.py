import sys
import os
import json
import time

# IMPORTANT: Set environment variables BEFORE importing vllm
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
        }
    elif args.method == "draft_model":
        assert args.draft_model is not None and args.draft_model != ""
        speculative_config = {
            "method": args.method,
            "model": args.draft_model,
            "num_speculative_tokens": args.num_spec_tokens,
            "disable_padded_drafter_batch": True,
            # "max_model_len": args.max_model_len,
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

    # For this experiment, we assume batch size is one and we will fit all requests in one batch.
    assert BATCH_SIZES == [1], "For acc benchmarking, the batch size is assumed to be 1."
    prompts = [req.prompt for req in all_requests]
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Duration: {duration} seconds")

    # wait for a short while to ensure all logs are flushed
    time.sleep(3)