export CUDA_VISIBLE_DEVICES=2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Set CUDA path
export VLLM_DISABLE_COMPILE_CACHE=1
export CUDA_HOME=/usr/local/cuda-12.8
export CUDADIR=/usr/local/cuda-12.8

# Add CUDA to PATH (for binaries)
export PATH=$CUDA_HOME/bin:$PATH

# Add CUDA to LD_LIBRARY_PATH (for libraries)
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_V1=1

MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Create a timestamped output directory for this run
timestamp=$(date +"%Y%m%d_%H%M%S")
output_dir="$SCRIPT_DIR/results/run_$timestamp"
mkdir -p "$output_dir"

# Save this shell script to the output directory for reproducibility
script_name=$(basename "$0")
cp "$0" "$output_dir/$script_name"

# Start timing
start_time=$(date +%s)

# Dataset configurations: dataset_name|dataset_path|output_len|num_prompts
DATASETS=(
    "hf|likaixin/InstructCoder|512|200"
    "sharegpt|/data/lily/ShareGPT_V3_unfiltered_cleaned_split.json|512|200"
    "hf|abisee/cnn_dailymail|512|200"
    "hf|openai/gsm8k|512|200"
)

# Speculative configurations: method_name|config_json
SPEC_CONFIGS=(
    'ngram|{"method": "ngram", "num_speculative_tokens": 20, "prompt_lookup_min": 3, "prompt_lookup_max": 7}'
    'eagle|{"method": "eagle", "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B", "num_speculative_tokens": 20}'
    'eagle3|{"method": "eagle3", "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "num_speculative_tokens": 20}'
)

# Loop over datasets
for dataset_config in "${DATASETS[@]}"; do
    IFS='|' read -r dataset_name dataset_path output_len num_prompts <<< "$dataset_config"

    # Extract a short name for the dataset for logging
    dataset_short=$(basename "$dataset_path" | sed 's/\.json$//')

    echo "========================================"
    echo "Dataset: $dataset_path"
    echo "========================================"

    # Loop over speculative configurations
    for spec_config in "${SPEC_CONFIGS[@]}"; do
        IFS='|' read -r spec_method spec_json <<< "$spec_config"

        log_file="$output_dir/${dataset_short}_${spec_method}.log"
        echo "====Running with method: $spec_method on dataset: $dataset_path" | tee -a "$log_file"
        run_start_time=$(date +%s)

        if python "$REPO_ROOT/benchmarks/benchmark_throughput.py" \
            --model "$MODEL" \
            --dataset-name "$dataset_name" \
            --dataset-path "$dataset_path" \
            --prefix-len 0 \
            --output-len "$output_len" \
            --num-prompts "$num_prompts" \
            --speculative_config "$spec_json" 2>&1 | tee -a "$log_file" > /dev/null; then
            run_end_time=$(date +%s)
            run_elapsed=$((run_end_time - run_start_time))
            echo "SUCCESS: $dataset_short, $spec_method, Time: ${run_elapsed}s" | tee -a "$output_dir/overview.log"
        else
            run_end_time=$(date +%s)
            run_elapsed=$((run_end_time - run_start_time))
            echo "FAILURE: $dataset_short, $spec_method, Time: ${run_elapsed}s" | tee -a "$output_dir/overview.log"
        fi
    done
done

# End timing and print elapsed time
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total benchmarking time: ${elapsed} seconds." | tee -a "$output_dir/overview.log"
