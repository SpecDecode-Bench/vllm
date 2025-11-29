export CUDA_VISIBLE_DEVICES=3

export VLLM_DISABLE_COMPILE_CACHE=1
# Set CUDA path
export CUDA_HOME=/usr/local/cuda-12.8
export CUDADIR=/usr/local/cuda-12.8

# Add CUDA to PATH (for binaries)
export PATH=$CUDA_HOME/bin:$PATH

# Add CUDA to LD_LIBRARY_PATH (for libraries)
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export PATH=/data/lily/miniconda3/envs/draft-sd-vllm/bin:$PATH

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_V1=1

# model=meta-llama/Llama-3.1-8B-Instruct
# model="meta-llama/Meta-Llama-3-70B-Instruct"
model="Qwen/Qwen3-8B"

# Create a timestamped output directory for this run
timestamp=$(date +"%Y%m%d_%H%M%S")
output_dir="results/run_$timestamp"
mkdir -p "$output_dir"

# Save this shell script to the output directory for reproducibility
script_name=$(basename "$0")
cp "$0" "$output_dir/$script_name"

# Start timing
start_time=$(date +%s)

# Set default values for num_reqs and max_tokens
num_reqs="500"
max_tokens="8"
# batch_sizes="1 8 16 32 64 128"
batch_sizes="1 16 64 128"

# Warmup run
python bench_latency.py --model "$model" \
                         --method "none"  \
                         --dataset "sharegpt" \
                         --num_spec_tokens "-1" \
                         --num_reqs "$num_reqs" \
                         --max_tokens "$max_tokens" \
                         --is_warmup 2>&1 | tee "$output_dir/warmup.log" > /dev/null
echo "Warmup done."

for dataset in cnndailymail gsm8k
do
    for method in none
    do
        # Set possible spec_tokens values for each method
        if [ "$method" = "draft_model" ]; then
            spec_tokens_list="3"
        elif [ "$method" = "none" ]; then
            spec_tokens_list="-1"
        fi

        for num_spec_tokens in $spec_tokens_list
        do
            log_file="$output_dir/${dataset}_${method}_${num_spec_tokens}.log"
            echo "====Running with method: $method on dataset: $dataset, num_spec_tokens: $num_spec_tokens" | tee -a "$log_file"
            run_start_time=$(date +%s)

            if [ "$method" = "none" ]; then
                # Run without draft model for method "none"
                if python bench_latency.py --model "$model" \
                    --method "$method" \
                    --dataset "$dataset" \
                    --results_dir "$output_dir" \
                    --num_spec_tokens "$num_spec_tokens" \
                    --num_reqs "$num_reqs" \
                    --batch_sizes $batch_sizes \
                    --max_tokens "$max_tokens" 2>&1 | tee -a "$log_file" > /dev/null; then
                    run_end_time=$(date +%s)
                    run_elapsed=$((run_end_time - run_start_time))
                    echo "SUCCESS: $dataset, $method, $num_spec_tokens, Time: ${run_elapsed}s" | tee -a "$output_dir/overview.log"
                else
                    run_end_time=$(date +%s)
                    run_elapsed=$((run_end_time - run_start_time))
                    echo "FAILURE: $dataset, $method, $num_spec_tokens, Time: ${run_elapsed}s" | tee -a "$output_dir/overview.log"
                fi
            else
                # Run with draft model for other methods
                if python bench_latency.py --model "$model" \
                    --draft_model Qwen/Qwen3-0.6B \
                    --method "$method" \
                    --dataset "$dataset" \
                    --results_dir "$output_dir" \
                    --num_spec_tokens "$num_spec_tokens" \
                    --num_reqs "$num_reqs" \
                    --batch_sizes $batch_sizes \
                    --max_tokens "$max_tokens" 2>&1 | tee -a "$log_file" > /dev/null; then
                    run_end_time=$(date +%s)
                    run_elapsed=$((run_end_time - run_start_time))
                    echo "SUCCESS: $dataset, $method, $num_spec_tokens, Time: ${run_elapsed}s" | tee -a "$output_dir/overview.log"
                else
                    run_end_time=$(date +%s)
                    run_elapsed=$((run_end_time - run_start_time))
                    echo "FAILURE: $dataset, $method, $num_spec_tokens, Time: ${run_elapsed}s" | tee -a "$output_dir/overview.log"
                fi
            fi
        done
    done
done

# End timing and print elapsed time
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total profiling time: ${elapsed} seconds." | tee -a "$output_dir/overview.log"