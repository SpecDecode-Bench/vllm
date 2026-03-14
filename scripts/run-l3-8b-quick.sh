#!/bin/bash
# Profile Llama-3.1-8B-Instruct with all speculative decoding methods,
# then generate speedup plots.
# Usage: bash run-l3-8b-quick.sh

export CUDA_VISIBLE_DEVICES=0
export VLLM_DISABLE_COMPILE_CACHE=1
export CUDA_HOME=/usr/local/cuda-12.8
export CUDADIR=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_V1=1

model="meta-llama/Llama-3.1-8B-Instruct"
num_reqs=100
max_tokens=8
batch_sizes="1 16 64 128"

# Ensure ShareGPT dataset is available
SHAREGPT_FILE="data/ShareGPT_V3_unfiltered_cleaned_split.json"
if [ ! -f "$SHAREGPT_FILE" ]; then
    echo "Downloading ShareGPT dataset..."
    mkdir -p data
    huggingface-cli download anon8231489123/ShareGPT_Vicuna_unfiltered \
        ShareGPT_V3_unfiltered_cleaned_split.json \
        --repo-type dataset --local-dir data/
fi
export SHAREGPT_PATH="$SHAREGPT_FILE"

timestamp=$(date +"%Y%m%d_%H%M%S")
output_dir="results/run_${timestamp}"
mkdir -p "$output_dir"
cp "$0" "$output_dir/$(basename $0)"
start_time=$(date +%s)

# Warmup
python bench_latency.py --model "$model" --method "none" --dataset "instructcoder" \
    --num_spec_tokens -1 --num_reqs "$num_reqs" --max_tokens "$max_tokens" --is_warmup \
    2>&1 | tee "$output_dir/warmup.log" > /dev/null
echo "Warmup done."

declare -A spec_tokens_map
spec_tokens_map["none"]="-1"
spec_tokens_map["ngram"]="3"
spec_tokens_map["eagle"]="3"
spec_tokens_map["eagle3"]="3"

# for dataset in instructcoder cnndailymail sharegpt gsm8k; do
for dataset in gsm8k; do
    for method in none ngram eagle eagle3; do
        # ngram5 (ngram k=5) is instructcoder-only
        k_list="${spec_tokens_map[$method]}"
        if [ "$method" = "ngram" ] && [ "$dataset" = "instructcoder" ]; then
            k_list="3 5"
        fi
        for k in $k_list; do
            log_file="$output_dir/${dataset}_${method}_${k}.log"
            echo "==== $dataset | $method | k=$k" | tee -a "$log_file"
            run_start=$(date +%s)
            if python bench_latency.py --model "$model" \
                --method "$method" --dataset "$dataset" \
                --results_dir "$output_dir" \
                --num_spec_tokens "$k" --num_reqs "$num_reqs" \
                --batch_sizes $batch_sizes --max_tokens "$max_tokens" \
                2>&1 | tee -a "$log_file" > /dev/null; then
                echo "SUCCESS: $dataset $method k=$k  time=$(($(date +%s) - run_start))s" \
                    | tee -a "$output_dir/overview.log"
            else
                echo "FAILURE: $dataset $method k=$k  time=$(($(date +%s) - run_start))s" \
                    | tee -a "$output_dir/overview.log"
            fi
        done
    done
done

elapsed=$(($(date +%s) - start_time))
echo "Total profiling time: ${elapsed}s" | tee -a "$output_dir/overview.log"

# Generate plots
echo "Generating plots..."
python vis_speedup.py --results_dir "$output_dir" --model_key llama3-8b --datasets gsm8k
echo "Done. Results in $output_dir, figures in $output_dir/figures/"
