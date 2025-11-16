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

export CUDA_VISIBLE_DEVICES=4,5,6,7


VLLM_USE_V1=1 python examples/offline_inference/spec_decode.py \
    --model-dir meta-llama/Meta-Llama-3-70B-Instruct \
    --draft-model meta-llama/Llama-3.2-1B \
    --method draft_model \
    --num_spec_tokens 3 \
    --dataset-name sharegpt \
    --dataset-path /data/lily/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num_prompts 128 \
    --temp 0.0 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --output-len 128 \
    --tp 4 \
    --enforce-eager
