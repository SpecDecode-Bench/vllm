# Set CUDA path
export CUDA_HOME=/usr/local/cuda-12.8
export CUDADIR=/usr/local/cuda-12.8

# Add CUDA to PATH (for binaries)
export PATH=$CUDA_HOME/bin:$PATH

# Add CUDA to LD_LIBRARY_PATH (for libraries)
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# export PATH=/data/lily/miniconda3/envs/draft-sd-vllm/bin:$PA

# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=0

MAX_CONCURRENCY=1

vllm bench serve \
    --model Qwen/Qwen3-8B \
    --dataset-name hf \
    --dataset-path likaixin/InstructCoder \
    --num-prompts 20 \
    --seed 42 \
    --max-concurrency ${MAX_CONCURRENCY} \
    --temperature 0.0 \
    --top-p 1.0 \
    --skip-chat-template