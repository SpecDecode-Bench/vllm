export CUDA_VISIBLE_DEVICES=2
export VLLM_DISABLE_COMPILE_CACHE=1
# Set CUDA path
export CUDA_HOME=/usr/local/cuda-12.8
export CUDADIR=/usr/local/cuda-12.8

# Add CUDA to PATH (for binaries)
export PATH=$CUDA_HOME/bin:$PATH

# Add CUDA to LD_LIBRARY_PATH (for libraries)
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# export PATH=/data/lily/miniconda3/envs/draft-sd-vllm/bin:$PA

# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=0

# vllm serve Qwen/Qwen3-8B \
#     --speculative_config.method draft_model \
#     --speculative_config.model Qwen/Qwen3-0.6B \
#     --speculative_config.num_speculative_tokens 3 \
#     --speculative_config.disable_padded_drafter_batch true \
#     --no-enable-prefix-caching \
#     --disable-uvicorn-access-log

vllm serve Qwen/Qwen3-8B \
    --no-enable-prefix-caching \
    --disable-uvicorn-access-log