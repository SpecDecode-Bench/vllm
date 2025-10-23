export CUDA_VISIBLE_DEVICES=2
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


VLLM_USE_V1=1 python examples/offline_inference/spec_decode.py \
    --model-dir Qwen/Qwen3-8B \
    --method none \
    --dataset-name hf \
    --dataset-path likaixin/InstructCoder \
    --num_prompts 20 \
    --temp 0.0 \
    --gpu-memory-utilization 0.9 \
    --seed 42 \
    --output_len 8192 \
    --print_output \
    --enable-chunked-prefill \


# VLLM_USE_V1=1 python examples/offline_inference/spec_decode.py \
#     --model-dir Qwen/Qwen3-8B \
#     --draft-model Qwen/Qwen3-0.6B \
#     --method draft_model \
#     --num_spec_tokens 3 \
#     --dataset-name hf \
#     --dataset-path likaixin/InstructCoder \
#     --num_prompts 20 \
#     --temp 0.0 \
#     --gpu-memory-utilization 0.9 \
#     --seed 42 \
#     --output_len 8192 \
#     --print_output \
#     --enable-chunked-prefill \
