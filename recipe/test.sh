#!/usr/bin/env bash

set -e
set -u
set -o pipefail

. ./path.sh

model_id=qwen3_0.3b
ngpus=-1
use_quant=false
model_args=""
is_vlm=false
max_num_batched_tokens=2048
use_tensor_parallel=false

. ./utils/parse_options.sh

if [ ${ngpus} -le -1 ]; then
    ngpus=$(nvidia-smi --list-gpus | wc -l)
fi

output_dir=$(basename ${model_id})
export HF_DATASETS_CACHE="./data"

# Seems to be required by vLLM on lm_eval
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# vllm_model=vllm
log "Evaluation of model ${output_dir} using vLLM-v1 for base tasks"

_model_args="model_name=${model_id}"
_model_args+=",dtype=bfloat16"
_model_args+=",max_num_batched_tokens=${max_num_batched_tokens}"
if ${use_tensor_parallel}; then
    _model_args+=",tensor_parallel_size=${ngpus}"
elif [ ${ngpus} -gt 1 ]; then
    _model_args+=",data_parallel_size=${ngpus}"
fi
if ${use_quant}; then
    _model_args+=",quantization=bitsandbytes"
fi
if [ -n ${model_args} ]; then
    _model_args+=",${model_args}"
fi

# Following are the kv_cache_memory for each model,
# to avoid OOM during the logprobs prediction, with a GPU VRAM=12G.
# | model-id   | kv_cache_memory_bytes | modality | gpu_memory_utilization |
# | Model 0.3B | --         | text | 0.85 |
# | Model 0.6B | 7410471372 | text |
# | Model 1.7B | 5015367116 | text |
# | Model 4B   | 3651455718 | text |

# lighteval|gpqa|5|0 < requires gpu utilization of 0.78
# custom|bbh|3|0 < takes longer time currently

lighteval vllm  \
    ${_model_args} \
    "local/base_llm.txt" \
    --custom-tasks "local/tasks.py" \
    --save-details \
    --output-dir exp/

# if ${is_vlm}; then
#     vllm_model=vllm-vlm
#     txt_tasks=mmlu,gsm8k,mmlu_pro,chartqa,mmmu_val

#     _model_args="model=${model_id},"
#     _model_args+="base_url=http://127.0.0.1:8000/v1/completions,"
#     _model_args+="num_concurrent=2,"
#     _model_args+="max_retries=3,"
#     _model_args+="tokenized_requests=False,"

#     log "Evaluation of model ${output_dir} using vLLM-v1 for task: ${txt_tasks}"
#     lm_eval --model local-completions \
#         --tasks ${txt_tasks} \
#         --log_samples \
#         --model_args ${_model_args} \
#         --output_path exp/
# else


