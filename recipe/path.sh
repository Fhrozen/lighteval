WORKDIR=$(readlink -e ${PWD}/../)

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:"${WORKDIR}/.venv/lib/python3.12/site-packages/torch/lib"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:"${WORKDIR}/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
source ${WORKDIR}/.venv/bin/activate
export OMP_NUM_THREADS=1

export PYTHONIOENCODING=UTF-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

# You need to change or unset NCCL_SOCKET_IFNAME according to your network environment
# https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html#nccl-socket-ifname
export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet"

# for debug
# export NCCL_P2P_LEVEL=2
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_TIMEOUT=22
# export TORCH_NCCL_BLOCKING_WAIT=0

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}

export -f log
export -f min

