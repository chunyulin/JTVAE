#!/bin/bash
### Nano5
#ml python/3.13.8
#source /work/p00lcy01/ENV/jtvae/bin/activate
### Qantum
#ml python/3.13.8
#source /work/p00lcy01/ENV/jtvae/bin/activate



JTVAE_GPU_PATH="/work/p00lcy01/JTVAE/JTVAE/GPU-P3"
export PYTHONPATH=$JTVAE_GPU_PATH

echo PORT: ${MASTER_PORT}

WRK=`pwd`
echo "WORK DIR: ${WRK}"
cd $WRK

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

JTVAE="${JTVAE_GPU_PATH}/fast_molvae/vae_train_gpu.py"
#JTVAE="${JTVAE_GPU_PATH}/fast_molvae/test_data.py"
TRA="${HOME}/share_DR_LIN/fda_drug_train"
VOC="${HOME}/share_DR_LIN/all_fda_vocab.txt"

DIR=$(date +%s)
mkdir -p ${DIR}

#NSYS=/work/HPC_software/LMOD/nvidia/packages/hpc_sdk-24.7/Linux_x86_64/24.7/profilers/Nsight_Systems/bin/nsys
NSYS=/work/HPC_software/LMOD/nvidia/packages/cuda-12.6/nsight-systems-2024.5.1/bin/nsys

${NSYS} profile -t cuda,nvtx,cudnn,cublas --stats true --show-output=true  -o g2  \
torchrun --nnodes=${SLURM_NNODES} --nproc_per_node=gpu --node-rank=${SLURM_NODEID} \
         --master-addr=${MASTER_ADDR} --master-port=${MASTER_PORT} \
   ${JTVAE} --train ${TRA} --vocab ${VOC} --save_dir ${DIR} \
            --mult_gpus --print_iter 1 --num_workers 4 \
            --epoch 1 --warmup 0 \
            --batch_size 57