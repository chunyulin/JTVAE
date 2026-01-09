#!/bin/bash
### Qantum
source /raid/lincy/ENV/jtvae/bin/activate

JTVAE_GPU_PATH="../../JTVAE/GPU-P3"
export PYTHONPATH=$JTVAE_GPU_PATH

echo PORT: ${MASTER_PORT}

WRK=`pwd`
echo "WORK DIR: ${WRK}"
cd $WRK

#export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

JTVAE="${JTVAE_GPU_PATH}/fast_molvae/vae_train_gpu.py"
TRA="/beegfs/home/lincy/JTVAE_data/fda_drug_train"
VOC="/beegfs/home/lincy/JTVAE_data/all_fda_vocab.txt"
#TRA="/beegfs/home/lincy/JTVAE_data/oom"
#VOC="/beegfs/home/lincy/JTVAE_data/all_vocab_passed_2.txt"


DIR=$(date +%s)
mkdir -p ${DIR}

#NSYS=/opt/nvidia/nsight-systems/2024.7.1/bin/nsys
#${NSYS} profile -t cuda,nvtx,osrt,cudnn -s cpu --delay=15 --stats=false -w true -f true -o jtvae_g1  \

CUDA_VISIBLE_DEVICES=7 \
torchrun --nnodes=1 --nproc_per_node=gpu \
   ${JTVAE} --train ${TRA} --vocab ${VOC} --save_dir ${DIR} \
            --mult_gpus --print_iter 1 --num_workers 4 \
            --epoch 1 --warmup 0 \
            --batch_size 57 --use_amp