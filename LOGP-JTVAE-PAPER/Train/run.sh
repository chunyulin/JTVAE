#!/bin/bash
### Qantum
source /raid/lincy/ENV/jtvae/bin/activate

JTVAE_GPU_PATH="/beegfs/home/lincy/JTVAE/JTVAE/GPU-P3"
export PYTHONPATH=$JTVAE_GPU_PATH

echo PORT: ${MASTER_PORT}

WRK=`pwd`
echo "WORK DIR: ${WRK}"
cd $WRK

#export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

JTVAE="${JTVAE_GPU_PATH}/fast_molvae/vae_train_gpu.py"
TRA="${HOME}/JTVAE/share_DR_LIN/fda_drug_train"
VOC="${HOME}/JTVAE/share_DR_LIN/all_fda_vocab.txt"

DIR=$(date +%s)
mkdir -p ${DIR}

CUDA_VISIBLE_DEVICES=6,7 torchrun --nnodes=1 --nproc_per_node=gpu \
   ${JTVAE} --train ${TRA} --vocab ${VOC} --save_dir ${DIR} \
            --mult_gpus --print_iter 1 --num_workers 4 \
            --epoch 1 --warmup 0 \
            --batch_size 25