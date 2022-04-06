#!/bin/bash

# We assume to have a parameter indicating whether to use overlap (0 or 1)
port=$(python get_free_port.py)
echo ${port}
alias exp='python -m torch.distributed.launch --nproc_per_node=2 --master_port ${port} run.py --num_workers 4 --sample_num 8'
shopt -s expand_aliases
overlap=$1

dataset=voc
epochs=40
task=15-5
lr_init=0.01
lr=0.001

if [ ${overlap} -eq 0 ]; then
  path=checkpoints/step/${dataset}-${task}/
  ov=""
else
  path=checkpoints/step/${dataset}-${task}-ov/
  ov="--overlap"
  echo "Overlap"
fi

dataset_pars="--dataset ${dataset} --task ${task} --batch_size 24 $ov --val_interval 2"
exp --name FT_bce --step 0 --bce --lr ${lr_init} ${dataset_pars}  --epochs 30

pretr_FT=${path}FT_bce_0.pth

exp --name OURS --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr_FT \
 --loss_de 1 --lr_policy warmup --affinity --epochs ${epochs}
