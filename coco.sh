#!/bin/bash

port=$(python get_free_port.py)
echo ${port}
alias exp='python -m torch.distributed.launch --nproc_per_node=2 --master_port ${port} run.py --num_workers 4 --sample_num 8'
shopt -s expand_aliases

dataset=coco-voc
epochs=30
task=voc
lr_init=0.01
lr=0.001

path=checkpoints/step/${dataset}-${task}/
dataset_pars="--dataset ${dataset} --task ${task} --batch_size 24 --epochs ${epochs} $ov --val_interval 2"

exp --name FTwide_bce --step 0 --lr ${lr_init} ${dataset_pars} --bce

pretr_FT=${path}FTwide_bce_0.pth
exp --name OURS --step 1 --weakly ${dataset_pars} --alpha 0.9 --lr ${lr} --step_ckpt $pretr_FT \
 --loss_de 1 --lr_policy warmup --affinity
