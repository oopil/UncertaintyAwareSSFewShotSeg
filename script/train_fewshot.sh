#!/usr/bin/env bash
# VOC 1-way 1-shot
output_dir="./outputs/PANet"
DATE=`date "+%Y-%m-%d"`
GPU_ID=$1
WAY=1
SHOT=$2
LR=5e-4
data="VOC"
#5e-4

for FOLD in $(seq 0 0)
do
python train_fewshot.py with \
dataset="$data" \
gpu_id=$GPU_ID \
mode='train' \
label_sets=$FOLD \
model.part=False \
task.n_ways=$WAY \
task.n_shots=$SHOT \
task.n_unlabels=0 \
optim.lr=$LR \
evaluate_interval=4000 \
infer_max_iters=1000 \
num_workers=8 \
n_steps=24000 \
eval=0 \
segments=True \
model.resnet=True \
model.sem=False \
ckpt_dir="$output_dir/$DATE-$data-w$WAY-s$SHOT-lr$LR-resnet-F$FOLD" \
| tee logs/"$DATE-$data-w$WAY-s$SHOT-lr$LR-resnet-F$FOLD".txt
done



