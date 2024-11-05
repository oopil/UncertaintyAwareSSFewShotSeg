#!/usr/bin/env bash
# VOC 1-way 1-shot
output_dir="./outputs/PANet"
# output_dir="./outputs/PANet_notFix"
# output_dir="./outputs/PANet_noalign"
DATE=`date "+%Y-%m-%d"`
GPU_ID=$1
WAY=1
SHOT=$2
LR=1e-3
# LR=5e-4
center=5


for FOLD in $(seq 0 3)
do
python train_fewshot.py with \
gpu_id=$GPU_ID \
mode='train' \
label_sets=$FOLD \
model.part=True \
task.n_ways=$WAY \
task.n_shots=$SHOT \
optim.lr=$LR \
evaluate_interval=4000 \
infer_max_iters=1000 \
num_workers=8 \
n_steps=24000 \
eval=0 \
model.resnet=True \
model.sem=False \
center=$center \
fix=False \
sample_dir="PANet_orig" \
ckpt_dir="$output_dir/$DATE-voc-w$WAY-s$SHOT-lr$LR-part$center-resnet-fix-F$FOLD" \
| tee logs/"$DATE-voc-w$WAY-s$SHOT-lr$LR-part$center-resnet-fix-F$FOLD".txt
done

#path.log_dir="$output_dir" \
#n_steps=24000 \
#align_loss_scaler=0 \ ##
#fix=True