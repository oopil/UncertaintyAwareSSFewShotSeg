#!/usr/bin/env bash
# VOC 1-way 1-shot
# use /bin/bash script/train_semi_sigma.sh 0 1 0 0 3


# name="PPNet_semi_train_w_sigma_un3_qproto"
DATE=`date "+%Y-%m-%d"`
GPU_ID=$1
WAY=1
SHOT=$2
LR=1e-3
center=5
qcenter=5
UN=$5
UN_test=$5
# name="PPNet_entropy_no_train_pseudo_qp_un$UN"
name="PPNet_entropy_no_train_w_sigma_qp_un$UN"
# name="PPNet_entropy_no_train_w_sigma_qp_max_un$UN"
output_dir="/data/soopil/FSS_uncertainty/outputs/$name"

if [ $3 = 0 ]
then
    gpu_list=(0 1 2 3)
else
    gpu_list=(4 5 6 7)
fi

if [ $4 = 1 ]
then
    sample_dir="$name-s$SHOT"
else
    sample_dir="try"
fi

idx=0
for FOLD in $(seq 0 3)
do

# gpu=${gpu_list[$idx]}
gpu=$GPU_ID
python train_fewshot_entropy_semi_sep.py with \
gpu_id=$gpu \
label_sets=$FOLD \
task.n_ways=$WAY \
task.n_shots=$SHOT \
task.n_unlabels=$UN \
test_unlabels=$UN_test \
optim.lr=$LR \
num_workers=8 \
eval=1 \
n_steps=8000 \
evaluate_interval=2000 \
min_sigma=0.1 \
is_pcm=False \
is_conv=False \
is_sigma=True \
is_qproto=True \
is_refine=False \
pre_refine=False \
refine_ckpt_dir="./outputs/PPNet_semi_3_refine_un0/pretrained_b" \
center=$center \
qcenter=$qcenter \
sample_dir=$sample_dir \
path.log_dir=$output_dir \
dist_ckpt_dir="./outputs/pretrained/PPNet_orig" \
ckpt_dir="$output_dir/$DATE-voc-w$WAY-s$SHOT-lr$LR-F$FOLD" \
| tee logs/"$DATE-$name-voc-w$WAY-s$SHOT-lr$LR-part$center-F$FOLD".txt

sleep 0.5
idx=$(($idx+1))
done

# dist_ckpt_dir="./outputs/pretrained/PPNet_Logp_sep_3sigma" \
# dist_ckpt_dir="./outputs/pretrained/PPNet_Logp_sep_3sigma_pcm_4th" \
# dist_ckpt_dir="./outputs/pretrained/PPNet_Logp_sep_1sigma_re" \
# dist_ckpt_dir="./outputs/pretrained/PPNet_Logp_sep_1sigma_pcm_wo_proj_3rdft" \
# dist_ckpt_dir="./outputs/pretrained/PPNet_Logp_sep_1sigma_conv" \
# dist_ckpt_dir="./outputs/PPNet_Logp_sep_1sigma/pretrained_b" \

# num_workers=8 \
# freeze_bn=True \
# mode='train' \
# model.part=True \
# model.resnet=True \
# model.sem=False \
# fix=False \
# sigma_interval=0.3 \
# infer_max_iters=1000 \


# sample_dir="try" \
# sample_dir="$name-s$SHOT" \
## default setting
# is_sigma=True \
# is_refine=True \
# is_single_proto=True \ - False is exclusive prototype
# is_sproto=False \
# is_qproto=True \
# sigma_interval=0.15 \
# dist_ckpt_dir="./outputs/PPNet_Logp/ppnet_logp_pretrained" \
# dist_ckpt_dir="./outputs/PPNet_Logp/ppnet_logp_pretrained_16000" \
# dist_ckpt_dir="./outputs/PPNet_Logp_2.5sigma/ppnet_logp_pretrained" \
# ckpt_dir="$output_dir/$DATE-voc-w$WAY-s$SHOT-lr$LR-F$FOLD" \

