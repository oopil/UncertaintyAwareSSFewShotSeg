DATE=`date "+%Y-%m-%d"`
GPU_ID=$1
WAY=1
SHOT=$2
LR=5e-4
center=5
qcenter=5
UN=$3
UN_test=$3

name="PPNet_res101_Logp_additional_semi_w_sigma_qp_un$UN"
output_dir="./$name"

if [ $4 = 1 ]
then
    sample_dir="$name-s$SHOT"
else
    sample_dir="try"
fi

idx=0
for FOLD in $(seq 0 3)
do
gpu=$GPU_ID
python train_fewshot_semi_sep_adv_iter.py with \
gpu_id=$gpu \
label_sets=$FOLD \
task.n_ways=$WAY \
task.n_shots=$SHOT \
task.n_unlabels=$UN \
test_unlabels=$UN_test \
optim.lr=$LR \
num_workers=8 \
eval=1 \
n_steps=4000 \
evaluate_interval=2000 \
min_sigma=0.1 \
enc="res101" \
is_sigma=True \
is_qproto=True \
is_refine=False \
center=$center \
qcenter=$qcenter \
sample_dir=$sample_dir \
path.log_dir=$output_dir \
dist_ckpt_dir="./PPNet_res101_Logp_additional/pretrained_best" \
ckpt_dir="$output_dir/$DATE-voc-w$WAY-s$SHOT-lr$LR-F$FOLD" \
| tee logs/"$DATE-$name-voc-w$WAY-s$SHOT-lr$LR-part$center-F$FOLD".txt
done