DATE=`date "+%Y-%m-%d"`
GPU_ID=$1
WAY=1
SHOT=$2
LR=5e-4
center=5
qcenter=5
name="PPNet_res101_Logp_additional"
output_dir="./$name"
sample_dir="$name-s$SHOT"
idx=0
for FOLD in $(seq 0 3)
do

gpu=$GPU_ID
python train_fewshot_dist_sep_additional.py with \
gpu_id=$gpu \
label_sets=$FOLD \
task.n_ways=$WAY \
task.n_shots=$SHOT \
optim.lr=$LR \
num_workers=8 \
eval=0 \
n_steps=4000 \
min_sigma=0.1 \
enc="res101" \
center=$center \
qcenter=$qcenter \
sample_dir=$sample_dir \
path.log_dir=$output_dir \
dist_ckpt_dir="./PPNet_res101_Logp/pretrained_24000" \
ckpt_dir="$output_dir/$DATE-voc-w$WAY-s$SHOT-lr$LR-F$FOLD" \
| tee logs/"$DATE-$name-voc-w$WAY-s$SHOT-lr$LR-part$center-F$FOLD".txt 
done