output_dir="./PPNet_res101_Logp"
DATE=`date "+%Y-%m-%d"`
GPU_ID=$1
WAY=1
SHOT=$2
LR=5e-4
center=5


for FOLD in $(seq 0 3)
do
gpu=$GPU_ID
python train_fewshot_dist.py with \
gpu_id=$gpu \
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
min_sigma=0.3 \
enc="res101" \
freeze_bn=True \
eval=0 \
model.resnet=True \
model.sem=False \
center=$center \
fix=False \
sample_dir="PPNet_res101_Logp_s$SHOT" \
path.log_dir=$output_dir \
ckpt_dir="$output_dir/$DATE-voc-w$WAY-s$SHOT-lr$LR-F$FOLD" \
| tee logs/"$DATE-voc-w$WAY-s$SHOT-lr$LR-part$center-F$FOLD".txt
done