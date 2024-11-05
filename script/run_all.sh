
source activate torch
cd tmp/semi_med_seg/codes/t9_set_transform/

source activate torch
cd /data/soopil/FSS_uncertainty

watch -n 1 gpustat

/bin/bash script/train_semi_sigma_sep.sh 0 1 0 0 1

/bin/bash script/train_semi_sigma_sep.sh 0 1 0 0 3 &; sleep 3; /bin/bash script/train_semi_sigma_sep.sh 0 5 0 0 3

/bin/bash script/train_semi_sigma_sep.sh 0 5 0 1 0 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 0 1 1 1 0 &
sleep 5

/bin/bash script/train_coco_logp.sh 0 5 0 &
sleep 5
/bin/bash script/train_coco_logp.sh 0 1 1 &
sleep 5

/bin/bash script/train_logp_sep.sh 0 1 0 &
sleep 30
/bin/bash script/train_logp_sep.sh 0 5 0 &
sleep 5

/bin/bash script/train_logp_sep_semi_sep.sh 0 1 0 1 0 &
sleep 8
/bin/bash script/train_logp_sep_semi_sep.sh 1 5 0 0 0 &
sleep 8
/bin/bash script/train_logp_sep_semi_sep.sh 2 1 0 0 6 &
sleep 8
/bin/bash script/train_logp_sep_semi_sep.sh 3 5 0 0 6 &
sleep 8

/bin/bash script/train_logp_sep_semi_sep.sh 6 1 0 0 6 &
sleep 10
/bin/bash script/train_logp_sep_semi_sep.sh 7 5 0 0 6 &
sleep 10

## additional decoder training
/bin/bash script/train_logp_additional.sh 0 5 0 &
sleep 8
/bin/bash script/train_logp_additional.sh 0 1 1 &
sleep 8


## original semi-supervision
/bin/bash script/train_semi_sigma_sep.sh 0 1 0 0 0 &
sleep 8
/bin/bash script/train_semi_sigma_sep.sh 0 5 0 0 0 &
sleep 8
/bin/bash script/train_semi_sigma_sep.sh 4 1 0 0 3 &
sleep 8
/bin/bash script/train_semi_sigma_sep.sh 7 5 0 0 3 &
sleep 8
/bin/bash script/train_semi_sigma_sep.sh 0 1 0 1 6 &
sleep 8
/bin/bash script/train_semi_sigma_sep.sh 0 5 1 0 6 &
sleep 8
/bin/bash script/train_semi_sigma_sep.sh 6 1 0 0 9 &
sleep 8
/bin/bash script/train_semi_sigma_sep.sh 5 5 0 0 9 &
sleep 8
/bin/bash script/train_semi_sigma_sep.sh 7 1 0 0 12 &
sleep 8
/bin/bash script/train_semi_sigma_sep.sh 4 5 0 0 12 &
sleep 8

/bin/bash script/train_semi_sigma_sep.sh 1 1 0 0 0 &
sleep 8
/bin/bash script/train_semi_sigma_sep.sh 0 5 0 0 0 &
sleep 8
/bin/bash script/train_semi_sigma_sep.sh 0 1 0 0 3 &
sleep 8
/bin/bash script/train_semi_sigma_sep.sh 1 5 1 0 3 &
sleep 8

/bin/bash script/train_semi_sigma_sep.sh 0 1 0 0 6 &
sleep 8
/bin/bash script/train_semi_sigma_sep.sh 0 5 1 0 6 &
sleep 8

/bin/bash script/train_semi_sigma_sep.sh 4 1 0 0 9 &
sleep 
/bin/bash script/train_semi_sigma_sep.sh 5 5 0 0 9 &
sleep 8
/bin/bash script/train_semi_sigma_sep.sh 6 1 1 0 12 &
sleep 8
/bin/bash script/train_semi_sigma_sep.sh 7 5 0 0 12 &
sleep 8

# more unlabeled images
/bin/bash script/train_semi_sigma_sep.sh 0 1 0 0 10 &
sleep 8
/bin/bash script/train_semi_sigma_sep.sh 1 5 0 0 10 &
sleep 8
/bin/bash script/train_semi_sigma_sep.sh 2 1 0 0 20 &
sleep 8
/bin/bash script/train_semi_sigma_sep.sh 3 5 0 0 20 &
sleep 8
/bin/bash script/train_semi_sigma_sep.sh 4 1 1 0 30 &
sleep 8
/bin/bash script/train_semi_sigma_sep.sh 5 5 0 0 30 &
sleep 8

/bin/bash script/train_semi_sigma_sep.sh 0 1 0 0 0 &
sleep 8
/bin/bash script/train_semi_sigma_sep.sh 0 5 0 0 0 &
sleep 8

/bin/bash script/train_semi_sigma_sep.sh 2 1 0 0 30 &
sleep 8
/bin/bash script/train_semi_sigma_sep.sh 3 5 0 0 30 &
sleep 8
/bin/bash script/train_semi_sigma_sep.sh 6 1 0 0 6 &
sleep 8
/bin/bash script/train_semi_sigma_sep.sh 7 5 0 0 6 &
sleep 8

/bin/bash script/train_coco_semi_sigma_sep.sh 4 1 0 0 0 &
sleep 8
/bin/bash script/train_coco_semi_sigma_sep.sh 4 5 0 0 0 &
sleep 8

/bin/bash script/train_coco_semi_sigma_sep.sh 6 1 0 0 30 &
sleep 8
/bin/bash script/train_coco_semi_sigma_sep.sh 7 5 0 0 30 &
sleep 8
/bin/bash script/train_coco_semi_sigma_sep.sh 4 1 0 0 6 &
sleep 8
/bin/bash script/train_coco_semi_sigma_sep.sh 5 5 0 0 6 &
sleep 8


## 144 visualize
/bin/bash script/train_semi_sigma_sep.sh 0 1 0 1 1 &
sleep 5


/bin/bash script/train_semi_sigma_sep.sh 0 1 0 1 1 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 1 1 0 1 6 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 2 5 0 1 6 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 3 1 0 1 0 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 4 1 0 1 6 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 5 5 0 1 6 &
sleep 5


python compare_samples_v2.py --model_dir vis_w_sigma_qp_un0-s1_f0,vis_refine_adv_w_sigma_qp_un0-s1_f0 --out_dir vis_none_and_refine_s1_un0_f0 --n_imgs 1000 &
python compare_samples_v2.py --model_dir vis_w_sigma_qp_un0-s1_f1,vis_refine_adv_w_sigma_qp_un0-s1_f1 --out_dir vis_none_and_refine_s1_un0_f1 --n_imgs 1000 &
python compare_samples_v2.py --model_dir vis_w_sigma_qp_un0-s1_f2,vis_refine_adv_w_sigma_qp_un0-s1_f2 --out_dir vis_none_and_refine_s1_un0_f2 --n_imgs 1000 &
python compare_samples_v2.py --model_dir vis_w_sigma_qp_un0-s1_f3,vis_refine_adv_w_sigma_qp_un0-s1_f3 --out_dir vis_none_and_refine_s1_un0_f3 --n_imgs 1000 &
python compare_samples_v2.py --model_dir vis_w_sigma_qp_un6-s1_f0,vis_refine_adv_w_sigma_qp_un6-s1_f0 --out_dir vis_none_and_refine_s1_un6_f0 --n_imgs 1000 &
python compare_samples_v2.py --model_dir vis_w_sigma_qp_un6-s1_f1,vis_refine_adv_w_sigma_qp_un6-s1_f1 --out_dir vis_none_and_refine_s1_un6_f1 --n_imgs 1000 &
python compare_samples_v2.py --model_dir vis_w_sigma_qp_un6-s1_f2,vis_refine_adv_w_sigma_qp_un6-s1_f2 --out_dir vis_none_and_refine_s1_un6_f2 --n_imgs 1000 &
python compare_samples_v2.py --model_dir vis_w_sigma_qp_un6-s1_f3,vis_refine_adv_w_sigma_qp_un6-s1_f3 --out_dir vis_none_and_refine_s1_un6_f3 --n_imgs 1000 &
python compare_samples_v2.py --model_dir vis_w_sigma_qp_un6-s5_f0,vis_refine_adv_w_sigma_qp_un6-s5_f0 --out_dir vis_none_and_refine_s5_un6_f0 --n_imgs 1000 &
python compare_samples_v2.py --model_dir vis_w_sigma_qp_un6-s5_f1,vis_refine_adv_w_sigma_qp_un6-s5_f1 --out_dir vis_none_and_refine_s5_un6_f1 --n_imgs 1000 &
python compare_samples_v2.py --model_dir vis_w_sigma_qp_un6-s5_f2,vis_refine_adv_w_sigma_qp_un6-s5_f2 --out_dir vis_none_and_refine_s5_un6_f2 --n_imgs 1000 &
python compare_samples_v2.py --model_dir vis_w_sigma_qp_un6-s5_f3,vis_refine_adv_w_sigma_qp_un6-s5_f3 --out_dir vis_none_and_refine_s5_un6_f3 --n_imgs 1000 &

python split_imgs.py --model_dir vis_none_and_refine_s1_un0_f0 &
python split_imgs.py --model_dir vis_none_and_refine_s1_un0_f1 &
python split_imgs.py --model_dir vis_none_and_refine_s1_un0_f2 &
python split_imgs.py --model_dir vis_none_and_refine_s1_un0_f3 &
python split_imgs.py --model_dir vis_none_and_refine_s1_un6_f0 &
python split_imgs.py --model_dir vis_none_and_refine_s1_un6_f1 &
python split_imgs.py --model_dir vis_none_and_refine_s1_un6_f2 &
python split_imgs.py --model_dir vis_none_and_refine_s1_un6_f3 &

/bin/bash script/train_coco_semi_sigma_sep.sh 0 1 0 1 1 &
sleep 5
/bin/bash script/train_coco_semi_sigma_sep.sh 1 1 0 1 6 &
sleep 5
/bin/bash script/train_coco_semi_sigma_sep.sh 2 5 0 1 6 &
sleep 5
/bin/bash script/train_coco_semi_sigma_sep.sh 3 1 0 1 0 &
sleep 5
/bin/bash script/train_coco_semi_sigma_sep.sh 4 1 0 1 6 &
sleep 5
/bin/bash script/train_coco_semi_sigma_sep.sh 5 5 0 1 6 &
sleep 5

python copy_imgs.py --in_dir COCO-COCO-vis_orig_w_sigma_qp_un0-s1_f0 --out_dir s1_un0_f0
python copy_imgs.py --in_dir COCO-COCO-vis_orig_w_sigma_qp_un0-s1_f1 --out_dir s1_un0_f1
python copy_imgs.py --in_dir COCO-COCO-vis_orig_w_sigma_qp_un0-s1_f2 --out_dir s1_un0_f2
python copy_imgs.py --in_dir COCO-COCO-vis_orig_w_sigma_qp_un0-s1_f3 --out_dir s1_un0_f3

python copy_imgs.py --in_dir vis_orig_w_sigma_qp_un0-s1_f0 --out_dir s1_un0_f0
python copy_imgs.py --in_dir vis_orig_w_sigma_qp_un0-s1_f1 --out_dir s1_un0_f1
python copy_imgs.py --in_dir vis_orig_w_sigma_qp_un0-s1_f2 --out_dir s1_un0_f2
python copy_imgs.py --in_dir vis_orig_w_sigma_qp_un0-s1_f3 --out_dir s1_un0_f3

python split_imgs.py --model_dir s1_un0_f0
python split_imgs.py --model_dir s1_un0_f1
python split_imgs.py --model_dir s1_un0_f2
python split_imgs.py --model_dir s1_un0_f3

python make_figure_v2.py --model_dir s1_un0_f0_split
python make_figure_v2.py --model_dir s1_un0_f1_split
python make_figure_v2.py --model_dir s1_un0_f2_split
python make_figure_v2.py --model_dir s1_un0_f3_split

python copy_imgs.py --in_dir vis_orig_7center_w_sigma_qp_un0-s1_f0 --out_dir s1_7center_un0_f0
python copy_imgs.py --in_dir vis_orig_7center_w_sigma_qp_un0-s1_f1 --out_dir s1_7center_un0_f1
python copy_imgs.py --in_dir vis_orig_7center_w_sigma_qp_un0-s1_f2 --out_dir s1_7center_un0_f2
python copy_imgs.py --in_dir vis_orig_7center_w_sigma_qp_un0-s1_f3 --out_dir s1_7center_un0_f3

python split_imgs.py --model_dir s1_7center_un0_f0
python split_imgs.py --model_dir s1_7center_un0_f1
python split_imgs.py --model_dir s1_7center_un0_f2
python split_imgs.py --model_dir s1_7center_un0_f3

python make_figure_v2.py --model_dir s1_7center_un0_f0_split
python make_figure_v2.py --model_dir s1_7center_un0_f1_split
python make_figure_v2.py --model_dir s1_7center_un0_f2_split
python make_figure_v2.py --model_dir s1_7center_un0_f3_split

python copy_imgs.py --in_dir s1_un0_f0_split_figure --out_dir f0
python copy_imgs.py --in_dir s1_un0_f1_split_figure --out_dir f1
python copy_imgs.py --in_dir s1_un0_f2_split_figure --out_dir f2
python copy_imgs.py --in_dir s1_un0_f3_split_figure --out_dir f3

python copy_imgs.py --in_dir s1_7center_un0_f0_split_figure --out_dir 7center_f0
python copy_imgs.py --in_dir s1_7center_un0_f1_split_figure --out_dir 7center_f1
python copy_imgs.py --in_dir s1_7center_un0_f2_split_figure --out_dir 7center_f2
python copy_imgs.py --in_dir s1_7center_un0_f3_split_figure --out_dir 7center_f3

python split_imgs.py --model_dir vis_orig_w_sigma_qp_un0-s1_f0 &
python split_imgs.py --model_dir vis_orig_w_sigma_qp_un0-s1_f1 &
python split_imgs.py --model_dir vis_orig_w_sigma_qp_un0-s1_f2 &
python split_imgs.py --model_dir vis_orig_w_sigma_qp_un0-s1_f3 &

python make_figure_v2.py --model_dir COCO-COCO-vis_orig_w_sigma_qp_un0-s1_f0_split
python make_figure_v2.py --model_dir COCO-COCO-vis_orig_w_sigma_qp_un0-s1_f1_split
python make_figure_v2.py --model_dir COCO-COCO-vis_orig_w_sigma_qp_un0-s1_f2_split
python make_figure_v2.py --model_dir COCO-COCO-vis_orig_w_sigma_qp_un0-s1_f3_split

## COCO training
/bin/bash script/train_coco_logp.sh 0 1 0 &
sleep 30
/bin/bash script/train_coco_logp.sh 0 5 1 &
sleep 30

## COCO refinement module training
/bin/bash script/train_coco_semi_sigma_sep.sh 0 1 1 0 0 &
sleep 10
/bin/bash script/train_coco_semi_sigma_sep.sh 0 5 1 0 0 &
sleep 10

/bin/bash script/train_coco_semi_sigma_sep.sh 0 1 0 1 0 &
sleep 8
/bin/bash script/train_coco_semi_sigma_sep.sh 1 5 0 0 0 &
sleep 8

/bin/bash script/train_coco_semi_sigma_sep.sh 0 1 0 0 3 &
sleep 8
/bin/bash script/train_coco_semi_sigma_sep.sh 3 5 0 0 3 &
sleep 8
/bin/bash script/train_coco_semi_sigma_sep.sh 1 1 0 0 6 &
sleep 8
/bin/bash script/train_coco_semi_sigma_sep.sh 2 5 1 0 6 &
sleep 8
/bin/bash script/train_coco_semi_sigma_sep.sh 2 1 0 0 9 &
sleep 8
/bin/bash script/train_coco_semi_sigma_sep.sh 1 5 0 0 9 &
sleep 8
/bin/bash script/train_coco_semi_sigma_sep.sh 3 1 0 0 12 &
sleep 8
/bin/bash script/train_coco_semi_sigma_sep.sh 0 5 0 0 12 &
sleep 8

/bin/bash script/train_coco_semi_sigma_sep.sh 0 1 1 0 30 &
sleep 8
/bin/bash script/train_coco_semi_sigma_sep.sh 1 5 1 0 30 &
sleep 8
/bin/bash script/train_coco_semi_sigma_sep.sh 4 1 1 0 45 &
sleep 8
/bin/bash script/train_coco_semi_sigma_sep.sh 5 5 1 0 45 &
sleep 8
/bin/bash script/train_coco_semi_sigma_sep.sh 4 1 1 0 30 &
sleep 8
/bin/bash script/train_coco_semi_sigma_sep.sh 5 5 1 0 30 &
sleep 8
/bin/bash script/train_coco_semi_sigma_sep.sh 6 1 1 0 45 &
sleep 8
/bin/bash script/train_coco_semi_sigma_sep.sh 7 5 1 0 45 &
sleep 8
## entropy setting
/bin/bash script/train_entropy_semi_sep.sh 7 1 0 1 0 &
sleep 8
/bin/bash script/train_entropy_semi_sep.sh 6 5 0 0 0 &
sleep 8
/bin/bash script/train_entropy_semi_sep.sh 5 1 0 0 6 &
sleep 8
/bin/bash script/train_entropy_semi_sep.sh 4 5 0 0 6 &
sleep 8

/bin/bash script/train_entropy_semi_sep.sh 0 1 0 0 6 &
sleep 8
/bin/bash script/train_entropy_semi_sep.sh 1 5 0 0 6 &
sleep 8

/bin/bash script/train_entropy_semi_sep.sh 4 1 0 0 3 &
sleep 8
/bin/bash script/train_entropy_semi_sep.sh 7 5 0 0 3 &
sleep 8
/bin/bash script/train_entropy_semi_sep.sh 5 1 0 0 9 &
sleep 8
/bin/bash script/train_entropy_semi_sep.sh 6 5 0 0 9 &
sleep 8
/bin/bash script/train_entropy_semi_sep.sh 6 1 0 0 12 &
sleep 8
/bin/bash script/train_entropy_semi_sep.sh 5 5 0 0 12 &
sleep 8
/bin/bash script/train_entropy_semi_sep.sh 7 1 0 0 15 &
sleep 8
/bin/bash script/train_entropy_semi_sep.sh 4 5 0 0 15 &
sleep 8

/bin/bash script/train_entropy_semi_sep.sh 4 1 0 0 3 &
sleep 8
/bin/bash script/train_entropy_semi_sep.sh 7 5 0 0 3 &
sleep 8
/bin/bash script/train_entropy_semi_sep.sh 5 1 0 0 9 &
sleep 8
/bin/bash script/train_entropy_semi_sep.sh 5 5 0 0 9 &
sleep 8
/bin/bash script/train_entropy_semi_sep.sh 7 1 0 0 12 &
sleep 8
/bin/bash script/train_entropy_semi_sep.sh 4 5 0 0 12 &
sleep 8

/bin/bash script/train_entropy_semi_sep.sh 4 1 0 0 15 &
sleep 8
/bin/bash script/train_entropy_semi_sep.sh 7 5 0 0 15 &
sleep 8

## refine model training
/bin/bash script/train_logp_sep_semi_sep.sh 0 1 1 0 0 &
sleep 10
/bin/bash script/train_logp_sep_semi_sep.sh 0 5 1 0 0 &
sleep 10

/bin/bash script/train_logp_sep_v2.sh 0 5 1 &
sleep 5
/bin/bash script/train_logp_sep_v2.sh 0 1 0 &
sleep 5

/bin/bash script/train_logp_sep_semi_sep.sh 0 1 0 0 9 &
sleep 10
/bin/bash script/train_logp_sep_semi_sep.sh 1 5 0 0 9 &
sleep 10
/bin/bash script/train_logp_sep_semi_sep.sh 2 1 0 0 12 &
sleep 10
/bin/bash script/train_logp_sep_semi_sep.sh 3 5 0 0 12 &
sleep 10


## sv 240 normal setting
/bin/bash script/train_logp_sep_semi_sep.sh 0 1 0 0 0 &
sleep 5
/bin/bash script/train_logp_sep_semi_sep.sh 1 5 0 0 0 &
sleep 5
/bin/bash script/train_logp_sep_semi_sep.sh 2 1 0 0 6 &
sleep 5
/bin/bash script/train_logp_sep_semi_sep.sh 4 5 0 0 6 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 2 1 0 0 6 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 2 5 0 0 6 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 3 1 0 0 9 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 3 5 0 0 9 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 4 1 0 0 12 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 4 5 0 0 12 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 5 1 0 0 15 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 5 5 0 0 15 &
sleep 5

## more unlabeled images
/bin/bash script/train_semi_sigma_sep.sh 0 1 0 0 15 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 1 5 0 0 15 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 2 1 0 0 30 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 3 5 0 0 30 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 4 1 0 0 45 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 5 5 0 0 45 &
sleep 5


## no train testing in sv144
/bin/bash script/train_semi_sigma_sep.sh 2 1 0 0 1 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 3 5 0 0 1 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 4 1 0 0 3 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 5 5 0 0 3 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 6 1 0 0 9 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 7 5 0 0 9 &
sleep 5


/bin/bash script/train_semi_sigma_sep.sh 1 1 0 0 1 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 2 5 0 0 1 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 3 1 0 0 3 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 4 5 0 0 3 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 5 1 0 0 9 &
sleep 5
/bin/bash script/train_semi_sigma_sep.sh 6 5 0 0 9 &
sleep 5

# qcenter
/bin/bash script/train_semi_sigma_sep.sh 0 1 0 0 1 &
sleep 15
/bin/bash script/train_semi_sigma_sep.sh 1 5 0 0 1 &
sleep 15
/bin/bash script/train_semi_sigma_sep.sh 2 1 0 0 3 &
sleep 15
/bin/bash script/train_semi_sigma_sep.sh 3 5 0 0 3 &
sleep 15
/bin/bash script/train_semi_sigma_sep.sh 4 1 0 0 9 &
sleep 15
/bin/bash script/train_semi_sigma_sep.sh 5 5 0 0 6 &
sleep 15
/bin/bash script/train_semi_sigma_sep.sh 6 1 0 0 6 &
sleep 15
/bin/bash script/train_semi_sigma_sep.sh 7 5 0 0 9 &
sleep 15


/bin/bash script/train_semi_sigma_sep.sh 0 5 0 0 1 &
sleep 15
/bin/bash script/train_semi_sigma_sep.sh 1 5 0 0 3 &
sleep 15
/bin/bash script/train_semi_sigma_sep.sh 2 5 0 0 9 &
sleep 15


/bin/bash script/train_part_coco.sh 0 1 0 &
sleep 5
/bin/bash script/train_part_coco.sh 0 5 1 &
sleep 5