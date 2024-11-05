# Uncertainty Aware Semi-Supervised FSS
This repository includes inplementation of the paper ["Uncertainty-aware semi-supervised few shot segmentation"](https://www.sciencedirect.com/science/article/pii/S0031320322007713) published in Pattern Recognition.

## Data Preparation for VOC Dataset

**1. Download Pascal VOC dataset**
set the path for the directory in config.py file

**2. Download pretrained model**
[ResNet50](https://download.pytorch.org/models/resnet50-19c8e357.pth) 
[Resnet101](https://download.pytorch.org/models/resnet50-19c8e357.pth)
put them under FewShotSeg-dataset/cache/ folder.

## Training & Evaluation in Command Line for Pascal VOC
- Train 5-shot model 
```
/bin/bash script/train_logp.sh 0 5
python aggr_pth.py --model_dir PPNet_res101_Logp --fname 24000
```
- Fine-tune uncertainty estimation module
```
/bin/bash script/train_logp_additional.sh 0 5
python aggr_pth.py --model_dir PPNet_res101_Logp_additional
```

-Test 5-shot semi-supervised prediction using 6 unlabeled images
```
/bin/bash script/train_semi_sigma_sep.sh 0 5 6 0
```
- summarize test results
```
python aggr_json.py --model_dir PPNet_res101_Logp_additional_semi_w_sigma_qp_un6
```

## Citation
If this code is helpful for your study, please cite:
```
@article{kim2023uncertainty,
  title={Uncertainty-aware semi-supervised few shot segmentation},
  author={Kim, Soopil and Chikontwe, Philip and An, Sion and Park, Sang Hyun},
  journal={Pattern Recognition},
  volume={137},
  pages={109292},
  year={2023},
  publisher={Elsevier}
}
```
