import os
import shutil

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose
from models.FewShotSegResnet import FewShotSeg as FewshotSegResnet
from models.FewShotSegPartResnet import FewShotSegPart as FewshotSegPartResnet
from models.FewShotSegPartResnetDistSep_v2 import FewShotSegPartDist as FewshotSegPartResnetDist

from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders.transforms import RandomMirror, Resize, ToTensorNormalize
from util.utils import set_seed, CLASS_LABELS, get_params
from config import ex
from util.metric import Metric
import tqdm
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import os.path as osp
import pdb
from gaussianloss import *

@ex.automain
def main(_run, _config, _log):

    logdir = f'{_run.observers[0].dir}/'
    print(logdir)
    category = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    if not os.path.exists(_config["ckpt_dir"]):
        os.makedirs(_config["ckpt_dir"])

    _run.log_scalar('ID', _run._id)
    data_name = _config['dataset']
    max_label = 20 if data_name == 'VOC' else 80

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    print(_config['ckpt_dir'])
    # tbwriter = SummaryWriter(osp.join(_config['ckpt_dir']))

    training_tags = {'loss': "ATraining/total_loss", "query_loss": "ATraining/query_loss",
                     'aligned_loss': "ATraining/aligned_loss", 'base_loss': "ATraining/base_loss",}
    infer_tags = {
        'mean_iou': "MeanIoU/mean_iou",
        "mean_iou_binary": "MeanIoU/mean_iou_binary",
    }

    _log.info('###### Create model ######')
    model = FewshotSegPartResnetDist(pretrained_path=_config['path']['init_path'], cfg=_config)
    _log.info('Model: FewShotSegPartDist')
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])

    ckpt = f'{_config["dist_ckpt_dir"]}/{_config["task"]["n_shots"]}s_f{_config["label_sets"]}.pth'
    print(f'Load {ckpt}')
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval() ## freeze bn

    _log.info('###### Load data ######')
    data_name = _config['dataset']
    if data_name == 'VOC':
        make_data = voc_fewshot
    elif data_name == 'COCO':
        make_data = coco_fewshot
    else:
        raise ValueError('Wrong config for dataset!')
    labels = CLASS_LABELS[data_name][_config['label_sets']]
    transforms = Compose([Resize(size=_config['input_size']),RandomMirror()])
    dataset = make_data(
        base_dir=_config['path'][data_name]['data_dir'],
        split=_config['path'][data_name]['data_split'],
        transforms=transforms,
        to_tensor=ToTensorNormalize(),
        labels=labels,
        max_iters=_config['n_steps'] * _config['batch_size'],
        n_ways=_config['task']['n_ways'],
        n_shots=_config['task']['n_shots'],
        n_queries=_config['task']['n_queries'],
        n_unlabel=_config['task']['n_unlabels'],
        cfg=_config

    )
    trainloader = DataLoader(
        dataset,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=_config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    _log.info('###### Set optim ######')
    print('Optimizer: Not fix')
    _config['optim']['lr'] = _config['optim']['lr']/10
    optim = torch.optim.SGD(model.module.decoder.parameters(), **_config['optim'])

    _log.info('###### Training Decoder Only ######')
    for i_iter, batch in enumerate(trainloader):
        if _config['eval']:
            if i_iter == 0:
                break
        # Prepare input
        support_images = [[shot.cuda() for shot in way] for way in batch['support_images']]
        support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way] for way in batch['support_mask']]
        support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]for way in batch['support_mask']]

        query_images = [query_image.cuda() for query_image in batch['query_images']]
        query_labels = torch.cat([query_label.long().cuda() for query_label in batch['query_labels']], dim=0)#1*417*417

        base_loss = torch.zeros(1).to(torch.device('cuda'))
        # Forward and Backward
        optim.zero_grad()
        query_pred, _, align_loss, sigma = model(support_images, support_fg_mask, support_bg_mask, query_images)
        logp = loglikelihood(query_pred, sigma, query_labels)
        # query_loss = criterion(query_pred.log(), query_labels) #1*3*417*417, 1*417*417
        loss = logp * _config['base_loss_scaler']
        loss.backward()
        optim.step()

    torch.save(model.state_dict(), os.path.join(f'{_config["ckpt_dir"]}/best.pth'))

    print()
    print(_config['ckpt_dir'])

    _log.info(' --------- Testing begins ---------')
    labels = CLASS_LABELS[data_name]['all'] - CLASS_LABELS[data_name][_config['label_sets']]
    transforms = [Resize(size=_config['input_size'])]
    transforms = Compose(transforms)
    # ckpt = os.path.join(f'{_config["ckpt_dir"]}/{_config["n_steps"]}.pth')
    # ckpt = os.path.join(f'{_config["ckpt_dir"]}/best_all.pth')
    ckpt = os.path.join(f'{_config["ckpt_dir"]}/best.pth')
    print(ckpt)

    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()

    metric = Metric(max_label=max_label, n_runs=5)
    with torch.no_grad():
        for run in range(1): # run=1 is only for sample check
            n_iter = 0
            _log.info(f'### Run {run + 1} ###')
            set_seed(_config['seed'] + run)

            _log.info(f'### Load data ###')
            dataset = make_data(
                base_dir=_config['path'][data_name]['data_dir'],
                split=_config['path'][data_name]['data_split'],
                transforms=transforms,
                to_tensor=ToTensorNormalize(),
                labels=labels,
                max_iters=_config['infer_max_iters'],
                n_ways=_config['task']['n_ways'],
                n_shots=_config['task']['n_shots'],
                n_queries=_config['task']['n_queries'],
                n_unlabel=_config['task']['n_unlabels'],
                cfg=_config

            )
            if _config['dataset'] == 'COCO':
                coco_cls_ids = dataset.datasets[0].dataset.coco.getCatIds()
            testloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                    num_workers=_config['num_workers'], pin_memory=True, drop_last=False)
            _log.info(f"Total # of Data: {len(dataset)}")
            for batch in tqdm.tqdm(testloader):
                if _config['dataset'] == 'COCO':
                    label_ids = [coco_cls_ids.index(x) + 1 for x in batch['class_ids']]
                else:
                    label_ids = batch['class_ids']
                    if type(label_ids[0]) == torch.Tensor:
                        label_ids = list(label_ids[0].numpy())
                    else:
                        label_ids = list(label_ids)

                support_images = [[shot.cuda() for shot in way] for way in batch['support_images']]
                suffix = 'mask'
                support_fg_mask = [[shot[f'fg_{suffix}'].float().cuda() for shot in way] for way in batch['support_mask']]
                support_bg_mask = [[shot[f'bg_{suffix}'].float().cuda() for shot in way] for way in batch['support_mask']]

                query_images = [query_image.cuda() for query_image in batch['query_images']]
                query_labels = torch.cat([query_label.cuda() for query_label in batch['query_labels']], dim=0)
                query_pred, _, _, sigma = model(support_images, support_fg_mask, support_bg_mask, query_images)

                curr_iou = metric.record(query_pred.argmax(dim=1)[0], query_labels[0], labels=label_ids, n_run=run)
                if run == 0 and _config['sample_dir'] != "try":
                    save_sample_img(support_images,
                                    support_fg_mask,
                                    query_images,
                                    query_labels,
                                    query_pred,
                                    n_iter,
                                    sigma = sigma,
                                    dir_name = f"{_config['sample_dir']}_f{_config['label_sets']}")
                n_iter += 1

            classIoU, meanIoU = metric.get_mIoU(labels=sorted(labels), n_run=run)
            classIoU_binary, meanIoU_binary = metric.get_mIoU_binary(n_run=run)

            _run.log_scalar('classIoU', classIoU.tolist())
            _run.log_scalar('meanIoU', meanIoU.tolist())
            _run.log_scalar('classIoU_binary', classIoU_binary.tolist())
            _run.log_scalar('meanIoU_binary', meanIoU_binary.tolist())
            _log.info(f'classIoU: {classIoU}')
            _log.info(f'meanIoU: {meanIoU}')
            _log.info(f'classIoU_binary: {classIoU_binary}')
            _log.info(f'meanIoU_binary: {meanIoU_binary}')

    classIoU, classIoU_std, meanIoU, meanIoU_std = metric.get_mIoU(labels=sorted(labels))
    classIoU_binary, classIoU_std_binary, meanIoU_binary, meanIoU_std_binary = metric.get_mIoU_binary()

    _run.log_scalar('meanIoU', meanIoU.tolist())
    _run.log_scalar('meanIoU_binary', meanIoU_binary.tolist())

    _run.log_scalar('final_classIoU', classIoU.tolist())
    _run.log_scalar('final_classIoU_std', classIoU_std.tolist())
    _run.log_scalar('final_meanIoU', meanIoU.tolist())
    _run.log_scalar('final_meanIoU_std', meanIoU_std.tolist())
    _run.log_scalar('final_classIoU_binary', classIoU_binary.tolist())
    _run.log_scalar('final_classIoU_std_binary', classIoU_std_binary.tolist())
    _run.log_scalar('final_meanIoU_binary', meanIoU_binary.tolist())
    _run.log_scalar('final_meanIoU_std_binary', meanIoU_std_binary.tolist())

    _log.info('----- Final Result ----- ')
    _log.info(f'classIoU mean: {classIoU}')
    _log.info(f'classIoU std: {classIoU_std}')
    _log.info(f'meanIoU mean: {meanIoU}')
    _log.info(f'meanIoU std: {meanIoU_std}')
    _log.info(f'classIoU_binary mean: {classIoU_binary}')
    _log.info(f'classIoU_binary std: {classIoU_std_binary}')
    _log.info(f'meanIoU_binary mean: {meanIoU_binary}')
    _log.info(f'meanIoU_binary std: {meanIoU_std_binary}')

    _log.info("## ------------------------------------------ ##")
    _log.info(f'###### Setting: {_run.observers[0].dir} ######')
    num_run=5
    _log.info(f"Running {num_run} runs, meanIoU:{meanIoU:.4f}, meanIoU_binary:{meanIoU_binary:.4f} "
                     f"meanIoU_std:{meanIoU_std:.4f}, meanIoU_binary_std:{meanIoU_std_binary:.4f}")
    _log.info(f"Current setting is {_run.observers[0].dir}")


    print(f"Running {num_run} runs, meanIoU:{meanIoU:.4f}, meanIoU_binary:{meanIoU_binary:.4f} "
                     f"meanIoU_std:{meanIoU_std:.4f}, meanIoU_binary_std:{meanIoU_std_binary:.4f}")
    print(f"Current setting is {_run.observers[0].dir}")
    print(_config['ckpt_dir'])
    print(logdir)
