"""Experiment Configuration"""
import os
import re
import glob
import itertools

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('PANet')
ex.captured_out_filter = apply_backspaces_and_linefeeds

source_folders = ['.', './dataloaders', './models', './util']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))
for source_file in sources_to_save:
    ex.add_source_file(source_file)
import datetime

@ex.config
def cfg():
    """Default configurations"""
    size=417
    input_size = (size, size) # 417, 321
    seed = 1234
    cuda_visable = '0,1,2,3'
    gpu_id = 7
    mode = 'train' # 'train' or 'test'


    if mode == 'train':
        dataset = 'VOC' # 'COCO'
        n_steps = 24000 # 40000
        num_workers = 8
        label_sets = 0
        batch_size = 2
        lr_milestones = [10000, 20000]
        # lr_milestones = [10000, 20000, 30000]
        align_loss_scaler = 1
        base_loss_scaler = 1
        ignore_label = 255
        print_interval = 100
        save_pred_every = 4000
        evaluate_interval = 4000
        n_runs = 1
        eval = 0
        eval_dir='.'

        ## model configuration
        freeze_bn = True
        refine_iter = 1
        is_sigma = True
        is_pcm = False
        is_conv = False
        is_refine = False
        is_single_proto = False
        is_sproto = False
        is_qproto = False
        test_unlabels = 3

        iter_proto = 0
        enc = "res50"
        sigma_interval = 0.3
        min_sigma = 0.3
        th = 0.5

        center = 5
        qcenter = 5
        un_bs = 3
        topk = 30
        resnet = 50 #101
        segments = False
        skip_ways = 'v1'

        ## optimizer configuration
        fix = False

        ## directories
        ppn_ckpt_dir = '.'
        dist_ckpt_dir = '.'
        pre_refine = False
        refine_ckpt_dir = '.'
        ckpt_dir = '.'
        sample_dir = 'try'
        n_sample = 500
        
        output_sem_size = 417
        infer_max_iters = 1000
        share = 3
        pt_lambda = 0.8
        global_const = 0.8
        align_loss_cs_scaler = 0
        p_value_thres = 0
        output_dir='.'

        model = {
            'part': True,
            'semi': False,
            'sem': False,
            'resnet': True,
            'slic': False,

        }

        task = {
            'n_ways': 1,
            'n_shots': 1,
            'n_queries': 1,
            'n_unlabels': 0,
        }

        optim = {
            'lr': 1e-3,
            'momentum': 0.9,
            'weight_decay': 0.0005,
        }

        slic = {
            'num_components': 80,
            'compactness': 80,
        }


    else:
        raise ValueError('Wrong configuration for "mode" !')

    exp_str = '_'.join(
        [dataset, ]
        + [key for key, value in model.items() if value]
        + [f'w{task["n_ways"]}s{task["n_shots"]}_lr{optim["lr"]}_cen{center}_F{label_sets}'])

    path = {
        'log_dir': './outputs/PANet/',
        'init_path': './FewShotSeg-dataset/cache/vgg16-397923af.pth',
        'VOC':{'data_dir': '/data/soopil/PASCAL_VOC/VOC2012/',
               'data_split': 'trainaug',},
        'COCO':{'data_dir': '/data/soopil/coco',
                'data_split': 'train',},
    }
            # 'VOC':{'data_dir': './FewShotSeg-dataset/Pascal/VOC2012/',
            #    'data_split': 'trainaug',},


@ex.config_hook
def add_observer(config, command_name, logger):
    """A hook fucntion to add observer"""
    exp_name = f'{ex.path}_{config["exp_str"]}'
    observer = FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config