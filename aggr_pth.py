import os
import pdb
import csv
import json
import shutil
import argparse
from glob import glob

def get_arguments():
    parser = argparse.ArgumentParser(description='aggregate the metrics.json files into a new directory')
    parser.add_argument("--root_dir", type=str,default='./outputs') #
    parser.add_argument("--model_dir", type=str,default='PPNet_refine_exclusive_qproto1_sproto_w_sigma') #
    parser.add_argument("--fname", type=str,default='24000') #
    return parser.parse_args()

def main():
    ## aggregate results
    args = get_arguments()
    src_dir = f"{args.root_dir}/{args.model_dir}"
    assert os.path.exists(src_dir)
    trg_dir = f"{src_dir}/pretrained_{args.fname}"
    print(trg_dir)
    if not os.path.exists(trg_dir):
        os.makedirs(trg_dir)
    
    dpaths = glob(f"{src_dir}/*/{args.fname}.pth")
    dpaths.sort()
    for dpath in dpaths:
        dname = dpath.split("/")[-2]
        way = dname.split("-")[4]
        shot = dname.split("-")[5][1]
        fold = dname[-1]
        opath = f"{trg_dir}/{shot}s_f{fold}.pth"
        shutil.copyfile(dpath, opath)
        print(dpath)
    

if __name__ == '__main__':
    main()