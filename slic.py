import os
import pdb
import csv
import json
import shutil
import argparse
from glob import glob
import numpy as np
# from fast_slic import Slic
from fast_slic.avx2 import SlicAvx2
import pickle
from PIL import Image

def get_arguments():
    parser = argparse.ArgumentParser(description='aggregate the metrics.json files into a new directory')
    parser.add_argument("--root_dir", type=str,default='./outputs') #
    parser.add_argument("--model_dir", type=str,default='PPNet_refine_exclusive_qproto1_sproto_w_sigma') #
    return parser.parse_args()

def main():
    ## sample check
    # with open('/data/soopil/sample/2007_000027.pkl', 'rb') as f:
	#     data = pickle.load(f)
    # print(data)
    # print(data.keys())
    # with open('filename.pkl', 'wb') as f:
	#     pickle.dump(temp_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    # assert False

    ## aggregate results
    args = get_arguments()
    src_dir = "/data/soopil/coco"
    assert os.path.exists(src_dir)
    print(src_dir)
    trg_dir = "/data/soopil/coco/superpixel"
    print(trg_dir)
    if not os.path.exists(trg_dir):
        os.makedirs(trg_dir)
    
    slic = SlicAvx2(num_components=100, compactness=10)
    dirs = ['train2017', 'test2017', 'val2017', 'unlabeled2017']
    # 'train2014', 'test2014', 'val2014', 
    for dir in dirs:
        print(dir)
        dir_path = f"{src_dir}/{dir}"
        fpaths = glob(f"{dir_path}/*.jpg")
        for i,fpath in enumerate(fpaths):
            fname = fpath.split('/')[-1]
            name = fname.split('.')[0]
            opath = f"{trg_dir}/{name}.pkl"
            with Image.open(fpath) as f:
                image = np.array(f)
            if len(image.shape) == 2:
                image = np.stack([image,image,image], axis=2)
            print(f"{i}/{len(fpaths)}, {image.shape}", end='\r')
            # import cv2; image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)   # You can convert the image to CIELAB space if you need.
            assignment = slic.iterate(image) # Cluster Map
            # print(assignment)
            with open(opath, 'wb') as f:
                pickle.dump(assignment, f, protocol=pickle.HIGHEST_PROTOCOL)

            # assert False
        

if __name__ == '__main__':
    main()