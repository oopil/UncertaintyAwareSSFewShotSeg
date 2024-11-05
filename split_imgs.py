import os
from PIL import Image
import numpy as np
import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='OneShot')
    parser.add_argument("--root_dir", type=str,default='./sample')
    parser.add_argument("--out_dir", type=str,default='compare')
    parser.add_argument("--model_dir", type=str,
    default='logp_refine_aspp_small_dil_1shot_f0,logp_id4')
    return parser.parse_args()

def read_img(path):
    im = Image.open(path)
    return np.array(im)    

def main():
    args = get_arguments()
    size = 320

    root_path = args.root_dir
    src_dir = f"{root_path}/{args.model_dir}"
    trg_dir = f"{root_path}/{args.model_dir}_split"

    if not os.path.exists(f"{trg_dir}"):
        os.makedirs(f"{trg_dir}")

    fnames = os.listdir(src_dir)
    fnames.sort()

    for i,fname in enumerate(fnames):
        name = fname.split(".")[0]

        arr = read_img(f"{src_dir}/{fname}")
        x,y,_ = arr.shape
        n_x = x//size
        n_y = y//size
        arrs1 = np.split(arr, n_x, axis=0)

        for xi, arr1 in enumerate(arrs1):
            arrs2 = np.split(arr1,n_y,axis=1)
            for yi, arr2 in enumerate(arrs2):
                im = Image.fromarray(arr2)
                im.save(f"{trg_dir}/{name}_{xi}_{yi}.png")

        print(f"{i}/{len(fnames)}",end='\r')

if __name__ == '__main__':
    main()