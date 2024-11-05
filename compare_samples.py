import os
from sys import maxsize
import cv2
import pdb
import argparse
from glob import glob
import numpy as np

def get_arguments():
    parser = argparse.ArgumentParser(description='OneShot')
    parser.add_argument("--root_dir", type=str,default='./sample')
    parser.add_argument("--out_dir", type=str,default='compare')
    parser.add_argument("--n_imgs", type=int,default=300)
    parser.add_argument("--model_dir", type=str,
    default='logp_refine_aspp_small_dil_1shot_f0,logp_id4')
    return parser.parse_args()

def main():
    args = get_arguments()
    models = args.model_dir.split(",")
    trg_dir = f"{args.root_dir}/{args.out_dir}"
    
    for i,m in enumerate(models):
        print(i,m)
    print(trg_dir)

    if not os.path.exists(trg_dir):
        os.makedirs(trg_dir)
    
    for i in range(1,args.n_imgs+1):
        ims = []
        for model in models:
            fpath = f"{args.root_dir}/{model}/{i}.png"
            ims.append(cv2.imread(fpath,cv2.IMREAD_COLOR))
        # ims.append(cv2.imread(f"{args.root_dir}/{models[0]}/{i+1}.png",cv2.IMREAD_COLOR))
        # ims.append(cv2.imread(f"{args.root_dir}/{models[1]}/{i}.png",cv2.IMREAD_COLOR))
        # ims.append(cv2.imread(f"{args.root_dir}/{models[2]}/{i}.png",cv2.IMREAD_COLOR))
        # ims.append(cv2.imread(f"{args.root_dir}/{models[3]}/{i}.png",cv2.IMREAD_COLOR))
        sizes = np.array([im.shape[1] for im in ims])
        max_size = np.max(sizes)
        # pdb.set_trace()
        if np.sum((sizes-max_size)==0) != len(models):
            for m_idx,model in enumerate(models):
                if ims[m_idx].shape[1] != max_size:
                    diff_size = max_size - ims[m_idx].shape[1]
                    newone = np.pad(ims[0],((0,0),(0,diff_size),(0,0)),'constant', constant_values=0)
                    ims[m_idx] = newone

        new_im = np.concatenate(ims,axis=0)
        cv2.imwrite(f"{trg_dir}/{i}.png", new_im)
        print(f"{i}/{args.n_imgs}", end='\r')

if __name__ == '__main__':
    main()