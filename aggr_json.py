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
    return parser.parse_args()

def find_last_exp(dpath):
    ids = os.listdir(dpath)
    ids = list(map(int, ids))
    ids.sort()
    return str(ids[-1])

def main():
    ## aggregate results
    args = get_arguments()
    src_dir = f"{args.root_dir}/{args.model_dir}"
    assert os.path.exists(src_dir)
    json_dir = f"{src_dir}/metric_json"
    print(json_dir)
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    
    dpaths = glob(f"{src_dir}/PANet*")
    dpaths.sort()
    for dpath in dpaths:
        dname = dpath.split("/")[-1]
        wayshot = dname.split("_")[4]
        fold = dname[-1]
        id = find_last_exp(dpath)
        json_fpath = f"{dpath}/{id}/metrics.json"
        opath = f"{json_dir}/{wayshot}_f{fold}.json"
        shutil.copyfile(json_fpath, opath)
        print(json_fpath)
    
    ## csv results
    trg_dir = json_dir
    print(f"target dir : {trg_dir}")
    # files = glob(f"{trg_dir}/*.json")
    fnames = os.listdir(trg_dir)
    fnames.sort()

    for fname in reversed(fnames):
        if fname[-5:] != ".json":
            fnames.remove(fname)

    m1,m2,m3,m4 = [["final_meanIoU"]],[["final_meanIoU_std"]],[["final_meanIoU_binary"]],[["final_meanIoU_std_binary"]]
    r1,r2,r3,r4 = [["final_classIoU"]],[["final_classIoU_std"]],[["final_classIoU_binary"]],[["final_classIoU_std_binary"]]

    for fname in fnames:
        # print(fname)
        name = fname.split(".")[0]
        fpath = f"{trg_dir}/{fname}"
        with open(fpath, "r") as json_fd:
            metric = json.load(json_fd)
        
        # print(metric.keys())
        final_classIoU = metric['final_classIoU']['values'][0]
        final_classIoU_std = metric['final_classIoU_std']['values'][0]
        final_meanIoU = metric['final_meanIoU']['values']
        final_meanIoU_std = metric['final_meanIoU_std']['values']
        final_classIoU_binary = metric['final_classIoU_binary']['values'][0]
        final_classIoU_std_binary = metric['final_classIoU_std_binary']['values'][0]
        final_meanIoU_binary = metric['final_meanIoU_binary']['values']
        final_meanIoU_std_binary = metric['final_meanIoU_std_binary']['values']
        # pdb.set_trace()
        m1.append([name, ]+final_meanIoU)
        m2.append([name, ]+final_meanIoU_std)
        m3.append([name, ]+final_meanIoU_binary)
        m4.append([name, ]+final_meanIoU_std_binary)
        r1.append([name, ]+final_classIoU)
        r2.append([name, ]+final_classIoU_std)
        r3.append([name, ]+final_classIoU_binary)
        r4.append([name, ]+final_classIoU_std_binary)

    categories = [m1,m2,m3,m4,r1,r2,r3,r4]
    f = open(f"{trg_dir}/results.csv",'w', newline='')
    wr = csv.writer(f)
    wr.writerow([args.model_dir])
    for c in categories:
        for l in c:
            wr.writerow(l)
        wr.writerow([])
    f.close()
    # print()

    

if __name__ == '__main__':
    main()