import os
import cv2
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt, pi

oneDivSqrtTwoPI = 1.0/sqrt(2.0*pi)

# This guassian func and loss codes are refered to https://kangbk0120.github.io/articles/2018-05/MDN

def gaussian_distribution(y, mu, sigma):
    # sigma += 1e-6 # prevent zero-division
    result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
    result = -0.5 * (result * result)
    return (torch.exp(result) * torch.reciprocal(sigma)) * oneDivSqrtTwoPI
    
def loglikelihood(mu, sigma, y):
    """
    mu,sigma : [B,2,w,h]
    y : [B,w,h]
    """
    # pdb.set_trace()
    with torch.no_grad(): ## to one-hot label
        y1 = y.clamp(0,1).unsqueeze(1) # remove 255 label
        y2 = torch.ones_like(y1)-y1
        y = torch.cat([y2,y1],dim=1)
    loss = gaussian_distribution(y, mu, sigma)
    return -torch.log(loss).mean()

def normalize_arr(a):
    if np.min(a) == np.max(a):
        return np.zeros_like(a, dtype=np.uint8)
    else:
        a = a - np.min(a)
        a = a / np.max(a)
        a = np.uint8(255 * a)
        return a

def convert3ch(a, color=0):
    if color==-1:
        zeros = np.stack((a,a,a),axis=2)
    else:
        zeros = np.zeros_like(a)
        zeros = np.stack((zeros,zeros,zeros),axis=2)
        zeros[:, :, color] = a
    return zeros

def img_process(im, dsize):
    im = normalize_arr(im)
    im = cv2.resize(im, dsize=dsize)
    return im

def mask_process(mask, dsize, color=1):
    mask = convert3ch(mask, color=color)
    # mask = cv2.resize(mask, dsize=dsize)
    mask = cv2.resize(mask, dsize=dsize, interpolation=cv2.INTER_NEAREST)
    return mask

def save_sample_img(sx,sy,qx,qy,qpred,\
                    n_iter, sigma = None, mean = None, \
                    dsize = (320,320), alpha = 0.5,\
                    dir_name = "test", overlap = True):

    sx = sx[0][0][0].permute(dims=(1, 2, 0)).cpu().numpy()
    sy = sy[0][0][0].cpu().numpy().astype(np.uint8)*255
    qx = qx[0][0].permute(dims=(1, 2, 0)).cpu().numpy()
    qy = qy[0].cpu().numpy().astype(np.uint8)*255
    qpred = qpred[0].argmax(dim=0).cpu().numpy().astype(np.uint8)*255

    sx = img_process(sx, dsize=dsize)
    qx = img_process(qx, dsize=dsize)
    sy = mask_process(sy, dsize=dsize, color=1)
    qy = mask_process(qy, dsize=dsize, color=1)
    qpred = mask_process(qpred, dsize=dsize, color=2)

    sout = cv2.addWeighted(sx, alpha, sy, 1-alpha, 0, 0)
    q_over_pred = cv2.addWeighted(qx, alpha, qpred, 1-alpha, 0, 0)
    q_over_y = cv2.addWeighted(qx, alpha, qy, 1-alpha, 0, 0)

    outimg = np.concatenate((sx,sout,qx,q_over_pred,q_over_y),axis=1)
    
    # pdb.set_trace()
    if type(mean) == torch.Tensor:
        ## visualize mean
        # mean_prob = np.uint8(mean[0][1].cpu().numpy()*255)
        # mean_prob = mask_process(mean_prob, dsize=dsize, color=-1)
        # mean_mask = np.round(mean_prob.astype(np.float32)/255).astype(np.uint8)*255
        # q_over_mean = cv2.addWeighted(qx, alpha, mean_mask, 1-alpha, 0, 0)
        # outimg = np.concatenate((outimg, mean_prob, q_over_mean), axis=1)

        ## check fg/bg calibrated mask
        fg_mask = np.uint8(mean[0][1].cpu().numpy()*255)
        fg_mask = mask_process(fg_mask, dsize=dsize, color=-1)
        q_over_fg = cv2.addWeighted(qx, alpha, fg_mask, 1-alpha, 0, 0)
        bg_mask = np.uint8(mean[0][0].cpu().numpy()*255)
        bg_mask = mask_process(bg_mask, dsize=dsize, color=-1)
        q_over_bg = cv2.addWeighted(qx, alpha, bg_mask, 1-alpha, 0, 0)
        outimg = np.concatenate((outimg, q_over_fg, q_over_bg), axis=1)

    if type(sigma) == torch.Tensor:
        ## normalize together
        sigma = normalize_arr(sigma[0].cpu().numpy())
        # sigma = np.uint8(255 * sigma[0].cpu().numpy())
        bg_sigma, fg_sigma = sigma
        bg_sigma = mask_process(bg_sigma, dsize=dsize, color=-1)
        fg_sigma = mask_process(fg_sigma, dsize=dsize, color=-1)
        outimg = np.concatenate((outimg, bg_sigma, fg_sigma), axis=1)

    # root_dir = "./sample"
    root_dir = "/data/soopil/FSS_uncertainty/sample"
    if not os.path.exists(f"{root_dir}/{dir_name}"):
        os.makedirs(f"{root_dir}/{dir_name}")

    cv2.imwrite(f"{root_dir}/{dir_name}/{n_iter}.png", outimg)

## visualize - no overlay
def save_sample_img_v3(sx,sy,qx,qy,qpred,\
                    n_iter, sigma = None, mean = None, \
                    dsize = (320,320), alpha = 0.5,\
                    dir_name = "test", overlap = True):
    
    sx = sx[0][0][0].permute(dims=(1, 2, 0)).cpu().numpy()
    sy = sy[0][0][0].cpu().numpy().astype(np.uint8)*255
    qx = qx[0][0].permute(dims=(1, 2, 0)).cpu().numpy()
    qy = qy[0].cpu().numpy().astype(np.uint8)*255
    qpred = qpred[0].argmax(dim=0).cpu().numpy().astype(np.uint8)*255

    sx = img_process(sx, dsize=dsize)
    qx = img_process(qx, dsize=dsize)
    sy = mask_process(sy, dsize=dsize, color=-1)
    qy = mask_process(qy, dsize=dsize, color=-1)
    qpred = mask_process(qpred, dsize=dsize, color=-1)

    sout = cv2.addWeighted(sx, alpha, sy, 1-alpha, 0, 0)
    q_over_pred = cv2.addWeighted(qx, alpha, qpred, 1-alpha, 0, 0)
    q_over_y = cv2.addWeighted(qx, alpha, qy, 1-alpha, 0, 0)

    outimg = np.concatenate((sx,sy,qx,qpred,qy),axis=1)
    
    # pdb.set_trace()
    if type(mean) == torch.Tensor:
        ## visualize mean
        # mean_prob = np.uint8(mean[0][1].cpu().numpy()*255)
        # mean_prob = mask_process(mean_prob, dsize=dsize, color=-1)
        # mean_mask = np.round(mean_prob.astype(np.float32)/255).astype(np.uint8)*255
        # q_over_mean = cv2.addWeighted(qx, alpha, mean_mask, 1-alpha, 0, 0)
        # outimg = np.concatenate((outimg, mean_prob, q_over_mean), axis=1)

        ## check fg/bg calibrated mask
        fg_mask = np.uint8(mean[0][1].cpu().numpy()*255)
        fg_mask = mask_process(fg_mask, dsize=dsize, color=-1)
        q_over_fg = cv2.addWeighted(qx, alpha, fg_mask, 1-alpha, 0, 0)
        bg_mask = np.uint8(mean[0][0].cpu().numpy()*255)
        bg_mask = mask_process(bg_mask, dsize=dsize, color=-1)
        q_over_bg = cv2.addWeighted(qx, alpha, bg_mask, 1-alpha, 0, 0)
        outimg = np.concatenate((outimg, q_over_fg, q_over_bg), axis=1)

    if type(sigma) == torch.Tensor:
        ## normalize together
        sigma = normalize_arr(sigma[0].cpu().numpy())
        # sigma = np.uint8(255 * sigma[0].cpu().numpy())
        bg_sigma, fg_sigma = sigma
        bg_sigma = mask_process(bg_sigma, dsize=dsize, color=-1)
        fg_sigma = mask_process(fg_sigma, dsize=dsize, color=-1)
        outimg = np.concatenate((outimg, bg_sigma, fg_sigma), axis=1)

    # root_dir = "./sample"
    root_dir = "/data/soopil/FSS_uncertainty/sample"
    if not os.path.exists(f"{root_dir}/{dir_name}"):
        os.makedirs(f"{root_dir}/{dir_name}")

    cv2.imwrite(f"{root_dir}/{dir_name}/{n_iter}.png", outimg)

# def save_sample_img_v1(sx,sy,qx,qy,qpred,n_iter,sigma=None, dsize = (320,320), alpha = 0.5, dir_name = "test"):
#     sx = normalize_arr(sx[0][0][0].permute(dims=(1, 2, 0)).cpu().numpy())
#     sy = sy[0][0][0].cpu().numpy().astype(np.uint8)*255
#     qx = normalize_arr(qx[0][0].permute(dims=(1, 2, 0)).cpu().numpy())
#     qy = qy[0].cpu().numpy().astype(np.uint8)*255
#     qpred = qpred[0].argmax(dim=0).cpu().numpy().astype(np.uint8)*255

#     sy = convert3ch(sy,color=1)
#     qy = convert3ch(qy,color=1)
#     qpred = convert3ch(qpred,color=2)

#     sx = cv2.resize(sx, dsize=dsize)
#     sy = cv2.resize(sy, dsize=dsize)
#     qx = cv2.resize(qx, dsize=dsize)
#     qy = cv2.resize(qy, dsize=dsize)
#     qpred = cv2.resize(qpred, dsize=dsize)

#     sout = cv2.addWeighted(sx, alpha, sy, 1-alpha, 0, 0)
#     qout = cv2.addWeighted(qx, alpha, qpred, 1-alpha, 0, 0)
#     qout2 = cv2.addWeighted(qx, alpha, qy, 1-alpha, 0, 0)
#     outimg = np.concatenate((sx,sout,qx,qout,qout2),axis=1)

#     # pdb.set_trace()
#     # if sigma != None:
#     if type(sigma) == torch.Tensor:
#         ## normalize together
#         sigma = normalize_arr(sigma[0].cpu().numpy())
#         bg_sigma, fg_sigma = sigma
#         ## normalize individually
#         # bg_sigma, fg_sigma = sigma[0].cpu().numpy()
#         # bg_sigma = normalize_arr(bg_sigma)
#         # fg_sigma = normalize_arr(fg_sigma)

#         bg_sigma = convert3ch(bg_sigma,color=-1)
#         fg_sigma = convert3ch(fg_sigma,color=-1)
#         bg_sigma = cv2.resize(bg_sigma, dsize=dsize)
#         fg_sigma = cv2.resize(fg_sigma, dsize=dsize)
#         outimg = np.concatenate((outimg, bg_sigma, fg_sigma), axis=1)

#     if not os.path.exists(f"./sample/{dir_name}"):
#         os.makedirs(f"./sample/{dir_name}")
#     cv2.imwrite(f"./sample/{dir_name}/{n_iter}.png", outimg)

def cal_prob(self,mu,sigma,x):
    return torch.exp(torch.square((x-mu)/sigma)*(-1/2))/(sigma*sqrt(2*pi)+1e-10)

def get_loss(self, logits, qy, idx):
    sft, qft, _, mu, sigma, out, out_softmax = logits
    loss = self.gaussian_distribution(qy, mu, sigma)# * pi
    loglikelihood = -torch.log(loss).mean()
    if self.is_seg:
        bce_logits_func = nn.CrossEntropyLoss()
        loss_bce = bce_logits_func(out, qy.squeeze(1).long())
    else:
        loss_bce = 0
    total_loss = loss_bce + loglikelihood#*0.5
    return total_loss,loss_bce,0

def accur(preds, targets):
    preds = preds.argmax(dim=1).view(targets.shape)
    return (preds == targets).sum().float() / targets.size(0)

def tensor2float(tensor, n):
    return round(float(tensor),n)

def IoU(preds, y):
    """
    :param preds: [N, n_cls, w, h]
    :param y: [N, w, h]
    """
    # pdb.set_trace()
    pred = preds.argmax(dim=1)
    intersection = (pred*y)
    iou = intersection.sum(dim=(1,2))/((pred+y-intersection).sum(dim=(1,2)) + 1e-10)
    # print(iou)
    return iou.mean()

def fast_adapt(batch, learner, loss, s_steps, fast_optim, device):
    sx, sy = batch['sx'], batch['sy']
    qx, qy = batch['qx'], batch['qy']
    # pdb.set_trace()

    # Adapt the model
    for step in range(s_steps):
        # fast_optim.zero_grad
        s_error = loss(learner(sx), sy)
        s_error.backward()
        fast_optim.step()
        # learner.adapt(s_error)

    # Evaluate the adapted model
    preds = learner(qx)
    q_error = loss(preds, qy)
    q_accur = IoU(preds, qy)
    return q_error, q_accur
