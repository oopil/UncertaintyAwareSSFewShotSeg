from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from collections import OrderedDict
from util.kmeans import KmeansClustering
# from .vgg import Encoder
from .ResNetBackbone import resnet50, resnet101
from .Aspp import _ASPP
from .decoder import *
import numpy as np
import pdb


class FewShotRefine(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.GLOBAL_CONST = 0.8
        self.config = cfg
        self.hdim = 2048
        self.out_size = cfg['input_size']
        self.sigma_interval = cfg['sigma_interval']
        self.min_sigma = cfg['min_sigma']
        self.is_sigma = cfg['is_sigma']
        self.is_refine = cfg['is_refine']
        self.is_qproto = cfg['is_qproto']
        self.n_unlabels = cfg['task']['n_unlabels']
        # qcenter = self.n_unlables + 1 if self.is_qproto else self.n_unlabels
        # self.kmeans = KmeansClustering(num_cnt=qcenter, iters=10, init='random')
        self.kmeans = KmeansClustering(num_cnt=self.config['qcenter'], iters=10, init='random')

        self.low_level_ft=1024
        if self.is_refine:
            self.refiner = ShallowRefineNet_v2()

    def forward(self, pred, dec=None):
        n_ways=1
        mean = pred["mean"]
        sigma = pred["sigma"]
        qft_low = pred["qft_low"]
        qft_high = pred["qft_high"]
        sprotos = pred["prototypes"]
        un_fts = pred["un_fts"]
        s_low_protos = pred["s_low_protos"]
        
        n_queries = len(qft_high)
        B,_,w,h = sigma.shape

        # pdb.set_trace()
        masks = torch.zeros_like(mean)
        if self.n_unlabels == 0 and not self.is_qproto and self.is_refine:
            output = self.refiner(qft_low, mean, s_low_protos)
            output = F.interpolate(output, size=self.out_size, mode='bilinear', align_corners=True)
            return output, masks

        outputs = []
        masks = []
        for epi in range(B):
            bg_prototype, fg_prototypes = sprotos[epi][0], [sprotos[epi][1]]
            qft_high_epi = qft_high[:, epi]

            if self.n_unlabels == 0:
                un_fts_epi = qft_high_epi
            else:
                un_fts_epi = un_fts[epi*self.n_unlabels:(epi+1)*self.n_unlabels]
                un_fts_epi = torch.cat([qft_high_epi, un_fts_epi], dim=0)

            ###### Compute the distance in unlabeled images ######
            prototypes = [bg_prototype,] + fg_prototypes #2, 5*512 ; p5*512
            un_dists = [self.calDist(un_fts_epi, prototype) for prototype in prototypes] #3, 1*53*53
            un_pred = torch.stack([d[0] for d in un_dists], dim=1)  # N x (1 + Wa) x H' x W'
            un_prob = F.softmax(un_pred,dim=1)

            ###### Compute the sigma of B/F distance regression ######
            bg_selected_protos = self.get_selected_prototypes(prototypes[0], un_dists[0][1])
            fg_selected_protos = self.get_selected_prototypes(prototypes[1], un_dists[1][1])
            un_sigma = dec.forward_batch([bg_selected_protos, fg_selected_protos], un_fts_epi)

            # un_sigma = un_sigma * un_sigma # use variance instead of sigma

            ###### Compute entropy as uncertainty ######
            un_entropy = - un_prob * torch.log(un_prob)
            un_entropy = torch.sum(un_entropy, 1, keepdim=True)
            un_entropy = torch.cat([un_entropy,un_entropy],dim=1)

            if self.n_unlabels == 0 and not self.is_qproto:
                pred = un_pred[:n_queries]
                sigma = un_sigma[:n_queries]
            else:
                ##### calculate mask for unlabeled image feature selection #####
                if self.is_sigma:
                    # un_sigma = torch.ones_like(un_sigma)*0.3 ## for simple 0.3 sigma
                    # sigma_norm = (1.0-un_sigma).clamp(0,1) # [B,2,w,h] ## use fg/bg sigma together
                    sigma_norm = (1.0-un_sigma[:,1:2,...]).clamp(0,1) # [B,2,w,h] ## use only fg sigma
                    soft_mask = un_prob*sigma_norm
                    hard_mask = soft_mask.round()

                    un_back_mask, un_fore_mask = hard_mask[:,0], hard_mask[:,1]

                else:
                    hard_mask = un_prob.round()
                    # hard_mask = (un_prob > 1-self.sigma_interval)*1.0

                masks.append(hard_mask[0:1,...])
                if not self.is_qproto:
                    un_fts_epi = un_fts_epi[n_queries:]
                    hard_mask = hard_mask[n_queries:]

                un_back_mask, un_fore_mask = hard_mask[:,0], hard_mask[:,1]

                ##### get prototypes from unlabeled image and add it to the original prototypes #####
                if un_back_mask.sum() > 1 and un_fore_mask.sum() > 1:
                    un_fg_fts = [[self.getFeaturesArray(un_fts_epi[shot:shot+1], un_fore_mask[shot:shot+1])
                                for shot in range(len(un_fts_epi))] for way in range(n_ways)]
                    un_bg_fts = [[self.getFeaturesArray(un_fts_epi[shot:shot+1], un_back_mask[shot:shot+1], 1)
                                for shot in range(len(un_fts_epi))] for way in range(n_ways)]
                    un_fg_prototypes, un_bg_prototype = self.kmeansPrototype(un_fg_fts, un_bg_fts)

                    bg_prototype = torch.cat([bg_prototype, un_bg_prototype], dim=0)
                    fg_prototypes[0] = torch.cat([fg_prototypes[0], un_fg_prototypes[0]], dim=0) # only for 1 way task

                ###### Compute the distance ######
                prototypes = [bg_prototype,] + fg_prototypes #2, 5*512 ; p5*512
                dists = [self.calDist(qft_high_epi, prototype) for prototype in prototypes] #3, 1*53*53
                pred = torch.stack([dist[0] for dist in dists], dim=1)  # N x (1 + Wa) x H' x W'

                ###### Compute the sigma of B/F distance regression ######
                # bg_selected_proto = self.get_selected_prototypes(prototypes[0], dists[0][1])
                # fg_selected_proto = self.get_selected_prototypes(prototypes[1], dists[1][1])
                # sigma = dec.forward_batch([bg_selected_proto, fg_selected_proto], qft_high_epi)
                # sigmas.append(sigma)

            outputs.append(pred)

        if self.n_unlabels != 0 or self.is_qproto:
            masks = torch.cat(masks, dim=0)
        else:
            masks = torch.zeros_like(mean)

        output = torch.cat(outputs, dim=0) # still cosine similarity map
        output = F.softmax(output,dim=1) # reduced size

        if self.is_refine:
            output = self.refiner(qft_low, output, s_low_protos)
        
        output = F.interpolate(output, size=self.out_size, mode='bilinear', align_corners=True)
        return output, masks


    def get_selected_prototypes(self, prototypes, idx):
        N,w,h = idx.shape
        selected_protos = prototypes[idx.view(N,w*h)]
        return selected_protos.view(N,w,h,self.hdim)


    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
            / (mask[None, ...].sum(dim=(2, 3)) + 1e-5) # 1 x C
        return masked_fts


    def getPrototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype

    
    def calDist(self, fts, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
            
        returns:
            dist: [1 x H x W]
            dix: [1 x H x W]
        """
        dist, idx = F.cosine_similarity(fts.unsqueeze(1), prototype[..., None, None].unsqueeze(0), dim=2).max(1)
        dist *= scaler
        return dist, idx


    def getFeaturesArray(self, fts, mask, upscale=2):

        """
        Extract foreground and background features
        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        c, h1, w1 = fts.shape[1:]
        h, w = mask.shape[1:]

        fts1 = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)
        masked_fts = torch.sum(fts1 * mask[None, ...], dim=(2, 3)) \
                     / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)  # 1 x C

        mask_bilinear = F.interpolate(mask.unsqueeze(0), size=(h1*upscale, w1*upscale), mode='nearest').view(-1)

        if mask_bilinear.sum(0) <= 10:
            fts = fts1.squeeze(0).permute(1, 2, 0).view(h * w, c)  ## l*c
            mask1 = mask.view(-1)
            if mask1.sum() == 0:
                fts = fts[[0]]*0  # 1 x C
            else:
                fts = fts[mask1>0]
        else:
            fts = F.interpolate(fts, size=(h1*upscale, w1*upscale), mode='bilinear', align_corners=True).squeeze(0).permute(1, 2, 0).view(h1*w1*upscale**2, c)
            fts = fts[mask_bilinear>0]

        return (fts, masked_fts)

    def kmeansPrototype(self, fg_fts, bg_fts):
        fg_fts_loc = [torch.cat([tr[0] for tr in way], dim=0) for way in fg_fts] ## concat all fg_fts
        bg_fts_loc = torch.cat([torch.cat([tr[0] for tr in way], dim=0) for way in bg_fts], dim=0)
        fg_prop_cls = [self.kmeans.cluster(way) if way.size(0) >= self.config['qcenter'] else way for way in fg_fts_loc]
        bg_prop_cls = self.kmeans.cluster(bg_fts_loc) if bg_fts_loc.size(0) >= self.config['qcenter'] else bg_fts_loc

        # fg_prop_cls = [self.kmeans.cluster(way) for way in fg_fts_loc]
        # bg_prop_cls = self.kmeans.cluster(bg_fts_loc)

        # fg_fts_glo = [[tr[1] for tr in way] for way in fg_fts]  ## all global
        # bg_fts_glo = [[tr[1] for tr in way] for way in bg_fts]
        # fg_prop_glo, bg_prop_glo = self.getPrototype(fg_fts_glo, bg_fts_glo)
        # fg_propotypes = [fg_c + self.GLOBAL_CONST * fg_g for (fg_c, fg_g) in zip(fg_prop_cls, fg_prop_glo)]
        # bg_propotypes = bg_prop_cls + self.GLOBAL_CONST * bg_prop_glo
        # return fg_propotypes, bg_propotypes  ## 2, 5*512; 5*512
        return fg_prop_cls, bg_prop_cls  ## 2, 5*512; 5*512



class FewShotSegPartDist(nn.Module):
    """
    Fewshot Segmentation model
    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """
    def __init__(self, pretrained_path=None, cfg=None):
        super().__init__()
        self.GLOBAL_CONST = 0.8
        self.config = cfg #self.config = cfg #

        self.hdim = 2048
        if cfg['enc'] == "res50":
            self.encoder = resnet50(cfg=cfg)
        elif cfg['enc'] == "res101":
            self.encoder = resnet101(cfg=cfg)
        else:
            print(f"Wrong encoder configuration : {cfg['enc']}")
            assert False
        self.kmeans = KmeansClustering(num_cnt=self.config['center'], iters=10, init='random')
        self.decoder = Decoder(dim=self.hdim, sigma_min=0.1, sigma_scale=1)
        self.n_unlabels = cfg['task']['n_unlabels']


    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, un_imgs):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """

        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs] + [torch.cat(qry_imgs, dim=0),], dim=0)
        # img_fts = self.encoder(imgs_concat) #X*512*53*53
        ###### Extract low and high level features together ######
        img_fts, img_low_fts = self.get_low_and_high_features(imgs_concat)
        qry_low_fts = img_low_fts[-n_queries*batch_size:]

        if self.n_unlabels == 0:
            un_imgs = torch.zeros_like(un_imgs[0][0]).unsqueeze(0)
        else:
            un_imgs = torch.cat([torch.cat(way, dim=0) for way in un_imgs], dim=0)

        with torch.no_grad():
            un_fts = self.encoder(un_imgs) #20,2048,53,53

        ###### Reduce feature size ######
        # img_fts = self.conv(img_fts)
        fts_size = img_fts.shape[-2:]

        supp_fts = img_fts[:n_ways * n_shots * batch_size].view(n_ways, n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        supp_low_fts = img_low_fts[:n_ways * n_shots * batch_size].view(n_ways, n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * batch_size:].view(n_queries, batch_size, -1, *fts_size)   # N x B x C x H' x W'
        fore_mask = torch.stack([torch.stack(way, dim=0) for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W'
        back_mask = torch.stack([torch.stack(way, dim=0) for way in back_mask], dim=0)  # Wa x Sh x B x H' x W'

        align_loss = torch.zeros(1).to(torch.device('cuda'))
        outputs = []
        sigmas = []
        s_low_protos = []

        ###### Compute loss ######
        out_dict = {
            "prototypes":[],
        }
        # pdb.set_trace()
        for epi in range(batch_size):
            ###### Extract prototype ######
            supp_fg_fts = [[self.getFeaturesArray(supp_fts[way, shot, [epi]], fore_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]
            supp_bg_fts = [[self.getFeaturesArray(supp_fts[way, shot, [epi]], back_mask[way, shot, [epi]], 1)
                            for shot in range(n_shots)] for way in range(n_ways)]

            ###### Obtain the prototypes######
            fg_prototypes, bg_prototype = self.kmeansPrototype(supp_fg_fts, supp_bg_fts)
            # fg_prototypes, bg_prototype = self.getPrototype(supp_fg_fts, supp_bg_fts)

            ###### obtain prototype of support low level features ######
            supp_fg_low_fts = [[self.getFeatures(supp_low_fts[way, shot, [epi]], fore_mask[way, shot, [epi]])
                for shot in range(n_shots)] for way in range(n_ways)]
            s_low_proto = [sum(way) / n_shots for way in supp_fg_low_fts][0] # only 1 way setting
            s_low_protos.append(s_low_proto)

            ###### Compute the distance ######
            prototypes = [bg_prototype,] + fg_prototypes #2, 5*512 ; p5*512
            out_dict["prototypes"].append(prototypes)
            dists = [self.calDist(qry_fts[:, epi], prototype) for prototype in prototypes] #3, 1*53*53
            pred = torch.stack([d[0] for d in dists], dim=1)  # N x (1 + Wa) x H' x W'
            outputs.append(pred)

            ###### Compute the sigma of B/F distance regression ######
            bg_selected_proto = self.get_selected_prototypes(prototypes[0], dists[0][1])
            fg_selected_proto = self.get_selected_prototypes(prototypes[1], dists[1][1])
            sigma = self.decoder.forward_batch([bg_selected_proto, fg_selected_proto], qry_fts[:, epi])
            sigmas.append(sigma)

        sigmas = torch.cat(sigmas, dim=0)
        output = torch.cat(outputs, dim=0) # still cosine similarity map
        output = F.softmax(output,dim=1) # reduced size
        
        out_dict["mean"] = output
        out_dict["sigma"] = sigmas
        out_dict["qft_high"] = qry_fts
        out_dict["qft_low"] = qry_low_fts
        out_dict["un_fts"] = un_fts
        out_dict["s_low_protos"] = s_low_protos
        return out_dict


    def get_low_and_high_features(self,x):
        net = self.encoder
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)
        x1 = net.layer1(x)
        x2 = net.layer2(x1)
        x3 = net.layer3(x2)
        x4 = net.layer4(x3)
        x_low = torch.cat([x2,x3], dim=1)
        return x4, x_low


    def get_low_and_high_features_save(self,x):
        net = self.encoder
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)
        x = net.layer1(x)
        x = net.layer2(x)
        x_low = net.layer3(x)
        x_high = net.layer4(x_low)
        return x_high, x_low

    
    def get_selected_prototypes(self, prototypes, idx):
        N,w,h = idx.shape
        selected_protos = prototypes[idx.view(N,w*h)]
        return selected_protos.view(N,w,h,self.hdim)


    def calDist(self, fts, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
            
        returns:
            dist: [1 x H x W]
            dix: [1 x H x W]
        """
        dist, idx = F.cosine_similarity(fts.unsqueeze(1), prototype[..., None, None].unsqueeze(0), dim=2).max(1)
        dist *= scaler
        return dist, idx


    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
            / (mask[None, ...].sum(dim=(2, 3)) + 1e-5) # 1 x C
        return masked_fts


    def getFeaturesArray(self, fts, mask, upscale=2):

        """
        Extract foreground and background features
        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        c, h1, w1 = fts.shape[1:]
        h, w = mask.shape[1:]

        fts1 = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)
        masked_fts = torch.sum(fts1 * mask[None, ...], dim=(2, 3)) \
                     / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)  # 1 x C

        mask_bilinear = F.interpolate(mask.unsqueeze(0), size=(h1*upscale, w1*upscale), mode='nearest').view(-1)

        if mask_bilinear.sum(0) <= 10:
            fts = fts1.squeeze(0).permute(1, 2, 0).view(h * w, c)  ## l*c
            mask1 = mask.view(-1)
            if mask1.sum() == 0:
                fts = fts[[0]]*0  # 1 x C
            else:
                fts = fts[mask1>0]
        else:
            fts = F.interpolate(fts, size=(h1*upscale, w1*upscale), mode='bilinear', align_corners=True).squeeze(0).permute(1, 2, 0).view(h1*w1*upscale**2, c)
            fts = fts[mask_bilinear>0]

        return (fts, masked_fts)


    def kmeansPrototype(self, fg_fts, bg_fts, is_query=False):
        fg_fts_loc = [torch.cat([tr[0] for tr in way], dim=0) for way in fg_fts] ## concat all fg_fts
        fg_fts_glo = [[tr[1] for tr in way] for way in fg_fts]  ## all global
        bg_fts_loc = torch.cat([torch.cat([tr[0] for tr in way], dim=0) for way in bg_fts], dim=0)
        bg_fts_glo = [[tr[1] for tr in way] for way in bg_fts]
        fg_prop_cls = [self.kmeans.cluster(way) if way.size(0) >= self.config['center'] else way for way in fg_fts_loc]
        bg_prop_cls = self.kmeans.cluster(bg_fts_loc) if bg_fts_loc.size(0) >= self.config['center'] else bg_fts_loc

        # fg_prop_cls = [self.kmeans.cluster(way) for way in fg_fts_loc]
        # bg_prop_cls = self.kmeans.cluster(bg_fts_loc)

        fg_prop_glo, bg_prop_glo = self.getPrototype(fg_fts_glo, bg_fts_glo)

        fg_propotypes = [fg_c + self.GLOBAL_CONST * fg_g for (fg_c, fg_g) in zip(fg_prop_cls, fg_prop_glo)]
        bg_propotypes = bg_prop_cls + self.GLOBAL_CONST * bg_prop_glo
        return fg_propotypes, bg_propotypes  ## 2, 5*512; 5*512



    def getPrototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype
    
if __name__ == '__main__':
    pdb.set_trace()
    