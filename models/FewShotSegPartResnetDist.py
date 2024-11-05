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
from .decoder import Decoder, RefineNet
import numpy as np
import pdb

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
        # Encoder
        # self.encoder = nn.Sequential(OrderedDict([('backbone', Encoder(in_channels)),]))
        # self.conv = nn.Sequential(
        #     nn.Conv2d(2048, self.hdim, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(self.hdim, affine=False),)

            # nn.ReLU(inplace=True),)
        # self.conv = nn.Conv2d(2048, self.hdim, kernel_size=1, stride=1, bias=False)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        if cfg['enc'] == "res50":
            self.encoder = resnet50(cfg=cfg)
        elif cfg['enc'] == "res101":
            self.encoder = resnet101(cfg=cfg)
        else:
            print(f"Wrong encoder configuration : {cfg['enc']}")
            assert False
            
        self.kmeans = KmeansClustering(num_cnt=self.config['center'], iters=10, init='random')
        self.decoder = Decoder(dim=self.hdim, sigma_min=self.config['min_sigma'], sigma_scale=1)


    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs):
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
        # pdb.set_trace()

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs] + [torch.cat(qry_imgs, dim=0),], dim=0)
        img_fts = self.encoder(imgs_concat) #X*512*53*53
        ###### Extract low and high level features together ######
        # img_fts, img_low_fts = self.get_low_and_high_features(imgs_concat)
        # qry_low_fts = img_low_fts[-n_queries*batch_size:]
        # del(img_low_fts)

        ###### Reduce feature size ######
        # img_fts = self.conv(img_fts)

        fts_size = img_fts.shape[-2:]

        supp_fts = img_fts[:n_ways * n_shots * batch_size].view(n_ways, n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * batch_size:].view(n_queries, batch_size, -1, *fts_size)   # N x B x C x H' x W'
        fore_mask = torch.stack([torch.stack(way, dim=0) for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W'
        back_mask = torch.stack([torch.stack(way, dim=0) for way in back_mask], dim=0)  # Wa x Sh x B x H' x W'

        align_loss = torch.zeros(1).to(torch.device('cuda'))
        outputs = []
        sigmas = []
        ###### Compute loss ######

        for epi in range(batch_size):
            ###### Extract prototype ######
            supp_fg_fts = [[self.getFeaturesArray(supp_fts[way, shot, [epi]], fore_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]
            supp_bg_fts = [[self.getFeaturesArray(supp_fts[way, shot, [epi]], back_mask[way, shot, [epi]], 1)
                            for shot in range(n_shots)] for way in range(n_ways)]

            ###### Obtain the prototypes######
            fg_prototypes, bg_prototype = self.kmeansPrototype(supp_fg_fts, supp_bg_fts)
            # fg_prototypes, bg_prototype = self.getPrototype(supp_fg_fts, supp_bg_fts)

            ###### Compute the distance ######
            prototypes = [bg_prototype,] + fg_prototypes #2, 5*512 ; p5*512
            dists = [self.calDist(qry_fts[:, epi], prototype) for prototype in prototypes] #3, 1*53*53
            pred = torch.stack([d[0] for d in dists], dim=1)  # N x (1 + Wa) x H' x W'
            outputs.append(pred)

            ###### Compute the sigma of B/F distance regression ######
            bg_selected_protos = self.get_selected_prototypes(prototypes[0], dists[0][1])
            fg_selected_protos = self.get_selected_prototypes(prototypes[1], dists[1][1])
            sigma = self.decoder.forward([bg_selected_protos, fg_selected_protos], qry_fts[:, epi])
            sigmas.append(sigma)

        sigmas = torch.cat(sigmas, dim=0)
        output = torch.cat(outputs, dim=0) # still cosine similarity map
        # output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = F.softmax(output,dim=1)
        output_refine = None

        sigmas = F.interpolate(sigmas, size=img_size, mode='bilinear', align_corners=True)
        output = F.interpolate(output, size=img_size, mode='bilinear', align_corners=True)
        # output = output.view(-1, *output.shape[2:])
        return output, output_refine, align_loss / batch_size, sigmas


    def get_selected_prototypes(self, prototypes, idx):
        _,w,h = idx.shape
        selected_protos = prototypes[idx.view(w*h)]
        return selected_protos.view(w,h,self.hdim)


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


    def get_low_and_high_features(self,x):
        net = self.encoder
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)
        x = net.layer1(x)
        x_low = net.layer2(x)
        x = net.layer3(x_low)
        x_high = net.layer4(x)
        return x_high, x_low


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


    def kmeansPrototype(self, fg_fts, bg_fts):
        fg_fts_loc = [torch.cat([tr[0] for tr in way], dim=0) for way in fg_fts] ## concat all fg_fts
        fg_fts_glo = [[tr[1] for tr in way] for way in fg_fts]  ## all global
        bg_fts_loc = torch.cat([torch.cat([tr[0] for tr in way], dim=0) for way in bg_fts], dim=0)
        bg_fts_glo = [[tr[1] for tr in way] for way in bg_fts]
        fg_prop_cls = [self.kmeans.cluster(way) if way.size(0) >= self.config['center'] else way for way in fg_fts_loc]
        bg_prop_cls = self.kmeans.cluster(bg_fts_loc) if bg_fts_loc.size(0) >= self.config['center'] else bg_fts_loc

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
    model = FewShotSegPartDist()
    x = torch.randn([2,3,421,421])
    # model.get_low_and_high_features(x)
    