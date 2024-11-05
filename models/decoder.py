import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class Decoder_save(nn.Module):
    def __init__(self, dim=2048, sigma_min = 0.3, sigma_scale=1):
        super(Decoder, self).__init__()
        self.dim = dim
        self.hdim = dim//4
        self.enc = nn.Sequential(
            nn.Conv2d(dim, self.hdim, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.sigma = nn.Sequential(
            nn.Conv2d(self.hdim*2, self.hdim//2, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hdim//2, self.hdim//4, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hdim//4, 1, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.sigma_min = sigma_min # 0.3 sigma equals to 0.09 variance
        self.sigma_scale = sigma_scale

    # def _init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    
    def forward(self,protos,qft):
        """
        protos: 2 x [w,h,c] - bg_prototype, fg_prototype
        qft:    [1,c,w,h]
        """
        # pdb.set_trace()
        protos = torch.stack(protos,dim=0) # [2,w,h,c]
        protos = protos.permute(0,3,1,2).contiguous()
        x = torch.cat([protos, qft], dim=0) # [3,c,w,h]
        bg_proto, fg_proto, qft = self.enc(x).unsqueeze(1)
        bg_sigma = self.sigma(torch.cat([qft,bg_proto],dim=1))*self.sigma_scale + self.sigma_min
        fg_sigma = self.sigma(torch.cat([qft,fg_proto],dim=1))*self.sigma_scale + self.sigma_min
        return torch.cat([bg_sigma, fg_sigma], dim=1)

    def forward_batch(self,protos,qft):
        """
        protos: 2 x [N,w,h,c] - bg_prototype, fg_prototype
        qft:    [N,c,w,h]
        """
        # pdb.set_trace()
        N,c,w,h = qft.shape
        protos = torch.cat(protos,dim=0) # [2*N,w,h,c]
        protos = protos.permute(0,3,1,2).contiguous()
        x = torch.cat([protos, qft], dim=0) # [2*N+N,c,w,h]
        embeddings = self.enc(x)
        bg_proto, fg_proto, qft = embeddings[:N],embeddings[N:N*2],embeddings[N*2:]
        bg_sigma = self.sigma(torch.cat([qft,bg_proto],dim=1))*self.sigma_scale + self.sigma_min
        fg_sigma = self.sigma(torch.cat([qft,fg_proto],dim=1))*self.sigma_scale + self.sigma_min
        return torch.cat([bg_sigma, fg_sigma], dim=1)

        
class Decoder(nn.Module):
    def __init__(self, dim=2048, sigma_min = 0.3, sigma_scale=1):
        super(Decoder, self).__init__()
        self.dim = dim
        self.hdim = dim//4
        self.enc = nn.Sequential(
            nn.Conv2d(dim, self.hdim, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.sigma = nn.Sequential(
            nn.Conv2d(self.hdim*2, self.hdim//2, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hdim//2, self.hdim//4, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hdim//4, 1, kernel_size=1, stride=1, bias=False),
        )

        self.sigma_min = sigma_min # 0.3 sigma equals to 0.09 variance
        self.sigma_scale = sigma_scale
        # self.pcm = PCM(ft_dim=2048, h_dim=512)

    # def _init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    
    def forward(self,protos,qft):
        """
        protos: 2 x [w,h,c] - bg_prototype, fg_prototype
        qft:    [1,c,w,h]
        """
        # pdb.set_trace()
        protos = torch.stack(protos,dim=0) # [2,w,h,c]
        protos = protos.permute(0,3,1,2).contiguous()
        x = torch.cat([protos, qft], dim=0) # [3,c,w,h]
        bg_proto, fg_proto, qft = self.enc(x).unsqueeze(1)
        bg_sigma = self.sigma(torch.cat([qft,bg_proto],dim=1))*self.sigma_scale + 0.3
        fg_sigma = self.sigma(torch.cat([qft,fg_proto],dim=1))*self.sigma_scale + 0.3
        bg_sigma = torch.max(torch.ones_like(bg_sigma)*self.sigma_min, bg_sigma)
        fg_sigma = torch.max(torch.ones_like(fg_sigma)*self.sigma_min, fg_sigma)
        return torch.cat([bg_sigma, fg_sigma], dim=1)

    def forward_batch(self,protos,qft):
        """
        protos: 2 x [N,w,h,c] - bg_prototype, fg_prototype
        qft:    [N,c,w,h]
        """
        # pdb.set_trace()
        N,c,w,h = qft.shape
        protos = torch.cat(protos,dim=0) # [2*N,w,h,c]
        protos = protos.permute(0,3,1,2).contiguous()
        x = torch.cat([protos, qft], dim=0) # [2*N+N,c,w,h]
        embeddings = self.enc(x)
        bg_proto, fg_proto, qft = embeddings[:N],embeddings[N:N*2],embeddings[N*2:]
        bg_sigma = self.sigma(torch.cat([qft,bg_proto],dim=1))*self.sigma_scale + 0.3
        fg_sigma = self.sigma(torch.cat([qft,fg_proto],dim=1))*self.sigma_scale + 0.3
        bg_sigma = torch.max(torch.ones_like(bg_sigma)*self.sigma_min, bg_sigma)
        fg_sigma = torch.max(torch.ones_like(fg_sigma)*self.sigma_min, fg_sigma)
        return torch.cat([bg_sigma, fg_sigma], dim=1)


class Decoder_v2(nn.Module):
    def __init__(self,
                dim=2048, 
                sigma_min = 0.3, 
                sigma_scale=1, 
                is_pcm=False, 
                pcm_dim=512, 
                is_conv=False):
        super(Decoder_v2, self).__init__()
        self.dim = dim
        self.hdim = dim//4
        self.enc = nn.Sequential(
            nn.Conv2d(dim, self.hdim, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.sigma = nn.Sequential(
            nn.Conv2d(self.hdim*2, self.hdim//2, kernel_size=1, padding=0,  stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hdim//2, self.hdim//4, kernel_size=1, padding=0,  stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hdim//4, 1, kernel_size=1, padding=0,  stride=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.sigma_min = sigma_min # 0.3 sigma equals to 0.09 variance
        self.sigma_scale = sigma_scale

        self.is_pcm = is_pcm
        if self.is_pcm:
            self.pcm = PCM(ft_dim=pcm_dim, h_dim=512)

    # def _init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    
    def forward(self,protos,qft,qft_low=None):
        """
        protos: 2 x [w,h,c] - bg_prototype, fg_prototype
        qft:    [1,c,w,h]
        """
        # pdb.set_trace()
        protos = torch.stack(protos,dim=0) # [2,w,h,c]
        protos = protos.permute(0,3,1,2).contiguous()
        x = torch.cat([protos, qft], dim=0) # [3,c,w,h]
        bg_proto, fg_proto, qft = self.enc(x).unsqueeze(1)
        bg_sigma = self.sigma(torch.cat([qft,bg_proto],dim=1))*self.sigma_scale + self.sigma_min
        fg_sigma = self.sigma(torch.cat([qft,fg_proto],dim=1))*self.sigma_scale + self.sigma_min
        if self.is_pcm:
            bg_sigma = self.pcm(bg_sigma, qft_low)
            fg_sigma = self.pcm(fg_sigma, qft_low)
        bg_sigma = bg_sigma.clamp(0,1)
        fg_sigma = fg_sigma.clamp(0,1)
        return torch.cat([bg_sigma, fg_sigma], dim=1)

    def forward_batch(self,protos,qft,qft_low=None):
        """
        protos: 2 x [N,w,h,c] - bg_prototype, fg_prototype
        qft:    [N,c,w,h]
        """
        # pdb.set_trace()
        N,c,w,h = qft.shape
        protos = torch.cat(protos,dim=0) # [2*N,w,h,c]
        protos = protos.permute(0,3,1,2).contiguous()
        x = torch.cat([protos, qft], dim=0) # [2*N+N,c,w,h]
        embeddings = self.enc(x)
        bg_proto, fg_proto, qft = embeddings[:N],embeddings[N:N*2],embeddings[N*2:]
        bg_sigma = self.sigma(torch.cat([qft,bg_proto],dim=1))*self.sigma_scale + self.sigma_min
        fg_sigma = self.sigma(torch.cat([qft,fg_proto],dim=1))*self.sigma_scale + self.sigma_min
        if self.is_pcm:
            bg_sigma = self.pcm(bg_sigma, qft_low)
            fg_sigma = self.pcm(fg_sigma, qft_low)
        bg_sigma = bg_sigma.clamp(0,1)
        fg_sigma = fg_sigma.clamp(0,1)
        return torch.cat([bg_sigma, fg_sigma], dim=1)

class Decoder_v3(nn.Module):
    def __init__(self,
                dim=2048, 
                sigma_min = 0.3, 
                sigma_scale=1, 
                is_pcm=False, 
                pcm_dim=1024, 
                is_conv=False):

        super(Decoder_v3, self).__init__()
        self.dim = dim
        self.hdim = dim//4
        self.enc = nn.Sequential(
            nn.Conv2d(dim, self.hdim, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )

        if is_conv:
            self.sigma = nn.Sequential(
                nn.Conv2d(self.hdim*2, self.hdim//2, kernel_size=3, padding=1,  stride=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hdim//2, self.hdim//4, kernel_size=3, padding=1,  stride=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hdim//4, 1, kernel_size=3, padding=1,  stride=1, bias=False),
            )
        else:
            self.sigma = nn.Sequential(
                nn.Conv2d(self.hdim*2, self.hdim//2, kernel_size=1, padding=0,  stride=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hdim//2, self.hdim//4, kernel_size=1, padding=0,  stride=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hdim//4, 1, kernel_size=1, padding=0,  stride=1, bias=False),
            )

        self.sigma_min = sigma_min # 0.3 sigma equals to 0.09 variance
        self.sigma_scale = sigma_scale

        self.is_pcm = is_pcm
        if self.is_pcm:
            self.pcm = PCM(ft_dim=pcm_dim, h_dim=512)

    
    def forward(self,protos,qft,qft_low=None):
        """
        protos: 2 x [w,h,c] - bg_prototype, fg_prototype
        qft:    [1,c,w,h]
        """
        # pdb.set_trace()
        protos = torch.stack(protos,dim=0) # [2,w,h,c]
        protos = protos.permute(0,3,1,2).contiguous()
        x = torch.cat([protos, qft], dim=0) # [3,c,w,h]
        bg_proto, fg_proto, qft = self.enc(x).unsqueeze(1)
        bg_sigma = self.sigma(torch.cat([qft,bg_proto],dim=1))*self.sigma_scale + self.sigma_min
        fg_sigma = self.sigma(torch.cat([qft,fg_proto],dim=1))*self.sigma_scale + self.sigma_min
        if self.is_pcm:
            bg_sigma = self.pcm(bg_sigma, qft_low)
            fg_sigma = self.pcm(fg_sigma, qft_low)
        
        bg_sigma = F.hardtanh(bg_sigma, max_val=1., min_val=self.sigma_min)
        fg_sigma = F.hardtanh(fg_sigma, max_val=1., min_val=self.sigma_min)
        return torch.cat([bg_sigma, fg_sigma], dim=1)


    def forward_batch(self,protos,qft,qft_low=None):
        """
        protos: 2 x [N,w,h,c] - bg_prototype, fg_prototype
        qft:    [N,c,w,h]
        """
        # pdb.set_trace()
        N,c,w,h = qft.shape
        protos = torch.cat(protos,dim=0) # [2*N,w,h,c]
        protos = protos.permute(0,3,1,2).contiguous()
        x = torch.cat([protos, qft], dim=0) # [2*N+N,c,w,h]
        embeddings = self.enc(x)
        bg_proto, fg_proto, qft = embeddings[:N],embeddings[N:N*2],embeddings[N*2:]
        bg_sigma = self.sigma(torch.cat([qft,bg_proto],dim=1))*self.sigma_scale + self.sigma_min
        fg_sigma = self.sigma(torch.cat([qft,fg_proto],dim=1))*self.sigma_scale + self.sigma_min
        if self.is_pcm:
            bg_sigma = self.pcm(bg_sigma, qft_low)
            fg_sigma = self.pcm(fg_sigma, qft_low)

        bg_sigma = F.hardtanh(bg_sigma, max_val=1., min_val=self.sigma_min)
        fg_sigma = F.hardtanh(fg_sigma, max_val=1., min_val=self.sigma_min)
        return torch.cat([bg_sigma, fg_sigma], dim=1)

## this uses support prototype expansion
class ShallowRefineNet_v2(nn.Module):
    def __init__(self, in_dim=(1024+512), h_dim=512, dilations = [6,12,18]):
        super(ShallowRefineNet_v2, self).__init__()
        self.f0 = nn.Conv2d(in_dim, h_dim, kernel_size=1, stride=1, padding=0, bias=True)

        residule_dim = h_dim*2 + 2

        self.residule1 = nn.Sequential(
            nn.Conv2d(residule_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )

        self.residule2 = nn.Sequential(
            nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )

        self.residule3 = nn.Sequential(
            nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )

        ## ASPP module
        self.layer1 = PSPnet(in_channels=h_dim,out_channels=h_dim, dilations=dilations)
        self.layer2 = nn.Sequential(
            nn.Conv2d(h_dim*5, h_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
        self.layer3 = nn.Conv2d(h_dim, 2, kernel_size=1, stride=1, bias=True)  # numclass = 2

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    
    def forward(self, x_in, pred, s_proto):
        """
        x:      [B,c,w',h'] low level feature of query image
        pred:   [B,2,w',h'] softmax of cosine similarity
        s_proto: B x [1,c] sigma predicted by decoder
        """
        if x_in.shape[-2:] != pred.shape[-2:]:
            x_in = F.interpolate(x_in, size=pred.shape[-2:], align_corners=True, mode='bilinear')
        B,_,w,h = pred.shape
        # pdb.set_trace()
        s_proto = torch.cat(s_proto, dim=0).unsqueeze(2).unsqueeze(3) # [B,c,1,1]
        s_proto = s_proto.expand(-1, -1, w, h)
        
        ## linear projection
        x_in = self.f0(x_in)
        s_proto = self.f0(s_proto)
        ## refinement
        x_list = [x_in, s_proto, pred]
        x = torch.cat(x_list, dim=1)
        x = x_in + self.residule1(x)
        x = x + self.residule2(x)
        x = x + self.residule3(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        pred = F.softmax(x,dim=1)
        return pred

class RefineNet(nn.Module):
    def __init__(self, in_dim=512, h_dim=512, dilations = [6,12,18], is_pcm = False, is_sigma = True):
        super(RefineNet, self).__init__()
        self.is_pcm = is_pcm
        self.is_sigma = is_sigma
        residule_dim = in_dim + 2*2 # 2*2

        if self.is_pcm:
            self.pcm = PCM(in_dim, in_dim)
            residule_dim = in_dim + 2*2 + 1

        if not self.is_sigma:
            residule_dim = in_dim + 2

        self.residule1 = nn.Sequential(
            nn.Conv2d(residule_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )

        self.residule2 = nn.Sequential(
            nn.Conv2d(in_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )

        self.residule3 = nn.Sequential(
            nn.Conv2d(in_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )

        ## ASPP module
        self.layer1 = PSPnet(in_channels=h_dim,out_channels=h_dim, dilations=dilations)
        self.layer2 = nn.Sequential(
            nn.Conv2d(h_dim*5, h_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
        self.layer3 = nn.Conv2d(h_dim, 2, kernel_size=1, stride=1, bias=True)  # numclass = 2

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    
    def forward(self, x_in, pred, sigma, n_iter=1):
        """
        x:      [B,c,w',h'] low level feature of query image
        pred:   [B,2,w',h'] softmax of cosine similarity
        sigma:  [B,2,w',h'] sigma predicted by decoder
        """
        if x_in.shape[-2:] != pred.shape[-2:]:
            x_in = F.interpolate(x_in, size=pred.shape[-2:], align_corners=True, mode='bilinear')

        # pdb.set_trace()
        if self.is_pcm:
            ## v1
            # pcm_map = self.pcm(pred[:, 1:2], x_in)
            ## v2
            sigma_norm = (1.3-sigma[:,1:2]) # [B,1,w,h]
            B,_,w,h = sigma_norm.shape
            flat_ = sigma_norm.view(B,w*h)
            mini = flat_.min(dim=1)[0].view(B,1,1,1)

            # if int(mini.item()) == 1:
            #     candidate = pred[:, 1:2]
            # else:
            sigma_norm = (sigma_norm-mini)*(1/(1-mini))
            candidate = pred[:, 1:2]*sigma_norm
            pcm_map = self.pcm(candidate, x_in)

        for iter in range(n_iter):
            x_list = [x_in,pred]

            if self.is_sigma:
                x_list.append(sigma) 
                # x_list.append(sigma-0.3) # subtract minimum sigma value
                # x_list.append((1.3-sigma)*3)
                # x_list.append(1.3-sigma)
            if self.is_pcm:
                x_list.append(pcm_map)

            x = torch.cat(x_list, dim=1)
            x = x_in + self.residule1(x)
            x = x + self.residule2(x)
            x = x + self.residule3(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            pred = F.softmax(x,dim=1)

        return pred

class ShallowRefineNet(nn.Module):
    def __init__(self, in_dim=512, h_dim=512, dilations = [6,12,18]):
        super(ShallowRefineNet, self).__init__()
        residule_dim = in_dim + 2

        self.residule1 = nn.Sequential(
            nn.Conv2d(residule_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )

        self.residule2 = nn.Sequential(
            nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )

        self.residule3 = nn.Sequential(
            nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )

        ## ASPP module
        self.layer1 = PSPnet(in_channels=h_dim,out_channels=h_dim, dilations=dilations)
        self.layer2 = nn.Sequential(
            nn.Conv2d(h_dim*5, h_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
        self.layer3 = nn.Conv2d(h_dim, 2, kernel_size=1, stride=1, bias=True)  # numclass = 2

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    
    def forward(self, x_in, pred):
        """
        x:      [B,c,w',h'] low level feature of query image
        pred:   [B,2,w',h'] softmax of cosine similarity
        sigma:  [B,2,w',h'] sigma predicted by decoder
        """
        if x_in.shape[-2:] != pred.shape[-2:]:
            x_in = F.interpolate(x_in, size=pred.shape[-2:], align_corners=True, mode='bilinear')

        x_list = [x_in,pred]
        x = torch.cat(x_list, dim=1)
        x = x_in + self.residule1(x)
        x = x + self.residule2(x)
        x = x + self.residule3(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        pred = F.softmax(x,dim=1)
        return pred


class RefineNetProto(nn.Module):
    def __init__(self, 
                in_dim=512, 
                h_dim=512, 
                dilations = [6,12,18], 
                is_pcm = False, 
                is_sigma = True,
                is_qproto = True):
        super(RefineNetProto, self).__init__()
        self.is_pcm = is_pcm
        self.is_sigma = is_sigma
        self.is_qproto = is_qproto
        if self.is_qproto:
            residule_dim = in_dim + 2*2 # 2*2
        else:
            residule_dim = in_dim + 2*1

        self.residule1 = nn.Sequential(
            nn.Conv2d(residule_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )

        self.residule2 = nn.Sequential(
            nn.Conv2d(in_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )

        self.residule3 = nn.Sequential(
            nn.Conv2d(in_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )

        ## ASPP module
        self.layer1 = PSPnet(in_channels=h_dim,out_channels=h_dim, dilations=dilations)
        self.layer2 = nn.Sequential(
            nn.Conv2d(h_dim*5, h_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
        self.layer3 = nn.Conv2d(h_dim, 2, kernel_size=1, stride=1, bias=True)  # numclass = 2

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    
    def forward(self, x_in, pred, new_pred, n_iter=1):
        """
        x:      [B,c,w',h'] low level feature of query image
        pred:   [B,2,w',h'] softmax of cosine similarity
        new_pred:  [B,2,w',h'] new_pred predicted by new prototypes (support + query prototype)
        """
        if x_in.shape[-2:] != pred.shape[-2:]:
            x_in = F.interpolate(x_in, size=pred.shape[-2:], align_corners=True, mode='bilinear')

        # pdb.set_trace()
        for iter in range(n_iter):
            x_list = [x_in,pred]
            if self.is_qproto:
                x_list.append(new_pred)

            x = torch.cat(x_list, dim=1)
            x = x_in + self.residule1(x)
            x = x + self.residule2(x)
            x = x + self.residule3(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            pred = F.softmax(x,dim=1)

        return pred


class PSPnet(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, dilations = [6,12,18]):
        super(PSPnet, self).__init__()
        self.layer6_0 = nn.Sequential(
            nn.Conv2d(in_channels , out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            )
        self.layer6_1 = nn.Sequential(
            nn.Conv2d(in_channels , out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            )
        self.layer6_2 = nn.Sequential(
            nn.Conv2d(in_channels , out_channels , kernel_size=3, stride=1, padding=dilations[0],dilation=dilations[0], bias=True),
            nn.ReLU(),
            )
        self.layer6_3 = nn.Sequential(
            nn.Conv2d(in_channels , out_channels, kernel_size=3, stride=1, padding=dilations[1], dilation=dilations[1], bias=True),
            nn.ReLU(),
            )
        self.layer6_4 = nn.Sequential(
            nn.Conv2d(in_channels , out_channels , kernel_size=3, stride=1, padding=dilations[2], dilation=dilations[2], bias=True),
            nn.ReLU(),
            )

    def forward(self, x):
        feature_size = x.shape[-2:]
        global_feature = F.avg_pool2d(x, kernel_size=feature_size)
        global_feature = self.layer6_0(global_feature)
        global_feature = global_feature.expand(-1, -1, feature_size[0], feature_size[1])
        out = torch.cat(
            [global_feature, self.layer6_1(x), self.layer6_2(x), self.layer6_3(x), self.layer6_4(x)], dim=1)
        return out


## PCM module is from https://github.com/YudeWang/SEAM/blob/master/network/resnet38_SEAM.py
class PCM(nn.Module):
    def __init__(self, ft_dim, h_dim):
        """
        :param h_dim_k: dimension of hidden 
        """
        super(PCM, self).__init__()
        # self.proj = torch.nn.Conv2d(ft_dim, h_dim, 1, bias=False) # bias=True
        # torch.nn.init.xavier_uniform_(self.proj.weight)
        # torch.nn.init.xavier_uniform_(self.proj.weight, gain=4)

    def forward(self, cam, ft):
        n,c,h,w = ft.size()
        if cam.shape[-2:] != ft.shape[-2:]:
            cam = F.interpolate(cam, (h,w), mode='bilinear', align_corners=True)
        cam = cam.view(n,-1,h*w)

        # ft = self.proj(ft)
        ft = ft.view(n,-1,h*w)
        ft = ft/(torch.norm(ft,dim=1,keepdim=True)+1e-5)

        aff = F.relu(torch.matmul(ft.transpose(1,2), ft),inplace=True)
        aff = aff/(torch.sum(aff,dim=1,keepdim=True)+1e-5)
        cam_rv = torch.matmul(cam, aff).view(n,-1,h,w)
        # if torch.isnan(cam_rv).sum() > 0:
        #     pdb.set_trace()

        return cam_rv


if __name__ == '__main__':
    pcm = PCM(512,512)
    ft = torch.randn([2,512,32,32])
    cam = torch.randn([2,1,32,32])
    pdb.set_trace()
    pcm.forward(cam, ft)
