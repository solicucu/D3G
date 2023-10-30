import enum
import torch
from torch import nn
from torch.functional import F
from .featpool import build_featpool  # downsample 1d temporal features to desired length
from .feat2d import build_feat2d  # use MaxPool1d/Conv1d to generate 2d proposal-level feature map from 1d temporal features
from .loss import  build_bce_with_weight_loss, GroupContrastiveLoss 
from .text_encoder import build_text_encoder
from .proposal_conv import build_proposal_conv
from .common import gen_gaussian_weight, gen_multi_glances_gaussian_weight
        
class D3G(nn.Module):
    def __init__(self, cfg):
        super(D3G, self).__init__()
        self.joint_space_size = cfg.MODEL.D3G.JOINT_SPACE_SIZE
        self.encoder_name = cfg.MODEL.D3G.TEXT_ENCODER.NAME
        self.sigma = cfg.SOLVER.SIGMA 
        self.thresh = cfg.SOLVER.THRESH 
        self.window = cfg.SOLVER.WINDOW 
        self.use_dga = cfg.SOLVER.USE_DGA
        self.memory = {}
        self.sim_memory = {}
        self.first_time = True 
        self.continuous_num = 1
        
        # 2D-TAN
        self.featpool = build_featpool(cfg) 
        self.feat2d = build_feat2d(cfg)
        self.text_encoder = build_text_encoder(cfg)
        self.proposal_conv = build_proposal_conv(cfg, self.feat2d.mask2d)
        # SA-GCL 
        self.group_contrastive_loss = GroupContrastiveLoss(cfg, self.feat2d.mask2d)
        self.bce_loss = build_bce_with_weight_loss(cfg, self.feat2d.mask2d)
        
    
    def init_memory(self, memory):
        self.memory = memory 
    
    @torch.no_grad()
    def update_memory(self, batches, idxs, feats):
        """
        Args:
            batches.glances (list): list[item], each item with shape [num_sent], indcate the position of each glance annotation.
            idxs (tensor): [B], indexs of current sampled videos. 
            feats (tensor): [B, C, N], video features after pooling.
        Returns:
            counters (list): list[item], each item with shape [num_sent, N]
            sims_split (list): list[item], each item with shape [num_sent, N], record the relavance between glance feat and neighbor frames.
        """
        # [B, C, N]
        feats_norm = F.normalize(feats, dim=1)
        sims = []
        num_sent_list = []
        for vid_feats, glances, idx in zip(feats_norm, batches.glances, idxs):
            # [C, num_sent]
            glances_feats = vid_feats.index_select(1, glances)
            
            # sim: [num_sent, C] x [C, N] -> [num_sent, N]
            sim = torch.mm(glances_feats.t(), vid_feats)
            
            # update sim with momentum 
            alpha = 0.7
            if not self.first_time:
                previous_sim = self.sim_memory[idx]
                sim = sim * alpha + previous_sim * (1 - alpha) 
            self.sim_memory[idx] = sim  
            
            sims.append(sim)
            num_sent_list.append(sim.size(0))
            
        # [num_sent_sum, N]
        sims = torch.cat(sims, dim=0)
        # get glances counter 
        counters = [self.memory[idx] for idx in idxs]
        # [num_sent_sum, N]
        counters_all = torch.cat(counters, dim=0).to(sims.device)
        
        # select similar frames with similarities >= sim_thresh, ++
        # only frames within the specified window are valid.
        valid_mask = (counters_all >= 0).to(sims.device)
        pos = sims >= self.thresh 
        # [num_sent_sum, N]
        valid_pos = valid_mask * pos  
        counters_all[valid_pos] = counters_all[valid_pos] + 1 
        
        # previous pos, --
        neg = sims < self.thresh
        valid_mask = (counters_all >= 1).to(sims.device)
        valid_neg = valid_mask * neg 
        counters_all[valid_neg] = counters_all[valid_neg] - 1
        
        # update memory 
        counters = torch.split(counters_all, num_sent_list)
        sims_split = torch.split(sims, num_sent_list)
        for i, idx in enumerate(idxs):
            self.memory[idx] = counters[i].cpu()
            
        return counters, sims_split
        
     
    def forward(self, batches, cur_epoch=1, idxs=None):
        """
        Arguments:
           batches.gmasks2d (list): list[item], each item with shape [num_sent, N, N], where N = num_clip 
           batches.glances (list): list[item], each item with shape [num_sent]
           batches.feats (tensor): [B, C, pre_num_clip], video features 
        """
        # update first time 
        if cur_epoch > 1:
            self.first_time = False
        
        # backbone
        gmasks2d = batches.gmasks2d 
        assert len(gmasks2d) == batches.feats.size(0)
        for idx, (gmask, sent) in enumerate(zip(gmasks2d, batches.queries)):
            assert gmask.size(0) == sent.size(0)
            assert gmask.size(0) == batches.num_sentence[idx]
        feats = self.featpool(batches.feats)  # from pre_num_clip to num_clip with overlapped average pooling, e.g., 256 -> 128
        map2d = self.feat2d(feats)  # use MaxPool1d to generate 2d proposal-level feature map from 1d temporal features
       
        map2d = self.proposal_conv(map2d)
        sent_feat = self.text_encoder(batches.queries, batches.wordlens)

        # loss
        if self.training:
            if self.use_dga:
                length = gmasks2d[0].size(-1)
                glance_counters, sims = self.update_memory(batches, idxs, feats) 
                # list[item], item:[num_sent, N, N]
                weight2d = gen_multi_glances_gaussian_weight(glance_counters, sims, length, self.sigma, self.continuous_num)
            else:
                length = gmasks2d[0].size(-1)
                glances = batches.glances
                # list[item], item:[num_sent, N, N]
                weight2d = gen_gaussian_weight(glances, length, self.sigma)
                
            loss = self.group_contrastive_loss(map2d, sent_feat, gmasks2d, weight2d, cur_epoch)
            return loss 
        else:
            contrastive_scores = []
            for i, sf in enumerate(sent_feat):
                # contrastive part
                vid_feat = map2d[i, ...]  # C x T x T
                vid_feat_norm = F.normalize(vid_feat, dim=0)
                sf_norm = F.normalize(sf, dim=1)  # num_sent x C
                _, T, _ = vid_feat.size()
                contrastive_score = torch.mm(sf_norm, vid_feat_norm.reshape(vid_feat.size(0), -1)).reshape(-1, T, T) * self.feat2d.mask2d  # num_sent x T x T
                contrastive_scores.append(contrastive_score)
            return contrastive_scores 
