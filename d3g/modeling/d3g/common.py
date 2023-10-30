from random import sample
import torch 
import math 
import torch.distributed as dist


def iou(candidates, gts):
    start, end = candidates[:,0], candidates[:,1]
    gt_start, gt_end = gts[:, 0], gts[:, 1]
    inter = torch.min(end, gt_end) - torch.max(start, gt_start)
    union = torch.max(end, gt_end) - torch.min(start, gt_start)
    return inter.clamp(min=0).float() / union


def gen_candidate_moments(num_clips):
    map2d = torch.ones(num_clips, num_clips)
    start, end = torch.where(map2d)   
    end += 1
    candidates = torch.stack([start, end], dim=1)
    return candidates.view(-1, num_clips, 2) # [num_clips, num_clips, 2]


def init_glance_memory(dataloader, length, window):
    gt_init = 3
    memory = {}
    count_videos = 0
    count_moments = 0
    for batch, idxs in dataloader:
        glances_list = batch.glances
        for glances, idx in zip(glances_list, idxs):
            num_sent = len(glances)
            counter = torch.zeros(num_sent, length) - 1
            # mark valid frames with zero
            for i, glance in enumerate(glances):
                index = torch.arange(glance - window, glance + window + 1)
                index = torch.clip(index, 0, length-1)
                counter[i][index] = 0
            counter.scatter_(dim=1, index=glances.unsqueeze(1), value=gt_init)
            memory[idx] = counter 
            count_videos += 1
            count_moments += counter.size(0)
    print(f"Init glance counter memory with window:{window} and find {count_videos} videos and {count_moments} moments at rank: {dist.get_rank()}")
    return memory 


def gen_gaussian_weight(anchors, length, sigma):
    """
    Args:
        anchors (list): list[item], each item with shape [num_sent], indicate the positions of glance annotations.
        length (int): video length used to generate guassian weight
        sigma (float): sigma used for gaussian distribution 
    Returns:
        weight2d_split(list): list[item], item: [num_sent, length, length]
    """
    num_sent_list = [item.size(0) for item in anchors]
    anchors = torch.cat(anchors, dim=0)
    N = anchors.size(0)
    
    # [N, lenght]
    x = torch.linspace(-1, 1, steps=length).view(1, length).expand(N, length).to(anchors.device)
    
    # normalize glance ind into range [-1, 1]
    u = (anchors / length) * 2 - 1 
    u = u.view(N, 1)
    
    # compute the weight 
    # [N, length]
    weight = torch.exp(-(x - u) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    weight /= torch.max(weight, dim=1, keepdim=True)[0] # normalize weight 
    
    # generate 2d-map weight 
    weight2d = torch.ones(length, length)
    begin, end = torch.where(weight2d)
    end = end + 1 
    mid = ((begin + end) / 2).long()
    
    # [length * length]
    begin_weight = weight[:, begin]
    mid_weight = weight[:, mid]
    end = torch.clip(end, min=0, max=length-1)
    end_weight = weight[:, end]
    
    #  the average of triplet weights
    final_weight = (begin_weight + mid_weight + end_weight) / 3. 
    
    weight2d = final_weight.view(N, length, -1)
    weight2d_split = torch.split(weight2d, num_sent_list, dim=0)
    
    return weight2d_split 

    
def gen_multi_glances_gaussian_weight(glance_counters, sims, length, sigma, cnum=1):
    """
    Args:
        glance_counter (list): list[item], each item with shape [num_sent, length]
        sims (list): list[item], each item with [num_sent, length]
        length (int): video length used to generate guassian weight
        sigma (float): sigma used for gaussian distribution 
    """
    num_sent_list = [item.size(0) for item in glance_counters]
    glance_counters = torch.cat(glance_counters, dim=0)
    sims = torch.cat(sims, dim=0)
    num_sent_sum = glance_counters.size(0)
    device = glance_counters[0].device 
    
    # generate template gaussian distribution 
    # normalize glance ind into range [-1, 1]
    u = (torch.arange(0, length) / length) * 2 - 1 
    u = u.view(-1, 1).to(device)
    x = torch.linspace(-1, 1, steps=length).view(1, length).expand(u.size(0), length).to(device)
    
    # compute the weight 
    # [length, length]
    weight = torch.exp(-(x - u) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    weight /= torch.max(weight, dim=1, keepdim=True)[0] # normalize weight 
    
    # repeat for each sentence 
    # [num_sent_sum, length, length]
    weight_all = weight.unsqueeze(dim=0).repeat(num_sent_sum, 1, 1)
    # [num_sent_sum, length]
    mask = glance_counters >= cnum 
    mask_num = mask.sum(dim=1, keepdim=True)
   
    # reweight according to the similarity  
    weight_all = weight_all * sims.unsqueeze(dim=-1) 
    weight = weight_all * mask.unsqueeze(dim=-1)
    weight = weight.sum(dim=1) / mask_num
    
    # generate 2d-map weight 
    weight2d = torch.ones(length, length)
    begin, end = torch.where(weight2d)
    end = end + 1 
    mid = ((begin + end) / 2).long()
    
    # [length, length]
    mid_weight = weight[:, mid] 
    begin_weight = weight[:, begin]
    end = torch.clip(end, min=0, max=length-1)
    end_weight = weight[:, end]
    
    # triplet weight
    final_weight = (begin_weight + mid_weight + end_weight) / 3.
 
    # [num_sent_sum, length, length]
    weight2d = final_weight.view(num_sent_sum, length, -1)
    weight2d_split = torch.split(weight2d, num_sent_list, dim=0)
    
    return weight2d_split