"""
Based on Official Implementation of [CVPR 2024] Gradient-based Parameter Selection for Efficient Fine-Tuning
https://github.com/FightingFighting/GPS/blob/main/utils/pruning.py
"""

import torch
import numpy as np
from contextlib import suppress

import torch.nn as nn
import torch.nn.functional as F

import re

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, normalize=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.normalize = normalize

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device
        if self.normalize:
            features = F.normalize(features, dim=2)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        # print(logits)
        exp_logits = torch.exp(logits) * logits_mask
        # print(exp_logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # print(log_prob)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def gps_masking_params(model, loader, args, amp_autocast=suppress):
    
    criterion = SupConLoss().cuda()

    model.train()    
    model.zero_grad()
    
    for input, target in loader:
        
        input = torch.cat([input[0], input[1]], dim=0)
        
        batch_size = target.shape[0]
        input, target = input.cuda(), target.cuda()
        
        with amp_autocast():
            features = model.forward_features(input)
            x1, x2 = torch.split(features, [batch_size, batch_size], dim=0)
            features = torch.cat([x1.unsqueeze(1), x2.unsqueeze(1)], dim=1)
            loss = criterion(features, target)
            
        loss.backward()
    
    W = ['q', 'k', 'v', 'proj', 'fc1', 'fc2']
    mapping_dict = {ori_name: f'lora_{ori_name}' for ori_name in W}
    for name, param in model.named_parameters():
        if any(kwd+'.weight' in name for kwd in W):
            grad = param.grad.data.cpu().numpy()
            param.grad = None
            param.requires_grad = False
            
            name = name.rsplit('.',1)[0]     # e.g., blocks.1.attn.q.weight -> blocks.1.attn.q
            
            if name == 'patch_embed.proj':
                b, c, h, w = grad.shape
                grad = np.reshape(grad, [b, -1])      # [out_dim, C, H, W] -> [out_dim, C x H x W]
                attr_path = 'lora_' + name
                attr_path = attr_path.replace('.proj', '')
            else:
                attr_path = re.sub(r'\bblocks\.(\d+)', r'blocks[\1]', name)     # e.g., blocks.1.attn.q -> blocks[1].attn.q
                for old, new in mapping_dict.items():
                    attr_path = attr_path.replace(f'.{old}', f'.{new}')          # e.g., blocks[1].attn.q -> blocks[1].attn.lora_q
            
            max_index = [abs(grad).argsort(1)[:, -1+k] for k in range(args.topk)]
            
            if args.topk == 1:
                max_index = max_index[0]
            else:
                max_index = np.concatenate(max_index)

            if name == 'patch_embed.proj':
                eval(f'model.{attr_path}.init_mask(max_index, patch_size=(h,w), in_chans=c)')
            else:
                eval(f'model.{attr_path}.init_mask(max_index)') 
    
    for param in model.parameters():
        if param.grad is not None:
            param.grad = None
    
    model.freeze_stages()
    
    del criterion, loss
    
                
def gps_masking_params_from_npz(model, args):

    model.train()
    
    external_idx = np.load(f'{args.pre_idx}/{args.data_set}.npz')
    
    W = ['q', 'k', 'v', 'proj', 'fc1', 'fc2']                               
    mapping_dict = {ori_name:f'lora_{ori_name}' for ori_name in W}
    for name, param in model.named_parameters():
        if any(kwd+'.weight' in name for kwd in W):   
            param.requires_grad = False
            name = name.rsplit('.',1)[0]        
            
            if name == 'patch_embed.proj':
                b, c, h, w = param.shape
                attr_path = 'lora_' + name
                attr_path = attr_path.replace('.proj', '')
            else:    
                attr_path = re.sub(r'\bblocks\.(\d+)', r'blocks[\1]', name)
                for old, new in mapping_dict.items():
                    attr_path = attr_path.replace(f'.{old}', f'.{new}')
            
            max_index = external_idx[name]
            max_index = max_index.ravel()
            
            if name == 'patch_embed.proj':
                eval(f'model.{attr_path}.init_mask(max_index, patch_size=(h,w), in_chans=c)')
            else:
                eval(f'model.{attr_path}.init_mask(max_index)')
    
    model.freeze_stages()
