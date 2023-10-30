import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MoCo(torch.nn.modules.loss._Loss):
    def __init__(self, device, T=0.5):
        """
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        self.T = T
        self.device = device

    def forward(self, emb_anchor, emb_positive, queue):
        # print(queue.shape)
        # print(emb_anchor.shape)
        # L2 normalize
        emb_anchor = torch.mm(torch.diag(torch.sum(torch.pow(emb_anchor, 2), axis=1) ** (-0.5)), emb_anchor)
        emb_positive = torch.mm(torch.diag(torch.sum(torch.pow(emb_positive, 2), axis=1) ** (-0.5)), emb_positive)
        queue = torch.mm(torch.diag(torch.sum(torch.pow(queue, 2), axis=1) ** (-0.5)), queue)

        # positive logits: Nx1, negative logits: NxK
        l_pos = torch.einsum('nc,nc->n', [emb_anchor, emb_positive]).unsqueeze(-1)
        l_neg = torch.einsum('nc,kc->nk', [emb_anchor, queue])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        # loss
        # print(logits)
        loss = F.cross_entropy(logits, labels)
        
        return loss


class BYOL(torch.nn.modules.loss._Loss): 
    """

    """
    def __init__(self, device, T=0.5):
        """
        T: softmax temperature (default: 0.07)
        """
        super(BYOL, self).__init__()
        self.T = T
        self.device = device
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, emb_anchor, emb_positive):

        # L2 normalize

        emb_neg_1 = 0.5*(emb_anchor + emb_positive)
        emb_neg_1 = torch.mm(torch.diag(torch.sum(torch.pow(emb_neg_1, 2), axis=1) ** (-0.5)), emb_neg_1)
        emb_anchor = torch.mm(torch.diag(torch.sum(torch.pow(emb_anchor, 2), axis=1) ** (-0.5)), emb_anchor)
        emb_positive = torch.mm(torch.diag(torch.sum(torch.pow(emb_positive, 2), axis=1) ** (-0.5)), emb_positive)

        # compute instance-aware world representation, Nx1
        sim = torch.mm(emb_anchor, emb_positive.t()) / self.T
        weight = self.softmax(sim)
        emb_neg = torch.mm(weight, emb_positive)

        # positive logits: Nxk, negative logits: NxK
        l_pos = torch.einsum('nc,nc->n', [emb_anchor, emb_positive]).unsqueeze(-1)
        # l_neg = torch.mm(emb_anchor, emb_positive.t())
        l_neg = torch.einsum('nc,nc->n', [emb_anchor, emb_neg]).unsqueeze(-1)
        l_neg_1 = torch.einsum('nc,nc->n', [emb_anchor, emb_neg_1]).unsqueeze(-1)

        loss = - l_pos.sum() #原始BYOL

        #loss = l_neg_1.sum()-l_pos.sum() #改进版

        #
                
        return loss


class SimSiam(torch.nn.modules.loss._Loss):

    def __init__(self, device, T=0.5):
        """
        T: softmax temperature (default: 0.07)
        """
        super(SimSiam, self).__init__()
        self.T = T
        self.device = device

    def forward(self, p1, p2, z1, z2):

        # L2 normalize
        p1 = F.normalize(p1, p=2, dim=1)
        p2 = F.normalize(p2, p=2, dim=1)
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        # mutual prediction
        l_pos1 = torch.einsum('nc,nc->n', [p1, z2.detach()]).unsqueeze(-1)
        l_pos2 = torch.einsum('nc,nc->n', [p2, z1.detach()]).unsqueeze(-1)

        loss = - (l_pos1.sum() + l_pos2.sum())
                
        return loss


class OurLoss(torch.nn.modules.loss._Loss):  

    def __init__(self, device, margin=0.5, sigma=2.0, T=2.0):
        """
        T: softmax temperature (default: 0.07)
        """
        super(OurLoss, self).__init__()
        self.T = T
        self.device = device
        self.margin = margin
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigma = sigma

    def forward(self, emb_anchor, emb_positive):
        # L2 normalize, Nxk, Nxk
        emb_anchor = F.normalize(emb_anchor, p=2, dim=1)
        emb_positive = F.normalize(emb_positive, p=2, dim=1)
        # print('emb_anchor',emb_anchor.shape) #32,128
        # compute instance-aware world representation, Nx1
        sim = torch.mm(emb_anchor, emb_positive.t()) 
        weight = self.softmax(sim)
        # print(weight)
        neg = torch.mm(weight, emb_positive)
        #
        # # representation similarity of pos/neg pairs
        l_pos = torch.exp(-torch.sum(torch.pow(emb_anchor - emb_positive, 2), dim=1) / (2 * self.sigma ** 2))
        # print('l_pos',l_pos.shape) #32
        l_neg = torch.exp(-torch.sum(torch.pow(emb_anchor - neg, 2), dim=1) / (2 * self.sigma ** 2))
        #
        zero_matrix = torch.zeros(l_pos.shape).to(self.device)
        loss = - l_pos.mean()
        return loss




class SimCLR(torch.nn.modules.loss._Loss):

    def __init__(self, device, T=1.0):
        """
        T: softmax temperature (default: 0.07)
        """
        super(SimCLR, self).__init__()
        self.T = T
        self.device = device

    def forward(self, emb_anchor, emb_positive):
        
        # L2 normalize
        emb_anchor = torch.mm(torch.diag(torch.sum(torch.pow(emb_anchor, 2), axis=1) ** (-0.5)), emb_anchor)
        emb_positive = torch.mm(torch.diag(torch.sum(torch.pow(emb_positive, 2), axis=1) ** (-0.5)), emb_positive)
        N = emb_anchor.shape[0]
        emb_total = torch.cat([emb_anchor, emb_positive], dim=0)

        # representation similarity matrix, NxN
        logits = torch.mm(emb_total, emb_total.t())
        logits[torch.arange(2*N), torch.arange(2*N)] =  -1e10
        logits /= self.T

        # cross entropy
        labels = torch.LongTensor(torch.cat([torch.arange(N, 2*N), torch.arange(N)])).to(self.device)
        loss = F.cross_entropy(logits, labels)
                
        return loss


class SupCon(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, device, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupCon, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device

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
        # print('feature',features.shape)  #[128, 2, 128]
        # print('labels',labels.shape)  #[128]
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        # print('feature', features.shape)  # [128, 2, 128]
        batch_size = features.shape[0]  #128
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None: #走这个
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            # print('mask',mask.shape)  # [128, 128]
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  #2
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
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss