import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class CosLayer(nn.Module):
    def __init__(self, num_feature, num_class,
                loss_type='all',
                s=30.0,
                m0=1.35,
                m1=0.50,
                m2=0.35
                ):
        super(CosLayer, self).__init__()
        self.num_feature = num_feature
        self.n_classes = num_class
        self.loss_type = loss_type
        self.s = s
        self.m0 = m0 #SphereFace margin
        self.m1 = m1 #ArcFace margin
        self.m2 = m2 #CosFace margin
        self.W = Parameter(torch.FloatTensor(num_class, num_feature))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, label=None):
        if (self.loss_type == 'softmax') or \
            (self.loss_type == 'adacos'):
            self.m0 = 1.0
            self.m1 = 0.0
            self.m2 = 0.0
        if self.loss_type == 'sphereface':
            self.m1 = 0.0
            self.m2 = 0.0
        if self.loss_type == 'arcface':
            self.m0 = 1.0
            self.m2 = 0.0
        if self.loss_type == 'cosface':
            self.m0 = 1.0
            self.m1 = 0.0

        # normalize features and weights
        x = F.normalize(x)
        W = F.normalize(self.W)

        # return output
        logits = F.linear(x, W)
        if label is None:
            return logits

        # calc margin
        theta = torch.acos(torch.clamp(logits, -1.0+1e-7, 1.0-1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        target_logits = torch.cos(self.m0 * theta + self.m1) - self.m2
        output = logits * (1 - one_hot) + target_logits * one_hot

        # calc optimal scale
        with torch.no_grad():
            pre_s = math.sqrt(2)*math.log(self.num_class-1)
            B_ave = torch.where(one_hot < 1, torch.exp(pre_s * logits),
                                             torch.zeros_like(logits))
            B_ave = torch.sum(B_ave) / x.size(0)
            theta_med = torch.median(theta[one_hot == 1])
            theta_med = torch.min(math.pi/4*torch.ones_like(theta_med),
                                  theta_med)
            opt_s = torch.log(B_ave) / torch.cos(theta_med)

        if (self.loss_type == 'adacos') or (self.loss_type == 'all'):
            self.s = opt_s

        # scale feature
        output *= self.s

        return output


def feature_similarity(x, label, similarity='cos'):
    n = label.size(0)
    mask = label.expand(n, n).t().eq(label.expand(n, n)).float()
    pos_mask = mask.triu(diagonal=1)
    neg_mask = (mask - 1).abs_().triu(diagonal=1)

    # inspired by attention
    if similarity == 'dot':
        sim_mat = x.matmul(x.t())
    elif similarity == 'cos':
        x = F.normalize(x)
        sim_mat = x.matmul(x.t())

    return sim_mat[pos_mask == 1], sim_mat[neg_mask == 1]

class CircleLoss(nn.Module):
    def __init__(self, m=0.25, gamma=80, similarity='cos'):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.similarity = similarity
        self.soft_plus = nn.Softplus()

    def forward(self, x, label):
        sp, sn = feature_similarity(x, label, similarity=self.similarity)
        ap = torch.relu(- sp.detach() + 1 + self.m)
        an = torch.relu(sn.detach() + self.m)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss_p = torch.sum(torch.exp(logit_p))
        loss_n = torch.sum(torch.exp(logit_n))

        loss = torch.log(1 + loss_p * loss_n)

        return loss