import torch
import torch.nn as nn
import torch.nn.functional as F

def feature_similarity(x, label, similarity='cos'):
    n = label.size(0)
    mask = label.expand(n, n).t().eq(label.expand(n, n)).float()
    pos_mask = mask.triu(diagonal=1)
    neg_mask = (mask - 1).abs_().triu(diagonal=1)

    # inspired by attention
    x = x.squeeze()
    if similarity == 'dot':
        sim_mat = x.matmul(x.transpose(0, 1))
    elif similarity == 'cos':
        x = F.normalize(x)
        sim_mat = x.matmul(x.transpose(0, 1))

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