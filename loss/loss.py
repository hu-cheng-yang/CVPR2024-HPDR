import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import hyptorch.nn as hypnn
from hyptorch.pmath import dist_matrix


    
class Loss(nn.Module):
    def __init__(self, nb_proxies, sz_embed, mrg=0.1, tau=0.1, hyp_c=0.01, clip_r=2.3, nb_proxies_top=10):
        super().__init__()
        self.nb_proxies = nb_proxies
        self.nb_proxies_top = nb_proxies_top
        self.sz_embed = sz_embed
        self.tau = tau
        self.hyp_c = hyp_c
        self.mrg = mrg
        self.clip_r = clip_r
        self.first_time = 0
        
        self.lcas = torch.randn(self.nb_proxies, self.sz_embed)
        self.lcas = self.lcas / math.sqrt(self.sz_embed) * clip_r * 0.9
        self.lcas = torch.nn.Parameter(self.lcas)
        self.to_hyperbolic = hypnn.ToPoincare(c=hyp_c, ball_dim=sz_embed, riemannian=True, clip_r=clip_r, train_c=False)

        self.o = torch.zeros(1, self.sz_embed)

        self.lcas_top = torch.randn(self.nb_proxies_top, self.sz_embed)
        self.lcas_top = self.lcas_top / math.sqrt(self.sz_embed) * clip_r * 0.9
        self.lcas_top = torch.nn.Parameter(self.lcas_top)

        self.dist_f = lambda x, y: dist_matrix(x, y, c=hyp_c)

        self.linear = nn.Linear(self.nb_proxies, 2, bias=False)

        self.picl_loss = nn.CrossEntropyLoss()
        self.ii_loss = nn.MSELoss()

        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, z_s, z_s_1, y, y_d=None):
  
        bs = len(z_s)

        lcas = self.to_hyperbolic(self.lcas)

        o = self.to_hyperbolic(self.o).to(lcas.device)

        all_nodes = torch.cat([z_s, lcas])
        all_dist_matrix = self.dist_f(all_nodes, all_nodes)

        sim_matrix = - all_dist_matrix[:bs, bs:]
        sim_matrix_1 = - self.dist_f(z_s_1, lcas)

        sim_matrix_prototype = - self.dist_f(lcas, lcas)
        sim_matrix_origin = - self.dist_f(o, lcas)

        mask = torch.zeros_like(sim_matrix)

        for i in range(bs):
            if y[i] == 0:
                mask[i, self.nb_proxies // 2 :] = 1
            elif y[i] == 1:
                mask[i, : self.nb_proxies // 2] = 1
            else:
                raise NotImplementedError()
        sim_matrix_masky = sim_matrix.clone().masked_fill(mask.bool(), - float('inf'))

        distance_rank = torch.argsort(sim_matrix_masky, dim=-1, descending=True)
        select_prot = torch.zeros(bs).cuda()
        prototype_number = dict()
        for i in range(self.nb_proxies):
            prototype_number[i] = 0
        
        sample_per_prototype = (bs) // self.nb_proxies

        for i in range(bs):
            for j in range(self.nb_proxies):
                if prototype_number[distance_rank[i, j].item()] < sample_per_prototype:
                    select_prot[i] = distance_rank[i, j].item()
                    prototype_number[distance_rank[i, j].item()] += 1
                    break

        loss_PI = self.picl_loss(sim_matrix, select_prot.long())
        loss_PI_1 = self.picl_loss(sim_matrix_1, select_prot.long())

        loss_II = self.ii_loss(sim_matrix, sim_matrix_1.detach()) + self.ii_loss(z_s, z_s_1.detach())
        
        cls_logist = sim_matrix
        cls = self.linear(cls_logist)

        cls_logist_1 = sim_matrix_1
        cls_1 = self.linear(cls_logist_1)

        loss_pred = self.cls_loss(cls, y)
        loss_pred_1 = self.cls_loss(cls_1, y)

        loss_pred = (loss_pred + loss_pred_1) / 2

        lcas_top = self.to_hyperbolic(self.lcas_top)

        prots = torch.cat([lcas.detach(), lcas_top])
        all_dist_matrix = self.dist_f(prots, prots)
        sim_matrix_prot_leaf = torch.exp(-all_dist_matrix[:self.nb_proxies, :self.nb_proxies]).detach()
        label = torch.zeros(self.nb_proxies).cuda()
        label[self.nb_proxies // 2 :] = 1
        one_hot_mat = (label.unsqueeze(1) == label.unsqueeze(0))
        sim_matrix_prot_leaf[one_hot_mat] += 1
        sim_matrix_prot_top = torch.exp(-all_dist_matrix[self.nb_proxies:, self.nb_proxies:]).detach()


        indices_tuple = self.get_reciprocal_triplets(sim_matrix_prot_leaf, topk=3, t_per_anchor = 3)
        loss_PP = self.compute_gHHC(lcas, lcas_top, all_dist_matrix[:self.nb_proxies, self.nb_proxies:], indices_tuple, sim_matrix_prot_leaf)

        indices_tuple2 = self.get_reciprocal_triplets(sim_matrix_prot_top, topk=3, t_per_anchor = 3)
        loss_PP += self.compute_gHHC(lcas_top, lcas_top, all_dist_matrix[self.nb_proxies:, self.nb_proxies:], indices_tuple2, sim_matrix_prot_top)

        loss_PP_leaf = - torch.log(1 / (1 + torch.sum(torch.exp(sim_matrix_prototype), dim=-1))).mean()
        loss_PP_origin = - torch.log(1 / (1 + torch.sum(torch.exp(sim_matrix_origin), dim=-1))).mean()
        loss_PP = loss_PP + loss_PP_leaf + loss_PP_origin

        loss = (loss_PI + loss_PI_1) + loss_II + loss_PP + loss_pred

        loss_dict = dict(loss_PI=loss_PI.item(), loss_PI_1=loss_PI_1.item(), loss_II=loss_II.item(), loss_PP=loss_PP.item(), loss_all=loss.item(), loss_pred=loss_pred.item())
        
        return loss, loss_dict
    

    def forward_cls(self, z_s):

        lcas = self.to_hyperbolic(self.lcas)
        all_dist_matrix = self.dist_f(z_s, lcas)

        sim_matrix = torch.exp(- all_dist_matrix)
        cls = self.linear(sim_matrix)

        return cls
    



    def compute_gHHC(self, z_s, lcas, dist_matrix, indices_tuple, sim_matrix):
        i, j, k = indices_tuple
        bs = len(z_s)
        
        cp_dist = dist_matrix
        
        max_dists_ij = torch.maximum(cp_dist[i], cp_dist[j])
        lca_ij_prob = F.gumbel_softmax(-max_dists_ij / self.tau, dim=1, hard=True)
        lca_ij_idx = lca_ij_prob.argmax(-1)
        
        max_dists_ijk = torch.maximum(cp_dist[k], max_dists_ij)
        lca_ijk_prob = F.gumbel_softmax(-max_dists_ijk / self.tau, dim=1, hard=True)
        lca_ijk_idx = lca_ijk_prob.argmax(-1)
        
        dist_i_lca_ij, dist_i_lca_ijk = (cp_dist[i] * lca_ij_prob).sum(1), (cp_dist[i] * lca_ijk_prob).sum(1)
        dist_j_lca_ij, dist_j_lca_ijk = (cp_dist[j] * lca_ij_prob).sum(1), (cp_dist[j] * lca_ijk_prob).sum(1)
        dist_k_lca_ij, dist_k_lca_ijk = (cp_dist[k] * lca_ij_prob).sum(1), (cp_dist[k] * lca_ijk_prob).sum(1)
                    
        hc_loss = torch.relu(dist_i_lca_ij - dist_i_lca_ijk + self.mrg) \
                    + torch.relu(dist_j_lca_ij - dist_j_lca_ijk + self.mrg) \
                    + torch.relu(dist_k_lca_ijk - dist_k_lca_ij + self.mrg)
                                        
        hc_loss = hc_loss * (lca_ij_idx!=lca_ijk_idx).float()
        loss = hc_loss.mean()
                
        return loss
        
    def get_reciprocal_triplets(self, sim_matrix, topk=20, t_per_anchor = 100):
        anchor_idx, positive_idx, negative_idx = [], [], []

        topk_index = torch.topk(sim_matrix, topk)[1]
        nn_matrix = torch.zeros_like(sim_matrix).scatter_(1, topk_index, torch.ones_like(sim_matrix))
        sim_matrix = ((nn_matrix + nn_matrix.t())/2).float()
        sim_matrix = sim_matrix.fill_diagonal_(-1)
                
        for i in range(len(sim_matrix)):
            if len(torch.nonzero(sim_matrix[i]==1)) <= 1:
                continue
            pair_idxs1 = np.random.choice(torch.nonzero(sim_matrix[i]==1).squeeze().cpu().numpy(), t_per_anchor, replace=True)
            pair_idxs2 = np.random.choice(torch.nonzero(sim_matrix[i]<1).squeeze().cpu().numpy(), t_per_anchor, replace=True)              
            positive_idx.append(pair_idxs1)
            negative_idx.append(pair_idxs2)
            anchor_idx.append(np.ones(t_per_anchor) * i)
        anchor_idx = np.concatenate(anchor_idx)
        positive_idx = np.concatenate(positive_idx)
        negative_idx = np.concatenate(negative_idx)
        return anchor_idx, positive_idx, negative_idx