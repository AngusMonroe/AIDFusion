import torch
import torch.nn as nn
import dgl
import re
import math
import numpy as np
from torch.nn import init
import torch.nn.functional as F
from layers.attention_layer import PositionwiseFeedforwardLayer, IdentitylEncoding
from layers.gcn_layer import GCNLayer, simpleGCNLayer


class AIDFusionNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        edge_dim = net_params['edge_dim']
        self.hidden_dim = net_params['hidden_dim']
        dropout = net_params['dropout']
        self.n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.e_feat = net_params['edge_feat']
        self.dropout = nn.Dropout(dropout)
        self.embedding_hs = nn.ModuleList([])
        self.embedding_es = nn.ModuleList([])
        self.pos_embs = nn.ModuleList([])
        self.branches = nn.ModuleList([])
        self.views = net_params['views']
        self.ca_adj = None
        self.ia_adj = None
        self.transforms = nn.ModuleList([])
        self.node_num = []
        self.node_reprs = []
        for name in self.views:
            match = re.search(r'\d+', name)
            if match:
                in_dim = int(match.group())
                self.node_num.append(in_dim)
            else:
                raise ValueError('No number in the view name')

            self.embedding_hs.append(nn.Linear(in_dim, self.hidden_dim))
            self.embedding_es.append(nn.Linear(edge_dim, self.hidden_dim))
            self.pos_embs.append(IdentitylEncoding(d_model=self.hidden_dim, node_num=in_dim, dropout=dropout))
            layers = nn.ModuleList([TransformerEncoderLayer(self.hidden_dim, 1, dropout, 5) for _ in range(self.n_layers)])
            self.branches.append(layers)
        print('Node num:', self.node_num)
        self.ca_gcs = nn.ModuleList([simpleGCNLayer(self.hidden_dim, F.relu, 0.0, batch_norm=False, residual=True) for _ in range(self.n_layers)])
        self.hidden_node_num = sum(self.node_num) // (2 * len(self.node_num))
        for i in range(len(self.views)):
            self.transforms.append(NodeAlignment(self.hidden_dim, self.hidden_node_num, self.hidden_dim, dropout))
        self.recon_loss = 0.0
        self.pop_recon_loss = 0.0
        self.entropy_loss = 0.0
        self.orth_loss = 0.0

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, batched_graphs, device):
        hgs = []
        self.node_reprs = []
        hg_sim = []
        self.recon_loss = 0.0
        self.pop_recon_loss = 0.0
        self.entropy_loss = 0.0
        self.orth_loss = 0.0

        hs = []
        es = []
        for i in range(len(self.views)):
            g = batched_graphs[i].to(device)
            h = g.ndata['feat'].to(device)  # num x feat
            e = g.edata['feat'].to(device)

            h = self.embedding_hs[i](h).view(-1, self.node_num[i], self.hidden_dim)
            e = self.embedding_es[i](e)
            h = self.pos_embs[i](h)

            hs.append(h.clone())
            es.append(e.clone())

        for j in range(self.n_layers):
            for i in range(len(self.views)):
                h = hs[i]
                h = self.branches[i][j](h)
                hs[i] = h.clone()
            new_h = self.ca_gcs[j](torch.cat(hs, dim=1), self.ca_adj.to(device))
            hs = [new_h[:, sum(self.node_num[:i]):sum(self.node_num[:i + 1]), :] for i in range(len(self.views))]

        for i in range(len(self.views)):
            g = batched_graphs[i].to(device)
            h = hs[i].reshape(-1, self.hidden_dim)
            node_repr = self.transforms[i](g, h, e)
            self.node_reprs.append(node_repr.clone())

            g.ndata['h'] = h

            if self.readout == "sum":
                hg = dgl.sum_nodes(g, 'h')
            elif self.readout == "max":
                hg = dgl.max_nodes(g, 'h')
            elif self.readout == "mean":
                hg = dgl.mean_nodes(g, 'h')
            else:
                hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            hgs.append(hg.clone())
            hg_sim.append(self.self_compute_similarity(hg))

        for i in range(len(self.views)):
            self.entropy_loss = self.entropy_loss + self.transforms[i].entropy_loss
            for j in range(i, len(self.views)):
                if i != j:
                    if self.branches[i][-1].redundant_num > 0:
                        self.orth_loss = self.orth_loss + self.orthogonal_loss(self.branches[i][-1].redundant_token, self.branches[j][-1].redundant_token)
                    self.recon_loss = self.recon_loss + self.subj_contrastive_loss(self.node_reprs[i], self.node_reprs[j])
                    self.pop_recon_loss = self.pop_recon_loss + self.reconstruct_loss(hg_sim[i], hg_sim[j])

        return hgs

    def orthogonal_loss(self, tensor1, tensor2):
        # Compute dot product between normalized tensors
        norm_tensor1 = F.normalize(tensor1, dim=-1)
        norm_tensor2 = F.normalize(tensor2, dim=-1)
        dot_product = torch.sum(norm_tensor1 * norm_tensor2, dim=-1)
        loss = torch.mean(torch.abs(dot_product))

        return loss

    # apply reconstruction loss to compare representation and feature
    def reconstruct_loss(self, repr, feat):
        criterion = nn.MSELoss()
        loss = criterion(repr, feat)

        return loss

    def subj_contrastive_loss(self, repr, feat, tau=0.25):
        bz = repr.size(0)
        feat = feat.view(bz, -1)
        repr = repr.view(bz, -1)

        # compute similarity matrix between `repr` and `feat`
        sim_mat = self.compute_similarity(repr, feat)
        sim_mat = torch.exp(sim_mat / tau)

        pos_mask = torch.eye(feat.size(0), dtype=torch.bool, device=feat.device)
        pos_loss = sim_mat[pos_mask].sum()
        neg_loss = sim_mat[~pos_mask].sum()
        loss = -math.log(pos_loss / neg_loss + 1e-8)

        return loss

    def compute_similarity(self, h1, h2):

        # Compute dot product between each pair of vectors
        dot_products = torch.matmul(h1, h2.t())  # Shape: (num_vectors, num_vectors)

        # Compute the norms of each vector
        norm1 = torch.norm(h1, p=2, dim=1, keepdim=True)  # Shape: (num_vectors, 1)
        norm2 = torch.norm(h2, p=2, dim=1, keepdim=True)

        # Normalize the dot products to obtain cosine similarity
        cosine_similarity = dot_products / (norm1 * norm2 + 1e-8)  # Add a small epsilon to avoid division by zero

        return cosine_similarity

    def self_compute_similarity(self, hs):

        # Compute dot product between each pair of vectors
        dot_products = torch.matmul(hs, hs.t())  # Shape: (num_vectors, num_vectors)

        # Compute the norms of each vector
        norms = torch.norm(hs, p=2, dim=1, keepdim=True)  # Shape: (num_vectors, 1)

        # Normalize the dot products to obtain cosine similarity
        cosine_similarity = dot_products / (norms * norms.t() + 1e-8)  # Add a small epsilon to avoid division by zero

        return cosine_similarity


class NodeAlignment(nn.Module):
    def __init__(self, in_dim, out_node_num, feat_dim, dropout, activation=F.relu, aggregator_type="maxpool",
                 batch_norm=True, layer='GCN'):
        super().__init__()
        self.layer = layer
        if self.layer == 'GCN':
            self.feat_gc = GCNLayer(in_dim, feat_dim, activation, dropout, batch_norm, residual=True)
            self.pool_gc = GCNLayer(in_dim, out_node_num, activation, dropout, batch_norm, residual=True)
        else:
            raise NotImplementedError
        self.in_dim = in_dim
        self.feat_dim = feat_dim
        self.out_node_num = out_node_num
        self.entropy_loss = 0.0

    def forward(self, g, h, e):
        device = h.device
        bz = g.batch_size

        if self.layer in ['GCN']:
            feat, e = self.feat_gc(g, h, e)
            assign_tensor, e = self.pool_gc(g, h, e)
        else:
            raise NotImplementedError

        assign_tensor = torch.nn.functional.softmax(assign_tensor.view(bz, -1, self.out_node_num), dim=-1)
        feat = torch.matmul(assign_tensor.permute(0, 2, 1), feat.reshape(bz, -1, self.feat_dim)).reshape(-1, self.feat_dim)
        self.cal_entropy_loss(assign_tensor)

        scale = torch.sqrt(torch.FloatTensor([feat.shape[0]])).to(device)
        feat = torch.matmul(feat, feat.t()) / scale
        return feat

    def cal_entropy_loss(self, attn):
        entropy = (torch.distributions.Categorical(logits=attn).entropy()).mean()
        assert not torch.isnan(entropy)
        self.entropy_loss = entropy


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, n_heads, dropout, no_params=False, learnable_q=False):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.no_params = no_params

        assert hid_dim % n_heads == 0

        if not self.no_params:
            self.w_q = nn.Linear(in_dim, hid_dim)
            self.w_k = nn.Linear(in_dim, hid_dim)
            self.w_v = nn.Linear(in_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):

        scale = torch.sqrt(torch.FloatTensor([self.hid_dim // self.n_heads])).to(query.device)

        # Q,K,V计算与变形：
        bsz = query.shape[0]

        if not self.no_params:
            Q = self.w_q(query)
            K = self.w_k(key)
            V = self.w_v(value)
        else:
            Q = query
            K = key
            V = value

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.dropout(torch.softmax(energy, dim=-1))

        x = torch.matmul(attention, V)

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        x = self.fc(x)

        return x, attention.squeeze()


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, redundant_num=0):
        super().__init__()
        self.redundant_num = redundant_num
        if redundant_num > 0:
            self.redundant_token = nn.Parameter(torch.rand(redundant_num, hid_dim), requires_grad=True)
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, hid_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        #  src = [batch size, src len, hid dim]
        #  src_mask = [batch size, 1, 1, src len]
        if self.redundant_num > 0:
            src = torch.cat([src, self.redundant_token.repeat(src.size(0), 1, 1)], dim=1)
        _src, _ = self.self_attention(src, src, src, src_mask)

        #  dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        #  src = [batch size, src len, hid dim]

        #  positionwise feedforward
        _src = self.positionwise_feedforward(src)

        #  dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]
        return src[:, :-self.redundant_num, :] if self.redundant_num > 0 else src
