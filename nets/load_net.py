"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.aidfusion import AIDFusionNet
import re
import torch
import torch.nn as nn
from layers.mlp_readout_layer import MLPReadout


def AIDFusion(net_params, trainset):
    return AIDFusionNet(net_params)


def gnn_model(MODEL_NAME, net_params, trainset):

    model = GNNNet(MODEL_NAME, net_params, trainset)
        
    return model


models = {
    "AIDFusion": AIDFusion
}

data_name2n_class = {
    'abide': 2,
    'adni': 6,
    'ppmi': 4,
    'taowu': 2,
    'neurocon': 2,
    'matai': 2
}


class GNNNet(nn.Module):
    def __init__(self, MODEL_NAME, net_params, trainset):
        super().__init__()
        self.name = MODEL_NAME
        views = net_params['views']
        out_dim = net_params['out_dim']
        data_name = views[0].split('_')[0]
        self.n_classes = data_name2n_class[data_name]
        net_params['n_classes'] = self.n_classes
        net_params['edge_dim'] = 1

        self.model = models[MODEL_NAME](net_params, trainset)
        self.lambda1 = net_params['lambda1']
        self.lambda2 = net_params['lambda2']
        self.lambda3 = net_params['lambda3']
        self.lambda4 = net_params['lambda4']
        self.MLP_layer = MLPReadout(out_dim, self.n_classes)

    def set_spatial_adj(self, ca_adj, ia_adj):
        # if 'CA' not in self.name:
        #     raise ValueError('This function is only for Multi models')
        self.model.ca_adj = ca_adj
        self.model.ia_adj = ia_adj

    def forward(self, batched_graphs, device):
        hgs = []
        hgs = self.model(batched_graphs, device)
        scores = self.MLP_layer(torch.stack(hgs).sum(dim=0).to(device))
        return scores

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        loss = loss + self.lambda1 * self.model.recon_loss + self.lambda2 * self.model.pop_recon_loss + self.lambda3 * self.model.entropy_loss + self.lambda4 * self.model.orth_loss

        return loss
