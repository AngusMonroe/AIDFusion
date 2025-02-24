import torch
import torch.utils.data
from torch.nn import functional as F
import time
import os
import numpy as np
import csv
import dgl
from dgl.data.utils import load_graphs, save_graphs
import networkx as nx
from tqdm import tqdm
from copy import deepcopy
from scipy.signal import butter, filtfilt
import random
random.seed(42)


name2path = {

    'abide_full_schaefer100': '/path/to/brain_binfile/abide_full_schaefer100.bin',
    'abide_full_AAL116': '/path/to/brain_binfile/abide_full_aal116.bin',

    'adni_AAL116': '/path/to/brain_binfile/adni_AAL116.bin',
    'adni_harvard48': '/path/to/brain_binfile/adni_harvard48.bin',
    'adni_kmeans100': '/path/to/brain_binfile/adni_kmeans100.bin',
    'adni_schaefer100': '/path/to/brain_binfile/adni_schaefer100.bin',
    'adni_schaefer100_bak': '/path/to/brain_binfile/adni_schaefer100_bak.bin',
    'adni_ward100': '/path/to/brain_binfile/adni_ward100.bin',

    'neurocon_AAL116': '/path/to/brain_binfile/neurocon_AAL116.bin',
    'neurocon_harvard48': '/path/to/brain_binfile/neurocon_harvard48.bin',
    'neurocon_kmeans100': '/path/to/brain_binfile/neurocon_kmeans100.bin',
    'neurocon_schaefer100': '/path/to/brain_binfile/neurocon_schaefer100.bin',
    'neurocon_ward100': '/path/to/brain_binfile/neurocon_ward100.bin',

    'ppmi_AAL116': '/path/to/brain_binfile/ppmi_AAL116.bin',
    'ppmi_harvard48': '/path/to/brain_binfile/ppmi_harvard48.bin',
    'ppmi_kmeans100': '/path/to/brain_binfile/ppmi_kmeans100.bin',
    'ppmi_schaefer100': '/path/to/brain_binfile/ppmi_schaefer100.bin',
    'ppmi_ward100': '/path/to/brain_binfile/ppmi_ward100.bin',

    'taowu_AAL116': '/path/to/brain_binfile/taowu_AAL116.bin',
    'taowu_harvard48': '/path/to/brain_binfile/taowu_harvard48.bin',
    'taowu_kmeans100': '/path/to/brain_binfile/taowu_kmeans100.bin',
    'taowu_schaefer100': '/path/to/brain_binfile/taowu_schaefer100.bin',
    'taowu_ward100': '/path/to/brain_binfile/taowu_ward100.bin',

    'matai_AAL116': '/path/to/brain_binfile/matai_AAL116.bin',
    'matai_harvard48': '/path/to/brain_binfile/matai_harvard48.bin',
    'matai_kmeans100': '/path/to/brain_binfile/matai_kmeans100.bin',
    'matai_schaefer100': '/path/to/brain_binfile/matai_schaefer100.bin',
    'matai_ward100': '/path/to/brain_binfile/matai_ward100.bin'
}

name2coor_path = {
    'atlas_200regions_5mm': '/path/to/brain_coordinate/coordinate_atlas200.csv',
    'atlas_200regions_8mm': '/path/to/brain_coordinate/coordinate_atlas200.csv',
    'schaefer100': '/path/to/brain_coordinate/schaefer100_coordinates.csv',
    'schaefer200': '/path/to/brain_coordinate/schaefer200_coordinates.csv',
    'schaefer500': '/path/to/brain_coordinate/schaefer500_coordinates.csv',
    'schaefer1000': '/path/to/brain_coordinate/schaefer1000_coordinates.csv',
    'AAL116': '/path/to/brain_coordinate/aal_coordinates.csv',
    'harvard48': '/path/to/brain_coordinate/harvard_coordinates.csv'
}


class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """

    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]
        self.graph_labels = lists[1]

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']


        This function is called inside a function in TUsDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']

    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    # print(non_self_edges_idx)
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)

    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g


class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, names, threshold=0.3, edge_ratio=0.0, node_feat_transform='original', norm=False, conbine_view=False):
        t0 = time.time()
        self.name = deepcopy(names)

        if conbine_view:
            self.create_combined_view(names)
        datasets = [self.load_single_view(name, threshold, edge_ratio, node_feat_transform, norm) for name in self.name]

        assert all(torch.tensor(datasets[0].graph_labels).equal(torch.tensor(dataset.graph_labels)) for dataset in datasets)
        dataset = []
        for i in range(len(datasets[0].graph_labels)):
            dataset.append([[d[i][0] for d in datasets], datasets[0][i][1]])
        # this function splits data into train/val/test and returns the indices
        self.all_idx = self.get_all_split_idx(datasets, names[0])

        self.all = dataset
        self.train = [self.format_dataset([dataset[idx] for idx in self.all_idx['train'][split_num]]) for split_num in range(10)]
        self.val = [self.format_dataset([dataset[idx] for idx in self.all_idx['val'][split_num]]) for split_num in range(10)]
        self.test = [self.format_dataset([dataset[idx] for idx in self.all_idx['test'][split_num]]) for split_num in range(10)]
        
        print("Time taken: {:.4f}s".format(time.time()-t0))

    def load_single_view(self, name, threshold=0.3, edge_ratio=0.0, node_feat_transform='original', norm=False):
        G_dataset, Labels = load_graphs(name2path[name])

        self.node_num = G_dataset[0].ndata['N_features'].size(0)

        print("[!] Dataset: ", name)

        # transfer DGLHeteroGraph to DGLFormDataset
        data = []
        min_feat_dim = G_dataset[0].ndata['N_features'].shape[-1]
        for i in range(len(G_dataset)):
            if G_dataset[i].ndata['N_features'].shape[-1] < min_feat_dim:
                min_feat_dim = G_dataset[i].ndata['N_features'].shape[-1]

        for i in tqdm(range(len(G_dataset))):
            if len(G_dataset[i].edata['E_features']) != self.node_num ** 2:
                G = nx.DiGraph(np.ones([self.node_num, self.node_num]))
                graph_dgl = dgl.from_networkx(G)
                graph_dgl.ndata['N_features'] = G_dataset[i].ndata['N_features']
                G_dataset[i] = graph_dgl
            G_dataset[i].edata['E_features'] = torch.from_numpy(
                np.corrcoef(G_dataset[i].ndata['N_features'].numpy())).clone().flatten().float()
            if edge_ratio:
                threshold_idx = int(len(G_dataset[i].edata['E_features']) * (1 - edge_ratio))
                threshold = sorted(G_dataset[i].edata['E_features'].tolist())[threshold_idx]

            G_dataset[i].remove_edges(
                torch.squeeze((torch.abs(G_dataset[i].edata['E_features']) < float(threshold)).nonzero()))
            # G_dataset[i].edata['E_features'][G_dataset[i].edata['E_features'] < 0] = 0
            G_dataset[i].edata['feat'] = G_dataset[i].edata['E_features'].unsqueeze(-1).clone()

            if name[:-7] == 'pearson' or node_feat_transform == 'original':
                G_dataset[i].ndata['feat'] = G_dataset[i].ndata['N_features'][:, :min_feat_dim].clone()
            elif node_feat_transform == 'one_hot':
                G_dataset[i].ndata['feat'] = torch.eye(self.node_num).clone()
            elif node_feat_transform == 'pearson':
                if norm:
                    corr = np.corrcoef(G_dataset[i].ndata['N_features'].numpy())
                    mean = np.mean(corr)
                    std = np.std(corr)
                    G_dataset[i].ndata['feat'] = torch.from_numpy((corr - mean) / std).clone()
                else:
                    G_dataset[i].ndata['feat'] = torch.from_numpy(
                        np.corrcoef(G_dataset[i].ndata['N_features'].numpy())).clone()
                G_dataset[i].ndata['feat'] = torch.nan_to_num(G_dataset[i].ndata['feat'], nan=0.0)
            else:
                raise NotImplementedError

            G_dataset[i].ndata.pop('N_features')
            G_dataset[i].edata.pop('E_features')
            data.append([G_dataset[i], Labels['glabel'].tolist()[i]])

        graphs = [d[0] for d in data]
        labels = [d[1] for d in data]

        for graph in graphs:
            graph.ndata['feat'] = graph.ndata['feat'].float()  # dgl 4.0
            # adding edge features for Residual Gated ConvNet, if not there
            if 'feat' not in graph.edata.keys():
                # graph.edata['feat'] = self.distances[graph.edges()]
                edge_feat_dim = graph.ndata['feat'].shape[1] # dim same as node feature dim
                graph.edata['feat'] = torch.ones(graph.number_of_edges(), edge_feat_dim)

        dataset = DGLFormDataset(graphs, labels)

        return dataset

    def create_combined_view(self, names):
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name = names[i] + '+' + names[j]
                self.name.append(name)
                if not os.path.exists(name2path[name]):
                    print('Creating combined view: ', name)
                    G_dataset1, Labels1 = load_graphs(name2path[names[i]])
                    G_dataset2, Labels2 = load_graphs(name2path[names[j]])
                    assert Labels1['glabel'].tolist() == Labels2['glabel'].tolist()
                    G_dataset = []
                    for G1, G2 in zip(G_dataset1, G_dataset2):
                        n1 = G1.ndata['N_features'].size(0)
                        n2 = G2.ndata['N_features'].size(0)
                        G = nx.DiGraph(np.ones([n1 + n2, n1 + n2]))
                        graph_dgl = dgl.from_networkx(G)
                        graph_dgl.ndata['N_features'] = torch.cat([G1.ndata['N_features'], G2.ndata['N_features']], dim=0)
                        graph_dgl.edata['E_features'] = torch.from_numpy(
                            np.corrcoef(graph_dgl.ndata['N_features'].numpy())).clone().flatten().float()
                        G_dataset.append(graph_dgl)
                    save_graphs(name2path[name], G_dataset, Labels1)

    def contruct_spatial_adj(self, k=10):
        coors = [self.get_3d_corr(name) for name in self.name]
        node_nums = [coor.shape[0] for coor in coors]
        ca_adj = torch.zeros(sum(node_nums), sum(node_nums))
        ia_adj = torch.zeros(sum(node_nums), sum(node_nums))
        for i in range(len(node_nums)):
            for j in range(len(node_nums)):
                dist = torch.cdist(torch.from_numpy(coors[i]), torch.from_numpy(coors[j]), p=2)

                if i == j:
                    dist = dist + torch.eye(dist.shape[0]) * 1e8

                # keep the largest k entries in each row of `dist`
                _, indices = torch.topk(dist, k, dim=1, largest=False)
                adj_mask = torch.zeros_like(dist)
                adj_mask.scatter_(1, indices, 1)
                if i == j:
                    ia_adj[sum(node_nums[:i]):sum(node_nums[:i + 1]), sum(node_nums[:j]):sum(node_nums[:j + 1])] = adj_mask.clone()
                else:
                    ca_adj[sum(node_nums[:i]):sum(node_nums[:i + 1]), sum(node_nums[:j]):sum(node_nums[:j + 1])] = adj_mask.clone()
        return ca_adj, ia_adj

    def get_all_split_idx(self, dataset, name):
        """
            - Split total number of graphs into 3 (train, val and test) in 80:10:10
            - Stratified split proportionate to original distribution of data with respect to classes
            - Using sklearn to perform the split and then save the indexes
            - Preparing 10 such combinations of indexes split to be used in Graph NNs
            - As with KFold, each of the 10 fold have unique test set.
        """
        root_idx_dir = './data/{}/'.format(name)
        if not os.path.exists(root_idx_dir):
            os.makedirs(root_idx_dir)
        all_idx = {}

        # If there are no idx files, do the split and store the files
        if not (os.path.exists(root_idx_dir + 'train.index')):
            print("[!] No data split for {}".format(name))
            raise NotImplementedError

        # reading idx from the files
        for section in ['train', 'val', 'test']:
            with open(root_idx_dir + section + '.index', 'r') as f:
                reader = csv.reader(f)
                all_idx[section] = [list(map(int, idx)) for idx in reader]
        return all_idx

    def format_dataset(self, dataset):  
        """
            Utility function to recover data,
            INTO-> dgl/pytorch compatible format 
        """
        graphs = [data[0] for data in dataset]
        labels = [data[1] for data in dataset]

        # for graph in graphs:
        #     #graph.ndata['feat'] = torch.FloatTensor(graph.ndata['feat'])
        #     graph.ndata['feat'] = graph.ndata['feat'].float() # dgl 4.0
        #     # adding edge features for Residual Gated ConvNet, if not there
        #     if 'feat' not in graph.edata.keys():
        #         # graph.edata['feat'] = self.distances[graph.edges()]
        #
        #         edge_feat_dim = graph.ndata['feat'].shape[1] # dim same as node feature dim
        #         graph.edata['feat'] = torch.ones(graph.number_of_edges(), edge_feat_dim)

        return DGLFormDataset(graphs, labels)

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))

        multi_graphs = [[] for i in range(len(graphs[0]))]
        for i in range(len(graphs)):
            for j in range(len(graphs[i])):
                multi_graphs[j].append(graphs[i][j])
        batched_graphs = []
        for i in range(len(multi_graphs)):
            batched_graphs.append(dgl.batch(multi_graphs[i]))

        return batched_graphs, labels

    def _sym_normalize_adj(self, adj):
        deg = torch.sum(adj, dim = 0)#.squeeze()
        deg_inv = torch.where(deg>0, 1./torch.sqrt(deg), torch.zeros(deg.size()))
        deg_inv = torch.diag(deg_inv)
        return torch.mm(deg_inv, torch.mm(adj, deg_inv))

    def _add_self_loops(self):

        # function for adding self loops
        # this function will be called only if self_loop flag is True
        for split_num in range(10):
            self.train[split_num].graph_lists = [self_loop(g) for g in self.train[split_num].graph_lists]
            self.val[split_num].graph_lists = [self_loop(g) for g in self.val[split_num].graph_lists]
            self.test[split_num].graph_lists = [self_loop(g) for g in self.test[split_num].graph_lists]
            
        for split_num in range(10):
            self.train[split_num] = DGLFormDataset(self.train[split_num].graph_lists, self.train[split_num].graph_labels)
            self.val[split_num] = DGLFormDataset(self.val[split_num].graph_lists, self.val[split_num].graph_labels)
            self.test[split_num] = DGLFormDataset(self.test[split_num].graph_lists, self.test[split_num].graph_labels)

    def get_3d_corr(self, name):
        atlas = name.split('_')[-1]
        path = name2coor_path[atlas]
        with open(path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='\n')
            if 'schaefer' not in name and 'AAL' not in name and 'harvard' not in name:
                coor = [row[1:] for row in spamreader][1:]
            else:
                coor = [row[1:] for row in spamreader]
        return np.array(coor, dtype='float')
