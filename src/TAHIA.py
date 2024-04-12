import math
import pickle
import dgl
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from util_funcs import cos_sim, restart_random_walk
import os


class TAHIA(nn.Module):
    """
    Decode neighbors of input graph.
    """

    def __init__(self, cf, g, dataset):
        super(TAHIA, self).__init__()
        self.__dict__.update(cf.get_model_conf())
        # ! Init variables
        self.dev = cf.dev
        self.ti, self.ri, self.types, self.ud_rels = g.t_info, g.r_info, g.types, g.undirected_relations
        feat_dim, mp_emb_dim = g.features.shape[1], list(g.mp_emb_dict.values())[0].shape[1]
        self.non_linear = nn.ReLU()
        # ! Graph Structure Learning
        MD = nn.ModuleDict
        self.dataset = cf.dataset
        self.fgg_direct, self.fgg_left, self.fgg_right, self.fg_agg, self.sgg_gen, self.sg_agg, self.overall_g_agg = \
            MD({}), MD({}), MD({}), MD({}), MD({}), MD({}), MD({})
        # Feature encoder
        self.encoder = MD(dict(zip(g.types, [nn.Linear(g.features.shape[1], cf.com_feat_dim) for _ in g.types])))
        self.lstm_encoder = MD(dict(zip(g.types, [nn.LSTM(cf.com_feat_dim, int(cf.com_feat_dim/2), 1, bidirectional = True) for _ in g.types])))
        self.lstm_agg = GraphChannelAttLayer(len(self.types))
        for r in g.undirected_relations:
            # ! Feature Graph Generator
            self.fgg_direct[r] = GraphGenerator(cf.com_feat_dim, cf.num_head, cf.fgd_th, self.dev)
            self.fgg_left[r] = GraphGenerator(feat_dim, cf.num_head, cf.fgh_th, self.dev)
            self.fgg_right[r] = GraphGenerator(feat_dim, cf.num_head, cf.fgh_th, self.dev)
            self.fg_agg[r] = GraphChannelAttLayer(3)  # 3 = 1 (first-order/direct) + 2 (second-order)
            # self.fg_agg[r] = GraphChannelAttLayer(2)  # 3 = 1 (first-order/direct) + 2 (second-order)

            # ! Semantic Graph Generator
            self.sgg_gen[r] = MD(dict(
                zip(cf.mp_list, [GraphGenerator(mp_emb_dim, cf.num_head, cf.sem_th, self.dev) for _ in cf.mp_list])))
            self.sg_agg[r] = GraphChannelAttLayer(len(cf.mp_list))

            # ! Overall Graph Generator
            # self.overall_g_agg[r] = GraphChannelAttLayer(3, [1, 1, 10])  # 3 = feat-graph + sem-graph + ori_graph
            self.overall_g_agg[r] = GraphChannelAttLayer(2)  # 3 = feat-graph + sem-graph + ori_graph
            # self.overall_g_agg[r] = GraphChannelAttLayer(1)  # 3 = feat-graph + sem-graph + ori_graph

        # ! Graph Convolution
        if cf.conv_method == 'gcn':
            # self.GCN = GCN(g.n_feat, cf.emb_dim, g.n_class, cf.dropout)
            self.GCN = GCN(g.n_feat, cf.emb_dim, cf.com_feat_dim, cf.dropout)
        self.classify = Classifier(cf.com_feat_dim, 2)
        self.norm_order = cf.adj_norm_order
        current_path = os.getcwd()
        with open(current_path + '/results/' + dataset + '/random_walk/epoch.pkl', 'rb') as file:
            self.neighbors = pickle.load(file)
        # with open(current_path + '/results/acm/random_walk/epoch.pkl', 'rb') as file:
        #     self.neighbors = pickle.load(file)



    def forward(self, features, adj_ori, mp_emb, e):
        def get_rel_mat(mat, r):
            return mat[self.ri[r][0]:self.ri[r][1], self.ri[r][2]:self.ri[r][3]]

        def get_type_rows(mat, type):
            return mat[self.ti[type]['ind'], :]

        def gen_g_via_feat(graph_gen_func, mat, r):
            return graph_gen_func(get_type_rows(mat, r[0]), get_type_rows(mat, r[-1]))
        # ! Heterogeneous Feature Mapping
        # 不同类型的节点分别编码
        com_feat_mat = torch.cat([self.non_linear(self.encoder[t](features[self.ti[t]['ind']])) for t in self.types])

        # ! Heterogeneous Graph Generation

        new_adj = torch.zeros_like(adj_ori).to(self.dev)
        for r in self.ud_rels:
            ori_g = get_rel_mat(adj_ori, r)
            # ! Feature Graph Generation
            fg_direct = gen_g_via_feat(self.fgg_direct[r], com_feat_mat, r)

            fmat_l, fmat_r = features[self.ti[r[0]]['ind']], features[self.ti[r[-1]]['ind']]
            sim_l, sim_r = self.fgg_left[r](fmat_l, fmat_l), self.fgg_right[r](fmat_r, fmat_r)
            fg_left, fg_right = sim_l.mm(ori_g), sim_r.mm(ori_g.t()).t()

            feat_g = self.fg_agg[r]([fg_direct, fg_left, fg_right])

            # ! Semantic Graph Generation
            
            # sem_g_list = [gen_g_via_feat(self.sgg_gen[r][mp], mp_emb[mp], r) for mp in mp_emb]
            # sem_g = self.sg_agg[r](sem_g_list)
            # ! Overall Graph
            # Update relation sub-matixs
            # new_adj[self.ri[r][0]:self.ri[r][1], self.ri[r][2]:self.ri[r][3]] = self.overall_g_agg[r]([feat_g, sem_g, ori_g])  # update edge  e.g. AP
            new_adj[self.ri[r][0]:self.ri[r][1], self.ri[r][2]:self.ri[r][3]] = self.overall_g_agg[r]([feat_g, ori_g])  # update edge  e.g. AP
            # new_adj[self.ri[r][0]:self.ri[r][1], self.ri[r][2]:self.ri[r][3]] = self.overall_g_agg[r]([feat_g, sem_g])  # update edge  e.g. AP
            # new_adj[self.ri[r][0]:self.ri[r][1], self.ri[r][2]:self.ri[r][3]] = self.overall_g_agg[r]([feat_g])  # update edge  e.g. AP

        new_adj += new_adj.clone().t()  # sysmetric
        # ! Aggregate
        new_adj = F.normalize(new_adj, dim=0, p=self.norm_order)

        # logits = self.GCN(features, new_adj)
        gcnfeature = self.GCN(features, new_adj, self.dataset)
        add = torch.tensor([1e-8] * com_feat_mat.shape[1], requires_grad=True).type(torch.FloatTensor).to(self.dev)
        add = add.unsqueeze(0)
        com_feat_mat = torch.cat([com_feat_mat, add])
        lens = self.lens
        target_type = self.target_type
        # neighbors = restart_random_walk(adj_ori, 100, self.ti, lens, target_type, e)
        neighbors = self.neighbors
        fe_agg = []
        fe_agg.append(com_feat_mat[self.ti[target_type]['ind']])
        for i, t in enumerate(self.types):
            if t == target_type:
                continue
            neighbor = neighbors[i]
            fe = com_feat_mat[numpy.array(neighbor)].view(-1, lens[i], com_feat_mat.shape[1])
            all_state, last_state = self.lstm_encoder[t](fe)
            fe_t = torch.mean(all_state, 1)
            fe_agg.append(fe_t.clone())
        fe_agg_all = self.lstm_agg(fe_agg)
        concat = torch.cat([fe_agg_all, gcnfeature[: fe_agg_all.shape[0]]], dim=1)
        logits = self.classify(concat)
        return logits, new_adj


class MetricCalcLayer(nn.Module):
    def __init__(self, nhid):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, nhid))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, h):
        return h * self.weight


class GraphGenerator(nn.Module):
    """
    Generate graph using similarity.
    """

    def __init__(self, dim, num_head=2, threshold=0.1, dev=None):
        super(GraphGenerator, self).__init__()
        self.threshold = threshold
        self.metric_layer = nn.ModuleList()
        for i in range(num_head):
            self.metric_layer.append(MetricCalcLayer(dim))
        self.num_head = num_head
        self.dev = dev

    def forward(self, left_h, right_h):
        """

        Args:
            left_h: left_node_num * hidden_dim/feat_dim
            right_h: right_node_num * hidden_dim/feat_dim
        Returns:

        """
        if torch.sum(left_h) == 0 or torch.sum(right_h) == 0:
            return torch.zeros((left_h.shape[0], right_h.shape[0])).to(self.dev)
        s = torch.zeros((left_h.shape[0], right_h.shape[0])).to(self.dev)
        zero_lines = torch.nonzero(torch.sum(left_h, 1) == 0)
        # The ReLU function will generate zero lines, which lead to the nan (devided by zero) problem.
        if len(zero_lines) > 0:
            left_h[zero_lines, :] += 1e-8
        for i in range(self.num_head):
            weighted_left_h = self.metric_layer[i](left_h)
            weighted_right_h = self.metric_layer[i](right_h)
            s += cos_sim(weighted_left_h, weighted_right_h)
        s /= self.num_head
        s = torch.where(s < self.threshold, torch.zeros_like(s), s)
        return s


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, d):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        if d == 'credit':
            return F.log_softmax(x, dim=1)
        return x


class GraphChannelAttLayer(nn.Module):

    def __init__(self, num_channel, weights=None):
        super(GraphChannelAttLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_channel, 1, 1))
        nn.init.constant_(self.weight, 0.1)  # equal weight
        # if weights != None:
        #     # self.weight.data = nn.Parameter(torch.Tensor(weights).reshape(self.weight.shape))
        #     with torch.no_grad():
        #         w = torch.Tensor(weights).reshape(self.weight.shape)
        #         self.weight.copy_(w)

    def forward(self, adj_list):
        adj_list = torch.stack(adj_list)
        # Row normalization of all graphs generated
        adj_list = F.normalize(adj_list, dim=1, p=1)
        # Hadamard product + summation -> Conv
        return torch.sum(adj_list * F.softmax(self.weight, dim=0), dim=0)

class GraphConvolution(nn.Module):  # GCN AHW
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        support = torch.spmm(inputs, self.weight)  # HW in GCN
        output = torch.spmm(adj, support)  # AHW
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class Classifier(nn.Module):
    def __init__(self, dim, out):
        super(Classifier, self).__init__()
        self.fun1 = nn.Linear(dim * 2, dim)
        self.act = nn.LeakyReLU()
        self.fun2 = nn.Linear(dim, out)
    def forward(self, x):
        x = self.fun1(x)
        x = self.fun2(self.act(x))
        return x

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout#dropout参数
        self.in_features = in_features#结点向量的特征维度
        self.out_features = out_features#经过GAT之后的特征维度
        self.alpha = alpha#LeakyReLU参数
        self.concat = concat# 如果为true, 再进行elu激活

        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)# xavier初始化
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)# xavier初始化

        # 定义leakyReLU激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        '''
        adj图邻接矩阵，维度[N,N]非零即一
        h.shape: (N, in_features), self.W.shape:(in_features,out_features)
        Wh.shape: (N, out_features)
        '''
        Wh = torch.mm(h, self.W) # 对应eij的计算公式
        e = self._prepare_attentional_mechanism_input(Wh)#对应LeakyReLU(eij)计算公式

        zero_vec = -9e15*torch.ones_like(e)#将没有链接的边设置为负无穷
        attention = torch.where(adj > 0, e, zero_vec)#[N,N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留
        # 否则需要mask设置为非常小的值，因为softmax的时候这个最小值会不考虑
        attention = F.softmax(attention, dim=1)# softmax形状保持不变[N,N]，得到归一化的注意力全忠！
        attention = F.dropout(attention, self.dropout, training=self.training)# dropout,防止过拟合
        h_prime = torch.matmul(attention, Wh)#[N,N].[N,out_features]=>[N,out_features]

        # 得到由周围节点通过注意力权重进行更新后的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        # 先分别与a相乘再进行拼接
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha=0.2, nheads=2):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        # 加入Multi-head机制
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

class GraphSAGEConvLayer(nn.Module):
    def __init__(self, in_feats, out_feats, aggregator_type='mean'):
        super(GraphSAGEConvLayer, self).__init__()
        self.aggregator_type = aggregator_type
        if self.aggregator_type == 'mean':
            self.aggregator = dgl.nn.pytorch.SAGEConv(in_feats, out_feats, aggregator_type='mean')
        else:
            self.aggregator = dgl.nn.pytorch.SAGEConv(in_feats, out_feats, aggregator_type='pool')

    def forward(self, g, h):
        
        return self.aggregator(g, h)

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, n_class, drop, num_layer=3):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(GraphSAGEConvLayer(in_feats, hidden_feats))
        for _ in range(num_layer - 2):
            self.layers.append(GraphSAGEConvLayer(hidden_feats, hidden_feats))
        self.layers.append(GraphSAGEConvLayer(hidden_feats, n_class))
        
    def forward(self, h, adj):
        # print(adj)
        index = torch.nonzero(adj.detach(), as_tuple=True)
        g = dgl.graph(index, num_nodes=adj.shape[0])
        g.edata['weight'] = adj[index[0], index[1]].detach()
        for layer in self.layers:
            h = layer(g, h)
        return h