"""
GAT_nn.py
=========
Implementasi Graph Attention Network (GAT) untuk memodelkan hubungan spasial
antar persimpangan lalu lintas dalam jaringan jalan.

GAT digunakan sebagai alternatif atau pelengkap Transformer untuk menangkap
ketergantungan antar agen berdasarkan topologi graf jaringan jalan.

Referensi:
    Veličković et al., "Graph Attention Networks", ICLR 2018.
"""
import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np
# from torch_sparse import SparseTensor

class GraphAttentionLayer(nn.Module):
    """
    Satu lapisan Graph Attention Network (GAT).

    Lapisan ini menghitung representasi baru setiap node dengan cara
    mengagregasi fitur node-node tetangganya menggunakan bobot attention.

    :param in_features:  (int) Dimensi fitur masukan setiap node.
    :param out_features: (int) Dimensi fitur keluaran setiap node.
    :param dropout:      (float) Probabilitas dropout untuk regularisasi.
    :param alpha:        (float) Parameter negatif kemiringan LeakyReLU.
    :param concat:       (bool) Jika True, terapkan aktivasi ELU pada output.
    :param device:       (str) Perangkat komputasi ('cpu' atau 'cuda').
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, device='cpu'):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features   # 节点表示向量的输入特征维度
        self.out_features = out_features   # 节点表示向量的输出特征维度
        self.dropout = dropout    # dropout参数
        self.alpha = alpha     # leakyrelu激活的参数
        self.concat = concat   # 如果为true, 再进行elu激活
        self.device = device
        
        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features), device=self.device))  
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1), device=self.device))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)   # xavier初始化
        
        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.to(device)
    
    def forward(self, inp, adj):
        """
        Hitung attention antar node dan hasilkan representasi baru.

        :param inp: (torch.Tensor) Fitur masukan node, bentuk [N, in_features].
        :param adj: (torch.Tensor) Matriks adjacency graf, bentuk [N, N],
                    bernilai 1 jika ada edge dan 0 jika tidak.
        :return:    (torch.Tensor) Fitur keluaran node, bentuk [N, out_features].
        """
        h = torch.mm(inp, self.W)   # [N, out_features]
        N = h.size()[0]    # N 图的节点数
        
        a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2*self.out_features)
        # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # [N, N, 1] => [N, N] 图注意力的相关系数（未归一化）
        
        zero_vec = -1e12 * torch.ones_like(e, device=self.device)    # 将没有连接的边置为负无穷
        attention = torch.where(adj>0, e, zero_vec)   # [N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=1)    # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)   # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime 
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    
    
    
class GAT(nn.Module):
    """
    Multi-head Graph Attention Network (Dense version).

    Menggabungkan beberapa GraphAttentionLayer secara paralel (multi-head attention),
    lalu menghubungkan keluarannya untuk mendapatkan representasi node yang kaya.

    :param n_feat:   (int) Dimensi fitur masukan node.
    :param n_hid:    (int) Dimensi tersembunyi setiap head attention.
    :param n_class:  (int) Dimensi fitur keluaran akhir.
    :param dropout:  (float) Probabilitas dropout.
    :param alpha:    (float) Parameter kemiringan LeakyReLU.
    :param n_heads:  (int) Jumlah head attention yang diparalelkan.
    :param node_num: (int) Jumlah node (persimpangan) dalam satu lingkungan.
    :param n_thr:    (int) Jumlah thread / lingkungan paralel.
    :param device:   (str) Perangkat komputasi ('cpu' atau 'cuda').
    """
    def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads, node_num, n_thr, device='cpu'):
        super(GAT, self).__init__()
        self.dropout = dropout 
        self.node_num = node_num
        self.node_num_batch = node_num * n_thr
        self.device = device
        
        # 定义multi-head的图注意力层
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True, device=self.device) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)   # 加入pytorch的Module模块
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout,alpha=alpha, concat=False, device=self.device)
    
    def to_adj(self, edge_index):
        """Ubah daftar edge (edge_index) menjadi matriks adjacency padat."""
        # adj = torch.zeros(self.node_num_batch, self.node_num_batch, device=self.device) ## 其实是 node_num* n_thr
        adj = torch.zeros(edge_index.shape[0], edge_index.shape[0], device=self.device) ## 其实是 node_num* n_thr
        for i in range(edge_index.shape[0]):
            for j in edge_index[i]:
                if j != -1:
                    div_n = i // self.node_num
                    if div_n >= 1:
                        j_ind = j + self.node_num * div_n
                    else:
                        j_ind = j
                    adj[i][j_ind.long()] = 1
                    
        return adj.long()
    
    def forward(self, x, edge_index, backward=False):
        """
        Jalankan forward pass GAT.

        :param x:          (torch.Tensor) Fitur node, bentuk [N, n_feat].
        :param edge_index: (torch.Tensor) Indeks edge antar node.
        :param backward:   (bool) Tidak digunakan saat ini (reserved).
        :return:           (torch.Tensor) Log-softmax distribusi kelas, bentuk [N, n_class].
        """
        adj = self.to_adj(edge_index)
        # if not backward:
        #     adj = self.to_adj(edge_index)
        # else:
        #     adj = torch.eye(x.shape[0]).long().to(self.device)
        #     n = x.shape[0]
        #     m = n*n//2
        #     indices = torch.randperm(n * n)[:m]
        #     adj.view(-1)[indices] = 1
            
        x = F.dropout(x, self.dropout, training=self.training)   # dropout，防止过拟合
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # 将每个head得到的表示进行拼接
        x = F.dropout(x, self.dropout, training=self.training)   # dropout，防止过拟合
        x = F.elu(self.out_att(x, adj))   # 输出并激活
        return F.log_softmax(x, dim=1)  # log_softmax速度变快，保持数值稳定



# # 定义一个小图
# in_channels, hidden_channels, out_channels, dropout, alpha, heads, node_num = 16, 8, 32, 0.8, 0.2, 2, 8
# model = GAT(in_channels, hidden_channels, out_channels, dropout, alpha, heads, node_num)
# # edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 0, 5, 4]], dtype=torch.long)
# edge_index = torch.tensor([[ 3., -1.,  1.,  6.], [-1., -1.,  2.,  7.]])



# x = torch.randn(8, 16)  # 节点特征向量维度为 16
# out = model(x, edge_index)

# print(out.shape)
    