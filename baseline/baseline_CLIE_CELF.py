import random
import os
from tqdm import tqdm
import numpy as np 
import glob
import pandas as pd
import torch
import scipy.sparse as sp 
import torch.nn as nn
import os
import time
import re

from diffuse import IC

random.seed(1)


class GNN_skip_small(nn.Module):
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_hidden_3, dropout):
        super(GNN_skip_small, self).__init__()
        self.fc1 = nn.Linear(2*n_feat, n_hidden_1)
        self.fc2 = nn.Linear(2*n_hidden_1, n_hidden_2)
        self.fc4 = nn.Linear(n_feat+n_hidden_1+n_hidden_2, 1)#+n_hidden_3
        
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(n_hidden_1)
        self.bn2 = nn.BatchNorm1d(n_hidden_2)

        
    def forward(self, adj,x_in,idx):
        lst = list()

        # 1st message passing layer
        lst.append(x_in)
        
        x = self.relu(self.fc1( torch.cat( (x_in, torch.mm(adj, x_in)),1 ) ) )
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)

        # 2nd message passing layer
        x = self.relu(self.fc2( torch.cat( (x, torch.mm(adj, x)),1) ))
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)

        # output layer
        x = torch.cat(lst, dim=1)
        
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx)+1 , x.size(1)).to(x_in.device)
        x = out.scatter_add_(0, idx, x)
        
        #print(out.size())
        x = self.relu(self.fc4(x))

        return x

def gnn_eval(model,A,tmp,feature,idx,device):
    
    feature[tmp,:] = 1
    
    output = model( A,feature,idx).squeeze()
    return output.cpu().detach().numpy().item()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def numerical_sort(value):
    """
    提取文件名中的数字部分，并将其转换为整数，用于排序。
    """
    # 提取文件名中的数字部分
    numbers = re.findall(r'\d+', os.path.basename(value))
    return int(numbers[0]) if numbers else -1


random.seed(1)
torch.manual_seed(1) 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    
    feat_d = 50
    dropout = 0.4
    hidden=64
    model = GNN_skip_small(feat_d,hidden,int(hidden/2),int(hidden/4),dropout).to(device)
    checkpoint = torch.load('models/model_best1.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    seed_size = 20
    
    fw = open("celf_glie_results.csv","a")
    fw.write("graph,nodes,infl,time\n")

    # 1. 获取所有模拟图文件路径
    train_dir = "data/test_set"
    graph_paths = sorted(glob.glob(os.path.join(train_dir, "g*.txt")))
    graph_paths = sorted(graph_paths, key=numerical_sort)

    print(graph_paths)
    # graph_paths 会是 ['data/sim_graphs/train/g0.txt', ..., 'g29.txt']

    # 2. 遍历所有模拟图
    for graph_path in tqdm(graph_paths):
        # 从文件名中提取图的 ID，比如 'g0'
        g = os.path.splitext(os.path.basename(graph_path))[0]
        print(f"Processing {g} …")

        # 3. 读取边列表
        G = pd.read_csv(graph_path, header=None, sep=" ")
        G.columns = ["source", "target", "weight"]

        # # 4. 构建稀疏邻接矩阵
        # nodes = set(G.source).union(G.target)
        # adj = sp.coo_matrix(
        #     (G.weight, (G.target, G.source)),
        #     shape=(len(nodes), len(nodes))
        # )
        # adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)

        # # 5. 度过滤（与之前一致）
        # outdegree = G.groupby("source")["target"].count().reset_index()
        # deg_thres = np.histogram(outdegree.target, bins=20)[1][1]
        # candidates = outdegree.source[outdegree.target > deg_thres].values

        # 读取后
        nodes = sorted(set(G.source) | set(G.target))
        node2idx = {nid: i for i, nid in enumerate(nodes)}

        # 映射边列表
        G['source_idx'] = G.source.map(node2idx)
        G['target_idx'] = G.target.map(node2idx)
        adj_sp = sp.coo_matrix(
            (G.weight, (G.target_idx, G.source_idx)),
            shape=(len(nodes), len(nodes))
        )
        adj = sparse_mx_to_torch_sparse_tensor(adj_sp).to(device)
            
            # 6) 度过滤得到候选集
        outdeg = G.groupby("source_idx")["target_idx"].count()
        deg_thres = np.histogram(outdeg, bins=20)[1][1]
        candidates = outdeg[outdeg > deg_thres].index.values  # 已是连续索引        


        if len(candidates) == 0:
            print(f"Skipping {g} because no candidates found.")
            continue

        # 6. CELF + GNN 选种子（同原逻辑，只要把 nodes 换成 candidates 即可）
        S = []
        Q = []
        idx = torch.LongTensor([0] * adj.shape[0]).to(device)
        feature = torch.zeros(adj.shape[0], feat_d).to(device)

        with torch.no_grad():
            # 初始化 Q
            for u in candidates:
                score = gnn_eval(model, adj, [u], feature.clone(), idx, device)
                Q.append([u, score, 0])
        Q.sort(key=lambda x: x[1], reverse=True)

        # 选第一个种子
        S.append(Q[0][0])
        infl_spread = Q[0][1]
        Q = Q[1:]

        start = time.time()
        # 迭代挑选其余 seed_size-1 个
        while len(S) < seed_size:
            u = Q[0]
            if u[2] != len(S):
                # 延迟更新
                new_score = gnn_eval(model, adj, S + [u[0]], feature.clone(), idx, device) - infl_spread
                u[1] = new_score
                u[2] = len(S)
                Q.sort(key=lambda x: x[1], reverse=True)
            else:
                infl_spread += u[1]
                S.append(u[0])
                Q = Q[1:]
        t = time.time() - start

        # 7. 真实传播评估
        ic  = IC(G, S)

        # 8. 写结果
        fw.write(f"{g},\"{','.join(map(str,S))}\",{ic:.4f},{t:.4f}\n")
        fw.flush()

    fw.close()

if __name__ == "__main__":
    main()