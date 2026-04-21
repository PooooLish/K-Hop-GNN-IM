import random
import os
from tqdm import tqdm
import numpy as np 
import glob
import pandas as pd
import torch
import scipy.sparse as sp 
import torch.nn as nn
import time
import re

from diffuse import IC
from train_khop import GNN_KHOP, sparse_mx_to_torch_sparse_tensor  # 导入新的KHOP-GNN

random.seed(1)
torch.manual_seed(1)

def gnn_eval_khop(model, A, active_nodes, feature, idx, device):
    """
    对给定的激活节点集 active_nodes，使用 KHOP-GNN 计算影响力评分。
    """
    # 1) 清零特征并标记 active_nodes
    feature.zero_()
    feature[active_nodes, :] = 1.0

    # 2) 前向计算
    with torch.no_grad():
        out = model(A, feature, idx)  # 返回 [num_graphs]，这里通常 num_graphs=1
        return out.squeeze().item()


def numerical_sort(value):
    nums = re.findall(r'\d+', os.path.basename(value))
    return int(nums[0]) if nums else -1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 超参
    feat_d   = 50
    hidden   = 64
    dropout  = 0.4
    k        = 3          # KHOP 阶数
    seed_size = 20

    # 1) 加载 KHOP-GNN 模型
    model = GNN_KHOP(
        in_features = feat_d,
        hidden_dim  = hidden,
        out_features= 1,
        k           = k,
        dropout     = dropout
    ).to(device)
    checkpoint = torch.load('models/k3_khop_model_best1.pth.tar', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 2) 打开结果文件
    with open("k3_celf_khop_results.csv", "w") as fw:
        fw.write("graph,nodes,infl,time\n")

        # 3) 遍历所有模拟图
        graph_dir = "data/test_set"
        graph_paths = sorted(
            glob.glob(os.path.join(graph_dir, "g*.txt")),
            key=numerical_sort
        )

        for graph_path in tqdm(graph_paths, desc="Graphs"):
            g = os.path.splitext(os.path.basename(graph_path))[0]
            # 4) 读取边列表
            G = pd.read_csv(graph_path, header=None, sep=" ")
            G.columns = ["source", "target", "weight"]

            # 5) 构建稀疏邻接矩阵并转换为 torch.sparse

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

            # 7) CELF + KHOP-GNN 选择种子集
            S = []
            idx = torch.zeros(adj.shape[0], dtype=torch.long, device=device)
            feature = torch.zeros(adj.shape[0], feat_d, device=device)

            # 初始化优先队列 Q: [node, score, last_updated_round]
            Q = []
            # for u in candidates:
            #     score = gnn_eval_khop(model, adj, [u], feature.clone(), idx, device)
            #     Q.append([u, score, 0])
            for u_idx in candidates:
                score = gnn_eval_khop(model, adj, [u_idx], feature.clone(), idx, device)
                Q.append([u_idx, score, 0])
            
            Q.sort(key=lambda x: x[1], reverse=True)

            # 选第一个种子
            S.append(Q[0][0])
            infl_spread = Q[0][1]
            Q = Q[1:]

            # 计时
            start = time.time()
            while len(S) < seed_size:
                u, sc, last = Q[0]
                if last != len(S):
                    # 延迟更新评分
                    new_sc = gnn_eval_khop(model, adj, S + [u], feature.clone(), idx, device) - infl_spread
                    Q[0][1] = new_sc
                    Q[0][2] = len(S)
                    Q.sort(key=lambda x: x[1], reverse=True)
                else:
                    infl_spread += sc
                    S.append(u)
                    Q.pop(0)
            t = time.time() - start

            # 8) 使用 IC 模型验证真实影响力
            ic  = IC(G, S)

            # 9) 写入结果
            fw.write(f"{g},\"{','.join(map(str,S))}\",{ic:.4f},{t:.4f}\n")
            fw.flush()

if __name__ == "__main__":
    main()
