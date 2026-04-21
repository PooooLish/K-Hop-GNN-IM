import os
import glob
import pandas as pd
import random
import time
from tqdm import tqdm

# 父目录与数据目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
DATA        = os.path.join(parent_dir, "data") 
SIM_DIR     = os.path.join(DATA, "sim_graphs") 
GRAPH_DIR   = os.path.join(SIM_DIR, "graphs") 

# 导入 IC 函数
import sys
sys.path.append(parent_dir)
from diffuse import IC

# 固定随机种子
random.seed(1)

def process_graphs():
    seed_size   = 10

    labels_path = os.path.join(SIM_DIR, "influence_labels.csv")

    # 遍历所有模拟图
    graph_files = sorted(glob.glob(os.path.join(GRAPH_DIR, "g*.txt")))
    for graph_path in tqdm(graph_files, desc="Graphs"):
        g = os.path.splitext(os.path.basename(graph_path))[0]
        print(f"Processing {g} ...")

        # 读取边列表
        G_df = pd.read_csv(graph_path, header=None, sep=" ")
        G_df.columns = ["source", "target", "weight"]  # 与目标脚本列名一致

        nodes = set(G_df.target).union(G_df.source)

        # 初始化 CELF 环境
        Q = []
        S = []
        nid, mg, iteration = 0, 1, 2

        # 1. 计算每个节点的初始影响力
        for u in tqdm(nodes, desc=f"Init {g}"):
            Q.append([u, IC(G_df, [u]), 0])  # IC(G, [u])

        # 2. 按影响力降序排序并选第一个种子
        Q.sort(key=lambda x: x[1], reverse=True)
        S = [Q[0][0]]
        infl_spread = Q[0][1]
        Q = Q[1:]

        # 写入初始种子到 CSV
        with open(labels_path, "a") as labels:
            labels.write(f"{g},\"{S[0]}\",{infl_spread}\n")
            labels.flush()
            os.fsync(labels.fileno())

        # 3. 迭代选其余种子
        start_time = time.time()
        while len(S) < seed_size:
            u = Q[0]
            # 如果未在当前迭代计算过，更新边际增益
            if u[iteration] != len(S):
                u[mg] = IC(G_df, S + [u[nid]], 300) - infl_spread 
                u[iteration] = len(S)
                Q.sort(key=lambda x: x[1], reverse=True)
            else:
                # 接受该节点，更新总影响力、种子集
                infl_spread += u[mg]
                S.append(u[nid])
                Q = Q[1:]
                # 写入当前种子列表及影响力
                with open(labels_path, "a") as labels:
                    labels.write(f"{g},\"{','.join(map(str,S))}\",{infl_spread}\n")
                    labels.flush()
                    os.fsync(labels.fileno())

        elapsed = time.time() - start_time
        print(f"Finished {g} in {elapsed:.2f}s\n")

if __name__ == "__main__":
    process_graphs()
