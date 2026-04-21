import os
import igraph as ig
from tqdm import tqdm
import numpy as np 
import time
import glob
import pandas as pd
import random
import networkx as nx
import sys

# 将父目录加入模块搜索路径，以便导入 diffuse.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
DATA        = os.path.join(parent_dir, "data") 
SIM_DIR     = os.path.join(DATA, "sim_graphs") 
GRAPH_DIR   = os.path.join(SIM_DIR, "graphs") 
sys.path.append(parent_dir)

from diffuse import IC

# 固定随机种子
random.seed(1)

def main():
    # 输入与输出文件路径
    labels_input = os.path.join(SIM_DIR, "influence_labels.csv")
    labels_output = os.path.join(SIM_DIR, "influence_train_set.csv")

    # 读取已有标签
    x = pd.read_csv(labels_input, header=None)
    print("Loaded labels:", x.shape)

    x.columns = ["graph","node","infl"]
    x["len"] = x.node.apply(lambda s: len(s.split(",")))

    gs = x.graph.unique()
    neg_samples = 10

    # 打开输出文件
    with open(labels_output, "a") as fout:
        # 遍历每个图标识
        print("Graphs to process:", gs)

        for g in tqdm(gs, desc="Graphs"):
            tmp = x[x.graph == g]
            # 读取图数据（父目录下 data/sim_graphs）
            graph_path = os.path.join(GRAPH_DIR,f"{g}.txt")
            G = pd.read_csv(graph_path, header=None, sep=" ")
            G.columns = ["source","target","weight"]
            nodes = set(G["target"]).union(G["source"])

            # 对每条记录生成负样本并写入
            for _, row in tmp.iterrows():
                seeds = set(row["node"].split(","))
                size = row["len"]
                # 生成多组负样本
                for _ in range(neg_samples):
                    neg = set(random.sample(nodes, size))
                    attempts = 0
                    while neg == seeds and attempts < 10:
                        neg = set(random.sample(nodes, size))
                        attempts += 1
                    sigma = IC(G, list(neg))
                    fout.write(f"{g},\"{','.join(map(str,neg))}\",{sigma}\n")
                # 写入正样本
                fout.write(f"{g},\"{row['node']}\",{row['infl']}\n")

if __name__ == "__main__":
    main()
