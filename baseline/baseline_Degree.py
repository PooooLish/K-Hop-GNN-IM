import os
import glob
import re
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from diffuse import IC  # 调用 diffuse.py 中的 IC 函数
import networkx as nx

random.seed(1)


def numerical_sort(value):
    nums = re.findall(r'\d+', os.path.basename(value))
    return int(nums[0]) if nums else -1


def degree_centrality_seeds(G_df, seed_size):
    """
    使用度中心性方法选择种子节点。
    G_df: DataFrame with columns ['source','target','weight']
    seed_size: 目标种子数
    返回：seed 列表
    """
    # 将边列表加载为 NetworkX 有向图
    G = nx.DiGraph()
    edgelist = G_df[['source', 'target']].values.tolist()
    G.add_edges_from(edgelist)
    # 计算度中心性（出度或入度均可，根据需求）
    # 这里采用出度作为影响力指标
    deg = G.out_degree()
    # 按度从高到低排序，选前 seed_size 个节点
    sorted_nodes = sorted(deg, key=lambda x: x[1], reverse=True)
    seeds = [node for node, _ in sorted_nodes[:seed_size]]
    return seeds


def main():
    random.seed(1)
    mc_runs_eval = 50
    seed_size = 20
    graph_dir = "data/sim_graphs/graphs/"
    output_path = "degree_centrality_results.csv"

    with open(output_path, 'w') as fw:
        fw.write("graph,nodes,infl,time\n")
        paths = sorted(glob.glob(os.path.join(graph_dir, 'g*.txt')), key=numerical_sort)
        for path in tqdm(paths, desc="Graphs"):
            gname = os.path.splitext(os.path.basename(path))[0]
            G_df = pd.read_csv(path, header=None, sep=' ')
            G_df.columns = ['source','target','weight']

            # 度中心性选种子
            t_start = time.time()
            seeds = degree_centrality_seeds(G_df, seed_size)
            t = time.time() - t_start

            # 使用 diffuse.IC 评估真实影响力
            ic = IC(G_df, seeds, mc=mc_runs_eval)

            fw.write(f"{gname},\"{','.join(map(str,seeds))}\",{ic:.4f},{t:.4f}\n")
            fw.flush()

if __name__ == '__main__':
    main()
