import os
import glob
import re
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from diffuse import IC  # 调用 diffuse.py 中的 IC 函数

random.seed(1)


def numerical_sort(value):
    nums = re.findall(r'\d+', os.path.basename(value))
    return int(nums[0]) if nums else -1


def pagerank_seeds(G_df, seed_size, alpha=0.85, max_iter=100):
    """
    使用 PageRank 中心性方法选择种子节点。
    G_df: DataFrame with columns ['source','target','weight']
    seed_size: 目标种子数
    alpha:    阻尼因子（默认 0.85）
    max_iter: 最大迭代次数
    返回：seed 列表
    """
    # 构建有向图
    G = nx.DiGraph()
    edgelist = G_df[['source', 'target']].values.tolist()
    G.add_edges_from(edgelist)
    # 计算 PageRank 得分
    pr = nx.pagerank(G, alpha=alpha, max_iter=max_iter)
    # 按得分降序排序，取前 seed_size 个节点
    sorted_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    seeds = [node for node, _ in sorted_nodes[:seed_size]]
    return seeds


def main():
    random.seed(1)
    mc_runs_eval = 50
    seed_size = 20
    graph_dir = "data/sim_graphs/graphs/"
    output_path = "pagerank_results.csv"

    with open(output_path, 'w') as fw:
        fw.write("graph,nodes,infl,time\n")
        paths = sorted(glob.glob(os.path.join(graph_dir, 'g*.txt')), key=numerical_sort)
        for path in tqdm(paths, desc="Graphs"):
            gname = os.path.splitext(os.path.basename(path))[0]
            G_df = pd.read_csv(path, header=None, sep=' ')
            G_df.columns = ['source','target','weight']

            # PageRank 中心性选种子
            t_start = time.time()
            seeds = pagerank_seeds(G_df, seed_size)
            t = time.time() - t_start

            # 使用 diffuse.IC 评估真实影响力
            ic = IC(G_df, seeds, mc=mc_runs_eval)

            fw.write(f"{gname},\"{','.join(map(str,seeds))}\",{ic:.4f},{t:.4f}\n")
            fw.flush()

if __name__ == '__main__':
    main()
