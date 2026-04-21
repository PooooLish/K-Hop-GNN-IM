import os
import glob
import re
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from diffuse import IC  # 调用 diffuse.py 中的 IC 函数

random.seed(1)


def numerical_sort(value):
    nums = re.findall(r'\d+', os.path.basename(value))
    return int(nums[0]) if nums else -1

# Generate Reverse Reachable (RR) set from a random node
def generate_rr_set(rev_adj, nodes):
    """
    生成一个反向可达集（RR 集），从一个随机节点反向进行IC模型传播。
    rev_adj: dict{target: [(source, prob), ...]}
    nodes: list of all node ids
    返回RR集
    """
    v = random.choice(nodes)
    rr = {v}
    queue = [v]
    while queue:
        curr = queue.pop()
        for u, prob in rev_adj.get(curr, []):
            if u not in rr and random.random() <= prob:
                rr.add(u)
                queue.append(u)
    return rr

# IMM Algorithm using diffuse.IC for evaluation
def imm(G_df, seed_size, eps=0.1):
    """
    实现IMM算法选择种子节点。
    使用IC函数评估最终影响力。
    返回seed列表
    """
    nodes = sorted(set(G_df.source).union(G_df.target))
    n = len(nodes)
    # 构建反向邻接表一次
    rev_adj = {u: [] for u in nodes}
    for _, row in G_df.iterrows():
        prob = row.weight if row.weight > 0 else 0.1
        rev_adj[row.target].append((row.source, prob))

    # θ计算 (近似版)
    theta = int(seed_size * (np.log(n) + np.log(2)) * 2 / (eps**2))
    R = [generate_rr_set(rev_adj, nodes) for _ in range(theta)]

    cover_count = {u: 0 for u in nodes}
    for rr in R:
        for u in rr:
            cover_count[u] += 1

    seeds = []
    for _ in range(seed_size):
        u = max(cover_count, key=cover_count.get)
        seeds.append(u)
        # 移除已覆盖的RR集
        to_remove = [i for i, rr in enumerate(R) if u in rr]
        for idx in sorted(to_remove, reverse=True):
            for v in R[idx]:
                cover_count[v] -= 1
            R.pop(idx)
    return seeds

# Main

def main():
    random.seed(1)
    mc_runs_eval = 50
    seed_size = 20
    eps = 0.1
    graph_dir = "data/sim_graphs/graphs/"
    output_path = "imm_results.csv"

    with open(output_path, 'w') as fw:
        fw.write("graph,seeds,infl20,time20,infl100,time100\n")
        paths = sorted(glob.glob(os.path.join(graph_dir, 'g*.txt')), key=numerical_sort)
        for path in tqdm(paths, desc="Graphs"):
            gname = os.path.splitext(os.path.basename(path))[0]
            G_df = pd.read_csv(path, header=None, sep=' ')
            G_df.columns = ['source','target','weight']


            # IMM 选种子
            t_start = time.time()
            seeds = imm(G_df, seed_size, eps)
            t = time.time() - t_start

            # 使用 diffuse.IC 评估真实影响力
            ic = IC(G_df, seeds, mc=mc_runs_eval)

            fw.write(f"{gname},\"{','.join(map(str,seeds))}\",{ic:.4f},{t if t else 0:.4f}\n")
            fw.flush()

if __name__ == '__main__':
    main()
