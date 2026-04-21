# import os
# import random
# import numpy as np
# import networkx as nx
# import pandas as pd
# from tqdm import tqdm
# from graph_generator import graph_generator

# # 配置输出目录
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir  = os.path.dirname(current_dir)
# OUTPUT_DIR  = os.path.join(parent_dir, "data", "train_set", "graphs")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # 节点规模区间与每区间图数量
# size_ranges      = [(100,200), (300,500), (800,1000)]
# graphs_per_range = 200

# # 拓扑类型与权重类型
# topo_types   = ['erdos_renyi','small-world','powerlaw']
# weight_types = ['random','degree','degree_noise']

# for (n_min, n_max) in size_ranges:
#     combos    = [(t, w) for t in topo_types for w in weight_types]
#     per_combo = graphs_per_range // len(combos)
#     extra     = graphs_per_range % len(combos)
#     counts    = [per_combo + (1 if i < extra else 0) for i in range(len(combos))]

#     for (topo, wt), cnt in zip(combos, counts):
#         # 每次新建一个 generator，生成 cnt 张图
#         gen = graph_generator()
#         gen.gen_new_graphs(
#             min_nodes = n_min,
#             max_nodes = n_max,
#             graph_no  = cnt,
#             train     = True,
#             g_type    = topo,
#             w_type    = wt
#         )

#         # 直接对每张图命名并保存
#         for idx, g in enumerate(gen.TrainSet):
#             # 先将无权图写入临时 edgelist
#             tmp_path = os.path.join(OUTPUT_DIR, f"tmp_{topo}_{wt}_{n_min}_{n_max}_{idx}.edgelist")
#             nx.write_edgelist(g, tmp_path, data=False)

#             # 读取 DataFrame，做双向扩展和入度归一化
#             df = pd.read_csv(tmp_path, sep=' ', header=None, names=['node1','node2'])
#             os.remove(tmp_path)

#             # 双向扩展
#             df1 = df.rename(columns={'node1':'source','node2':'target'})
#             df2 = df.rename(columns={'node1':'target','node2':'source'}) \
#                     .rename(columns={'source':'target','target':'source'})
#             df_bi = pd.concat([df1, df2], ignore_index=True)

#             # 统计入度并归一化
#             deg = df_bi.groupby('target').size().reset_index(name='cnt')
#             deg['weight'] = (1.0 / deg['cnt']).round(6)

#             # 合并权重
#             df_bi = df_bi.merge(deg[['target','weight']], on='target', how='left')

#             # 唯一文件名：区间_拓扑_权重_索引.txt
#             fname = f"g_{n_min}-{n_max}_{topo}_{wt}_{idx}.txt"
#             out_path = os.path.join(OUTPUT_DIR, fname)

#             # 写入最终带权边列表
#             df_bi[['source','target','weight']].to_csv(
#                 out_path, sep=' ', header=False, index=False, float_format='%.6f'
#             )


import os
import glob
import re
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from graph_generator import graph_generator

# 配置输出目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
GRAPH_DIR   = os.path.join(parent_dir, "data", "real_train_set")
os.makedirs(GRAPH_DIR, exist_ok=True)

topo_types   = ['erdos_renyi','small-world','powerlaw']
weight_types = ['random','degree','degree_noise']

combos    = [(t, w) for t in topo_types for w in weight_types]

for (topo, wt) in combos:

    generator = graph_generator()
    generator.gen_new_graphs(800, 1000, 10, True, topo , wt)
    generator.save_graphs(GRAPH_DIR)

    # 2) 处理并重命名
    os.chdir(GRAPH_DIR)
    pattern = "g[0-9]*.txt"

    for g in tqdm(glob.glob(pattern), desc="归一化并重命名"):
        # 2.1 读入
        df = pd.read_csv(g, header=None, sep=" ", names=["node1","node2","w"])
        df = df.drop(columns="w")

        # 2.2 双向扩展
        tmp = df.copy()
        df = pd.DataFrame(
            np.concatenate([df.values, tmp[["node2","node1"]].values]),
            columns=["source","target"]
        )

        # 2.3 归一化
        outdeg = df.groupby("target").size().reset_index(name="cnt")
        outdeg["weight"] = (1.0 / outdeg["cnt"]).round(6)
        df = df.merge(outdeg[["target","weight"]], on="target")

        # 2.4 保存到原文件（覆盖内容）
        df[["source","target","weight"]].to_csv(
            g, sep=" ", header=False, index=False, float_format="%.6f"
        )

        # 2.5 构造新的、特征化的文件名
        #    假设原始文件名格式为 "g<idx>.txt"
        idx = re.findall(r"g(\d+)\.txt", os.path.basename(g))
        idx = idx[0] if idx else "0"

        #    重新读一次，得到真实节点数和边数
        df2 = pd.read_csv(g, header=None, sep=" ", names=["source","target","weight"])
        num_nodes = len(set(df2["source"]) | set(df2["target"]))
        num_edges = len(df2)

        scale = f"large"        

        new_name = f"g_{scale}_{topo}_{wt}_nodes{num_nodes}_edges{num_edges}_{idx}.txt"
        if os.path.exists(new_name):
            os.remove(new_name)       # 直接覆盖
        os.rename(g, new_name)


