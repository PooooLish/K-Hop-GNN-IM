# GNN-IM: Influence Maximization with k-Hop Graph Neural Networks

## Overview

This project implements a scalable solution to the **Influence Maximization (IM)** problem in social networks using **Graph Neural Networks (GNNs)**.

The proposed framework combines:
- Classical IM algorithms (Monte Carlo simulation, CELF greedy)
- A configurable **k-Hop GNN model**
- Efficient seed node selection strategy

The goal is to **predict node influence efficiently** and select high-quality seed sets without expensive simulations.

This project is based on my undergraduate thesis:  
**"Influence Maximization in Social Networks based on Graph Neural Networks"** 

---

## Problem Definition

Given a graph \( G = (V, E) \), the task is to select a set of \( k \) seed nodes such that the expected spread of influence is maximized:

$$
S^* = \arg\max_{|S|=k} \sigma(S)
$$

Traditional methods rely on Monte Carlo simulations, which are computationally expensive.  
This project reformulates the problem as a learning-based regression task using GNN.

---

## Key Contributions

- **k-Hop GNN Architecture**
  - Explicitly aggregates multi-hop neighbor information (1-hop to k-hop)
  - Captures long-range influence propagation

- **Hybrid Labeling Strategy**
  - Combines Monte Carlo simulation with CELF greedy algorithm
  - Generates high-quality training labels

- **GNN + CELF Hybrid Inference**
  - Uses GNN predictions to guide CELF selection
  - Significantly reduces computation cost

- **Scalable Framework**
  - Works on synthetic and real-world graphs
  - Supports configurable model size and parameters

---

## Model Architecture

The model consists of three main components:

1. **Feature Construction**
   - Seed nodes are encoded as binary node features

2. **k-Hop Message Passing**
   - Computes:
     ```
     X, AX, A²X, ..., A^k X
     ```
   - Concatenates all representations

3. **Prediction Head**
   - MLP + BatchNorm + Dropout
   - Graph-level pooling (mean aggregation)
   - Outputs influence score

---

## Project Structure

```
├── train_khop.py # GNN training script
├── celf_khop.py # Seed selection with CELF + GNN
├── diffuse.py # Independent Cascade model
├── data/
│ ├── sim_graphs/ # training graphs
│ ├── test_set/ # test graphs
├── models/ # trained checkpoints
├── errors/ # logs
```

---

## Training

Train the k-Hop GNN model:

```bash
python train_khop.py
```

Key hyperparameters:

1. k: number of propagation hops
2. hidden_dim: hidden layer dimension
3. dropout: regularization rate
4. batch_size: training batch size

The model is trained using MSE loss to regress influence scores.

## Inference & Seed Selection

Run CELF + GNN hybrid algorithm:

```bash
python celf_khop.py
```

Pipeline:

1. Load trained GNN
2. Generate candidate nodes (degree filtering)
3. Use GNN to estimate marginal gain
4. Apply CELF lazy greedy selection
5. Evaluate with Independent Cascade model

## Applications

1. Viral marketing
2. Social recommendation systems
3. Information diffusion modeling
4. Epidemic spread analysis

## Requirements

Python 3.x /
PyTorch /
NumPy /
SciPy /
pandas /
tqdm

## Future Work

1. Extend to dynamic / temporal graphs
2. Incorporate heterogeneous graph structures
3. Improve scalability for large-scale real-world networks