import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import add_self_loops, degree
import numpy as np


def load_and_preprocess_data(dataset_name, root="data"):
    """Load and preprocess graph datasets."""
    if dataset_name == "Cora":
        dataset = Planetoid(root=root, name="Cora", transform=NormalizeFeatures())
    elif dataset_name == "PubMed":
        dataset = Planetoid(root=root, name="PubMed", transform=NormalizeFeatures())
    elif dataset_name == "Reddit":
        dataset = Reddit(root=f"{root}/Reddit", transform=NormalizeFeatures())
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    data = dataset[0]
    
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    
    row, col = data.edge_index
    deg = degree(col, data.num_nodes, dtype=data.x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    data.deg_norm = deg_inv_sqrt
    
    if not hasattr(data, 'train_mask'):
        num_nodes = data.num_nodes
        train_ratio, val_ratio = 0.6, 0.2
        
        indices = torch.randperm(num_nodes)
        train_size = int(train_ratio * num_nodes)
        val_size = int(val_ratio * num_nodes)
        
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        data.train_mask[indices[:train_size]] = True
        data.val_mask[indices[train_size:train_size + val_size]] = True
        data.test_mask[indices[train_size + val_size:]] = True
    
    return data, dataset.num_classes


def compute_node_features(data):
    """Compute additional node features for MLSP."""
    row, col = data.edge_index
    deg = degree(col, data.num_nodes, dtype=torch.float)
    
    edge_features = []
    for i in range(data.edge_index.size(1)):
        src, dst = data.edge_index[0, i], data.edge_index[1, i]
        deg_i, deg_j = deg[src], deg[dst]
        
        x_i, x_j = data.x[src], data.x[dst]
        cos_sim = F.cosine_similarity(x_i.unsqueeze(0), x_j.unsqueeze(0))
        
        edge_features.append([deg_i.item(), deg_j.item(), cos_sim.item()])
    
    return torch.tensor(edge_features, dtype=torch.float)
