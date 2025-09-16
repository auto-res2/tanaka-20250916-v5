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


def compute_node_features(data, batch_size=100000, max_edges_for_features=10000000):
    """Compute additional node features for MLSP with memory-efficient processing."""
    row, col = data.edge_index
    deg = degree(col, data.num_nodes, dtype=torch.float)
    
    num_edges = data.edge_index.size(1)
    
    if num_edges > max_edges_for_features:
        print(f"Large graph detected ({num_edges} edges). Using simplified edge features.")
        deg_i = deg[row]
        deg_j = deg[col]
        degree_ratio = torch.minimum(deg_i, deg_j) / torch.maximum(deg_i, deg_j)
        degree_ratio = torch.nan_to_num(degree_ratio, nan=0.0)
        
        edge_features = torch.stack([deg_i, deg_j, degree_ratio], dim=1)
        return edge_features
    
    edge_features = torch.zeros((num_edges, 3), dtype=torch.float, device=data.x.device)
    
    for start_idx in range(0, num_edges, batch_size):
        end_idx = min(start_idx + batch_size, num_edges)
        
        batch_src = data.edge_index[0, start_idx:end_idx]
        batch_dst = data.edge_index[1, start_idx:end_idx]
        
        deg_i = deg[batch_src]
        deg_j = deg[batch_dst]
        
        x_i = data.x[batch_src]
        x_j = data.x[batch_dst]
        
        x_i_norm = F.normalize(x_i, p=2, dim=1)
        x_j_norm = F.normalize(x_j, p=2, dim=1)
        cos_sim = (x_i_norm * x_j_norm).sum(dim=1)
        
        edge_features[start_idx:end_idx, 0] = deg_i
        edge_features[start_idx:end_idx, 1] = deg_j
        edge_features[start_idx:end_idx, 2] = cos_sim
        
        torch.cuda.empty_cache()
    
    return edge_features
