import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
import numpy as np
import time
from tqdm import tqdm
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class LipschitzEdgeFreezer:
    """Lipschitz Edge-Freezer (LEF) component."""
    
    def __init__(self, eps_b=5e-3, K_freeze=6, negative_slope=0.2):
        self.eps_b = eps_b
        self.K_freeze = K_freeze
        self.negative_slope = negative_slope
        self.frozen_edges = {}
        self.freeze_counters = {}
    
    def compute_lipschitz_bound(self, delta_h_i, delta_h_j, attention_weights, temperature=1.0):
        """Compute Lipschitz bound for attention change."""
        d = delta_h_i.size(-1)
        L = torch.max(attention_weights.abs()) * max(1, self.negative_slope) / (np.sqrt(d) * temperature)
        bound = L * (delta_h_i.norm() + delta_h_j.norm())
        return bound
    
    def update_frozen_edges(self, edge_index, delta_h, attention_weights):
        """Update frozen edge status based on Lipschitz bounds."""
        newly_frozen = []
        unfrozen = []
        
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            edge_key = (src, dst)
            
            if edge_key in self.frozen_edges:
                self.freeze_counters[edge_key] -= 1
                if self.freeze_counters[edge_key] <= 0:
                    del self.frozen_edges[edge_key]
                    del self.freeze_counters[edge_key]
                    unfrozen.append(i)
            else:
                bound = self.compute_lipschitz_bound(
                    delta_h[src], delta_h[dst], attention_weights
                )
                if bound < self.eps_b:
                    self.frozen_edges[edge_key] = i
                    self.freeze_counters[edge_key] = self.K_freeze
                    newly_frozen.append(i)
        
        return newly_frozen, unfrozen


class MetaLearnedStabilityPrior(nn.Module):
    """Meta-Learned Stability Prior (MLSP) component."""
    
    def __init__(self, input_dim=3, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, edge_features):
        """Predict probability of edge being frozen."""
        return self.mlp(edge_features).squeeze(-1)


class SharedBasisHierarchicalSketch(nn.Module):
    """Shared-Basis Hierarchical Sketch (SBHS) component."""
    
    def __init__(self, d_model, r_basis=32):
        super().__init__()
        self.d_model = d_model
        self.r_basis = r_basis
        self.register_buffer('gaussian_matrix', torch.randn(d_model, r_basis) / np.sqrt(r_basis))
        self.cached_coefficients = {}
    
    def cache_message(self, edge_key, message):
        """Cache message using shared basis."""
        coeffs = torch.matmul(message, self.gaussian_matrix)
        self.cached_coefficients[edge_key] = coeffs
    
    def reconstruct_message(self, edge_key):
        """Reconstruct message from cached coefficients."""
        if edge_key in self.cached_coefficients:
            coeffs = self.cached_coefficients[edge_key]
            return torch.matmul(coeffs, self.gaussian_matrix.T)
        return None


class BEFGAT(MessagePassing):
    """BEF-GAT: Bounded-Error Freezing for Graph Attention Networks."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8,
                 dropout=0.6, eps_b=5e-3, K_freeze=6, r_basis=32, edge_batch_size=4000000):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.edge_batch_size = edge_batch_size
        
        self.lin_l = nn.Linear(in_channels, num_heads * hidden_channels, bias=False)
        self.lin_r = nn.Linear(in_channels, num_heads * hidden_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, num_heads, 2 * hidden_channels))
        self.lin_out = nn.Linear(num_heads * hidden_channels, out_channels)
        
        self.lef = LipschitzEdgeFreezer(eps_b, K_freeze)
        self.mlsp = MetaLearnedStabilityPrior()
        self.sbhs = SharedBasisHierarchicalSketch(hidden_channels, r_basis)
        
        self.reset_parameters()
        self.prev_h = None
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.xavier_uniform_(self.lin_out.weight)
    
    def forward(self, x, edge_index, edge_features=None):
        H, C = self.num_heads, self.hidden_channels
        
        x_l = self.lin_l(x).view(-1, H, C)
        x_r = self.lin_r(x).view(-1, H, C)
        
        if self.prev_h is not None:
            delta_h = x - self.prev_h
        else:
            delta_h = torch.zeros_like(x)
        self.prev_h = x.clone()
        
        num_edges = edge_index.size(1)
        if num_edges > self.edge_batch_size:
            out = self._batched_propagate(edge_index, x=(x_l, x_r), delta_h=delta_h, 
                                        edge_features=edge_features)
        else:
            out = self.propagate(edge_index, x=(x_l, x_r), delta_h=delta_h, 
                               edge_features=edge_features)
        
        out = out.view(-1, H * C)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.lin_out(out)
        
        return out
    
    def _batched_propagate(self, edge_index, x, delta_h, edge_features=None):
        """Memory-efficient batched message passing for large graphs."""
        num_edges = edge_index.size(1)
        num_nodes = x[0].size(0)
        H, C = self.num_heads, self.hidden_channels
        
        out = torch.zeros(num_nodes, H, C, device=edge_index.device, dtype=x[0].dtype)
        
        for start_idx in range(0, num_edges, self.edge_batch_size):
            end_idx = min(start_idx + self.edge_batch_size, num_edges)
            
            batch_edge_index = edge_index[:, start_idx:end_idx]
            batch_edge_features = edge_features[start_idx:end_idx] if edge_features is not None else None
            
            batch_out = self.propagate(batch_edge_index, x=x, delta_h=delta_h, 
                                     edge_features=batch_edge_features)
            
            out += batch_out
        
        return out
    
    def message(self, x_i, x_j, edge_index_i, edge_index_j, delta_h, edge_features=None):
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, edge_index_i)
        
        if edge_features is not None and self.training:
            stability_probs = self.mlsp(edge_features)
            
            edge_index = torch.stack([edge_index_i, edge_index_j])
            newly_frozen, unfrozen = self.lef.update_frozen_edges(
                edge_index, delta_h, alpha
            )
        
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)


class EnergyMeter:
    """Energy monitoring using pynvml."""
    
    def __init__(self):
        self.available = PYNVML_AVAILABLE
        if self.available:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.start_time = None
        self.start_energy = None
    
    def start(self):
        self.start_time = time.time()
        if self.available:
            self.start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
    
    def stop(self):
        if not self.start_time:
            return 0
        
        elapsed_time = time.time() - self.start_time
        if self.available and self.start_energy:
            end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            energy_joules = (end_energy - self.start_energy) / 1000.0
            return energy_joules
        return 0


def train_bef_gat(data, num_classes, config):
    """Train BEF-GAT model with given configuration."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BEFGAT(
        in_channels=data.num_features,
        hidden_channels=config['hidden_dim'],
        out_channels=num_classes,
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        eps_b=config['eps_b'],
        K_freeze=config['K_freeze'],
        r_basis=config['r_basis'],
        edge_batch_size=config.get('edge_batch_size', 4000000)
    ).to(device)
    
    data = data.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    energy_meter = EnergyMeter() if config.get('energy_monitoring', False) else None
    
    model.train()
    train_losses = []
    val_accuracies = []
    
    for epoch in tqdm(range(config['epochs']), desc="Training"):
        if energy_meter:
            energy_meter.start()
        
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_pred = val_out[data.val_mask].argmax(dim=1)
            val_acc = (val_pred == data.y[data.val_mask]).float().mean()
            val_accuracies.append(val_acc.item())
        model.train()
        
        if energy_meter:
            epoch_energy = energy_meter.stop()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}")
    
    model.eval()
    with torch.no_grad():
        test_out = model(data.x, data.edge_index)
        test_pred = test_out[data.test_mask].argmax(dim=1)
        test_acc = (test_pred == data.y[data.test_mask]).float().mean()
    
    frozen_edges_pct = len(model.lef.frozen_edges) / data.edge_index.size(1) * 100
    avg_freeze_length = np.mean(list(model.lef.freeze_counters.values())) if model.lef.freeze_counters else 0
    
    results = {
        'test_accuracy': test_acc.item(),
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'frozen_edges_percentage': frozen_edges_pct,
        'average_freeze_length': avg_freeze_length,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'final_loss': train_losses[-1] if train_losses else 0
    }
    
    return model, results
