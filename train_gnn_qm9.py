# -*- coding: utf-8 -*-
"""
Implementation of a simpler GNN model for predicting HOMO-LUMO gap (Δε)
on the QM9 dataset using PyTorch Geometric.
"""

import os
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.nn import GCNConv, NNConv

# --- Configuration ---
TARGET_PROPERTY_INDEX = 4  # Index for HOMO-LUMO gap (Δε) in QM9
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = Path(os.environ.get('QM9_DATA_DIR', Path(__file__).resolve().parent / 'data' / 'QM9'))
MODEL_SAVE_PATH = Path(os.environ.get('QM9_OUTPUT_PATH', Path(__file__).resolve().parent / 'runs' / 'gnn_qm9_homo_lumo.pt'))
NUM_EPOCHS = 30 # Reduced from SchNet model due to simpler architecture
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_WORKERS = int(os.environ.get('QM9_NUM_WORKERS', '0')) # portable default

# --- Model Definition ---
class EdgeNetwork(nn.Module):
    """Neural network for transforming edge features."""
    def __init__(self, edge_features, hidden_channels, out_channels):
        super(EdgeNetwork, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(edge_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels * out_channels)
        )
        
    def forward(self, x):
        return self.nn(x)

class GNNModel(torch.nn.Module):
    """Custom GNN model for molecular property prediction."""
    def __init__(self, node_features=8, edge_features=4, hidden_channels=64, out_channels=1):
        super(GNNModel, self).__init__()
        
        # Initial node embedding
        self.node_embedding = nn.Linear(node_features, hidden_channels)
        
        # Graph convolution layers
        self.conv1 = NNConv(
            hidden_channels, 
            hidden_channels, 
            EdgeNetwork(edge_features, hidden_channels, hidden_channels)
        )
        
        self.conv2 = NNConv(
            hidden_channels, 
            hidden_channels, 
            EdgeNetwork(edge_features, hidden_channels, hidden_channels)
        )
        
        self.conv3 = NNConv(
            hidden_channels, 
            hidden_channels, 
            EdgeNetwork(edge_features, hidden_channels, hidden_channels)
        )
        
        # Output layers
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr, batch):
        # Initial node embedding
        x = self.node_embedding(x)
        x = F.relu(x)
        
        # Graph convolution layers (no residual connections in the original architecture)
        x1 = F.relu(self.conv1(x, edge_index, edge_attr))
        x2 = F.relu(self.conv2(x1, edge_index, edge_attr))
        x3 = F.relu(self.conv3(x2, edge_index, edge_attr))
        
        # Global pooling (sum of node features)
        x = global_add_pool(x3, batch)
        
        # Output layers
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        
        return x

# --- Data Loading and Preprocessing ---
class CustomTransform:
    def __init__(self, target_index):
        self.target_index = target_index
        
    def __call__(self, data):
        # Select the target property
        data.y = data.y[:, self.target_index].unsqueeze(1)
        
        # Create node features (atomic number one-hot encoded)
        # QM9 uses z for atomic numbers
        z = data.z.long()
        
        # Create a simple one-hot encoding for atom types (H, C, N, O, F)
        # Atomic numbers: H=1, C=6, N=7, O=8, F=9
        atom_types = torch.zeros(z.size(0), 5, dtype=torch.float)
        for i, atom in enumerate(z):
            if atom == 1:  # H
                atom_types[i, 0] = 1
            elif atom == 6:  # C
                atom_types[i, 1] = 1
            elif atom == 7:  # N
                atom_types[i, 2] = 1
            elif atom == 8:  # O
                atom_types[i, 3] = 1
            elif atom == 9:  # F
                atom_types[i, 4] = 1
        
        # Add additional node features (position)
        # Normalize positions to have zero mean and unit variance
        pos = data.pos
        pos_mean = pos.mean(dim=0)
        pos_std = pos.std(dim=0, unbiased=pos.size(0) > 1)
        pos_normalized = (pos - pos_mean) / (pos_std + 1e-6)
        
        # Combine atom types and positions as node features
        data.x = torch.cat([atom_types, pos_normalized], dim=1)
        
        # Create edge features based on distances between atoms
        row, col = data.edge_index
        dist = torch.norm(data.pos[row] - data.pos[col], p=2, dim=1).view(-1, 1)
        
        # Create Gaussian basis for distances
        # Use 3 Gaussian functions centered at different distances
        centers = torch.tensor([0.0, 1.5, 3.0]).to(dist.device)
        width = 0.5
        gaussian_basis = torch.exp(-width * (dist.view(-1, 1) - centers.view(1, -1))**2)
        
        # Add distance as an additional feature
        data.edge_attr = torch.cat([dist, gaussian_basis], dim=1)
        
        return data


def main():
    if MODEL_SAVE_PATH.exists():
        raise FileExistsError('Choose a new QM9_OUTPUT_PATH; existing checkpoints are protected')
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    print(f"Using device: {DEVICE}")
    print("Loading QM9 dataset...")
    
    # Load the dataset
    dataset = QM9(str(DATA_PATH), transform=CustomTransform(TARGET_PROPERTY_INDEX))
    if len(dataset) <= 120000:
        raise ValueError('Original QM9 dataset is required for the intended split')
    
    print(f"Dataset loaded. Number of graphs: {len(dataset)}")
    
    # Split dataset (using standard QM9 splits: 110k train, 10k val, rest test)
    torch.manual_seed(42) # for reproducible splits
    perm = torch.randperm(len(dataset))
    train_idx = perm[:110000]
    val_idx = perm[110000:120000]
    test_idx = perm[120000:]
    
    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Check the data format
    for batch in train_loader:
        print("Sample batch:")
        print(f"Node features shape: {batch.x.shape}")
        print(f"Edge index shape: {batch.edge_index.shape}")
        print(f"Edge attributes shape: {batch.edge_attr.shape}")
        print(f"Target shape: {batch.y.shape}")
        print(f"Batch indices shape: {batch.batch.shape}")
        break
    
    # Initialize model
    print("Initializing GNN model...")
    # Get node and edge feature dimensions from the data
    sample_data = next(iter(train_loader))
    node_features = sample_data.x.size(1)
    edge_features = sample_data.edge_attr.size(1)
    
    model = GNNModel(
        node_features=node_features,
        edge_features=edge_features,
        hidden_channels=64,
        out_channels=1
    ).to(DEVICE)
    
    # --- Training Setup ---
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    criterion = F.l1_loss # Mean Absolute Error (MAE)
    
    # --- Training and Evaluation Functions ---
    def train(epoch):
        model.train()
        total_loss = 0
        start_time = time.time()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        avg_loss = total_loss / len(train_loader.dataset)
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train MAE: {avg_loss:.4f} | Time: {elapsed_time:.2f}s")
        return avg_loss
    
    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        total_error = 0
        for batch in loader:
            batch = batch.to(DEVICE)
            output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            total_error += criterion(output, batch.y).item() * batch.num_graphs
        avg_mae = total_error / len(loader.dataset)
        return avg_mae
    
    # --- Main Training Loop ---
    print("Starting training...")
    best_val_mae = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        train_mae = train(epoch)
        val_mae = evaluate(val_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Val MAE: {val_mae:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_mae)
        
        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved best model with Val MAE: {best_val_mae:.4f}")
    
    print("Training finished.")
    
    # --- Final Evaluation ---
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
    test_mae = evaluate(test_loader)
    print(f"Final Test MAE: {test_mae:.4f}")

if __name__ == '__main__':
    main()
