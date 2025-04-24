import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from performer_pytorch import Performer
from torch.utils.data import Dataset, DataLoader

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# ======================
# 1. Data Preparation (Fixed)
# ======================
class PortfolioDataset(Dataset):
    def __init__(self, price_data, volume_data, lookback=10):
        self.price_returns = price_data.pct_change().dropna()
        self.volume_returns = volume_data.pct_change().dropna()
        self.lookback = lookback

    def __len__(self):
        return len(self.price_returns) - self.lookback + 1

    def __getitem__(self, idx):
        prices = self.price_returns.iloc[idx:idx+self.lookback]
        volumes = self.volume_returns.iloc[idx:idx+self.lookback]
        seq_data = torch.stack([
            torch.stack([
                torch.tensor(prices[price_col].values),
                torch.tensor(volumes[vol_col].values)
            ], dim=1)
            for price_col, vol_col in zip(self.price_returns.columns, self.volume_returns.columns)
        ], dim=0).float()
        target = torch.tensor(self.price_returns.iloc[idx+self.lookback-1].values).float()
        return seq_data, target

# ======================
# 2. Performer Model (Unchanged)
# ======================
class PortfolioPerformer(nn.Module):
    def __init__(self, num_assets, lookback=10, dim=32, depth=2):
        super().__init__()
        self.num_assets = num_assets
        self.input_proj = nn.Linear(2, dim)
        self.performer = Performer(
            dim=dim,
            depth=depth,
            heads=4,
            dim_head=16,
            causal=True
        )
        self.weight_net = nn.Sequential(
            nn.Linear(dim, 16),
            nn.ReLU(inplace=False),  # Ensure not in-place
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        asset_embeddings = []
        for i in range(self.num_assets):
            asset_seq = x[:, i, :, :]
            projected = self.input_proj(asset_seq)
            embeddings = self.performer(projected)
            asset_embeddings.append(embeddings[:, -1, :])
        combined = torch.stack(asset_embeddings, dim=1)
        weights = self.weight_net(combined).squeeze(-1)
        weights = torch.softmax(weights, dim=1)
        return weights

# ======================
# 3. Training Setup (Updated)
# ======================
def sharpe_loss(weights, returns):
    portfolio_returns = torch.sum(weights * returns, dim=-1)
    # Add epsilon to denominator to avoid division by zero and prevent in-place ops
    sharpe = torch.mean(portfolio_returns) / (torch.std(portfolio_returns) + 1e-8)
    return -sharpe * np.sqrt(252)

def train_model(price_data, volume_data, epochs=10):
    dataset = PortfolioDataset(price_data, volume_data, lookback=10)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = PortfolioPerformer(num_assets=4, lookback=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        for seq, target in loader:
            optimizer.zero_grad()
            weights = model(seq)
            loss = sharpe_loss(weights, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} | Sharpe: {-loss.item():.4f}")
    return model

# ======================
# 4. Main Execution (Fixed)
# ======================
if __name__ == "__main__":
    data_path = os.path.join("data", "clean_data", "stock_data_clean.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    
    df = pd.read_csv(data_path)
    selected_assets = ['CSCO', 'KO', 'CRM', 'AVGO']  # Removed 'NVDA'
    
    # Create proper feature columns
    price_cols = [f"{asset}_Close" for asset in selected_assets]
    volume_cols = [f"{asset}_Volume" for asset in selected_assets]
    
    # Validate columns before processing
    missing = [col for col in price_cols + volume_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    price_data = df[price_cols]
    volume_data = df[volume_cols]
    
    # Validate data after preprocessing
    print("Price data shape:", price_data.shape)
    print("Volume data shape:", volume_data.shape)
    
    # Train model
    model = train_model(price_data, volume_data, epochs=10)
    
    # Get final weights with validation
    dataset = PortfolioDataset(price_data, volume_data, lookback=10)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check data and lookback parameters.")
    
    last_idx = len(dataset) - 1
    last_seq, _ = dataset[last_idx]  # Use explicit index instead of [-1]
    
    if last_seq.numel() == 0:
        raise ValueError("Final sequence is empty. Check data slicing.")
    
    with torch.no_grad():
        weights = model(last_seq.unsqueeze(0)).squeeze().numpy()
    
    print("\nOptimal Weights:")
    for asset, weight in zip(selected_assets, weights):
        print(f"{asset}: {weight:.4f}")

    # Calculate and print final Sharpe ratio
    # Use the last target as the returns for the final sequence
    _, last_target = dataset[last_idx]
    final_portfolio_return = np.dot(weights, last_target.numpy())
    # For Sharpe, use mean and std of all portfolio returns in the dataset
    all_weights = []
    all_returns = []
    for seq, target in dataset:
        w = model(seq.unsqueeze(0)).squeeze().detach().numpy()
        all_weights.append(w)
        all_returns.append(np.dot(w, target.numpy()))
    all_returns = np.array(all_returns)
    sharpe = np.mean(all_returns) / (np.std(all_returns) + 1e-8) * np.sqrt(252)
    print(f"\nFinal Sharpe Ratio: {sharpe:.4f}")