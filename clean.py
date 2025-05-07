import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Define the Transformer Model for Time-Series Forecasting
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.fc_out = nn.Linear(d_model, input_dim)

    def forward(self, src):
        src = self.embedding(src)
        output = self.transformer(src, src)
        return self.fc_out(output[-1])

# Simulated Historical Data (Replace with Real Stock Data)
def generate_synthetic_data(num_assets=5, num_days=500):
    np.random.seed(42)
    returns = np.random.randn(num_days, num_assets) * 0.02  # Simulated daily returns
    prices = 100 * np.cumprod(1 + returns, axis=0)
    return pd.DataFrame(prices, columns=[f'Asset_{i}' for i in range(num_assets)])

data = generate_synthetic_data()
returns = data.pct_change().dropna()

# Data Preprocessing
scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns)

def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 30
X, y = create_sequences(returns_scaled, seq_length)

# Convert to PyTorch Tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(input_dim=X.shape[2]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_X.transpose(0, 1))
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Predict Future Returns
model.eval()
x_test = torch.tensor(returns_scaled[-seq_length:], dtype=torch.float32).unsqueeze(1).to(device)
predicted_returns = model(x_test).cpu().detach().numpy()
predicted_returns = scaler.inverse_transform(predicted_returns)

# Portfolio Optimization (Mean-Variance Optimization)
def optimize_portfolio(returns):
    cov_matrix = np.cov(returns.T)
    mean_returns = np.mean(returns, axis=0)
    inv_cov = np.linalg.inv(cov_matrix)
    weights = inv_cov @ mean_returns
    weights /= np.sum(weights)
    return weights

optimal_weights = optimize_portfolio(predicted_returns)
print("Optimal Portfolio Weights:", optimal_weights)
