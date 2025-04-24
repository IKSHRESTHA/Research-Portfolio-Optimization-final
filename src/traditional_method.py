import pandas as pd
import os
import random
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- Load data ---
clean_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'clean_data')
clean_data_path = os.path.join(clean_data_dir, 'stock_data_clean.csv')
df_clean = pd.read_csv(clean_data_path)

# Select specific assets instead of random
selected_shares = ['CSCO_Close', 'KO_Close', 'NVDA_Close', 'CRM_Close', 'AVGO_Close']
returns = df_clean[selected_shares].pct_change().dropna()

# Annualize risk-free rate (2%)
annual_risk_free_rate = 0.02
risk_free_rate = (1 + annual_risk_free_rate) ** (1/252) - 1  # Daily rate

# --- Portfolio performance function ---
def portfolio_performance(weights, mean_returns, cov_matrix, annual_risk_free_rate):
    annual_returns = np.sum(mean_returns * weights) * 252
    annual_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = (annual_returns - annual_risk_free_rate) / annual_volatility
    return annual_volatility, annual_returns, sharpe_ratio

# --- Sharpe-maximization optimization ---
def portfolio_optimization(mean_returns, cov_matrix, annual_risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, annual_risk_free_rate)

    def neg_sharpe_ratio(weights, *args):
        _, _, sharpe = portfolio_performance(weights, *args)
        return -sharpe

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.ones(num_assets) / num_assets

    result = minimize(
        neg_sharpe_ratio,
        initial_weights,
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")
    return result

# --- Run optimization for max Sharpe ---
mean_returns = returns.mean()
cov_matrix = returns.cov() * 252  # Annualize covariance
optimal_portfolio = portfolio_optimization(mean_returns, cov_matrix, annual_risk_free_rate)
optimal_weights = np.round(optimal_portfolio.x, 3)

# Print results
print("Selected assets:", selected_shares)
print(f"Optimal weights: {optimal_weights}")

volatility, annual_return, sharpe = portfolio_performance(
    optimal_weights, mean_returns, cov_matrix, annual_risk_free_rate
)
print("\nOptimal Portfolio Performance:")
print(f"Annualized Volatility (Risk): {volatility:.4f}")
print(f"Annualized Return: {annual_return:.4f}")
print(f"Sharpe Ratio: {sharpe:.4f}")

# --- Efficient frontier helper ---
def minimize_volatility_for_return(target_return, mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (cov_matrix,)

    def portfolio_variance(weights, cov_matrix):
        return np.dot(weights.T, np.dot(cov_matrix, weights))

    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) * 252 - target_return}
    )
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = np.ones(num_assets) / num_assets

    sol = minimize(portfolio_variance, init_guess, args=args,
                   method='SLSQP', bounds=bounds, constraints=constraints)
    if not sol.success:
        raise ValueError(f"Optimization failed for target return {target_return:.4f}: {sol.message}")
    return sol.x

# --- Compute and plot efficient frontier ---
min_ret = np.min(mean_returns) * 252
max_ret = np.max(mean_returns) * 252
target_returns = np.linspace(min_ret, max_ret, 50)

frontier_vols = []
frontier_rets = []
for tr in target_returns:
    w = minimize_volatility_for_return(tr, mean_returns, cov_matrix)
    vol, ann_ret, _ = portfolio_performance(w, mean_returns, cov_matrix, annual_risk_free_rate)
    frontier_vols.append(vol)
    frontier_rets.append(ann_ret)

# Create Capital Market Line
max_sharpe_vol = volatility  # From the optimal portfolio
max_sharpe_ret = annual_return  # From the optimal portfolio
cml_x = np.array([0, max_sharpe_vol * 1.5])  # Extend beyond the tangency point
cml_y = annual_risk_free_rate + (max_sharpe_ret - annual_risk_free_rate) / max_sharpe_vol * cml_x

plt.figure(figsize=(10, 6))
plt.plot(frontier_vols, frontier_rets, 'b-', linewidth=2, label='Efficient Frontier')
plt.plot(cml_x, cml_y, 'r--', linewidth=2, label='Capital Market Line')
plt.scatter(volatility, annual_return, marker='*', s=200, color='green', label='Maximum Sharpe Ratio')
plt.scatter(0, annual_risk_free_rate, marker='o', color='red', label='Risk-Free Rate')
plt.title('Efficient Frontier and Capital Market Line')
plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Return')
plt.legend()
plt.grid(True)
plt.show()