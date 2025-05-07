import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Download historical stock data for multiple tickers
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]  # Add more tickers as needed
start_date = "2018-01-01"
end_date = "2023-01-01"

# Download data (this will have a multi-level column structure)
data = yf.download(tickers, start=start_date, end=end_date)

# Step 2: Handle the multi-level column structure
# Extract the "Close" prices for all tickers
close_prices = data["Close"]

# Calculate daily returns
returns = close_prices.pct_change().dropna()

# Step 3: Modern Portfolio Theory (MPT)
def portfolio_performance(weights, mean_returns, cov_matrix):
    """Calculate portfolio return and volatility."""
    weights = np.array(weights)  # Convert weights to a NumPy array
    portfolio_return = np.dot(weights.T, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.0):
    """Optimize portfolio for maximum Sharpe ratio."""
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Sum of weights = 1
    bounds = tuple((0, 1) for _ in range(num_assets))  # No short selling
    initial_guess = np.array([1 / num_assets] * num_assets)  # Equal weights

    # Maximize Sharpe ratio
    def negative_sharpe_ratio(weights, mean_returns, cov_matrix):
        portfolio_return, portfolio_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
        return -(portfolio_return - risk_free_rate) / portfolio_volatility

    result = minimize(negative_sharpe_ratio, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Calculate mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Optimize portfolio using MPT
optimal_weights_mpt = optimize_portfolio(mean_returns, cov_matrix)
print("Optimal Weights (MPT):", dict(zip(tickers, optimal_weights_mpt)))

# Step 4: Random Forest Approach
def random_forest_portfolio(returns, tickers):
    """Predict future returns using Random Forest and construct a portfolio."""
    X = returns.iloc[:-1]  # Features: past returns
    y = returns.iloc[1:]   # Target: next day's returns

    predicted_weights = {}
    for ticker in tickers:
        X_train, X_test, y_train, y_test = train_test_split(X, y[ticker], test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"{ticker} - Mean Squared Error: {mse}")

        # Predict future returns
        future_return = model.predict(returns.iloc[-1:].values.reshape(1, -1))[0]
        predicted_weights[ticker] = future_return

    # Normalize weights to sum to 1
    total = sum(predicted_weights.values())
    predicted_weights = {k: v / total for k, v in predicted_weights.items()}
    return predicted_weights

# Get predicted weights using Random Forest
predicted_weights_rf = random_forest_portfolio(returns, tickers)
print("Predicted Weights (Random Forest):", predicted_weights_rf)

# Step 5: Create a table for portfolio weights and Sharpe Ratio
def calculate_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    """Calculate Sharpe Ratio for a given portfolio."""
    weights = np.array(weights)  # Convert weights to a NumPy array
    portfolio_return, portfolio_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return sharpe_ratio

# Calculate Sharpe Ratio for MPT and RF portfolios
mpt_sharpe = calculate_sharpe_ratio(optimal_weights_mpt, mean_returns, cov_matrix)
rf_sharpe = calculate_sharpe_ratio(list(predicted_weights_rf.values()), mean_returns, cov_matrix)

# Create a DataFrame for the weights and Sharpe Ratio
weights_df = pd.DataFrame({
    'Ticker': tickers,
    'MPT Weight': [optimal_weights_mpt[i] for i in range(len(tickers))],
    'RF Weight': [predicted_weights_rf[ticker] for ticker in tickers]
})

# Add Sharpe Ratio to the DataFrame
weights_df.loc[len(weights_df)] = ['Sharpe Ratio', mpt_sharpe, rf_sharpe]

# Display the table
print("Portfolio Weights and Sharpe Ratio:")
print(weights_df)

# Step 6: Compare MPT and Random Forest Portfolios
def calculate_portfolio_value(weights, returns):
    """Calculate cumulative portfolio value over time."""
    portfolio_returns = (returns * weights).sum(axis=1)
    return (1 + portfolio_returns).cumprod()

# MPT Portfolio
mpt_portfolio_value = calculate_portfolio_value(optimal_weights_mpt, returns)

# Random Forest Portfolio
rf_weights_array = np.array([predicted_weights_rf[ticker] for ticker in tickers])
rf_portfolio_value = calculate_portfolio_value(rf_weights_array, returns)

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(mpt_portfolio_value, label="MPT Portfolio")
plt.plot(rf_portfolio_value, label="Random Forest Portfolio")
plt.title("Portfolio Value Over Time: MPT vs Random Forest")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.legend()
plt.show()