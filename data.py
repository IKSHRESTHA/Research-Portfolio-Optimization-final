import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

# Step 1: Fetch historical data using Alpha Vantage API
def fetch_data(api_key, tickers):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={tickers}&apikey={api_key}"
    response = requests.get(url)
    json_data = response.json()
    print(json_data)  # Print the entire response
    df = pd.DataFrame(json_data["Time Series (Daily)"]).T
    return df
# Step 2: Prepare data for Random Forest
def prepare_data(data, window=30):
    """
    Prepare data for training the Random Forest model.
    """
    returns = data.pct_change().dropna()  # Calculate daily returns
    X = returns.shift(1).dropna()  # Features (lagged returns)
    y = returns.iloc[1:]  # Target (next day's returns)
    return X, y

# Step 3: Train Random Forest model
def train_random_forest(X, y):
    """
    Train a Random Forest Regressor to predict future returns.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 4: Predict future returns using Random Forest
def predict_returns(model, data):
    """
    Predict the next day's returns using the trained Random Forest model.
    """
    latest_returns = data.pct_change().iloc[-1].values.reshape(1, -1)  # Latest returns as input
    predicted_returns = model.predict(latest_returns)  # Predict next day's returns
    return predicted_returns

# Step 5: Portfolio optimization using Mean-Variance Theory (Manual Implementation)
def optimize_portfolio(returns, predicted_returns):
    """
    Optimize the portfolio using Mean-Variance Theory (MVT) without pyportfolioopt.
    """
    n_assets = returns.shape[1]
    cov_matrix = returns.cov()  # Covariance matrix
    expected_returns = predicted_returns  # Use predicted returns as expected returns

    # Objective function: Minimize portfolio risk (variance)
    def objective(weights):
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return portfolio_variance

    # Constraints: Sum of weights = 1, and each weight >= 0
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))  # No short selling

    # Initial guess: Equal weights
    initial_guess = np.ones(n_assets) / n_assets

    # Optimize using scipy's minimize function
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x  # Optimal weights

# Step 6: Main function
def main():
    """
    Main function to execute the portfolio optimization process.
    """
    # Define tickers and Alpha Vantage API key
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']  # Example tickers
    api_key = "Y1TEZ7EG8RRCDGYH"  # Replace with your Alpha Vantage API key

    # Step 1: Fetch data
    print("Fetching data...")
    data = fetch_data(api_key, tickers)
    print(data)

    # Step 2: Prepare data for Random Forest
   # print("Preparing data for Random Forest...")
    #X, y = prepare_data(data)

    # Step 3: Train Random Forest model
    #print("Training Random Forest model...")
    #model = train_random_forest(X, y)

    # Step 4: Predict returns
    #print("Predicting future returns...")
    #predicted_returns = predict_returns(model, data)

    # Step 5: Optimize portfolio
    #print("Optimizing portfolio...")
    #returns = data.pct_change().dropna()  # Historical returns for covariance matrix
    #weights = optimize_portfolio(returns, predicted_returns)

    # Display results
    #print("\nOptimal Portfolio Weights:")
    #for ticker, weight in zip(tickers, weights):
     #   print(f"{ticker}: {weight:.2%}")

if __name__ == "__main__":
    main()