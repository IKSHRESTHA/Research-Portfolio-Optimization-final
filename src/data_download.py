import yfinance as yf
import pandas as pd
import os

# List of stock tickers with more than 8 years of trading history
tickers = [
    'AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'PYPL', 'ADBE',
    'INTC', 'AMD', 'IBM', 'ORCL', 'CSCO', 'QCOM', 'TXN', 'BA', 'V', 'MA',
    'JPM', 'GS', 'BRK-B', 'WFC', 'C', 'XOM', 'CVX', 'JNJ', 'PFE', 'MRNA', 'UNH',
    'GE', 'T', 'PEP', 'KO', 'DIS', 'MCD', 'NKE', 'PG', 'CRM', 'HON', 'MDT',
    'ABBV', 'ACN', 'AMGN', 'AVGO', 'AXP', 'BMY', 'CAT', 'CL', 'COP', 'COST'
]

# Define the date range for data retrieval
start_date = '2023-03-15'
end_date = '2025-03-15'

# Download data
df = yf.download(tickers, start=start_date, end=end_date, interval='1d', group_by='ticker')

# Flatten the MultiIndex columns
df.columns = ['_'.join(col).strip() for col in df.columns]

# Reset index to keep Date as a column
df = df.reset_index()

# Select only columns containing 'Close' or 'Volume' along with 'Date'
df = df[['Date'] + [col for col in df.columns if 'Close' in col or 'Volume' in col]]

# Define the path to the data/raw_data directory
output_dir = 'c:/Users/krish/OneDrive/Desktop/Research-Portfolio Optimization/data/raw_data'

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the output file path
output_path = os.path.join(output_dir, 'stock_data_raw.csv')

# Write the DataFrame to a CSV file
df.to_csv(output_path, index=False)

# Display first few rows
print(df.head())