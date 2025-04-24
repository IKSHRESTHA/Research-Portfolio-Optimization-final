import pandas as pd
import os

# Define the path to the data/raw_data directory
input_dir = 'c:/Users/krish/OneDrive/Desktop/Research-Portfolio Optimization/data/raw_data'
input_path = os.path.join(input_dir, 'stock_data_raw.csv')

# Read the CSV file into a DataFrame
df = pd.read_csv(input_path)

# Display first few rows
print(df.head())

# Filter the data for the past 10 years
# We only took 10 years of data so that all companies have a history of it.
# Doing this as well because a few companies were not founded at that time.
df['Date'] = pd.to_datetime(df['Date'])
ten_years_ago = pd.Timestamp.now() - pd.DateOffset(years=10)
df_filtered = df[df['Date'] >= ten_years_ago]

# Remove columns corresponding to MRNA and PYPL
columns_to_remove = [col for col in df_filtered.columns if 'MRNA' in col or 'PYPL' in col]
df_filtered = df_filtered.drop(columns=columns_to_remove)

# Count the number of NaN values in each column
na_counts = df_filtered.isna().sum()

# Display the count of NaN values for each column
print(na_counts[na_counts > 0])

# Calculate the percentage of columns that have NaN values
total_columns = len(df_filtered.columns)
columns_with_na = len(na_counts[na_counts > 0])
percentage_with_na = (columns_with_na / total_columns) * 100

# Display the percentage of columns with NaN values
print(f"Percentage of columns with NaN values: {percentage_with_na:.2f}%")

# Define the path to the clean_data directory
output_dir = 'c:/Users/krish/OneDrive/Desktop/Research-Portfolio Optimization/data/clean_data'

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the output file path
output_path = os.path.join(output_dir, 'stock_data_clean.csv')

# Write the cleaned DataFrame to a CSV file
df_filtered.to_csv(output_path, index=False)

# Display the cleaned data
print(f"Here is the cleaned data which consists of {df_filtered.shape[0]} rows and {df_filtered.shape[1]} columns.")