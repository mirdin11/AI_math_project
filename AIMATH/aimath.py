import pandas as pd
import numpy as np
import glob
import os
from scipy.optimize import minimize

# Load and preprocess data
def load_data(data_path='.'):
    # Load files from ETFs and stocks folders
    csv_files = glob.glob(os.path.join(data_path, 'ETFs', '*.csv')) + glob.glob(os.path.join(data_path, 'stocks', '*.csv'))
    dataframes = []
    
    if not csv_files:
        print("No CSV files found in ETFs or stocks folders. Please check the dataset path.")
    else:
        print(f"Found {len(csv_files)} CSV files.")
    
    for file in csv_files:
        ticker = os.path.splitext(os.path.basename(file))[0]
        try:
            # Load only 'Date' and 'Close' columns
            df = pd.read_csv(file, usecols=['Date', 'Close'])
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.rename(columns={'Close': ticker}, inplace=True)
            dataframes.append(df)
            print(f"Loaded data for {ticker}.")
        except KeyError:
            print(f"Skipping {file}: missing 'Date' or 'Close' columns.")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if dataframes:
        data = pd.concat(dataframes, axis=1, join='inner')
        print("Data loaded successfully.")
    else:
        raise ValueError("No valid data files found.")
    
    return data

# Calculate daily returns for each asset
def calculate_returns(data):
    returns = data.pct_change().dropna()
    return returns

# Define the objective function to minimize risk (portfolio variance)
def portfolio_variance(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights

# Define constraints and bounds
def optimize_portfolio(returns):
    num_assets = len(returns.columns)
    cov_matrix = returns.cov()
    
    # Initial weights (equal distribution)
    init_guess = np.ones(num_assets) / num_assets
    # Constraints: weights must sum to 1
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    # Bounds for each weight (between 0 and 1)
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Perform optimization
    result = minimize(
        portfolio_variance, 
        init_guess, 
        args=(cov_matrix,), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    return result.x, result.fun

# Main function to run the analysis
def main():
    data_path = '.'  # Update path if needed
    try:
        data = load_data(data_path)
        returns = calculate_returns(data)
        
        # Expected returns (mean returns for each asset)
        expected_returns = returns.mean()
        
        # Optimize portfolio
        optimal_weights, portfolio_risk = optimize_portfolio(returns)
        
        # Display results
        print("Optimal Weights:", optimal_weights)
        print("Expected Portfolio Return:", np.dot(optimal_weights, expected_returns))
        print("Portfolio Risk (Variance):", portfolio_risk)
    except ValueError as e:
        print(e)

# Run the main function
if __name__ == '__main__':
    main()
