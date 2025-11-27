import pandas as pd
import os

def load_data(filepath):
    """
    Loads training data from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data with parsed dates.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} was not found.")

    # Load data
    df = pd.read_csv(filepath)

    # Convert date column to datetime objects
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    return df

if __name__ == "__main__":
    # Simple test to verify loading
    try:
        df = load_data("data.csv")
        print("Data loaded successfully:")
        print(df.head())
    except Exception as e:
        print(f"Error loading data: {e}")
