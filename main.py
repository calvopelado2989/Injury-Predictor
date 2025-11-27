from data_loader import load_data
from features import add_features
from model import train_model
from evaluate import evaluate_and_plot
import sys

def main():
    print("Starting Injury Risk Detection Pipeline...")

    # 1. Load Data
    print("\n[1/4] Loading data...")
    try:
        df = load_data("data.csv")
        print(f"Loaded {len(df)} records.")
    except Exception as e:
        print(f"Failed to load data: {e}")
        sys.exit(1)

    # 2. Feature Engineering
    print("\n[2/4] Generating features...")
    df = add_features(df)
    print("Features added.")

    # 3. Model Training
    print("\n[3/4] Training model...")
    model, results = train_model(df)
    
    # 4. Evaluation
    print("\n[4/4] Evaluating and plotting...")
    evaluate_and_plot(df)

    print("\nPipeline completed successfully!")
    print("Run 'python cli.py' to make new predictions.")

if __name__ == "__main__":
    main()
