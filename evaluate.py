import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def evaluate_and_plot(df, model_path='model.joblib'):
    """
    Loads the model, makes predictions on the full dataset, and generates plots.

    Args:
        df (pd.DataFrame): Dataframe with features.
        model_path (str): Path to the saved model.
    """
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Cannot evaluate.")
        return

    model = joblib.load(model_path)
    
    # Prepare features
    feature_cols = ['tennis_hours', 'gym_hours', 'sleep_hours', 'rpe', 'matches_played', 'total_hours', 'load_index', 'recovery_index', 'recovery_score']
    X = df[feature_cols]
    
    # Predict probabilities (if supported) or labels
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1] # Probability of class 1 (near_injury)
        df['injury_prob'] = probs
    else:
        df['injury_prob'] = model.predict(X)

    # --- Plot 1: Weekly Training Load ---
    plt.figure(figsize=(10, 6))
    # Resample to weekly sum if we had enough data, but for 20 days, daily bar chart is better
    # Let's do a daily load chart
    plt.bar(df['date'], df['total_hours'], color='skyblue', label='Total Training Hours')
    plt.xlabel('Date')
    plt.ylabel('Hours')
    plt.title('Daily Training Load (Tennis + Gym)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_load.png')
    print("Saved plot: training_load.png")
    plt.close()

    # --- Plot 2: Load vs Injury Risk ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='load_index', y='injury_prob', hue='near_injury', palette={0: 'green', 1: 'red'}, s=100)
    plt.xlabel('Load Index (Hours * RPE)')
    plt.ylabel('Predicted Injury Probability')
    plt.title('Training Load vs. Predicted Injury Risk')
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Decision Threshold')
    plt.legend(title='Actual Near Injury')
    plt.tight_layout()
    plt.savefig('risk_analysis.png')
    print("Saved plot: risk_analysis.png")
    plt.close()

if __name__ == "__main__":
    from data_loader import load_data
    from features import add_features

    try:
        df = load_data("data.csv")
        df = add_features(df)
        evaluate_and_plot(df)
    except Exception as e:
        print(f"Error during evaluation: {e}")
