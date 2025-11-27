import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model(df, model_path='model.joblib'):
    """
    Trains a machine learning model to predict injury risk.
    Compares Logistic Regression and Random Forest, saves the best one.

    Args:
        df (pd.DataFrame): Dataframe with features and target.
        model_path (str): Path to save the trained model.

    Returns:
        model: The best trained model.
        dict: Performance metrics of the best model.
    """
    # Define features and target
    # We use the raw inputs + derived features
    feature_cols = ['tennis_hours', 'gym_hours', 'sleep_hours', 'rpe', 'matches_played', 'total_hours', 'load_index', 'recovery_index', 'recovery_score']
    target_col = 'near_injury'

    X = df[feature_cols]
    y = df[target_col]

    # Split data (80% train, 20% test)
    # random_state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    best_model = None
    best_accuracy = 0
    best_name = ""
    results = {}

    print("Training models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        
        print(f"\n{name} Performance:")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        if acc >= best_accuracy:
            best_accuracy = acc
            best_model = model
            best_name = name

    print(f"\nBest model: {best_name} with accuracy {best_accuracy:.4f}")

    # Save the best model
    if best_model:
        joblib.dump(best_model, model_path)
        print(f"Model saved to {model_path}")

    return best_model, results

if __name__ == "__main__":
    from data_loader import load_data
    from features import add_features

    try:
        df = load_data("data.csv")
        df = add_features(df)
        train_model(df)
    except Exception as e:
        print(f"Error during training: {e}")
