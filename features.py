import pandas as pd

def add_features(df):
    """
    Adds derived features to the dataframe for model training.

    Args:
        df (pd.DataFrame): Raw dataframe containing daily metrics.

    Returns:
        pd.DataFrame: Dataframe with new feature columns.
    """
    df = df.copy()

    # 1. Total daily training load (hours)
    df['total_hours'] = df['tennis_hours'] + df['gym_hours']

    # 2. Load intensity (Hours * RPE) - a common metric in sports science
    df['load_index'] = df['total_hours'] * df['rpe']

    # 3. Recovery score (Sleep / RPE) - higher is better recovery relative to exertion
    # Avoid division by zero by ensuring RPE is at least 1 (it should be 1-10)
    df['recovery_index'] = df['sleep_hours'] / df['rpe'].replace(0, 1)

    # 4. Recovery Activity Score
    # Map categorical activity to a numerical score
    # None=0, Stretching=1, Foam Rolling=2, Ice Bath/Massage=3
    recovery_map = {
        'None': 0,
        'Stretching': 1,
        'Foam Rolling': 2,
        'Ice Bath': 3,
        'Massage': 3
    }
    # Use map, fill NaN with 0 (None)
    if 'recovery_activity' in df.columns:
        df['recovery_score'] = df['recovery_activity'].map(recovery_map).fillna(0)
    else:
        # If column missing (e.g. old data), assume 0
        df['recovery_score'] = 0

    return df

if __name__ == "__main__":
    # Test feature generation
    try:
        from data_loader import load_data
        df = load_data("data.csv")
        df_features = add_features(df)
        print("Features added successfully:")
        print(df_features[['date', 'total_hours', 'load_index', 'recovery_index']].head())
    except Exception as e:
        print(f"Error generating features: {e}")
