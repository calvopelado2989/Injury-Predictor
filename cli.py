import joblib
import pandas as pd
import os
import sys

def generate_recommendations(data):
    """
    Generates specific recommendations based on input data.
    """
    recs = []
    
    # Sleep recommendation
    if data['sleep_hours'] < 7.5:
        recs.append("- Priority: Increase sleep to at least 8 hours tonight for recovery.")
    
    # Intensity recommendation
    if data['rpe'] > 7:
        recs.append("- High Intensity detected: Consider a lighter technical session or active recovery tomorrow.")
    
    # Volume recommendation
    if data['total_hours'] > 3.0:
        recs.append("- High Volume detected: Ensure you are hydrating and fueling enough for this workload.")
        
    # Match play recommendation
    if data['matches_played'] > 0:
        recs.append("- Post-Match: Focus on cool-down, stretching, and protein intake immediately.")

    if not recs:
        recs.append("- General: Maintain your current routine but stay vigilant for soreness.")
        
    return recs

def predict_risk():
    """
    Interactive CLI to predict injury risk for a new day.
    """
    model_path = 'model.joblib'
    if not os.path.exists(model_path):
        print("Error: Model not found. Please run main.py first to train the model.")
        return

    print("--- Injury Risk Predictor ---")
    print("Enter your training details for the day:")

    try:
        tennis_hours = float(input("Tennis Hours (e.g., 2.0): "))
        gym_hours = float(input("Gym Hours (e.g., 1.0): "))
        sleep_hours = float(input("Sleep Hours (e.g., 8.0): "))
        rpe = int(input("RPE (1-10): "))
        matches_played = int(input("Matches Played (e.g., 0): "))
        
        print("\nPost-Training Recovery:")
        print("1: None")
        print("2: Stretching")
        print("3: Foam Rolling")
        print("4: Ice Bath / Massage")
        recovery_choice = int(input("Select activity (1-4): "))
        
        recovery_map_input = {1: 0, 2: 1, 3: 2, 4: 3}
        recovery_score = recovery_map_input.get(recovery_choice, 0)
        
    except ValueError:
        print("Invalid input. Please enter numbers.")
        return

    # Create a DataFrame for the input
    input_data = pd.DataFrame([{
        'tennis_hours': tennis_hours,
        'gym_hours': gym_hours,
        'sleep_hours': sleep_hours,
        'rpe': rpe,
        'matches_played': matches_played,
        'recovery_score': recovery_score
    }])

    # Calculate derived features (must match features.py logic)
    input_data['total_hours'] = input_data['tennis_hours'] + input_data['gym_hours']
    input_data['load_index'] = input_data['total_hours'] * input_data['rpe']
    input_data['recovery_index'] = input_data['sleep_hours'] / input_data['rpe'].clip(lower=1) # Avoid div by 0

    # Load model
    model = joblib.load(model_path)

    # Predict
    # Ensure columns match training order
    feature_cols = ['tennis_hours', 'gym_hours', 'sleep_hours', 'rpe', 'matches_played', 'total_hours', 'load_index', 'recovery_index', 'recovery_score']
    X_new = input_data[feature_cols]

    try:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_new)[0][1]
            print(f"\nPredicted Probability of Near Injury: {prob:.2%}")
            
            if prob < 0.3:
                print("Risk Level: LOW - You are good to go!")
            else:
                if prob < 0.7:
                    print("Risk Level: MODERATE - Be careful, monitor your body.")
                else:
                    print("Risk Level: HIGH - Consider resting or reducing intensity.")
                
                print("\nRECOMMENDATIONS:")
                # Pass the single row as a Series/dict for easier access
                recs = generate_recommendations(input_data.iloc[0])
                for rec in recs:
                    print(rec)

        else:
            prediction = model.predict(X_new)[0]
            print(f"\nPredicted Near Injury: {'YES' if prediction == 1 else 'NO'}")
            if prediction == 1:
                print("Risk Level: HIGH - Model predicts a risk of injury/fatigue.")
                print("\nRECOMMENDATIONS:")
                recs = generate_recommendations(input_data.iloc[0])
                for rec in recs:
                    print(rec)
            else:
                print("Risk Level: LOW - Model predicts no immediate risk.")

    except Exception as e:
        print(f"Error making prediction: {e}")

if __name__ == "__main__":
    predict_risk()
