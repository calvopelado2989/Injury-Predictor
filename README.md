# Tennis Injury Risk Predictor

A machine learning project to predict training fatigue and injury risk for competitive tennis players. This tool analyzes daily training metrics (tennis hours, gym hours, sleep, RPE) to estimate the likelihood of "near injury" or excessive fatigue.

## Project Structure

- `data.csv`: Synthetic daily training data.
- `data_loader.py`: Handles data loading and cleaning.
- `features.py`: Calculates derived metrics like Total Load and Recovery Index.
- `model.py`: Trains and compares Logistic Regression and Random Forest models.
- `evaluate.py`: Evaluates the model and generates performance plots.
- `cli.py`: Interactive command-line tool for daily risk prediction.
- `main.py`: Runs the full training and evaluation pipeline.

## Installation

1. Ensure you have Python installed.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn joblib
   ```

## Usage

### 1. Train the Model
Run the main pipeline to load data, train the model, and generate reports:
```bash
python main.py
```
This will create:
- `model.joblib`: The saved best model.
- `training_load.png`: Visualization of weekly workload.
- `risk_analysis.png`: Visualization of risk vs. load.

### 2. Make a Prediction
Use the CLI to input today's stats and get a risk assessment:
```bash
python cli.py
```
Example Input:
- Tennis Hours: 2.5
- Gym Hours: 1.0
- Sleep Hours: 7.0
- RPE: 8
- Matches Played: 0

## Methodology
The model uses **Logistic Regression** and **Random Forest Classifiers** to predict a binary `near_injury` label.
Key features include:
- **Load Index**: `(Tennis + Gym Hours) * RPE`
- **Recovery Index**: `Sleep Hours / RPE`

*This is a demonstration project for educational purposes and should not replace professional medical advice.*
