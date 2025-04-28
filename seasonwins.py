import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# === Load Data ===
df = pd.read_csv("cleaned_win_prob_data.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Season'] = df['Season'].astype(str).str.strip()
df['Rslt'] = df['Rslt'].astype(str).str.strip().str.upper()

# === Target Teams ===
TARGET_TEAMS = {
    'Arizona', 'Bethune-Cookman', 'Bowling Green', 'Charleston Southern',
    'Charlotte', 'Davidson', 'Dayton', 'Detroit Mercy', 'Duquesne',
    'East Tennessee State', 'Eastern Michigan', 'Fordham', 'George Mason',
    'George Washington', 'Gonzaga', 'La Salle', 'Loyola Chicago',
    'Massachusetts', 'Providence', 'Rhode Island', 'Richmond',
    "Saint Joseph's", 'Saint Louis', 'St. Bonaventure', 'Temple',
    'VCU', 'VMI'
}

# === Compute Season Averages ===
def compute_season_averages(data):
    season_stats = data.groupby(['Team', 'Season']).agg({
        'eFG%': 'mean',
        'Opp_eFG%': 'mean',
        'TOV%': 'mean',
        'Opp_TOV%': 'mean',
        'FT%': 'mean',
        'Opp_FT%': 'mean',
        'ORB%': 'mean',
        'Opp_ORB%': 'mean',
        'Rslt': lambda x: (x == 'W').sum()
    }).reset_index().rename(columns={'Rslt': 'Total_Wins'})

    season_stats['Net_eFG%'] = season_stats['eFG%'] - season_stats['Opp_eFG%']
    season_stats['Net_TOV%'] = season_stats['Opp_TOV%'] - season_stats['TOV%']
    season_stats['Net_FT%'] = season_stats['FT%'] - season_stats['Opp_FT%']
    season_stats['Net_ORB%'] = season_stats['ORB%'] - season_stats['Opp_ORB%']

    return season_stats

# === Build Training Data ===
def build_training_data_prior_year(data, prior_season="2023-2024", target_season="2024-2025"):
    season_data = compute_season_averages(data)

    prior_stats = season_data[season_data['Season'] == prior_season]
    target_wins = season_data[(season_data['Season'] == target_season) & (season_data['Total_Wins'] > 0)]

    merged = pd.merge(
        prior_stats,
        target_wins[['Team', 'Total_Wins']],
        on='Team',
        how='inner',
        suffixes=('_prior', '_target')
    )

    X = merged[['Net_eFG%', 'Net_TOV%', 'Net_FT%', 'Net_ORB%']]
    y = merged['Total_Wins']

    return X, y, merged[['Team', 'Total_Wins']]

# === Train Model ===
def train_linear_model(X, y):
    model = LinearRegression()
    model.fit(X, y)

    feature_names = ['Net_eFG%', 'Net_TOV%', 'Net_FT%', 'Net_ORB%']
    normalized_coefs = np.abs(model.coef_) / np.abs(model.coef_).sum()

    print("\n=== Normalized Feature Weights (Sum to 1) ===")
    for name, weight in zip(feature_names, normalized_coefs):
        print(f"{name}: {weight:.4f}")
    print(f"Intercept (not normalized): {model.intercept_:.4f}")
    print("============================================\n")

    return model

# === Evaluate Model ===
def evaluate_model(X, y, model, team_summary):
    y_pred = model.predict(X)
    r_squared = model.score(X, y)
    mae = np.mean(np.abs(y_pred - y))

    within_2_wins = np.sum(np.abs(y_pred - y) <= 2)
    total = len(y)
    within_2_accuracy = within_2_wins / total

    print("=== Model Evaluation ===")
    print(f"R-squared: {r_squared:.4f}")
    print(f"MAE: {mae:.2f} wins")
    print(f"Accuracy within 2 wins: {within_2_accuracy:.2%} ({within_2_wins}/{total} teams)")
    print("============================================\n")

    results = team_summary.copy()
    results['Predicted_Wins'] = np.round(y_pred, 1)
    print("=== Actual vs Predicted Wins ===")
    print(results[['Team', 'Total_Wins', 'Predicted_Wins']])

# === Main Execution ===
if __name__ == "__main__":
    X, y, season_summary = build_training_data_prior_year(df)

    if X is not None and y is not None:
        print("Training linear regression model...")
        model = train_linear_model(X, y)

        print("\nEvaluating model...")
        evaluate_model(X, y, model, season_summary)

    print("\n=== Total Wins Distribution ===")
    print(season_summary['Total_Wins'].value_counts())
