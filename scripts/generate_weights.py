# Used to get weights for the factors

import pandas as pd
from sklearn.linear_model import LinearRegression

# === Load Data ===
df = pd.read_csv("cleaned_win_prob_data.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# === Prepare Data ===
# Create a 'Win' column: 1 if result is W, 0 if L
df['Win'] = (df['Rslt'] == 'W').astype(int)

# === Regression Training Function ===
def train_four_factors_plus_steals_blocks_regression(all_data):
    # Define columns
    factor_cols = [
        'eFG%', 'TOV', 'TOV%', 'ORB%', 'FT%',
        'Opp_eFG%', 'Opp_TOV', 'Opp_TOV%', 'Opp_ORB%', 'Opp_FT%',
        'STL', 'BLK', 'FGA', 'FTA', 'Opp_FGA', 'Opp_FTA'
    ]
    
    # Drop missing rows
    all_data = all_data.dropna(subset=factor_cols + ['Win', 'Source_File', 'Season'])

    # Extract Team name from filename
    all_data['Team'] = all_data['Source_File'].apply(lambda x: x.replace('.csv', '').lower())

    # Group by Team and Season: mean stats (per game)
    team_season_stats = all_data.groupby(['Team', 'Season']).agg({
        'eFG%': 'mean',
        'TOV': 'mean',
        'TOV%': 'mean',
        'ORB': 'mean',      
        'ORB%': 'mean',
        'FT%': 'mean',
        'Opp_eFG%': 'mean',
        'Opp_TOV': 'mean',
        'Opp_TOV%': 'mean',
        'Opp_ORB': 'mean', 
        'Opp_ORB%': 'mean',
        'Opp_FT%': 'mean',
        'STL': 'mean',
        'BLK': 'mean',
        'FGA': 'mean',
        'FTA': 'mean',
        'Opp_FGA': 'mean',
        'Opp_FTA': 'mean',
        'Win': 'sum'
}).reset_index()


    # Calculate total games played
    team_season_stats['Games'] = all_data.groupby(['Team', 'Season'])['Win'].count().values

    # === SCALE per-game stats to season totals ===
    team_season_stats['Total_Steals'] = team_season_stats['STL'] * team_season_stats['Games']
    team_season_stats['Total_Blocks'] = team_season_stats['BLK'] * team_season_stats['Games']
    team_season_stats['Total_FGA'] = team_season_stats['FGA'] * team_season_stats['Games']
    team_season_stats['Total_FTA'] = team_season_stats['FTA'] * team_season_stats['Games']
    team_season_stats['Total_TOV'] = team_season_stats['TOV'] * team_season_stats['Games']
    team_season_stats['Total_ORB'] = team_season_stats['ORB'] * team_season_stats['Games']
    team_season_stats['Total_Opp_FGA'] = team_season_stats['Opp_FGA'] * team_season_stats['Games']
    team_season_stats['Total_Opp_FTA'] = team_season_stats['Opp_FTA'] * team_season_stats['Games']
    team_season_stats['Total_Opp_TOV'] = team_season_stats['Opp_TOV'] * team_season_stats['Games']

    # === Estimate Team and Opponent Possessions (fixed)
    team_season_stats['Team_Possessions'] = (
        team_season_stats['Total_FGA'] +
        0.44 * team_season_stats['Total_FTA'] -
        team_season_stats['Total_ORB'] +
        team_season_stats['Total_TOV']
    )
    
    team_season_stats['Opponent_Possessions'] = (
        team_season_stats['Total_Opp_FGA'] +
        0.44 * team_season_stats['Total_Opp_FTA'] +
        team_season_stats['Total_Opp_TOV']
    )

    # === Calculate Steal% and Block% (normalized by opponent possessions)
    team_season_stats['Steal%'] = (team_season_stats['Total_Steals'] / team_season_stats['Opponent_Possessions']) * 100
    team_season_stats['Block%'] = (team_season_stats['Total_Blocks'] / team_season_stats['Opponent_Possessions']) * 100

    # === Create Net Four Factors
    team_season_stats['Net_eFG%'] = team_season_stats['eFG%'] - team_season_stats['Opp_eFG%']
    team_season_stats['Net_TOV%'] = team_season_stats['Opp_TOV%'] - team_season_stats['TOV%']
    team_season_stats['Net_ORB%'] = team_season_stats['ORB%'] - team_season_stats['Opp_ORB%']
    team_season_stats['Net_FT%'] = team_season_stats['FT%'] - team_season_stats['Opp_FT%']

    # === Create Win%
    team_season_stats['Win%'] = team_season_stats['Win'] / team_season_stats['Games']

    # === Final Features (X) and Target (y)
    X = team_season_stats[['Net_eFG%', 'Net_TOV%', 'Net_ORB%', 'Net_FT%', 'Steal%', 'Block%']]
    y = team_season_stats['Win%']

    # Train Linear Regression
    model = LinearRegression()
    model.fit(X, y)

    # Output results (Normalized Coefficients)
    coefficients = model.coef_
    normalized_coefficients = coefficients / coefficients.sum()

    print("\n=== Normalized Regression Coefficients (Sum to 1) ===")
    for feature, coef in zip(X.columns, normalized_coefficients):
        print(f"{feature}: {coef:.4f}")

    print(f"\nIntercept (not normalized): {model.intercept_:.4f}")
    print(f"R-squared (model fit): {model.score(X, y):.4f}")
    
    return model

# === Run training when file is executed
if __name__ == "__main__":
    trained_model = train_four_factors_plus_steals_blocks_regression(df)
