import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
import tkinter as tk
from tkinter import messagebox, ttk

# === Load Data ===
df = pd.read_csv("cleaned_win_prob_data.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# === Target Teams (Only these matter for prediction) ===
TARGET_TEAMS = {
    'Arizona', 'Bethune-Cookman', 'Bowling Green', 'Charleston Southern',
    'Charlotte', 'Davidson', 'Dayton', 'Detroit Mercy', 'Duquesne',
    'East Tennessee State', 'Eastern Michigan', 'Fordham', 'George Mason',
    'George Washington', 'Gonzaga', 'La Salle', 'Loyola Chicago',
    'Massachusetts', 'Providence', 'Rhode Island', 'Richmond',
    "Saint Joseph's", 'Saint Louis', 'St. Bonaventure', 'Temple',
    'VCU', 'VMI'
}

def extract_team_list(data):
    teams = sorted(set(data['Team'].dropna().unique()))
    return [team for team in teams if team in TARGET_TEAMS]

TEAM_LIST = extract_team_list(df)

# === Core Functions ===
def get_pre_matchup_games(team_name, opponent_name, season, all_data):
    team_games = all_data.copy()
    team_games = team_games[
        (team_games['Team'] == team_name) & 
        (team_games['Season'] == season)
    ].dropna(subset=["Date"]).sort_values(by="Date")
    return team_games

def compute_advanced_stats(past_games):
    if past_games.empty:
        raise ValueError("No past games available to compute stats.")

    stat_columns = [
        'eFG%', 'TOV', 'TOV%', 'ORB', 'ORB%', 'FT%',
        'Opp_eFG%', 'Opp_TOV', 'Opp_TOV%', 'Opp_ORB%', 'Opp_FT%',
        'FGA', 'FTA', 'Opp_FGA', 'Opp_FTA',
        'STL', 'BLK'
    ]

    means = past_games[stat_columns].mean()

    team_possessions = (
        means['FGA'] +
        0.44 * means['FTA'] -
        means['ORB'] +
        means['TOV']
    )

    opponent_possessions = (
        means['Opp_FGA'] +
        0.44 * means['Opp_FTA'] +
        means['Opp_TOV']
    )

    steal_rate = (means['STL'] / opponent_possessions) * 100 if opponent_possessions else 0
    block_rate = (means['BLK'] / opponent_possessions) * 100 if opponent_possessions else 0

    stats = {
        'Net_TOV%': means['Opp_TOV%'] - means['TOV%'],
        'Net_eFG%': means['eFG%'] - means['Opp_eFG%'],
        'Net_FT%': means['FT%'] - means['Opp_FT%'],
        'Net_ORB%': means['ORB%'] - means['Opp_ORB%'],
        'Steal%': steal_rate,
        'Block%': block_rate
    }
    return stats

def build_training_data(all_data, season="2024-2025"):
    X = []
    y = []

    all_data = all_data.dropna(subset=['Date', 'Rslt', 'Team', 'Opp'])

    for idx, game in all_data.iterrows():
        team1 = game['Team']
        team2 = game['Opp']
        game_season = game['Season']
        actual_result = game['Rslt']

        if team1 not in TARGET_TEAMS or team2 not in TARGET_TEAMS:
            continue

        if game_season != season:
            continue

        try:
            team1_games = get_pre_matchup_games(team1, team2, season, all_data)
            team2_games = get_pre_matchup_games(team2, team1, season, all_data)

            if team1_games.empty or team2_games.empty:
                continue

            team1_stats = compute_advanced_stats(team1_games)
            team2_stats = compute_advanced_stats(team2_games)

            features = [
                team1_stats['Net_eFG%'] - team2_stats['Net_eFG%'],
                (team1_stats['Net_TOV%'] - team2_stats['Net_TOV%']) * 50,  # Increased weight
                team1_stats['Net_FT%'] - team2_stats['Net_FT%'],
                team1_stats['Net_ORB%'] - team2_stats['Net_ORB%'],
                team1_stats['Steal%'] - team2_stats['Steal%'],
                team1_stats['Block%'] - team2_stats['Block%']
            ]

            X.append(features)

            y.append(1 if actual_result == 'W' else 0)

        except Exception as e:
            continue

    return np.array(X), np.array(y)

def train_logistic_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)

    feature_names = [
        'Net_eFG%', 
        'Net_TOV%', 
        'Net_FT%', 
        'Net_ORB%', 
        'Steal%', 
        'Block%'
    ]

    raw_coefs = model.coef_[0]
    abs_coefs = np.abs(raw_coefs)
    normalized_coefs = abs_coefs / abs_coefs.sum()

    print("\n=== Normalized Feature Weights (Sum to 1) ===")
    for name, weight in zip(feature_names, normalized_coefs):
        print(f"{name}: {weight:.4f}")
    print(f"Intercept (not normalized): {model.intercept_[0]:.4f}")
    print("============================================\n")
    
    return model

def predict_game_winner(team1, team2, season, all_data, model):
    team1_games = get_pre_matchup_games(team1, team2, season, all_data)
    team2_games = get_pre_matchup_games(team2, team1, season, all_data)

    team1_stats = compute_advanced_stats(team1_games)
    team2_stats = compute_advanced_stats(team2_games)

    features = np.array([[
        team1_stats['Net_eFG%'] - team2_stats['Net_eFG%'],
        team1_stats['Net_TOV%'] - team2_stats['Net_TOV%'],
        team1_stats['Net_FT%'] - team2_stats['Net_FT%'],
        team1_stats['Net_ORB%'] - team2_stats['Net_ORB%'],
        team1_stats['Steal%'] - team2_stats['Steal%'],
        team1_stats['Block%'] - team2_stats['Block%']
    ]])

    prob = model.predict_proba(features)[0][1]
    winner = team1 if prob > 0.5 else team2
    win_prob = round(prob, 3) if winner == team1 else round(1 - prob, 3)

    return winner, win_prob

# === GUI ===
def run_gui(model):
    def get_prediction():
        team1 = team1_combo.get().strip()
        team2 = team2_combo.get().strip()

        if not team1 or not team2:
            messagebox.showerror("Input Error", "Please select both teams.")
            return

        try:
            winner, prob = predict_game_winner(team1, team2, "2024-2025", df, model)
            result_label.config(text=f"Predicted Winner: {winner}\nWin Probability: {prob*100:.1f}%")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    def reset_teams():
        team1_combo.set('')
        team2_combo.set('')
        result_label.config(text='')

    root = tk.Tk()
    root.title("FCS Game Predictor")

    instructions = (
        "Select teams using dropdown (format matches 'Team' column).\n"
        "Example: Arizona, Davidson, St. Bonaventure, VMI"
    )
    tk.Label(root, text=instructions, font=("Helvetica", 9), justify="left", wraplength=420).grid(
        row=0, column=0, columnspan=2, padx=10, pady=(10, 5)
    )

    tk.Label(root, text="Team 1:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    team1_combo = ttk.Combobox(root, values=TEAM_LIST, width=30)
    team1_combo.grid(row=1, column=1, padx=10, pady=5)

    tk.Label(root, text="Team 2:").grid(row=2, column=0, padx=10, pady=5, sticky="e")
    team2_combo = ttk.Combobox(root, values=TEAM_LIST, width=30)
    team2_combo.grid(row=2, column=1, padx=10, pady=5)

    button_frame = tk.Frame(root)
    button_frame.grid(row=3, column=0, columnspan=2, pady=10)

    predict_button = tk.Button(button_frame, text="Get Winner", command=get_prediction, width=15)
    predict_button.pack(side="left", padx=10)

    reset_button = tk.Button(button_frame, text="Reset Teams", command=reset_teams, width=15)
    reset_button.pack(side="left", padx=10)

    result_label = tk.Label(root, text="", font=("Helvetica", 12), justify="center")
    result_label.grid(row=4, column=0, columnspan=2, pady=10)

    root.mainloop()

# === Model Testing ===
def test_model_accuracy(all_data, season="2024-2025", model=None):
    correct = 0
    total = 0
    failed = 0

    all_data = all_data.dropna(subset=['Date', 'Rslt', 'Team', 'Opp'])

    for idx, game in all_data.iterrows():
        team1 = game['Team']
        team2 = game['Opp']
        game_season = game['Season']
        actual_result = game['Rslt']

        if team1 not in TARGET_TEAMS or team2 not in TARGET_TEAMS:
            continue

        if game_season != season:
            continue

        try:
            winner, _ = predict_game_winner(team1, team2, season, all_data, model)
            actual_winner = team1 if actual_result == 'W' else team2

            if winner == actual_winner:
                correct += 1
            total += 1

        except Exception as e:
            failed += 1
            continue

    if total > 0:
        print(f"Accuracy: {correct}/{total} = {correct / total:.2%}")
    else:
        print("No games could be tested.")
    print(f"Failed predictions: {failed}")

# === Main Execution ===
if __name__ == "__main__":
    print("Building training data...")
    X, y = build_training_data(df, season="2024-2025")

    print("Training logistic regression model...")
    model = train_logistic_model(X, y)

    print("\nTesting model accuracy...")
    test_model_accuracy(df, season="2024-2025", model=model)

    run_gui(model)
