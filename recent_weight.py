import pandas as pd
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox

# === Load Data ===
df = pd.read_csv("cleaned_win_prob_data.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# === A-10 Teams Only ===
A10_TEAMS = {
    'Davidson', 'Dayton', 'Duquesne', 'Fordham', 'George Mason',
    'George Washington', 'La Salle', 'Loyola Chicago', 'Massachusetts',
    'Rhode Island', 'Richmond', "Saint Joseph's", 'Saint Louis',
    'St. Bonaventure', 'VCU'
}

# === Define Weights Globally ===
WEIGHTS = {
    'Net_TOV%': 0.43,
    'Net_eFG%': 0.41,
    'Net_FT%': 0.11,
    'Net_ORB%': 0.04,
    'Block%': 0.005,
    'Steal%': 0.005
}

def extract_team_list(data):
    teams = sorted(set(data['Team'].dropna().unique()))
    return [team for team in teams if team in A10_TEAMS]

TEAM_LIST = extract_team_list(df)

# === Core Functions ===
def get_pre_matchup_games(team_name, season, all_data, matchup_date):
    prior_season = str(int(season.split('-')[0]) - 1) + '-' + str(int(season.split('-')[1]) - 1)
    
    team_games = all_data[
        (all_data['Team'] == team_name) &
        (all_data['Season'].isin([prior_season, season])) &
        (all_data['Date'] < matchup_date)
    ].dropna(subset=["Date"]).sort_values(by="Date")

    # Tag whether it's prior season or current season
    team_games['IsPriorSeason'] = team_games['Season'] == prior_season
    return team_games


def compute_advanced_stats(past_games, matchup_date):
    if past_games.empty:
        raise ValueError("No past games available to compute stats.")

    stat_columns = [
        'eFG%', 'TOV', 'TOV%', 'ORB', 'ORB%', 'FT%',
        'Opp_eFG%', 'Opp_TOV', 'Opp_TOV%', 'Opp_ORB%', 'Opp_FT%',
        'FGA', 'FTA', 'Opp_FGA', 'Opp_FTA',
        'STL', 'BLK'
    ]

    # Compute recency weights
    days_diff = (matchup_date - past_games['Date']).dt.days

    recency_weight = np.exp(-days_diff / 30)  # normal decay
    # If from prior season, penalize the weight further
    recency_weight = np.where(past_games['IsPriorSeason'], recency_weight * 0.1, recency_weight)
    recency_weight /= recency_weight.sum()  # Normalize to sum to 1

    weighted_means = (past_games[stat_columns].multiply(recency_weight, axis=0)).sum()

    opponent_possessions = (
        weighted_means['Opp_FGA'] +
        0.44 * weighted_means['Opp_FTA'] +
        weighted_means['Opp_TOV']
    )

    steal_rate = (weighted_means['STL'] / opponent_possessions) * 100 if opponent_possessions else 0
    block_rate = (weighted_means['BLK'] / opponent_possessions) * 100 if opponent_possessions else 0

    stats = {
        'Net_TOV%': weighted_means['Opp_TOV%'] - weighted_means['TOV%'],
        'Net_eFG%': weighted_means['eFG%'] - weighted_means['Opp_eFG%'],
        'Net_FT%': weighted_means['FT%'] - weighted_means['Opp_FT%'],
        'Net_ORB%': weighted_means['ORB%'] - weighted_means['Opp_ORB%'],
        'Steal%': steal_rate,
        'Block%': block_rate
    }
    return stats


def predict_game_winner(team1, team2, season, all_data, matchup_date):
    team1_games = get_pre_matchup_games(team1, season, all_data, matchup_date)
    team2_games = get_pre_matchup_games(team2, season, all_data, matchup_date)

    if team1_games.empty or team2_games.empty:
        raise ValueError("Not enough game data for one or both teams before matchup date.")

    team1_stats = compute_advanced_stats(team1_games, matchup_date)
    team2_stats = compute_advanced_stats(team2_games, matchup_date)

    prob_team1 = sum(WEIGHTS[stat] * team1_stats[stat] for stat in WEIGHTS)
    prob_team2 = sum(WEIGHTS[stat] * team2_stats[stat] for stat in WEIGHTS)

    if pd.isna(prob_team1) or pd.isna(prob_team2):
        raise ValueError("One or more computed probabilities resulted in NaN.")

    prob_team1 = max(prob_team1, 0)
    prob_team2 = max(prob_team2, 0)
    total = prob_team1 + prob_team2
    team1_win_prob = prob_team1 / total if total > 0 else 0.5

    winner = team1 if team1_win_prob > 0.5 else team2
    prob = round(team1_win_prob, 3) if winner == team1 else round(1 - team1_win_prob, 3)

    return winner, prob

# === GUI ===
def run_gui():
    def get_prediction():
        team1 = team1_combo.get().strip()
        team2 = team2_combo.get().strip()

        if not team1 or not team2:
            messagebox.showerror("Input Error", "Please select both teams.")
            return

        try:
            winner, prob = predict_game_winner(team1, team2, "2024-2025", df, pd.Timestamp('today'))
            result_label.config(text=f"Predicted Winner: {winner}\nWin Probability: {prob*100:.1f}%")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    def reset_teams():
        team1_combo.set("")
        team2_combo.set("")
        result_label.config(text="")

    root = tk.Tk()
    root.title("A-10 Basketball Game Predictor")

    tk.Label(root, text="Select Team 1:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    tk.Label(root, text="Select Team 2:").grid(row=1, column=0, padx=10, pady=5, sticky="e")

    team1_combo = ttk.Combobox(root, values=TEAM_LIST, width=30, state="readonly")
    team1_combo.grid(row=0, column=1, padx=10, pady=5)
    team2_combo = ttk.Combobox(root, values=TEAM_LIST, width=30, state="readonly")
    team2_combo.grid(row=1, column=1, padx=10, pady=5)

    button_frame = tk.Frame(root)
    button_frame.grid(row=2, column=0, columnspan=2, pady=10)

    predict_button = tk.Button(button_frame, text="Predict Winner", command=get_prediction, width=15)
    predict_button.pack(side="left", padx=10)

    reset_button = tk.Button(button_frame, text="Reset", command=reset_teams, width=15)
    reset_button.pack(side="left", padx=10)

    result_label = tk.Label(root, text="", font=("Helvetica", 12), justify="center")
    result_label.grid(row=3, column=0, columnspan=2, pady=10)

    root.mainloop()

# === Accuracy Testing ===
def test_model_accuracy(all_data, season="2024-2025"):
    correct = 0
    total = 0
    failed = 0

    all_data = all_data.dropna(subset=['Date', 'Rslt', 'Team', 'Opp'])

    for idx, game in all_data.iterrows():
        team1 = game['Team']
        team2 = game['Opp']
        matchup_date = game['Date']
        game_season = game['Season']
        actual_result = game['Rslt']

        if team1 not in A10_TEAMS or team2 not in A10_TEAMS:
            continue

        if game_season != season:
            continue

        try:
            winner, _ = predict_game_winner(team1, team2, season, all_data, matchup_date)
            actual_winner = team1 if actual_result == 'W' else team2

            if winner == actual_winner:
                correct += 1
            total += 1
        except Exception as e:
            failed += 1
            continue

    if total > 0:
        print(f"\nAccuracy: {correct}/{total} = {correct / total:.2%}")
    else:
        print("\nNo games could be tested.")
    print(f"Failed predictions: {failed}")

# === Main Execution ===
if __name__ == "__main__":
    print("Testing model accuracy (A-10 teams only)...")
    test_model_accuracy(df, season="2024-2025")
    run_gui()
