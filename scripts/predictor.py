import pandas as pd
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox

# === Load Data ===
df = pd.read_csv("cleaned_win_prob_data.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

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

def predict_game_winner(team1, team2, season, all_data):
    team1_games = get_pre_matchup_games(team1, team2, season, all_data)
    team2_games = get_pre_matchup_games(team2, team1, season, all_data)

    if team1_games.empty or team2_games.empty:
        raise ValueError("Not enough game data for one or both teams.")

    team1_stats = compute_advanced_stats(team1_games)
    team2_stats = compute_advanced_stats(team2_games)

    weights = {
        'Net_TOV%': 0.43,
        'Net_eFG%': 0.41,
        'Net_FT%': 0.11,
        'Net_ORB%': 0.04,
        'Block%': 0.005,
        'Steal%': 0.005
    }

    prob_team1 = sum(weights[stat] * team1_stats[stat] for stat in weights)
    prob_team2 = sum(weights[stat] * team2_stats[stat] for stat in weights)

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
# === GUI ===
def run_gui():
    def get_prediction():
        team1 = team1_combo.get().strip()
        team2 = team2_combo.get().strip()

        if not team1 or not team2:
            messagebox.showerror("Input Error", "Please select both teams.")
            return

        try:
            winner, prob = predict_game_winner(team1, team2, "2024-2025", df)
            result_label.config(text=f"Predicted Winner: {winner}\nWin Probability: {prob*100:.1f}%")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    def reset_teams():
        team1_combo.set("")
        team2_combo.set("")
        result_label.config(text="")

    root = tk.Tk()
    root.title("Basketball Game Predictor")

    tk.Label(root, text="Select Team 1:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    tk.Label(root, text="Select Team 2:").grid(row=1, column=0, padx=10, pady=5, sticky="e")

    team1_combo = ttk.Combobox(root, values=TEAM_LIST, width=30, state="readonly")  # <-- added state="readonly"
    team1_combo.grid(row=0, column=1, padx=10, pady=5)
    team2_combo = ttk.Combobox(root, values=TEAM_LIST, width=30, state="readonly")  # <-- added state="readonly"
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
        game_season = game['Season']
        actual_result = game['Rslt']

        if team1 not in TARGET_TEAMS or team2 not in TARGET_TEAMS:
            continue

        if game_season != season:
            continue

        try:
            winner, _ = predict_game_winner(team1, team2, season, all_data)
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
    print("Testing model accuracy...")
    test_model_accuracy(df, season="2024-2025")
    run_gui()
