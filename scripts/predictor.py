import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, ttk

# === Load Data ===
df = pd.read_csv("cleaned_win_prob_data.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

def extract_team_list(data):
    filenames = data['Source_File'].unique()
    teams = sorted(set(name.replace(".csv", "").split("/")[-1] for name in filenames))
    return teams

TEAM_LIST = extract_team_list(df)

# === Core Functions ===
def get_pre_matchup_games(team_name, opponent_name, season, all_data):
    team_file_fragment = team_name.lower()
    team_games = all_data[all_data['Source_File'].str.contains(team_file_fragment)]
    team_games = team_games.dropna(subset=["Date"])

    past_games = team_games[team_games['Season'] < season]
    post_sept_games = team_games[
        (team_games['Season'] == season) &
        (team_games['Date'] > pd.Timestamp("2025-09-01"))
    ].sort_values(by="Date")

    matchup_idx = post_sept_games[post_sept_games['Opp'].str.lower() == opponent_name.lower().replace("-", " ")].index

    if len(matchup_idx) == 0:
        valid_2025_games = post_sept_games[post_sept_games['Date'] < datetime.now()]
    else:
        matchup_position = post_sept_games.index.get_loc(matchup_idx[0])
        valid_2025_games = post_sept_games.iloc[:matchup_position]

    return pd.concat([past_games, valid_2025_games])

def compute_team_stats(team_name, opponent_name, season, all_data):
    past_games = get_pre_matchup_games(team_name, opponent_name, season, all_data)
    if past_games.empty:
        raise ValueError(f"No data found for {team_name} before the {opponent_name} matchup.")

    stat_cols = ['eFG%', 'TOV%', 'DRB%', 'FT%']
    opp_stat_cols = ['Opp_eFG%', 'Opp_TOV%', 'Opp_DRB%', 'Opp_FT%']
    team_stats = past_games[stat_cols].mean()
    opp_stats = past_games[opp_stat_cols].mean()

    if team_stats.isnull().any() or opp_stats.isnull().any():
        raise ValueError(f"Incomplete stat data for {team_name} before the {opponent_name} matchup.")

    return team_stats, opp_stats

def predict_game_winner(team1, team2, season, data):
    team1_stats, team1_opp_stats = compute_team_stats(team1, team2, season, data)
    team2_stats, team2_opp_stats = compute_team_stats(team2, team1, season, data)

    prob_team1 = (
        0.4 * (team1_stats['eFG%'] - team2_opp_stats['Opp_eFG%']) +
        0.3 * (team1_stats['ORB%'] - team2_opp_stats['Opp_ORB%']) -
        0.2 * (team1_stats['TOV%'] - team2_opp_stats['Opp_TOV%']) +
        0.1 * (team1_stats['FT%'] - team2_opp_stats['Opp_FT%'])
    )
    prob_team2 = (
        0.4 * (team2_stats['eFG%'] - team1_opp_stats['Opp_eFG%']) +
        0.3 * (team2_stats['ORB%'] - team1_opp_stats['Opp_ORB%']) -
        0.2 * (team2_stats['TOV%'] - team1_opp_stats['Opp_TOV%']) +
        0.1 * (team2_stats['FT%'] - team1_opp_stats['Opp_FT%'])
    )

    if pd.isna(prob_team1) or pd.isna(prob_team2):
        raise ValueError("One or more computed probabilities resulted in NaN. Please check data integrity.")

    prob_team1 = max(prob_team1, 0)
    prob_team2 = max(prob_team2, 0)
    total = prob_team1 + prob_team2
    team1_win_prob = prob_team1 / total if total > 0 else 0.5

    winner = team1 if team1_win_prob > 0.5 else team2
    prob = round(team1_win_prob, 3) if winner == team1 else round(1 - team1_win_prob, 3)

    return winner, prob

# === GUI Setup ===
def run_gui():
    def get_prediction():
        team1 = team1_combo.get().strip()
        team2 = team2_combo.get().strip()

        if not team1 or not team2:
            messagebox.showerror("Input Error", "Please select both teams.")
            return

        try:
            winner, prob = predict_game_winner(team1, team2, "2025-2026", df)
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
        "Select team names using dropdown with autocompletion.\n"
        "Format: lowercase, hyphens instead of spaces, no apostrophes.\n"
        "Examples: davidson, st-bonaventure, virginia-military-institute"
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

    # Create a new frame to hold the buttons
    button_frame = tk.Frame(root)
    button_frame.grid(row=3, column=0, columnspan=2, pady=10)

    predict_button = tk.Button(button_frame, text="Get Winner", command=get_prediction, width=15)
    predict_button.pack(side="left", padx=10)

    reset_button = tk.Button(button_frame, text="Reset Teams", command=reset_teams, width=15)
    reset_button.pack(side="left", padx=10)


    result_label = tk.Label(root, text="", font=("Helvetica", 12), justify="center")
    result_label.grid(row=4, column=0, columnspan=2, pady=10)

    root.mainloop()

if __name__ == "__main__":
    run_gui()
