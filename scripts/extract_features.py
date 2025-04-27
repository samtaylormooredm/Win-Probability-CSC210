import pandas as pd
import os

# Use these paths if you're running the script from *inside the data folder*
file_paths = [
    "data_csv/2024/arizona.csv",
    "data_csv/2025/arizona.csv",
    "data_csv/2024/bethune-cookman.csv",
    "data_csv/2025/bethune-cookman.csv",
    "data_csv/2024/bowling-green-state.csv",
    "data_csv/2025/bowling-green-state.csv",
    "data_csv/2024/charleston-southern.csv",
    "data_csv/2025/charleston-southern.csv",
    "data_csv/2024/charlotte.csv",
    "data_csv/2025/charlotte.csv",
    "data_csv/2024/davidson.csv",
    "data_csv/2025/davidson.csv",
    "data_csv/2024/dayton.csv",
    "data_csv/2025/dayton.csv",
    "data_csv/2024/detroit-mercy.csv",
    "data_csv/2025/detroit-mercy.csv",
    "data_csv/2024/duquesne.csv",
    "data_csv/2025/duquesne.csv",
    "data_csv/2024/east-tennessee-state.csv",
    "data_csv/2025/east-tennessee-state.csv",
    "data_csv/2024/eastern-michigan.csv",
    "data_csv/2025/eastern-michigan.csv",
    "data_csv/2024/fordham.csv",
    "data_csv/2025/fordham.csv",
    "data_csv/2024/george-mason.csv",
    "data_csv/2025/george-mason.csv",
    "data_csv/2024/george-washington.csv",
    "data_csv/2025/george-washington.csv",
    "data_csv/2024/gonzaga.csv",
    "data_csv/2025/gonzaga.csv",
    "data_csv/2024/la-salle.csv",
    "data_csv/2025/la-salle.csv",
    "data_csv/2024/loyola-il.csv",
    "data_csv/2025/loyola-il.csv",
    "data_csv/2024/massachusetts.csv",
    "data_csv/2025/massachusetts.csv",
    "data_csv/2024/providence.csv",
    "data_csv/2025/providence.csv",
    "data_csv/2024/rhode-island.csv",
    "data_csv/2025/rhode-island.csv",
    "data_csv/2024/richmond.csv",
    "data_csv/2025/richmond.csv",
    "data_csv/2024/saint-josephs.csv",
    "data_csv/2025/saint-josephs.csv",
    "data_csv/2024/saint-louis.csv",
    "data_csv/2025/saint-louis.csv",
    "data_csv/2024/st-bonaventure.csv",
    "data_csv/2025/st-bonaventure.csv",
    "data_csv/2024/temple.csv",
    "data_csv/2025/temple.csv",
    "data_csv/2024/virginia-commonwealth.csv",
    "data_csv/2025/virginia-commonwealth.csv",
    "data_csv/2024/virginia-military-institute.csv",
    "data_csv/2025/virginia-military-institute.csv"
]

# List for storing cleaned DataFrames
cleaned_dataframes = []

for file_path in file_paths:
    df = pd.read_csv(file_path, header=1)
    df = df.iloc[:-1]  # Drop last row (season totals)

    df.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]

    # Calculate the stats we need
    df['TOV%'] = df['TOV'] / (df['FGA'] + 0.44 * df['FTA'] + df['TOV'])
    df['ORB%'] = df['ORB'] / (df['ORB'] + df['DRB.1'])
    df['FT%'] = df['FT'] / df['FGA']

    df['Opp_TOV%'] = df['TOV.1'] / (df['FGA.1'] + 0.44 * df['FTA.1'] + df['TOV.1'])
    df['Opp_ORB%'] = df['ORB.1'] / (df['ORB.1'] + df['DRB'])
    df['Opp_FT%'] = df['FT.1'] / df['FGA.1']

    # Select and rename
    selected_df = df[['Date', 'Rslt', 'Tm', 'Opp',
                  'eFG%', 'TOV', 'TOV%', 'ORB', 'ORB%', 'FT%',
                  'eFG%.1', 'TOV.1', 'Opp_TOV%', 'ORB.1', 'Opp_ORB%', 'Opp_FT%',
                  'STL', 'STL.1', 'BLK', 'BLK.1',
                  'FGA', 'FGA.1', 'FTA', 'FTA.1']] 

    selected_df = selected_df.rename(columns={
        'Tm': 'Points',
        'Opp': 'Opp_Points',
        'eFG%.1': 'Opp_eFG%',
        'TOV.1': 'Opp_TOV',
        'ORB.1': 'Opp_ORB',
        'STL.1': 'Opp_STL',
        'BLK.1': 'Opp_BLK',
        'FGA.1': 'Opp_FGA',   # <== Renaming FGA.1 → Opp_FGA
        'FTA.1': 'Opp_FTA',   # <== Renaming FTA.1 → Opp_FTA
    })

    # Add file and season info
    selected_df['Source_File'] = os.path.basename(file_path)
    year = file_path.split('/')[1]
    selected_df['Season'] = f"{int(year)-1}-{year}"

    cleaned_dataframes.append(selected_df)

# Combine all into one
combined_df = pd.concat(cleaned_dataframes, ignore_index=True)

team_columns = ['eFG%', 'TOV', 'TOV%', 'ORB', 'ORB%', 'FT%', 'STL', 'BLK', 'FGA', 'FTA']
opp_columns = ['Opp_Points', 'Opp_eFG%', 'Opp_TOV', 'Opp_TOV%', 'Opp_ORB', 'Opp_ORB%', 'Opp_FT%', 'Opp_STL', 'Opp_BLK', 'Opp_FGA', 'Opp_FTA']

# Full final column order
final_columns = ['Date', 'Season', 'Rslt', 'Points'] + team_columns + opp_columns + ['Source_File']
combined_df = combined_df[final_columns]

# Save
os.makedirs("outputs", exist_ok=True)
combined_df.to_csv("outputs/cleaned_win_prob_data.csv", index=False)
print("Cleaned data saved as 'cleaned_win_prob_data.csv'")
