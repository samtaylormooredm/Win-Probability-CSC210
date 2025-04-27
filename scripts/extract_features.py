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
    selected_df = df[['Date', 'Opp', 'eFG%', 'TOV%', 'ORB%', 'FT%',
                      'eFG%.1', 'Opp_TOV%', 'Opp_ORB%', 'Opp_FT%']]
    selected_df = selected_df.rename(columns={
        'eFG%': 'eFG%',
        'eFG%.1': 'Opp_eFG%'
    })

    # Add file and season info
    selected_df['Source_File'] = os.path.basename(file_path)
    year = file_path.split('/')[1]
    selected_df['Season'] = f"{int(year)-1}-{year}"

    cleaned_dataframes.append(selected_df)

# Combine all into one
combined_df = pd.concat(cleaned_dataframes, ignore_index=True)

# Reorder columns
cols = combined_df.columns.tolist()
cols.insert(1, cols.pop(cols.index('Season')))
combined_df = combined_df[cols]

# Save
os.makedirs("outputs", exist_ok=True)
combined_df.to_csv("outputs/cleaned_win_prob_data.csv", index=False)
print("Cleaned data saved as 'cleaned_win_prob_data.csv'")
