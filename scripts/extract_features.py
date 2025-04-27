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


# Column mapping: maps your final output column names to the CSV header structure
col_mapping = {
    "Date": "Date",
    "Opp": "Opp",
    "Type": "Type",
    "Rslt": "Rslt",
    "Tm": "Tm",
    "Opp_score": "Opp.1",
    "eFG%": "eFG%",
    "TOV%": "TOV",
    "DRB%": "DRB",
    "FT%": "FT%",  # optional
    "Opp_eFG%": "eFG%.1",
    "Opp_TOV%": "TOV.1",
    "Opp_DRB%": "DRB.1",
    "Opp_FT%": "FT%.1"
}




# List for storing cleaned DataFrames
cleaned_dataframes = []


# Loop through each CSV and process
for file_path in file_paths:
    df = pd.read_csv(file_path, header=1)
    # Drop the last row (season totals)
    df = df.iloc[:-1]

    print(df.columns.tolist())

    df.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]

    # Extract relevant columns and rename
    selected_df = df[list(col_mapping.values())].rename(columns={v: k for k, v in col_mapping.items()})
    selected_df["Source_File"] = os.path.basename(file_path)

     # Add season column
    year = file_path.split("/")[1]  # e.g., '2024'
    season = f"{int(year)-1}-{year}"  # e.g., '2023-2024'
    selected_df["Season"] = season

    cleaned_dataframes.append(selected_df)

# Combine everything into one DataFrame
combined_df = pd.concat(cleaned_dataframes, ignore_index=True)

# Move "Season" to second column
cols = combined_df.columns.tolist()
cols.insert(1, cols.pop(cols.index("Season")))
combined_df = combined_df[cols]

# Save or preview
combined_df.to_csv("outputs/cleaned_win_prob_data.csv", index=False)
print("Cleaned data saved as 'cleaned_win_prob_data.csv'")
