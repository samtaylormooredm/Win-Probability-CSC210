import pandas as pd
import os
import glob

def convert_excel_to_csv(excel_path, team, year):
    target_dir = f"data_csv/{year}"
    os.makedirs(target_dir, exist_ok=True)

    team_clean = team.lower().replace(" ", "_")
    csv_path = os.path.join(target_dir, f"{team_clean}.csv")

    try:
        df_list = pd.read_html(excel_path)
        df = df_list[0]
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
    except Exception as e:
        print(f"Failed: {excel_path} — {e}")


def batch_convert_excels(root_dir="."):
    # Find all Excel files recursively
    excel_files = glob.glob(f"{root_dir}/**/*.xls*", recursive=True)

    print(f"Found {len(excel_files)} Excel file(s).")

    for file_path in excel_files:
        print(f"Found file: {file_path}")  # Add this line

        file_name = os.path.basename(file_path).replace(".xlsx", "").replace(".xls", "")
        
        if "_" in file_name:
            try:
                team, year = file_name.split("_")
                year = int(year)
                convert_excel_to_csv(file_path, team, year)
            except ValueError:
                print(f"Skipping file with unexpected name: {file_name}")
        else:
            print(f"Skipping unrecognized filename: {file_name}")

# Run the batch converter
batch_convert_excels()
