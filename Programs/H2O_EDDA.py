import os
import pandas as pd
from io import StringIO
import glob
from datetime import datetime

# === Define the dynamic parts ===
base_dir = "/home/cow/Downloads/H2O_EDDA"

# Define the concentrations and subfolders (vT_1, vT_2, vT_3)
concentrations = ["20uM", "50uM", "75uM", "100uM", "125uM", "150uM"]
subfolders_vt = ["_vT_1", "_vT_2", "_vT_3"]

# Generate subfolder2 options dynamically from 5c to 50c with step size of 5
subfolder2_options = [f"{i}c" for i in range(5, 51, 5)]  # Generates 5c, 10c, ..., 50c

# === Loop over all combinations of concentration, subfolder, and subfolder2 ===
all_height_data = []  # List to hold all Height_df DataFrames

for concentration in concentrations:
    for subfolder in subfolders_vt:
        for subfolder2 in subfolder2_options:
            # Compose the search pattern to find the peakparam.dat file
            search_path = os.path.join(base_dir, concentration, concentration + subfolder, f"*{subfolder2}*unidecfiles/*peakparam.dat")
            print(f"Searching for files: {search_path}")  # Debugging output
            
            # Find the matching files
            matches = glob.glob(search_path)
            print(f"Found matches: {matches}")  # Debugging output
            
            if matches:
                file_path = matches[0]  # Use the first match
                print(f"Reading: {file_path}")
                
                # Read the file
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                # Parse header and data
                header = lines[0].lstrip('#').strip().split()
                data = lines[1:]

                # Read data into a DataFrame
                df = pd.read_csv(StringIO(''.join(data)), delim_whitespace=True, names=header)

                # Create Height_df with a dynamic column name
                height_column_name = f"{concentration}_{subfolder.strip('_')}_{subfolder2}"
                Height_df = pd.DataFrame({height_column_name: df['Height']})

                # Append to the list of all height data
                all_height_data.append(Height_df)
            else:
                print(f"No files found for: {search_path}")  # Debugging output

# Check if we have any data to concatenate
if all_height_data:
    # Combine all Height DataFrames into a single DataFrame
    final_height_df = pd.concat(all_height_data, axis=1)
    print(f"Initial DataFrame: \n{final_height_df.head()}")  # Debugging output

    # Now calculate sums for each column and create normalized columns
    for column in final_height_df.columns:
        column_sum = final_height_df[column].sum()
        mol_column_name = f"{column}_MOL"  # New column name
        final_height_df[mol_column_name] = final_height_df[column] / column_sum

    print(f"Final DataFrame with MOL columns: \n{final_height_df.head()}")

    # === Define output file path with the current date ===
    current_date = datetime.today().strftime('%Y-%m-%d')  # Get the current date in YYYY-MM-DD format
    output_file_path = f"/home/cow/Chem-Code/File_Input/AAD2O_thermodynamics/H2O_EDDA_MOL_{current_date}.csv"

    # Write the final DataFrame to a CSV file
    final_height_df.to_csv(output_file_path, index=False)
    print(f"Data written to {output_file_path}")
else:
    print("No data was found to concatenate.")
