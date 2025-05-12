import os
import pandas as pd
from io import StringIO
import glob
from datetime import datetime
import re
from collections import defaultdict

# === Define the dynamic parts ===
base_dir = "/home/cow/Downloads/H2O_EDDA"
output_base_dir = "/home/cow/Chem-Code/File_Input/AAD2O_thermodynamics/testing"
os.makedirs(output_base_dir, exist_ok=True)

bin_count = 8  # Number of bins per row

# Step 1: Get all concentration directories (e.g., 20uM, 50uM)
concentrations = sorted([
    name[:4] for name in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, name)) and "uM" in name
])

# Initialize nested dict to hold data
grouped_data = defaultdict(lambda: defaultdict(dict))

# Step 2: Load all height data
for concentration in sorted(concentrations):
    conc_path = os.path.join(base_dir, concentration)
    print(f"Checking concentration folder: {conc_path}")

    if not os.path.exists(conc_path):
        print(f"Concentration folder not found: {conc_path}")
        continue

    subfolders_vt = sorted([
        name for name in os.listdir(conc_path)
        if os.path.isdir(os.path.join(conc_path, name)) and "vT" in name
    ])

    for vt in subfolders_vt:
        vt_path = os.path.join(conc_path, vt)
        print(f"  Checking vT folder: {vt_path}")

        if not os.path.exists(vt_path):
            print(f"vT folder not found: {vt_path}")
            continue

        subfolders_C = sorted([
            name for name in os.listdir(vt_path)
            if os.path.isdir(os.path.join(vt_path, name)) and re.search(r'\d+[cC]_', name)
        ])

        for subfolder_C in subfolders_C:
            full_path = os.path.join(vt_path, subfolder_C)
            print(f"    Checking C_ folder: {full_path}")

            search_pattern = os.path.join(full_path, "*peakparam.dat")
            matches = sorted(glob.glob(search_pattern))

            if matches:
                file_path = matches[0]
                print(f"      Found data file: {file_path}")
                try:
                    with open(file_path, 'r') as file:
                        lines = file.readlines()

                    header = lines[0].lstrip('#').strip().split()
                    data = lines[1:]
                    df = pd.read_csv(StringIO(''.join(data)), sep=r'\s+', names=header)

                    # Extract the concentration label like "20C" or "15c"
                    identifier_match = re.search(r'_(\d+[cC])(?:_|$)', subfolder_C)
                    C_label = identifier_match.group(1) if identifier_match else 'Unknown'

                    grouped_data[concentration][vt][C_label] = df['Height'].values

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

# Step 3: Write output files
for concentration in sorted(grouped_data.keys()):
    conc_num_match = re.search(r'\d+', concentration)
    if not conc_num_match:
        continue
    conc_num = conc_num_match.group()

    for vt in sorted(grouped_data[concentration].keys()):
        vt_index_match = re.search(r'vT_(\d+)', vt)
        vt_index = vt_index_match.group(1) if vt_index_match else 'X'

        sorted_C_labels = sorted(
            grouped_data[concentration][vt].keys(),
            key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else -1
        )

        file_name = f"temp_{conc_num}_AA-D_{sorted_C_labels[0].lower()}_{vt_index}.txt"
        file_path = os.path.join(output_base_dir, file_name)

        with open(file_path, 'w') as f:
            # Header
            f.write("0\t" + "\t".join(str(i) for i in range(bin_count)) + "\n")
            f.write("0\t" + "\t".join(["1" if i == 0 else "0" for i in range(bin_count)]) + "\n")

            for C_label in sorted_C_labels:
                heights = grouped_data[concentration][vt][C_label]

                # Pad or trim to bin_count
                padded_heights = list(heights[:bin_count]) + [0.0] * (bin_count - len(heights))

                total = sum(padded_heights)
                if total == 0:
                    mol_values = ["0"] * bin_count
                else:
                    mol_values = [f"{h / total:.8f}" for h in padded_heights]

                # Get ligand concentration value from C_label (e.g., 5C → 0.000005)
                match = re.search(r'\d+', C_label)
                ligand_conc = float(match.group()) / 1_000_000 if match else 0.0

                f.write(f"{ligand_conc:.6f}\t" + "\t".join(mol_values) + "\n")

        print(f"Written: {file_path}")
