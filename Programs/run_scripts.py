import subprocess

# Base directory for your scripts
base_path = "/home/cow/Chem-Code/Programs/"

# Dictionary of scripts with full paths
scripts = {
    "1": ("Ligand_Equilibrium_061021", base_path + "Ligand_Equilibrium_061021.py"),
    "2": ("NLVH_ave", base_path + "NLVH_ave.py"),
    "3": ("NLVH_format_formatting_plots", base_path + "NLVH_format_formatting_plots.py"),
    "4": ("NLVH_stacked_plots_errorbar", base_path + "NLVH_stacked_plots_errorbar.py"),
    "A": ("Run All Scripts", None)  # Special option
}

# Show menu
print("Which scripts would you like to run?")
print("Separate choices with commas (e.g., 1,3 or A for all)\n")
for key, (desc, _) in scripts.items():
    print(f"{key}. {desc}")

# Get input
choices = input("\nEnter your choices: ").upper().split(",")

# Determine which scripts to run
to_run = []

if "A" in choices:
    to_run = [script for key, (_, script) in scripts.items() if script]
else:
    for choice in choices:
        choice = choice.strip()
        if choice in scripts and scripts[choice][1]:
            to_run.append(scripts[choice][1])
        else:
            if choice not in scripts:
                print(f"Ignoring invalid choice: {choice}")

# Run selected scripts
for script_path in to_run:
    # Find the program name from the path
    program_name = next((name for name, path in scripts.values() if path == script_path), script_path)

    print(f"\n============== {program_name} ==============\n")
    subprocess.run(["python3", script_path])
