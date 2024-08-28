import os

# Loop through model results and delete subgrid file
results_dir = r'Z:\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\02_sfincs_models_future'

for root, dirs, files in os.walk(results_dir):
    for name in files:
        if name.endswith("sfincs.sbg"):
            os.remove(os.path.join(root, name))