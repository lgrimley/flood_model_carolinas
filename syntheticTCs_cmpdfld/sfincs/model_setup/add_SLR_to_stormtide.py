import os
import pandas as pd

tc_ids_hist = [4347, 4796]
tc_ids_fut = [4550, 671, 5547, 5719, 4133, 4970, 1831, 1545, 3214, 4823, 4762, 5701, 2041, 5902, 2818]


bc_dir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\sfincs_initcond_mod'

slr = 1.12 # mean 1.12 95 CI 0.68 - 1.57m
waterlevel = pd.read_csv(os.path.join(bc_dir,'tides_sfincs.bzs'), header=None,sep="\s+", index_col=0)
waterlevel = waterlevel.astype(float)
waterlevel_slr = waterlevel + slr

os.chdir(bc_dir)

# Open the file in write mode
filename = f'tides_sfincs_SLR{int(slr*100)}.bzs'
with open(filename, 'w') as f:
    # Iterate through each row of the DataFrame
    for index, row in waterlevel_slr.iterrows():
        # Format the index with 9 spaces and columns with 8 spaces
        formatted_row = f"{index:9.1f}" + "".join([f"{val:8.2f}" for val in row]) + "\n"
        # Write the formatted row to the file
        f.write(formatted_row)