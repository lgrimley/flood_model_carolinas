import pandas as pd
import os

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\03_sfincs_models_SLRedit')

slr_ids = pd.read_csv(r'.\hindcast_slr_event_ids_DONOTDELETE.csv', header=None)
slr_ids.columns = ['event_id', 'slr_m']
slr_ids['storm'] = [x.split('_')[0] for x in slr_ids['event_id']]


def add_SLR(original_file, adjusted_file, slr_m):
    # Ass Sea Level to coastal water level BC
    wl = pd.read_csv(original_file, header=None, sep="\s+", index_col=0)
    wl = wl.astype(float)
    wl_slr = wl + slr_m

    with open(adjusted_file, 'w') as f:
        # Iterate through each row of the DataFrame
        for index, row in wl_slr.iterrows():
            # Format the index with 9 spaces and columns with 8 spaces
            formatted_row = f"{index:9.1f}" + "".join([f"{val:8.2f}" for val in row]) + "\n"
            # Write the formatted row to the file
            f.write(formatted_row)
    print(f'Created {adjusted_file}')

bc_inputs_dir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\tides'
for i in range(len(slr_ids)):
    storm = slr_ids.loc[i,'storm']
    slr_m = slr_ids.loc[i, 'slr_m'].astype(float)
    event_id = slr_ids.loc[i, 'event_id']

    add_SLR(original_file = os.path.join(bc_inputs_dir, f'{storm}_pres_tides_waterlevel.bzs'),
            adjusted_file = os.path.join(bc_inputs_dir, f'{event_id}_tides_waterlevel.bzs'),
            slr_m = slr_m)
