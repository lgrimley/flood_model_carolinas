import subprocess
import pandas as pd
import os
import shutil


def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


module_path = r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\syntheticTCs_cmpdfld'
root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL'
gcm = 'canesm'
ssp = 'ssp585'
dd = os.path.join(root, f'{gcm}_{ssp}_runs')
if os.path.exists(dd) is False:
    os.makedirs(dd)
os.chdir(dd)

# # List of TCs to create model inputs for
# tc_index_list = pd.read_csv(fr'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\CMIP6_585\tracks\{gcm}_{ssp}_stormTide_vstore100.csv', index_col=0)
# tc_index_list.set_index('tc_id', inplace=True)
# tc_index_list['vstore100'] = tc_index_list['vstore100'].astype(float)
# ordered_list = tc_index_list.sort_values(by='vstore100', ascending=False)
# storm_list = ordered_list.index.tolist()
#
# missing = []
# for root, dirs, files in os.walk(dd):
#     if files:
#         continue
#     elif len(files) < 6:
#         r = root.split(os.sep)[-2]
#         if r == 'runs_canesm_ssp585':
#             continue
#         else:
#             tc_index = int(r.split('_')[-1])
#             print(f"TC {tc_index} is missing files")
#             missing.append(tc_index)
#             #shutil.rmtree(os.path.join(dd, r))
#
# c = os.listdir(dd)[1:]
# tci = [int(s.split('_')[-1]) for s in c]
# tcdf = pd.DataFrame()
# tcdf['tc_id'] = tci
# tcdf.to_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\runs_remaining.csv')

storm_list = [1252, 1583, 2723, 4803, 5510, 6034, 6040]
timeout_duration=1200
failed = []
with open(fr"output.txt","a") as f:
    for tc_index in storm_list:
        if os.path.exists(os.path.join(dd,'completed_runs', f'TC_{str(tc_index).zfill(4)}')):
            print(f'{tc_index} already run in SFINCS, skipping.')
            continue
        else:
            track_dir = os.path.join(dd, 'runs', f'TC_{str(tc_index).zfill(4)}', 'sfincs_bc_inputs')
            if os.path.exists(track_dir) is False:
                os.makedirs(track_dir)
            files = os.listdir(track_dir)
            if len(files) < 6:
                print(f'Working on {tc_index}')
                try:
                    subprocess.run(['python', r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\syntheticTCs_cmpdfld\sfincs\model_setup\write_sfincs_track_inputs.py',
                                             str(tc_index), track_dir],
                                   env={**os.environ, 'PYTHONPATH': module_path},
                                   stdout=f, text=True, timeout=timeout_duration)
                except subprocess.TimeoutExpired:
                    print(f"Subprocess timed out after {timeout_duration} seconds for {tc_index}")
                    failed.append(tc_index)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    failed.append(tc_index)
            else:
                print(f'Already processed {tc_index}')


