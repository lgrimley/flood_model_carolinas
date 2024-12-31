import subprocess
import pandas as pd
import os

def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL')
# tc_index_list = pd.read_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\NCEP_Reanalysis\tracks\adcirc_modeled_TCs_all.csv')
# storm_list = tc_index_list['tc_id'].tolist()
# tc_list_by_vmax = pd.read_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\NCEP_Reanalysis\TCs_VmaxRanked_at_each_gage.csv')
# storm_list = tc_list_by_vmax['193'].dropna().astype(int).tolist()
# directory = r"Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\runs"
# missing = []
# for root, dirs, files in os.walk(directory):
#     if files:
#         continue
#     elif len(files) < 6:
#         r = root.split(os.sep)[-2]
#         if r == 'runs':
#             continue
#         else:
#             tc_index = int(r.split('_')[-1])
#             print(f"TC {tc_index} is missing files")
#             missing.append(tc_index)
# storm_list = missing

storm_list = [3150, 2209]#99, 1484, 2618, 3148, 3522, 3656, 3681, 3736, 3879, 3981,
              #4045,4159, 4442, 4732, 4755, 4919, 4981,
    #5008, 5017, 5018]

timeout_duration=300
failed = []
with open(
        r"Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\runs\output.txt",
        "a") as f:

    for tc_index in storm_list:
        track_dir = os.path.join(os.getcwd(), 'runs', f'TC_{str(tc_index).zfill(4)}', 'sfincs_bc_inputs')
        if os.path.exists(track_dir) is False:
            os.makedirs(track_dir)
        files = os.listdir(track_dir)
        if len(files) < 6:
            print(f'Working on {tc_index}')
            try:
                subprocess.run(['python', r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\syntheticTCs_cmpdfld\sfincs\model_setup\write_sfincs_track_inputs.py',
                                         str(tc_index), track_dir],stdout=f, text=True, timeout=timeout_duration)
            except subprocess.TimeoutExpired:
                print(f"Subprocess timed out after {timeout_duration} seconds for  {tc_index}")
                failed.append(tc_index)
            except Exception as e:
                print(f"An error occurred: {e}")
                failed.append(tc_index)
        else:
            print(f'Already processed {tc_index}')



