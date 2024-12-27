import os

directory = r"Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\runs"  # Replace with the actual path
missing = []
for root, dirs, files in os.walk(directory):
    if files:
        continue
    elif len(files) < 6:
        r = root.split(os.sep)[-2]
        if r == 'runs':
            continue
        else:
            tc_index = int(r.split('_')[-1])
            print(f"TC {tc_index} is missing files")
            missing.append(tc_index)

