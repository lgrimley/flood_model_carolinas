import sys
import subprocess
import os
import time
import traceback
import psutil
import pandas as pd
import shutil


task_id = int(sys.argv[1])

# Load in the TCs and rank them by landfall Vmax
tc_ids = pd.read_csv(r'/projects/sfincs/syntheticTCs_cmpdfld/SFINCS_model/ncep_TCs_vstore100.csv',
                     index_col=0)
tcs_ranked = tc_ids.sort_values(by='vstore100', ascending=False)
ordered_list_track = tcs_ranked['tc_id'].tolist()
track_id = ordered_list_track[task_id]


def get_usage():
    # Function to get CPU and memory usage
    mem_info = proc.memory_info()
    cpu_percent = proc.cpu_percent(interval=1)  # Get CPU usage in the last 1 second
    mem_usage = mem_info.rss / (1024 * 1024)  # Convert from bytes to MB
    return cpu_percent, mem_usage


def run_sfincs_models(scenario_dir, input_dir, base_model_dir, task_id):
    # Prepare the bind options for the Singularity command
    bind_str = f'-B {scenario_dir},{input_dir},{base_model_dir}'
    singularity_command = [
        "singularity", "run", 
        bind_str,  # Bind mount the folders into the container
        "/projects/sfincs/docker/sfincs.sif",      # The Singularity image file
    ]
    print(singularity_command)

    try:
        os.chdir(scenario_dir)

        with open(f'sfincs_log_{task_id}.log', 'w') as log_file:
            # Get the current process ID
            pid = os.getpid()
            proc = psutil.Process(pid)

            # Get the number of CPU cores
            logical_cores = psutil.cpu_count(logical=True)
            physical_cores = psutil.cpu_count(logical=False)
            log_file.write(f"Logical CPU cores: {logical_cores}\n")
            log_file.write(f"Physical CPU cores: {physical_cores}\n")
            log_file.flush()  # Ensure the log is written immediately

            # Start the task processing
            start_time = time.time()
            print("Starting task...")

            # Run the Singularity command using subprocess.Popen
            process = subprocess.Popen(singularity_command,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True)

            # Read stdout and stderr in real-time
            for line in process.stdout:
                log_file.write(line)  # Write output to log file
                log_file.flush()      # Ensure the log file is flushed immediately

            for line in process.stderr:
                log_file.write(line)  # Write error to log file
                log_file.flush()      # Ensure the log file is flushed immediately

            # Wait for the process to finish
            process.wait()

            # Check if the process was successful
            if process.returncode == 0:
                log_file.write(f"Successfully completed SFINCS run: TaskID {task_id}")
                cpu, mem = get_usage()
                # Log the results to the log file
                elapsed_time = time.time() - start_time
                log_file.write(f"Task {os.environ['SLURM_ARRAY_TASK_ID']} completed\n")
                log_file.write(f"Elapsed time: {elapsed_time:.2f}s, CPU: {cpu:.2f}%, Memory: {mem:.2f} MB\n")
                log_file.flush()  # Ensure the log is written immediately
            else:
                log_file.write(f"SFINCS {task_id} failed with exit code {process.returncode}")
                sys.exit(1)
                
    except Exception as e:
        print(f"Error running {scenario_dir}: {e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()

        # Extract line number
        tb = traceback.extract_tb(exc_traceback)[-1]
        line_number = tb.lineno

        # Print error information
        print(f"Error Type: {exc_type.__name__}")
        print(f"Error Message: {exc_value}")
        print(f"Occurred at line: {line_number}")
        return None
        pass


def modify_config_file(template_file, output_file, updates_dict):
    """
    Reads the template file, modifies values on the right side of the `=` sign based on `modifications`,
    and writes the updated content to an output file. Adds new keys from updates_dict that are not
    already present in the template file. Does not write lines with empty values.

    :param template_file: Path to the template file.
    :param output_file: Path to the output file where the modified content will be saved.
    :param updates_dict: Dictionary of modifications where keys are variable names on the left
                          and values are the new values for the right-hand side.
    """
    try:
        # Open the template file for reading
        with open(template_file, 'r') as infile:
            lines = infile.readlines()

        existing_keys = set()  # Track keys that are already in the template file
        updated_lines = []

        # Process each line in the template file
        for line in lines:
            # Split each line at the equal sign ('=') to separate the key and value
            parts = line.split("=")

            if len(parts) == 2:
                key = parts[0].strip()  # Left side (key), remove extra spaces
                value = parts[1].strip()  # Right side (value), remove extra spaces

                # Check if the key exists in the modifications dictionary
                if key in updates_dict:
                    # Modify the value based on the dictionary
                    value = updates_dict[key]
                existing_keys.add(key)

                # Only write the line if the value is non-empty
                if value:
                    k = "{:<20}".format(key)
                    updated_lines.append(f"{k} = {value}\n")
            else:
                # If the line doesn't match the expected format, write it as is (skip empty lines)
                updated_lines.append(line)

        # Add any new keys from updates_dict that weren't in the template
        for key, value in updates_dict.items():
            if key not in existing_keys and value:  # Only add the key if it has a non-empty value
                k = "{:<20}".format(key)
                updated_lines.append(f"{k} = {value}\n")

        # Write the updated content to the output file
        with open(output_file, 'w') as outfile:
            outfile.writelines(updated_lines)

        print(f"File modified and saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")


base_model_dir = r'/projects/sfincs/syntheticTCs_cmpdfld/SFINCS_model/base_model'
track_dir = rf'/projects/sfincs/syntheticTCs_cmpdfld/SFINCS_model/runs/TC_{str(track_id).zfill(4)}'
if not os.path.exists(track_dir):
    print(f"The folder does not exist for {track_id}. Exiting the python script.")
    sys.exit()

bc_inputs_dir = os.path.join(track_dir, 'sfincs_bc_inputs')
if len(os.listdir(bc_inputs_dir)) < 2:
    print(f"Boundary condition inputs not created for {track_id}. Exiting the python script.")
    sys.exit()

scenarios = ['runoff', 'coastal', 'compound']
basemodel_config_updates = {
    "depfile": os.path.join(base_model_dir, 'sfincs.dep'),
    "mskfile": os.path.join(base_model_dir, 'sfincs.msk'),
    "indexfile": os.path.join(base_model_dir, 'sfincs.ind'),
    "sbgfile": os.path.join(base_model_dir, 'sfincs_subgrid.nc'),
    "obsfile": os.path.join(base_model_dir, 'sfincs_obsLocs_gauge_riv_v2.obs'),
    "smaxfile": os.path.join(base_model_dir, 'sfincs.smax'),
    "sefffile": os.path.join(base_model_dir, 'sfincs.seff'),
    "ksfile": os.path.join(base_model_dir, 'sfincs.ks'),
    "weirfile": os.path.join(base_model_dir, 'sfincs.weir'),
    "rstfile": os.path.join(base_model_dir, 'sfincs.19990119.000000.rst'),
    "tspinup": '60',
    "dtout": '432000',
    "dtrstout": '',
    "twet_threshold": '0.1',
    "storetwet": '1',
    "storevelmax": '1',
    "storefluxmax": '1',
    'zsini': '',
    'dthisout':'1800',
    'dtmaxout':'99999999',
}

# Loop through the scenarios and run each one
for scenario in scenarios:
    scenario_dir = os.path.join(track_dir, scenario)
    if os.path.exists(scenario_dir) is False:
        os.makedirs(scenario_dir)

    # Make updates to the sfincs.inp file for each scenario
    template_inp = os.path.join(bc_inputs_dir, 'sfincs.inp')
    output_inp = os.path.join(scenario_dir, 'sfincs.inp')
    if scenario == 'runoff':
        config_updates = {
            "bndfile": os.path.join(bc_inputs_dir, 'tides_sfincs.bnd'),
            "bzsfile": os.path.join(bc_inputs_dir, 'tides_sfincs.bzs'),
            "srcfile": os.path.join(base_model_dir, 'sfincs.src'),
            "disfile": os.path.join(base_model_dir, 'sfincs.dis'),
            "netamprfile": os.path.join(bc_inputs_dir, 'precip_2d.nc'),
            "netamuamvfile": ''
        }
    elif scenario == 'coastal':
        config_updates = {
            "bndfile": os.path.join(bc_inputs_dir, 'sfincs.bnd'),
            "bzsfile": os.path.join(bc_inputs_dir, 'sfincs.bzs'),
            "srcfile": '',
            "disfile": '',
            "netamprfile": '',
            "netamuamvfile": os.path.join(bc_inputs_dir, 'wind_2d.nc')
        }
    else:
        config_updates = {
            "bndfile": os.path.join(bc_inputs_dir, 'sfincs.bnd'),
            "bzsfile": os.path.join(bc_inputs_dir, 'sfincs.bzs'),
            "srcfile": os.path.join(base_model_dir, 'sfincs.src'),
            "disfile": os.path.join(base_model_dir, 'sfincs.dis'),
            "netamprfile": os.path.join(bc_inputs_dir, 'precip_2d.nc'),
            "netamuamvfile": os.path.join(bc_inputs_dir, 'wind_2d.nc')
        }
    modify_config_file(template_file=template_inp, output_file=output_inp,
                       updates_dict={**basemodel_config_updates, **config_updates})
    print(f'SFINCS config file updated for {track_id} - {scenario}')

    # Print details of the current task
    print(f"Running {scenario} for track {track_id} (Slurm task {os.getenv('SLURM_ARRAY_TASK_ID')}) in\n {scenario_dir}")
    
    # Run the Singularity container for the current scenario
    run_sfincs_models(scenario_dir, bc_inputs_dir, base_model_dir, task_id)

print(f"All scenarios for storm {track_id} (Slurm task {os.getenv('SLURM_ARRAY_TASK_ID')}) completed.")


destination_folder = rf'/projects/sfincs/syntheticTCs_cmpdfld/SFINCS_model/completed_runs/TC_{str(track_id).zfill(4)}'
shutil.move(track_dir, destination_folder)
