import tqdm
import numpy as np
import os
import gc
import multiprocessing
import pickle
import shutil

# NOTE: This script must be run from the command line, not necessarily a Jupyter cell,
# as it spawns new processes.

# --- Worker Function to run in a separate process ---
# This function will be executed in a completely separate Python process.
def predict_and_save_worker(args):
    """
    A worker that performs prediction for a single item and saves the result to a temporary file.
    """
    # Unpack arguments
    i, chain_index, thetas, model_pkl, sps_pkl, obs_dict_pkl, temp_dir = args

    try:
        # Deserialize the objects in the new process
        model = pickle.loads(model_pkl)
        sps = pickle.loads(sps_pkl)
        observations_dict = pickle.loads(obs_dict_pkl)

        # Run the prediction - this is the memory-intensive part
        spec, mags, mass_frac = model.predict(thetas, sps=sps, obs=observations_dict)

        # Define temporary file paths
        spec_temp_path = os.path.join(temp_dir, f'spec_{i}.bin')
        mags_temp_path = os.path.join(temp_dir, f'mags_{i}.bin')
        mass_frac_temp_path = os.path.join(temp_dir, f'mass_frac_{i}.bin')

        # Save results to temporary binary files
        spec.astype('float32').tofile(spec_temp_path)
        mags.astype('float32').tofile(mags_temp_path)
        np.array([mass_frac], dtype='float32').tofile(mass_frac_temp_path)

        return (i, True, None)  # Return success
    except Exception as e:
        return (i, False, str(e))  # Return failure and the error message

# --- Main Script ---
def main():
    # ==================================================================
    # TODO: Load your actual sps, observations_dict, results, and model
    # objects here. The following is placeholder data.
    # ==================================================================
    class MockModel:
        def predict(self, thetas, sps, obs):
            n_wavelengths = sps.wavelengths.shape[0]
            n_mags = len(obs['filters'])
            # Simulate a memory-intensive operation
            _ = np.random.rand(500, 500) @ np.random.rand(500, 500)
            return np.random.rand(n_wavelengths), np.random.rand(n_mags), np.random.rand()

    class MockSPS:
        wavelengths = np.linspace(4000, 8000, 2000)

    model = MockModel()
    sps = MockSPS()
    observations_dict = {'filters': ['g', 'r', 'i', 'z']}
    results = {
        'chain': np.random.rand(2000, 5),
        'weights': np.random.rand(2000)
    }
    # ==================================================================
    # End of placeholder data
    # ==================================================================

    n_samples = 1000
    n_wavelengths = sps.wavelengths.shape[0]
    n_mags = len(observations_dict['filters'])

    # Define final output filenames and a temporary directory
    spec_bin_filename = 'seds_spec_array.bin'
    mags_bin_filename = 'seds_mag_array.bin'
    mass_frac_filename = 'surviving_mass_frac.bin'
    temp_dir = 'temp_results'

    # Clean up from previous runs
    for f in [spec_bin_filename, mags_bin_filename, mass_frac_filename]:
        if os.path.exists(f): os.remove(f)
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    weights = results.get('weights', None)
    if weights is not None:
        idx = np.argsort(weights)[-n_samples:]
    else:
        idx = np.arange(len(results['chain']) - n_samples, len(results['chain']))

    # Serialize large objects to pass to worker processes
    print("Serializing objects for subprocesses...")
    model_pkl = pickle.dumps(model)
    sps_pkl = pickle.dumps(sps)
    obs_dict_pkl = pickle.dumps(observations_dict)

    # Prepare arguments for the worker function
    tasks = []
    for i, chain_index in enumerate(idx):
        thetas = results['chain'][chain_index]
        tasks.append((i, chain_index, thetas, model_pkl, sps_pkl, obs_dict_pkl, temp_dir))

    # Use a multiprocessing Pool to run tasks in separate processes.
    # `processes=1` and `maxtasksperchild=1` is the safest (but slowest) mode.
    # It creates a fresh process for every single prediction.
    print(f"Starting prediction for {len(tasks)} samples with 1 worker process...")
    with multiprocessing.Pool(processes=1, maxtasksperchild=1) as pool:
        all_results = list(tqdm.tqdm(pool.imap(predict_and_save_worker, tasks), total=len(tasks)))

    # Check for errors
    errors = [r for r in all_results if not r[1]]
    if errors:
        print(f"\nEncountered {len(errors)} errors during processing.")
        for i, _, err_msg in errors:
            print(f"  - Task {i} failed: {err_msg}")

    print("\nConsolidating results...")
    # Consolidate results from temporary files
    with open(spec_bin_filename, 'ab') as spec_file, \
         open(mags_bin_filename, 'ab') as mags_file, \
         open(mass_frac_filename, 'ab') as mass_frac_file:
        for i in tqdm.tqdm(range(len(idx))):
            for suffix, out_file in [('spec', spec_file), ('mags', mags_file), ('mass_frac', mass_frac_file)]:
                temp_path = os.path.join(temp_dir, f'{suffix}_{i}.bin')
                if os.path.exists(temp_path):
                    with open(temp_path, 'rb') as f_in:
                        shutil.copyfileobj(f_in, out_file)

    # Clean up
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    print("Done.")

    # --- How to Read the Data Back ---
    print("\nTo read the data back:")
    print(f"spec_data = np.fromfile('{spec_bin_filename}', dtype='float32').reshape({n_samples}, {n_wavelengths})")
    print(f"mags_data = np.fromfile('{mags_bin_filename}', dtype='float32').reshape({n_samples}, {n_mags})")
    print(f"mass_frac_data = np.fromfile('{mass_frac_filename}', dtype='float32')")


if __name__ == '__main__':
    # This is crucial for multiprocessing to work correctly
    main()
