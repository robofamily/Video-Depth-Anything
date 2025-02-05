import os
import json
import argparse
import h5py
from tqdm import tqdm

def generate_trajectory_paths(dataset_dir, output_json, embodiments):
    trajectory_data = []

    for embodiment in embodiments:
        embodiment_dir = os.path.join(dataset_dir, embodiment)
        if not os.path.exists(embodiment_dir):
            print(f"Embodiment directory {embodiment_dir} does not exist. Skipping.")
            continue

        for env_name in tqdm(os.listdir(embodiment_dir), desc=f"Processing {embodiment}"):
            dataset_root = os.path.join(embodiment_dir, env_name, 'success_episodes/train')
            if not os.path.exists(dataset_root):
                print(f"Dataset root {dataset_root} does not exist. Skipping.")
                continue

            for trajectory_id in sorted(os.listdir(dataset_root)):
                data_dir = os.path.join(dataset_root, trajectory_id, 'data')
                if not os.path.exists(data_dir):
                    print(f"Data directory {data_dir} does not exist. Skipping.")
                    continue

                for file in os.listdir(data_dir):
                    if file.endswith('.hdf5'):
                        file_path = os.path.join(data_dir, file)

                        try:
                            with h5py.File(file_path, 'r') as h5_file:
                                camera_names = []
                                for cam in h5_file['observations']['rgb_images']:
                                    if h5_file['observations']['rgb_images'][cam][0].shape[0] > 0:
                                        camera_names.append(cam)
                        except RuntimeError as e:
                            print(f"Failed to read {file_path}: {e}")
                            continue
                        except Exception as e:
                            print(f"An error occurred while reading {file_path}: {e}")
                            continue

                        # Create a separate entry for each camera
                        for camera_name in camera_names:
                            trajectory_data.append({
                                "embodiment": embodiment,
                                "env_name": env_name,
                                "camera_name": camera_name,  # Only one camera per entry
                                "trajectory_path": file_path
                            })

    # Save the data to a JSON file
    with open(output_json, 'w') as f:
        json.dump(trajectory_data, f, indent=4)

    print(f"Saved {len(trajectory_data)} trajectory entries to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--output_json", type=str, required=True, help="Output JSON file to save trajectory data")
    args = parser.parse_args()

    # Define the embodiments to process
    embodiments = ["h5_franka_1rgb", "h5_franka_3rgb"]

    generate_trajectory_paths(args.dataset_path, args.output_json, embodiments)