import os
import json
import argparse

def generate_trajectory_paths(dataset_dir, output_json, embodiments):
    trajectory_data = []

    for embodiment in embodiments:
        embodiment_dir = os.path.join(dataset_dir, embodiment)
        if not os.path.exists(embodiment_dir):
            print(f"Embodiment directory {embodiment_dir} does not exist. Skipping.")
            continue

        for env_name in os.listdir(embodiment_dir):
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

                        # Determine the camera names based on the embodiment
                        if embodiment == "h5_ur_1rgb":
                            camera_names = ["camera_top"]
                        elif embodiment == "h5_franka_3rgb":
                            camera_names = ["camera_top", "camera_left", "camera_right"]
                        elif embodiment == "h5_franka_1rgb":
                            camera_names = ["camera_top"]
                        else:
                            raise ValueError(f"Unknown embodiment: {embodiment}")

                        # Create a separate entry for each camera
                        for camera_name in camera_names:
                            trajectory_data.append({
                                "embodiment": embodiment,
                                "env_name": env_name,
                                "camera_name": camera_name,  # Only one camera per entry
                                "trajectory_path": file_path
                            })
                        break  # Assuming only one HDF5 file per trajectory

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
    embodiments = ["h5_ur_1rgb", "h5_franka_3rgb", "h5_franka_1rgb"]

    generate_trajectory_paths(args.dataset_path, args.output_json, embodiments)