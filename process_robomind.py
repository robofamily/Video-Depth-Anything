import numpy as np
import json
import cv2
import copy
import argparse
import h5py
import torch
from collections import defaultdict
from video_depth_anything.video_depth import VideoDepthAnything
import argparse
import os

class ReadH5Files():
    def __init__(self, robot_infor):
        self.camera_names = robot_infor['camera_names']
        self.camera_sensors = robot_infor['camera_sensors']
        self.arms = robot_infor['arms']
        self.robot_infor = robot_infor['controls']

    def decoder_image(self, camera_rgb_images, camera_depth_images):
        if type(camera_rgb_images[0]) is np.uint8:
            rgb = cv2.imdecode(camera_rgb_images, cv2.IMREAD_COLOR)
            if camera_depth_images is not None:
                depth_array = np.frombuffer(camera_depth_images, dtype=np.uint8)
                depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
            else:
                depth = np.asarray([])
            return rgb, depth
        else:
            rgb_images = []
            depth_images = []
            for idx, camera_rgb_image in enumerate(camera_rgb_images):
                camera_rgb_image = np.array(camera_rgb_image)
                # print(f"camera_rgb_image: {camera_rgb_image.shape}")
                rgb = cv2.imdecode(camera_rgb_image, cv2.IMREAD_COLOR)
                if rgb is None:
                    rgb = np.frombuffer(camera_rgb_image, dtype=np.uint8)
                    try:
                        rgb = rgb.reshape(720, 1280, 3)
                    except:
                        rgb = rgb.reshape(480, 640, 3)

                if camera_depth_images is not None:
                    depth_array = np.frombuffer(camera_depth_images[idx], dtype=np.uint8)
                    depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
                else:
                    depth = np.asarray([])
                rgb_images.append(rgb)
                depth_images.append(depth)
            rgb_images = np.asarray(rgb_images)
            depth_images = np.asarray(depth_images)
            return rgb_images, depth_images

    def execute(self, file_path, camera_frame=None, control_frame=None):
        with h5py.File(file_path, 'r') as f:
            is_sim = f.attrs['sim']
            is_compress = f.attrs['compress']
            is_compress = True
            image_dict = defaultdict(dict)
            import ipdb;ipdb.set_trace()
            for cam_name in self.camera_names:
                if is_compress:
                    if camera_frame is not None:
                        if len(self.camera_sensors) >= 2:
                            decode_rgb, decode_depth = self.decoder_image(
                                camera_rgb_images=f['observations'][self.camera_sensors[0]][cam_name][camera_frame],
                                    camera_depth_images=f['observations'][self.camera_sensors[1]][cam_name][camera_frame])
                        else:
                            decode_rgb, decode_depth = self.decoder_image(
                                camera_rgb_images=f['observations'][self.camera_sensors[0]][cam_name][camera_frame],
                                camera_depth_images=None)
                    else:
                        if len(self.camera_sensors) >= 2:
                            rgb_images = f['observations'][self.camera_sensors[0]][cam_name][:]
                            depth_images =  None

                        else:
                            rgb_images = f['observations'][self.camera_sensors[0]][cam_name][:]
                            depth_images = None
                        print(f"rgb_images: {rgb_images.shape}")
                        decode_rgb, decode_depth = self.decoder_image(camera_rgb_images=rgb_images,camera_depth_images=depth_images)
                    
                    image_dict[self.camera_sensors[0]][cam_name] = decode_rgb
                    if len(self.camera_sensors) >= 2:
                        image_dict[self.camera_sensors[1]][cam_name] = decode_depth

                else:
                    if camera_frame:
                        image_dict[self.camera_sensors[0]][cam_name] = f[
                            'observations'][self.camera_sensors[0]][cam_name][camera_frame]
                        image_dict[self.camera_sensors[1]][cam_name] = f[
                            'observations'][self.camera_sensors[1]][cam_name][camera_frame]
                    else:
                        image_dict[self.camera_sensors[0]][cam_name] = f[
                           'observations'][self.camera_sensors[0]][cam_name][:]

            control_dict = defaultdict(dict)
            for arm_name in self.arms:
                for control in self.robot_infor:
                    if control_frame:
                        control_dict[arm_name][control] = f[arm_name][control][control_frame]
                    else:
                        control_dict[arm_name][control] = f[arm_name][control][:]
            base_dict = defaultdict(dict)
        return image_dict, control_dict, base_dict, is_sim, is_compress

# 定义函数：保存 RGB 图像为 MP4 文件
def save_rgb_images_to_mp4(rgb_images, output_path, fps=30):
    """
    将 RGB 图像序列保存为 MP4 文件
    :param rgb_images: f*h*w*3 的 numpy 数组，范围是 0-255
    :param output_path: 输出的 MP4 文件路径
    :param fps: 帧率，默认为 30
    """
    # 获取视频的宽度和高度
    height, width = rgb_images.shape[1:3]
    # 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 遍历每一帧并写入视频
    for frame in rgb_images:
        # OpenCV 需要 BGR 格式，因此需要将 RGB 转换为 BGR
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_frame)
    
    # 释放 VideoWriter 对象
    video_writer.release()
    print(f"RGB 图像已保存为 {output_path}")
    
def save_arrays_to_numpy(estimated_depth, state, action, output_prefix):
    """
    将 estimated_depth、state 和 action 保存为 NumPy 文件
    :param estimated_depth: f*h*w 的 numpy 数组
    :param state: 状态数组
    :param action: 动作数组
    :param output_prefix: 输出文件的前缀路径
    """
    np.save(output_prefix + '_depth.npy', estimated_depth)
    np.save(output_prefix + '_state.npy', state)
    np.save(output_prefix + '_action.npy', action)
    print(f"estimated_depth 已保存为 {output_prefix}_depth.npy")
    print(f"state 已保存为 {output_prefix}_state.npy")
    print(f"action 已保存为 {output_prefix}_action.npy")
    
def read_trajectory(json_file, indices, video_depth_anything):
    """
    Read and process trajectories from the JSON file based on the provided indices.
    """
    with open(json_file, 'r') as f:
        trajectory_data = json.load(f)

    for index in range(indices):
        if index >= len(trajectory_data):
            print(f"Index {index} is out of range. There are only {len(trajectory_data)} entries.")
            continue

        entry = trajectory_data[index]
        file_path = entry["trajectory_path"]
        # import ipdb;ipdb.set_trace()
        embodiment = entry["embodiment"]
        env_name = entry["env_name"]
        camera_name = entry["camera_name"]

        print(f"Reading trajectory {index} from {file_path}")
        print(f"Embodiment: {embodiment}, Environment: {env_name}, Camera: {camera_name}")

        # Define robot information based on the embodiment
        if embodiment == "h5_ur_1rgb":
            robot_infor = {
                "camera_names": [camera_name],
                "camera_sensors": ['rgb_images', 'depth_images'],
                "arms": ['puppet'],
                "controls": ['joint_position', 'end_effector'],
            }
        elif embodiment == "h5_franka_3rgb":
            robot_infor = {
                "camera_names": [camera_name],
                "camera_sensors": ['rgb_images', 'depth_images'],
                "arms": ['puppet'],
                "controls": ['joint_position', 'end_effector'],
            }
        elif embodiment == "h5_franka_1rgb":
            robot_infor = {
                "camera_names": [camera_name],
                "camera_sensors": ['rgb_images', 'depth_images'],
                "arms": ['puppet'],
                "controls": ['joint_position', 'end_effector'],
            }
        else:
            raise ValueError(f"Unknown embodiment: {embodiment}")

        # Read the HDF5 file
        read_h5files = ReadH5Files(robot_infor)
        image_dict, control_dict, base_dict, _, is_compress = read_h5files.execute(file_path)
        
        # Process the data as needed
        action_list = []
        for keys in control_dict.keys():
            control_list = []
            for control_key in control_dict[keys].keys():
                control = control_dict[keys][control_key]
                control_list.append(control)
            control = np.concatenate(control_list, axis=1)
            action_list.append(control)
        action = np.concatenate(action_list, axis=1)
        state = copy.deepcopy(action)

        action = action[1:, -7:]
        state = state[:-1, -7:]

        for sensor_type in image_dict:
            for camera in image_dict[sensor_type]:
                image_dict[sensor_type][camera] = image_dict[sensor_type][camera][0:-1]

        for camera in image_dict['rgb_images']:
            depth_list, fps = video_depth_anything.infer_video_depth(image_dict['rgb_images'][camera], target_fps=-1, device=DEVICE)
            image_dict['estimated_depth'][camera] = np.stack(depth_list, axis=0)

        parts = file_path.split('/')
        # 在指定位置插入 robotmind_v1
        parts.insert(5, 'robotmind_v1')  # 假设 robotmind_v1 插入在第 5 个位置
        # 构造输出目录路径（去掉文件名）
        output_dir = '/'.join(parts[:-2])  # 去掉最后的 data/trajectory.hdf5
        os.makedirs(output_dir, exist_ok=True)
        video_output_path = os.path.join(output_dir, 'video.mp4')
        depth_output_path = os.path.join(output_dir, 'depth.npy')
        save_rgb_images_to_mp4(image_dict['rgb_images'][camera], video_output_path)
    
        # 保存 estimated_depth、state 和 action 为 NumPy 文件
        save_arrays_to_numpy(image_dict['estimated_depth'][camera], state, action, depth_output_path.replace('_depth.npy', ''))
        print(f"Processed trajectory {index} successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str, default='1', help="JSON file containing trajectory data")
    parser.add_argument("--indices", type=int, nargs='+',default=12641, help="Indices of the trajectories to process")
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl'])
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    video_depth_anything.load_state_dict(torch.load('/mnt/ruili_2/videogames/robotmind/Video-Depth-Anything/video_depth_anything_vitl.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    read_trajectory(args.json_file, args.indices, video_depth_anything)
