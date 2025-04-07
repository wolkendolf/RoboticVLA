import os
import numpy as np
from sklearn.model_selection import train_test_split


# Data constants

data_dir = "/data/kazachkovda/trajectories/"  # path to folder with raw trajectories

def process_trajectory(traj):
    """
    traj - trajectory in .npy format.
    - 'world_vector' is delta xyz [action[:3]]
    - 'rot_axangle' is delta rotation in axis-angle representation [action[3:6]]
    - 'gripper' is the meaning of open / close depends on robot URDF [action[6:7]]
    - 'terminate_episode' is 3 dims, we take only 1, which show process terminating
    """
    vectors = []
    for step in traj:
        vector = np.concatenate([
            step['action']['world_vector'],  # 3 values
            step['action']['rot_axangle'],   # 3 values
            step['action']['gripper'],       # 1 value
            step['action']['terminate_episode'][0].reshape(1,)  # 1 value
        ])
        vectors.append(vector)
    return np.array(vectors)  # (trajectory_length, 8)

def process_data(data_dir):
    file_list = os.listdir(data_dir)
    file_list = [os.path.join(data_dir, f) for f in file_list if os.path.isfile(os.path.join(data_dir, f))]
    data = [np.load(file, allow_pickle=True) for file in file_list]
    data_processed = [process_trajectory(traj) for traj in data]
    return data_processed

# Data loader setup
train_data, val_data = train_test_split(process_data(data_dir), test_size=0.2) 
train_data = np.concatenate(train_data, axis=0)  # (total_train_steps, 8)
val_data = np.concatenate(val_data, axis=0)  # (total_val_steps, 8)

# Save to files
os.makedirs('data/robot_trajectories', exist_ok=True)
np.save(f"{data_dir}train.npy", train_data)
np.save(f"{data_dir}val.npy", val_data)