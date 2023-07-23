import os
import cv2
import numpy as np

from common.dataset import Dataset
from common.trajectory import Trajectory


def quaternion_to_rotation_matrix(quaternion):
    """
    Generate rotation matrix 3x3  from the unit quaternion.
    Input:
    qQuaternion -- tuple consisting of (qx,qy,qz,qw) where
         (qx,qy,qz,qw) is the unit quaternion.
    Output:
    matrix -- 3x3 rotation matrix
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    eps = np.finfo(float).eps * 4.0
    assert nq > eps
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3]),
        (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3]),
        (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1])
    ), dtype=np.float64)
    

def MSE(img1, img2):
    summed = np.sum((img1 - img2) ** 2)
    total_pixels = 3 * img1.shape[0] * img1.shape[1]
    mse = summed / total_pixels
    return mse

    
def predict_pose(unknown_frame_id, frames, known_frames, known_poses, weights=(0.8, 0.2)):
    """
    Find two most similar frames according to MSE and take weighted mean of their position and quaternion
    """
    similarities = {known_frame_id: MSE(frames[unknown_frame_id], frames[known_frame_id])
                    for known_frame_id in known_frames}
    similarities = dict(sorted(similarities.items(), key=lambda x: x[1]))
    top_1, top_2 = list(similarities.keys())[:2]
    weight_1, weight_2 = weights
    weighted_mean = np.array(known_poses[top_1], dtype=np.float32)*weight_1 + \
                    np.array(known_poses[top_2], dtype=np.float32)*weight_2
    return list(map(str, weighted_mean))


def estimate_trajectory(data_dir, out_dir):

    rgb = Dataset.read_dict_of_lists(os.path.join(data_dir, 'rgb.txt'))
    known_poses = Dataset.read_dict_of_lists(os.path.join(data_dir, 'known_poses.txt'))
    
    frames = {frame_id: cv2.imread(os.path.join(data_dir, frame_path)) 
              for frame_id, frame_path in rgb.items()}
    
    known_frames = list(known_poses.keys())
    unknown_frames = [frame_id for frame_id in rgb.keys() if frame_id not in known_frames]
    
    predicted_poses = {unknown_frame_id: predict_pose(unknown_frame_id, frames, known_frames, known_poses)
                       for unknown_frame_id in unknown_frames}
    
    all_poses = {**known_poses, **predicted_poses}
    
    trajectory = {frame_id: Trajectory.to_matrix4(positionAndQuaternion)
                  for frame_id, positionAndQuaternion in all_poses.items()}
    
    Trajectory.write(Dataset.get_result_poses_file(out_dir), trajectory)
