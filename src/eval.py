# Python implementation of the KITTI odometry leaderboard metric
# Based on the original KITTI devkit implementation (https://www.cvlibs.net/datasets/kitti/eval_odometry.php)
import numpy as np
from scipy.spatial.transform import Rotation
from SE3 import SE3


def __get_first_with_len(distances, length, first_frame):
    i = first_frame
    if length == -1:
        if i + 1 < len(distances):
            return i + 1
        else:
            return -1
    while i < len(distances):
        if distances[i] > distances[first_frame] + length:
            return i
        i += 1

    return -1


# Calculates the angle of pose_error
def __rotation_error(pose_error):
    abc = np.trace(pose_error[:-1, :-1])
    d = 0.5 * (abc - 1.0)

    return np.arccos(np.clip(d, -1, 1))


def __translation_error(pose_error):
    return np.linalg.norm(pose_error[:-1, -1])


def eval(
    poses_gt,
    poses_es,
    lengths=np.arange(100, 800, 100),
    step_size=10,
):

    if isinstance(poses_gt[0], SE3):
        poses_gt = np.array([T.matrix() for T in poses_gt])
        poses_es = np.array([T.matrix() for T in poses_es])

    assert poses_gt.shape == poses_es.shape
    assert poses_gt.ndim == 3
    # return list for all errors
    errors = []

    # Calculating distance of gt path
    distances = np.zeros(len(poses_gt))
    for i in range(1, len(poses_gt)):
        d_gt = poses_gt[i - 1, :-1, -1] - poses_gt[i, :-1, -1]
        distances[i] = distances[i - 1] + np.linalg.norm(d_gt)

    # For every start_frame (with step size of step_size)
    for start_frame in range(0, len(poses_gt), step_size):
        # for every length in lengths
        for l in lengths:
            # finds the first index that has a distance > len (starting from start_frame)
            end_frame = __get_first_with_len(distances, l, start_frame)
            # skipping if none is found
            if end_frame == -1:
                continue

            # Calculating error
            delta_gt = np.linalg.inv(poses_gt[start_frame]) @ poses_gt[end_frame]
            delta_es = np.linalg.inv(poses_es[start_frame]) @ poses_es[end_frame]
            error = np.linalg.inv(delta_es) @ delta_gt

            # Creating other metrics
            trans_err = __translation_error(error)
            rot_err = __rotation_error(error)
            num_frames = end_frame - start_frame + 1
            speed = l / (0.1 * num_frames)

            delta_trans_gt = __translation_error(delta_gt)
            delta_trans_es = __translation_error(delta_es)
            delta_rot_gt = __rotation_error(delta_gt)
            delta_rot_es = __rotation_error(delta_es)

            err = np.array(
                [
                    start_frame,  # first frame ob subsequence
                    end_frame,  # last frame of subsequence
                    num_frames,  # number of frames in subsequence, is end_frame-start_frame+1
                    l,  # length of the subsequence in [m]
                    trans_err / l * 100,  # translational error in %
                    rot_err * 180 / np.pi / l,  # rotational error in °/m
                    delta_trans_gt,  # distance between poses_gt[start_frame] to poses_gt[end_frame]
                    delta_trans_es,  # distance between poses_es[start_frame] to poses_es[end_frame]
                    delta_rot_gt,  # angle between poses_gt[start_frame] to poses_gt[end_frame]
                    delta_rot_es,  # angle between poses_es[start_frame] to poses_es[end_frame]
                    speed,
                ]
            )  # average speed over subsequence
            errors.append(err)

    # Handling if sequence is too short and no errors have been calculated
    errors = np.array(errors)
    if errors.ndim < 2:
        return np.zeros((0, 11))
    return errors


def get_total_length(poses):
    total_len = 0
    for i in range(1, len(poses)):
        total_len += np.linalg.norm(poses[i, :-1, -1] - poses[i - 1, :-1, -1])
    return total_len


def load_gt_poses_kitti_odometry(home_dir, seq):
    import pykitti
    drive = pykitti.odometry(home_dir, seq)
    T_cam0_velo = drive.calib.T_cam0_velo
    poses = np.array(drive.poses).astype(np.float32)
    left = np.einsum("...ij,...jk->...ik", np.linalg.inv(T_cam0_velo), poses)
    right = np.einsum("...ij,...jk->...ik", left, T_cam0_velo)
    timestamps = np.linspace(0, 0.1 * len(poses), len(poses))

    return timestamps, np.array([SE3.from_matrix(e) for e in right])

def print_eval(es_poses,gt_poses):
    es_poss = np.array([T.t for T in es_poses])
    gt_poss = np.array([T.t for T in gt_poses])
    ATEs3D = np.linalg.norm(es_poss - gt_poss,axis=-1)
    ATEs2D = np.linalg.norm((es_poss - gt_poss)[:,:2], axis=-1)
    print(f"{args.es_poses}: mean ATE 2D: {np.mean(ATEs2D)}")

if __name__ == "__main__":
    import argparse
    import json
    from SE3 import SE3
    from pathlib import Path
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(
        prog="Eval",
        description="Evaluates results",
        epilog="Text at the bottom of help",
    )

    parser.add_argument("es_poses") #Assumed to be a json outputted by Pipeline
    parser.add_argument("gt_poses")
    parser.add_argument("--dataloader",default="kitti_odometry",type=str)
    parser.add_argument("--plot",action="store_true")

    args = parser.parse_args()
    with open(args.es_poses) as f:
        data_dict = json.load(f)
        es_timestamps = np.array(data_dict["timestamps"])
        es_poses = np.array([SE3.exp(t) for t in data_dict["poses"]])

    if args.dataloader == "kitti_odometry":
        base_dir = Path(args.gt_poses).parent.parent
        seq = Path(args.gt_poses).name.split(".")[0]
        gt_timestamps, gt_poses = load_gt_poses_kitti_odometry(base_dir,seq)

    print_eval(es_poses,gt_poses)

    if args.plot:
        import matplotlib.pyplot as plt
        es_poss = np.array([T.t for T in es_poses])
        gt_poss = np.array([T.t for T in gt_poses])
        plt.plot(gt_poss[:, 0], gt_poss[:, 1], label="ground_truth",c="C0")
        plt.plot(es_poss[:,0],es_poss[:,1],label="estimated",c="C2")
        plt.axis("equal")
        plt.legend()
        plt.show()





