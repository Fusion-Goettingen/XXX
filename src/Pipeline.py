import icp_cpp

import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from SE3 import SE3

class Pipeline_P2P:
    def __init__(
        self,
        local_map,
        global_map,
        initial_threshold=1,
        min_dist=5,
        max_dist=80,
        min_Z=-3,
        voxelsize_map=0.5,
        voxelsize_frame=1.5,
        max_iterations=500,
        min_transfromation_magnitude=1E-5,
        alpha=0.03,
        deskew=False,
        visualize=False,
    ):
        self.local_map = local_map
        self.global_map = global_map

        self.poses = []
        self.timestamps = []

        self.current_threshold = initial_threshold

        self.min_dist = min_dist
        self.max_dist = max_dist
        self.min_Z = min_Z
        self.voxelsize_map = voxelsize_map
        self.voxelsize_frame = voxelsize_frame

        self.max_iterations = max_iterations
        self.min_transfromation_magnitude = min_transfromation_magnitude

        self.alpha = alpha
        self.deskew = deskew
        self.visualize = visualize

        self.visualizers = []
        # Create ros2 publishers if visualize is set to true
        if self.visualize:
            import VisualizerPublisher
            self.visualizers.append(VisualizerPublisher.PosePublisher("es_poses","/es_poses"))
            self.visualizers.append(VisualizerPublisher.PointCloudPublisher("local_map", "/local_map"))
            self.visualizers.append(VisualizerPublisher.PointCloudPublisher("global_map", "/global_map"))
            self.visualizers[-1].publish(np.array(self.global_map.get_all()))
            self.visualizers.append(VisualizerPublisher.TransformPublisher("campra_pose",""))


    def run_all(self, iterator):
        for timestamp, data in tqdm(iterator):
            self.run(timestamp, data)

    def run(self, timestamp, data):
        sigma = self.current_threshold
        tau = sigma / 3
        max_correspondence_distance = sigma * 3
        # Extrapolate position based on constant velocity motion model, and preprocess data
        V = self.predict_movement()
        T_L_pred = self.poses[-1] * V if len(self.poses) > 0 else SE3()
        frame_map, frame_icp = self.preprocess(data)

        # local ICP, use T_L_pred as initial guess for the pose
        T_ICP = icp_cpp.ICP(
            T_L_pred.apply(frame_icp),
            self.local_map,
            tau,
            max_correspondence_distance,
            self.max_iterations,
            self.min_transfromation_magnitude
        )
        T_ICP = SE3.from_matrix(T_ICP)
        T_L = T_ICP * T_L_pred

        T_L_pred = T_L
        _pc = (T_L_pred).apply(frame_icp)
        _pc[:, 2] = 0
        T_ICP_OSM = icp_cpp.ICP(
            _pc,
            self.global_map,
            tau,
            max_correspondence_distance,
            self.max_iterations,
            self.min_transfromation_magnitude,
        )
        T_ICP_OSM = SE3.from_matrix(T_ICP_OSM)
        T_G = T_ICP_OSM * T_L


        # Apply fusion between T_L and T_G
        V_R_A = (T_L.inv() * T_G)
        if np.linalg.norm(V_R_A.log()[:3]) < max_correspondence_distance:
            T_F = T_L * SE3.exp(self.alpha * (T_L.inv() * T_G).log())
        else:
            T_F = T_L

        # Update
        frame_map = T_F.apply(frame_map)
        added_to_local_map_mask = self.local_map.add(frame_map)
        self.local_map.remove_outside(T_F.t)

        # Save pose for next iteration
        self.poses.append(T_F)
        self.timestamps.append(timestamp)

        # Visualize
        if self.visualize:
            added_to_local_map = frame_map[added_to_local_map_mask]
            self.visualizers[0].publish(self.poses)
            self.visualizers[1].publish(added_to_local_map)
            self.visualizers[3].publish(self.poses[-1],"world","camera")


    def predict_movement(self):
        V = SE3()

        if len(self.poses) >= 2:
            V = self.poses[-2].inv() * self.poses[-1]

        return V

    def preprocess(self, frame_and_times):
        def remove_outliers(data, min_dist, max_dist, min_Z):
            dist = np.linalg.norm(data, axis=1)
            mask = np.logical_and(
                np.logical_and(dist > min_dist, dist < max_dist), data[:, -1] > min_Z
            )
            return data[mask]


        def subsample_voxel(data, voxelsize, max_elements_per_voxel, world_size):
            map = icp_cpp.VoxelHashMap3d(voxelsize, max_elements_per_voxel, world_size)
            mask = map.add(data)
            return data[mask]


        def deskew(frame, timestamps, start_pose, finish_pose):
            timestamps = timestamps - 0.5

            delta_pose = (start_pose.inv() * finish_pose).log()
            corrections = np.outer(timestamps, delta_pose)
            Rs = Rotation.from_rotvec(corrections[:, 3:], False).as_matrix()
            ts = corrections[:, :3]
            res = np.einsum("nij,nj->ni", Rs, frame).astype(frame.dtype) + ts
            return res

        frame = frame_and_times[:, :3]
        timestamps = frame_and_times[:,3]

        # Remove nans
        mask = ~np.isnan(frame[:, 0])
        frame = frame[mask]
        timestamps = timestamps[mask]
        # Deskew
        if self.deskew:
            finish_pose = self.poses[-1] if len(self.poses) > 0 else SE3()
            start_pose = self.poses[-2] if len(self.poses) > 1 else SE3()
            frame = deskew(frame,timestamps,start_pose, finish_pose)

        # Remove outliers and PERMUTE order of frame
        frame = remove_outliers(
            frame, self.min_dist, self.max_dist, self.min_Z
        )
        frame = np.random.permutation(frame)

        # Subsample twice
        frame_map = subsample_voxel(
            frame, self.voxelsize_map, 1, self.local_map.get_world_size()
        )
        frame_ICP = subsample_voxel(
            frame_map,
            self.voxelsize_frame,
            1,
            self.local_map.get_world_size(),
        )

        return frame_map, frame_ICP

    def get_poses(self):
        logs = np.array([T.log() for T in self.poses])
        return list(zip(self.timestamps, logs.tolist()))

if __name__ == "__main__":
    import argparse
    import glob
    import json
    import os
    import pathlib
    import pickle
    from PrettyJsonEncoder import PrettyJSONEncoder
    from pathlib import Path

    this_dir = pathlib.Path(__file__).parent.resolve()

    parser = argparse.ArgumentParser(
        prog="Pipeline_Original",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    parser.add_argument("data_dir")
    parser.add_argument("--map", default="", type=str)
    parser.add_argument("--dataloader", default="kitti", type=str)
    parser.add_argument("--out_path", default=f"{str(this_dir.parent)}/out/result.json", type=str)

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--min_dist", default=8, type=float)
    parser.add_argument("--max_dist", default=80, type=float)
    parser.add_argument("--min_Z", default=-3, type=float)
    parser.add_argument("--voxelsize_map", default=0.5, type=float)
    parser.add_argument("--voxelsize_frame", default=1.5, type=float)
    parser.add_argument("--initial_threshold", default=1, type=float)
    parser.add_argument("--max_iterations", default=500, type=int)
    parser.add_argument("--min_transfromation_magnitude", default=1e-5, type=float)

    parser.add_argument("--map_size", default=80, type=float)
    parser.add_argument("--map_max_elements_per_voxel", default=20, type=int)

    parser.add_argument("--alpha", default=0.03, type=float)
    parser.add_argument("--deskew", action="store_true")
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()

    if args.map_size < args.max_dist + 2 * args.voxelsize_map:
        print(
            "map_size was chosen smaller than max_dist + 2 * voxelsize_map. Increasing map_size to that value."
        )
        args.map_size = max(args.map_size, args.max_dist + 2 * args.voxelsize_map)

    np.random.seed(args.seed)

    local_map = icp_cpp.VoxelHashMap3d(
        args.voxelsize_map, args.map_max_elements_per_voxel, args.map_size
    )

    # Init. point cloud generators, depending on dataloader
    if args.dataloader == "kitti":
        import dataloader_kitti
        data_dir = args.data_dir
        frames = glob.glob(data_dir + "/*")
        frames.sort()
        pointcloud_iterator = dataloader_kitti.pointcloud_generator(frames, True)
    elif args.dataloader == "okular":
        import dataloader_okular
        pointcloud_iterator = dataloader_okular.pointcloud_generator(args.data_dir)
    else:
        raise Exception("Invalid dataloader")

    # Loading global map from file
    if args.map != "":
        sampled_osm_buildings = np.load(args.map)
    else:
        sampled_osm_buildings = np.zeros((0, 6), dtype=np.float64)

    global_map = icp_cpp.VoxelHashMap3d(3, 10000, 10000.0)
    global_map.add(sampled_osm_buildings[:, :3])

    # Create Piplen object and run
    pip = Pipeline_P2P(
        local_map,
        global_map,
        args.initial_threshold,
        args.min_dist,
        args.max_dist,
        args.min_Z,
        args.voxelsize_map,
        args.voxelsize_frame,
        args.max_iterations,
        args.min_transfromation_magnitude,
        args.alpha,
        args.deskew,
        args.visualize,
    )

    pip.run_all(pointcloud_iterator)
    # Persist result to out_dir
    timestamped_poses = pip.get_poses()
    out_path = Path(args.out_path)
    os.makedirs(out_path.parent, exist_ok=True)
    out_file = out_path.name.split(".")[0] + ".json"

    dict = vars(args)
    timestamps, poses = list(zip(*timestamped_poses))
    dict["timestamps"] = timestamps
    dict["poses"] = poses
    with open(f"{out_path.parent}/{out_file}", "w") as f:
        json.dump(vars(args), f, separators=("\n,", ":"),cls=PrettyJSONEncoder,indent=4)


