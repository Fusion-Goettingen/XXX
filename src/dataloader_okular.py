import time

import numpy as np
from poincloud2 import read_point_cloud
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import rosbags
from tqdm import tqdm
from util import llas_to_cart
from SE3 import SE3
from scipy.spatial.transform import Rotation
import pathlib

def closest_searchsorted(a,v):
    """
    Searches for the index where a is closest to v (for all v if v is a list). Assumes that a is sorted in ascending order.
    Parameters
    ----------
    a: the array where v is searched for in.
    v: the value (or array) that is searched for

    Returns
    -------
    the index where a is closest to v
    """
    N = len(a) - 1
    idx = np.searchsorted(a, v)
    d_l = np.full(idx.shape[0], 10000000000000000)  # TODO: replace with proper int max value
    d_r = np.full(idx.shape[0], 10000000000000000)  # TODO: replace with proper int max value
    d_l[idx > 0] = np.abs(a[idx[idx > 0] - 1] - v[idx > 0])
    d_r[idx < N] = np.abs(a[idx[idx < N] + 1] - v[idx < N])
    d = np.zeros(idx.shape, int)
    d[d_l < d_r] = -1
    d[d_l > d_r] = 1
    id = idx + d
    return id

def first_lidar_pose(bag_path):
    gen = pointcloud_generator(bag_path)
    lidar_timestamp, _ = next(gen)

    poses_timestamps, poses = gt_poses(bag_path)
    idx = closest_searchsorted(poses_timestamps,np.array([lidar_timestamp]))[0]
    lidar_pose = poses[idx]
    return lidar_pose

def gt_poses(bag_path,topics=["/ref_sys/NavSatFix_INS","/ref_sys/inat_imu"],to_cartesian=False,normalize_orientation=False):
    if not pathlib.Path(bag_path).joinpath("gt_poses.npy").exists():
        reader = Reader(bag_path)
        reader.open()
        connections = [x for x in reader.connections if x.topic in topics]
        msgs = reader.messages(connections=connections)
        typestore = get_typestore(Stores.ROS2_HUMBLE)

        nav_msgs = []
        nav_msgs_timestamps = []
        imu_msgs = []
        imu_msgs_timestamps = []
        for i, (connection, timestamp, rawdata) in tqdm(enumerate(msgs)):
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            if connection.msgtype == "sensor_msgs/msg/Imu":
                orientation = np.array([msg.orientation.x,msg.orientation.y,msg.orientation.z,msg.orientation.w])
                t = msg.header.stamp.sec * 1E9 + msg.header.stamp.nanosec + 2500000
                imu_msgs.append(orientation)
                imu_msgs_timestamps.append(t)
            elif connection.msgtype == "sensor_msgs/msg/NavSatFix":
                pos = np.array([msg.latitude,msg.longitude,msg.altitude])
                t = msg.header.stamp.sec * 1E9 + msg.header.stamp.nanosec
                nav_msgs.append(pos)
                nav_msgs_timestamps.append(t)

        nav_msgs = np.array(nav_msgs)
        imu_msgs = np.array(imu_msgs)
        nav_msgs_timestamps = np.array(nav_msgs_timestamps)
        imu_msgs_timestamps = np.array(imu_msgs_timestamps)
        id = closest_searchsorted(nav_msgs_timestamps,imu_msgs_timestamps)

        positions = nav_msgs[id]
        orientations = imu_msgs

        poses = np.column_stack((imu_msgs_timestamps,positions,orientations))
        np.save(pathlib.Path(bag_path).joinpath("gt_poses.npy"),poses)
    else:
        poses = np.load(pathlib.Path(bag_path).joinpath("gt_poses.npy"))
        imu_msgs_timestamps = poses[:,0]
        positions = poses[:,1:4]
        orientations = poses[:,4:]

    if to_cartesian:
        positions = llas_to_cart(positions)

    # Transforming orientation from IMU frame to lidar frame
    R_correction = Rotation.from_rotvec(np.array([0, 0, -np.pi / 2]))
    orientations = np.array([R_correction * Rotation.from_quat(quat,scalar_first=False) for quat in orientations])
    poses = np.full(orientations.shape,None)
    for i, (t,R) in enumerate(zip(positions,orientations)):
        euler = R.as_euler("XYZ")
        euler[1:] *= -1
        poses[i] = SE3(t, Rotation.from_euler("XYZ", euler))

    if normalize_orientation:
        R_init = SE3(R=poses[0].R)
        poses = np.array([R_init.inv() @ T for T in poses])

    return imu_msgs_timestamps, poses

def pointcloud_generator(bag_path,topic="/points"):
    reader = Reader(bag_path)
    reader.open()
    connection = [x for x in reader.connections if x.topic == topic]
    msgs = reader.messages(connections=connection)
    typestore = get_typestore(Stores.LATEST)

    for connection, timestamp, rawdata in msgs:
        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
        t = msg.header.stamp.sec * 1E9 + msg.header.stamp.nanosec
        points = read_point_cloud(msg)
        points[:,-1] /= 50000000 # Normalize timestamps to [0,1]
        yield t, points

    reader.close()