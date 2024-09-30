import rclpy
from rclpy.node import Node
import numpy as np
from scipy.spatial.transform import Rotation

from std_msgs.msg import String, ColorRGBA
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import (
    PoseArray,
    Pose,
    Point,
    Quaternion,
    Vector3,
    TransformStamped,
    Transform,
)

from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import MarkerArray, Marker
from SE3 import SE3
import tf2_ros

rclpy.init()


def SE3_to_Pose(se3: SE3):
    t = se3.t
    position = Point()
    position.x = t[0]
    position.y = t[1]
    position.z = t[2]
    q = se3.R.as_quat()
    orientation = Quaternion()
    orientation.x = q[0]
    orientation.y = q[1]
    orientation.z = q[2]
    orientation.w = q[3]
    return Pose(position=position, orientation=orientation)


# http://docs.ros.org/en/noetic/api/visualization_msgs/html/msg/Marker.html
def plane_to_marker(id, plane):
    header = Header(frame_id="world")
    ns = "0"
    type = 1
    action = 0  # add/modify
    pose = SE3_to_Pose(SE3(plane[:3], Rotation.from_rotvec(plane[3:])))
    scale = Vector3()
    scale.x = 1.0
    scale.y = 1.0
    scale.z = 0.1
    color = ColorRGBA()
    color.r = 1.0
    color.g = 0.0
    color.b = 1.0
    color.a = 0.5
    marker = Marker(
        header=header,
        ns=ns,
        id=id,
        type=type,
        action=action,
        pose=pose,
        scale=scale,
        color=color,
    )
    return marker


def del_marker():
    header = Header(frame_id="world")
    ns = "0"
    type = 1
    action = 3  # add/modify
    marker = Marker(
        header=header,
        ns=ns,
        id=-1,
        type=type,
        action=action,
    )
    return marker


class PointCloudPublisher(Node):
    def __init__(self, name, topic):
        super().__init__(name)
        self.publisher_ = self.create_publisher(PointCloud2, topic, 10)

    def publish(self, data):
        header = Header(frame_id="world")
        ros_dtype = PointField.FLOAT32
        itemsize = np.dtype(np.float32).itemsize
        fields = [
            PointField(name=n, offset=i * itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate("xyz")
        ]
        msg = PointCloud2(
            header=header,
            height=1,
            width=data.shape[0],
            is_bigendian=False,
            fields=fields,
            point_step=3 * itemsize,
            row_step=3 * itemsize * data.shape[0],
            data=data[:, :3].astype(np.float32).tobytes(),
        )

        # msg.data = data[:, :3]

        self.publisher_.publish(msg)


class PosePublisher(Node):
    def __init__(self, name, topic):
        super().__init__(name)
        self.publisher_ = self.create_publisher(PoseArray, topic, 10)

    def publish(self, poses):
        header = Header(frame_id="world")
        ros_dtype = PointField.FLOAT32
        itemsize = np.dtype(np.float32).itemsize
        fields = [
            PointField(name=n, offset=i * itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate("xyz")
        ]
        msg = PoseArray(header=header, poses=[SE3_to_Pose(pose) for pose in poses])

        self.publisher_.publish(msg)


class PlanePublisher(Node):
    def __init__(self, name, topic):
        super().__init__(name)
        self.publisher_ = self.create_publisher(MarkerArray, topic, 10)

    def publish(self, planes):
        msg = MarkerArray(
            markers=[del_marker()]
            + [plane_to_marker(i, plane) for i, plane in enumerate(planes)]
        )

        self.publisher_.publish(msg)


class TransformPublisher(Node):
    def __init__(self, name, topic):
        super().__init__(name)
        # self.publisher_ = self.create_publisher(TFMessage, topic, 10)

        self.broadcaster = tf2_ros.StaticTransformBroadcaster(self)

    def publish(self, pose, frame_id, child_frame_id):

        static_transformStamped = TransformStamped()

        # static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = frame_id
        static_transformStamped.child_frame_id = child_frame_id

        static_transformStamped.transform.translation.x = pose.t[0]
        static_transformStamped.transform.translation.y = pose.t[1]
        static_transformStamped.transform.translation.z = pose.t[2]
        self.broadcaster.sendTransform(static_transformStamped)
