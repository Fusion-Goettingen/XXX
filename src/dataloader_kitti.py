from scipy.spatial.transform import Rotation
import numpy as np

def correct_KITTI_pointcloud(points):
    VERTICAL_ANGLE_OFFSET = (0.205 * np.pi) / 180.0
    ups = np.zeros(points.shape, dtype=points.dtype)
    ups[:, -1] = 1
    rotation_vectors = np.cross(points, ups)
    rotation_vectors = rotation_vectors * (
        VERTICAL_ANGLE_OFFSET
        / np.linalg.norm(rotation_vectors, axis=1)[:, np.newaxis]
    )
    Rs = Rotation.from_rotvec(rotation_vectors).as_matrix()
    res = np.einsum("nij,nj->ni", Rs, points).astype(points.dtype)
    return res

def pointcloud_generator(files,correct=True):
    for i, file in enumerate(files):
        data = np.fromfile(file, dtype=np.float32).reshape((-1, 4)).astype(np.float64)
        # Overwriting reflectivity information with timestamps
        data[:, -1] = 0.5
        if correct:
            data[:, :3] = correct_KITTI_pointcloud(data[:, :3])
        yield 0.1 * i, data