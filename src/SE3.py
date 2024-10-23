import numpy as np
from scipy.spatial.transform import Rotation

class SE3:
    def __init__(self, t=np.zeros(3), R=np.eye(3)):
        """Creates a new SE3 object from a translation vertor (3) and either a scipy.spatial.transform.Rotation or o rotation matrix (3x3)."""
        self.t = t
        if isinstance(R, np.ndarray):
            self.R = Rotation.from_matrix(R)
        elif isinstance(R, Rotation):
            self.R = R

    def __mul__(self, other):
        """Multiplies self with other"""
        t_res = self.R.apply(other.t) + self.t
        R_res = self.R * other.R
        return SE3(t_res, R_res)


    def __matmul__(self, other):
        """Multiplies self with other"""
        t_res = self.R.apply(other.t) + self.t
        R_res = self.R * other.R
        return SE3(t_res, R_res)

    def __sub__(self, other):
        return SE3.exp(self.log() - other.log())

    @staticmethod
    def add(self, other):
        return SE3(self.t + other.t, self.R * other.R)

    @staticmethod
    def sub(self, other):
        return SE3(self.t - other.t, self.R.inv() * other.R)

    def inv(self):
        R_inv = self.R.inv()
        t_inv = -R_inv.apply(self.t)
        return SE3(t_inv, R_inv)

    # For backwards compatibility
    def log(self):
        return self.as_log()

    def as_log(self):
        res = np.zeros(6)
        res[:3] = self.t
        res[3:] = self.R.as_rotvec()
        return res

    def as_quat(self):
        res = np.zeros(7,dtype=float)
        res[:3] = self.t
        res[3:] = self.R.as_quat(scalar_first=True)
        return res

    # For backwards compatibility
    @staticmethod
    def exp(t_log):
        return SE3.from_log(t_log)

    @staticmethod
    def from_log(t_log):
        return SE3(t_log[:3], Rotation.from_rotvec(t_log[3:]).as_matrix())

    # For backwards compatibility
    def apply(self, vectors, pad_Z=False):
        if pad_Z:
            vectors = np.column_stack((vectors, np.zeros(len(vectors))))
        res = self.transform(vectors)
        if pad_Z:
            res = res[:, :2]
        return res

    def transform(self, vectors):
        return self.R.apply(vectors) + self.t

    def matrix(self):
        return self.as_matrix()

    # For backwards compatibility
    def as_matrix(self):
        res = np.eye(4)
        res[:3, :3] = self.R.as_matrix()
        res[:3, -1] = self.t
        return res

    @staticmethod
    def from_matrix(mat):
        if mat.shape == (4,4):
            return SE3(mat[:-1, -1], mat[:-1, :-1])
        else:
            return SE3(mat[:,-1],mat[:,:-1])

    @staticmethod
    def from_euler(position,angles,seq="XYZ"):
        R = Rotation.from_euler(seq,angles)
        return SE3(position,R)

    def as_euler(self,seq="XYZ"):
        return np.array([*self.t,*self.R.as_euler(seq)])

    def __str__(self):
        return str(self.log())

    def copy(self):
        return SE3(self.t.copy(), self.R)
