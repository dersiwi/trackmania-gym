from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as rot
from tminterface.structs import CheckpointData, SimStateData, CheckpointTime
from scipy.spatial.transform import Rotation 
import random
# we assume that the standard rotation is 
X = np.array([1,0,0]) # x = [1 0 0]
Y = np.array([0,1,0]) # y = [0 1 0]
Z = np.array([0,0,1]) # z = [0 0 1]
# where z points forward , y upwards and x to the side 
SIM_HAS_DYNA = 0x2  

class RandomRespawnManager:

    def __init__(self, reference_line):
        self.reference_line : np.ndarray = reference_line
        self.n_reference_points : int = self.reference_line.shape[0]
        self.current_reference_line_point = self.reference_line[0]

    def make_ssD_from_ref_point(self,ssD:SimStateData ):
        """
        Constructs a SimStateData object given reference point index. The new position corresponds 
        to the possition fo the ref_point_idx. The orientation of the resulting state is computed 
        such that the agent faces in the direction of the track — from the current reference point
        toward the next one — effectively pointing toward the finish line.

        Parameters:
            ref_point_idx (int): Index of the reference point to teleport to. 
        """
        print("rotation matrix before reset:", ssD.rotation_matrix)
        print("yaw pitch roll before reset:", ssD.yaw_pitch_roll)

        copy_ssD = ssD

        # randomly select a valid index (not the last one)
        ref_point_idx = np.random.randint(0, self.n_reference_points - 1)
        current_position = self.reference_line[ref_point_idx]
        next_position = self.reference_line[ref_point_idx + 1]

        # compute the forward direction vector from the current point to the next.
        new_z_direction = next_position - current_position

        # compute the rotation matrix such that the new z-direction points in direction (next_position - current_position)
        rotation_matrix = RandomRespawnManager.make_rotation_from_forward(forward=new_z_direction)
        #rotation_matrix = RandomRespawnManager.make_rotation_v1(Z,new_z_direction)

        # transform rotation matrix into euler angles and quaternion
        R = Rotation.from_matrix(rotation_matrix)
        angles = R.as_euler('yxz', degrees=False)
        quat = R.as_quat()
        
        # setting all the relevant SimStateData fields for respawning with new position and orientation
        # TODO check which fields are unnecessary
        copy_ssD.flags |= SIM_HAS_DYNA

        copy_ssD.position = current_position #+ np.array([0,5,0])

        copy_ssD.velocity =  np.zeros(3)
        copy_ssD.dyna.current_state.angular_speed =  np.zeros(3)
        copy_ssD.dyna.current_state.angular_velocity = np.zeros(3)
        copy_ssD.dyna.current_state.velocity = np.zeros(3) 

        copy_ssD.dyna.current_state.rotation = rotation_matrix
        copy_ssD.rotation_matrix = rotation_matrix
        copy_ssD.dyna.prev_state.rotation = rotation_matrix
        copy_ssD.dyna.previous_state.rotation = rotation_matrix
        copy_ssD.dyna.temp_state.rotation = rotation_matrix

        copy_ssD.dyna.current_state.quat = quat.tolist()
        copy_ssD.dyna.prev_state.quat = quat.tolist()
        copy_ssD.dyna.previous_state.quat = quat.tolist()
        copy_ssD.dyna.temp_state.quat = quat.tolist()

        copy_ssD.dyna.current_state_yaw_pitch_roll = angles

        #RandomRespawnManager.draw_rotation_and_identity(rotation_matrix,direction=direction)
        print("rotation matrix after reset:", copy_ssD.rotation_matrix)
        print("yaw pitch roll after reset:", copy_ssD.yaw_pitch_roll)
        print("manual computed euler angles:",angles)
        print("quat set to:", copy_ssD.dyna.current_state.quat.to_numpy())
        return copy_ssD


    @staticmethod
    def make_rotation_v0(a,b):
        """
        This functions creates a rotation matrix which rotates vector a onto  vector b
        from  https://math.stackexchange.com/a/2672702
        """
        # normalize vectors 
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        c = a+b
        d =  np.outer(c,c)/ np.dot(c,c)
        return 2.*d - np.eye(d.shape[0])
    
    @staticmethod
    def make_rotation_v1(a, b):
        """
        This function creates a rotation matrix that rotates vector `a` onto vector `b`
        using the method from https://math.stackexchange.com/a/476311.
        """
        # Normalize vectors
        assert a.shape[-1] == 3 and b.shape[-1] == 3
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)

        cos_theta = np.dot(a, b)
        v = np.cross(a, b)
        s = np.linalg.norm(v)

        # Handle special case: a and b are opposite
        if np.isclose(cos_theta, -1.0):
            # Find an orthogonal vector to `a` to construct a 180° rotation
            orthog = np.array([1, 0, 0]) if not np.allclose(a, [1, 0, 0]) else np.array([0, 1, 0])
            axis = np.cross(a, orthog)
            axis = axis / np.linalg.norm(axis)
            # Use Rodrigues formula with theta=pi
            K = np.array([[0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]])
            R = np.eye(3) + 2 * K @ K  # since sin(pi) = 0, (1 - cos(pi)) = 2
            return R

        # Skew-symmetric matrix of v
        v_skew = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

        # Rodrigues' rotation formula
        R = np.eye(3) + v_skew + (v_skew @ v_skew) * (1 / (1 + cos_theta))
        return R
    
    @staticmethod
    def make_rotation_from_forward(forward: np.ndarray, up_hint: np.ndarray = np.array([0, 1, 0])):
        """
        Constructs a right-handed rotation matrix from a forward vector and an up hint.
        Ensures a consistent and upright car orientation.
        """
        forward = forward / np.linalg.norm(forward)
        right = np.cross(up_hint, forward)
        if np.linalg.norm(right) < 1e-6:
            # Forward and up_hint are parallel — pick another up
            up_hint = np.array([1, 0, 0])
            right = np.cross(up_hint, forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)

        # Construct rotation matrix: columns are right (X), up (Y), forward (Z)
        return np.column_stack((right, up, forward))

    @staticmethod   
    def draw_rotation_and_identity(rotation_matrix, origin=np.array([0, 0, 0]),direction=np.array([0, 0, 0])):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Length of arrows
        length = 1.0
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Live Rotation Matrix Axes")
        # Initial dummy arrows
        ax.quiver(0, 0, 0, 1, 0, 0, linestyle = '--',color='r', label="X-axis")
        ax.quiver(0, 0, 0, 0, 1, 0, linestyle = '--', color='g', label="Y-axis")
        ax.quiver(0, 0, 0, 0, 0, 1, linestyle = '--' ,color='b', label="Z-axis")

        origin = np.array([0, 0, 0])
        x_axis = rotation_matrix[:, 0]
        y_axis = rotation_matrix[:, 1]
        z_axis = rotation_matrix[:, 2]

        ax.quiver(*origin, *x_axis, color='r', label="X-axis")
        ax.quiver(*origin, *y_axis, color='g', label="Y-axis")
        ax.quiver(*origin, *z_axis, color='b', label="Z-axis")
        dir_norm = direction / np.linalg.norm(direction)
        ax.quiver(*origin, *dir_norm, color='purple', length=2,alpha =0.5, label='Direction')

        ax.legend()
        plt.show()