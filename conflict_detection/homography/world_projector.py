import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Literal


class WorldProjector:
    '''
    Description
    -----------
    Image preprocessing class that performs homography on an array of points that represent pixel coordinates in image space.
    The points are projected into BEV (bird's eye view) space so that the points representing lane lines appear parallel. 
    This is done to improve fitting. 

    Parameters
    ----------
    src_pts : NDArray
        Four pixel coordinates for points in image space.

    dst_pts : NDArray
        Four real-world coordinates that correspond to the four pixel coordinates.
    '''
    def __init__(self, src_pts:NDArray, dst_pts:NDArray):
        '''
        Parameters
        ----------
        src_pts : NDArray
            Four pixel coordinates for points in image space.
            
        dst_pts : NDArray
            Four real-world coordinates that correspond to the four pixel coordinates.

        At Initialization
        -----------------
        src_pts / dst_pts shape, dtype, and point order validated
        Homography matrix / Inverse Homography matrix are computed
        '''
        self.src_pts = self._pts_validation(src_pts)
        self.dst_pts = self._pts_validation(dst_pts)
        self.H = self._calc_H_mat(self.src_pts, self.dst_pts)
        self.H_I = np.linalg.inv(self.H)

    def project(self, pts:NDArray, direction:Literal["forward", "backward"]):
        """
        Transform points between camera space and real-world geography space.
        
        Parameters
        ----------
        pts : NDArray, shape (n_points, 2)
            Points to transform in (x, y) pixel coordinates
        direction : {"forward", "backward"}
            "forward" = camera -> real-world, "backward" = real-world -> camera
            
        Returns
        -------
        transformed_pts : NDArray, shape (n_points, 2)
            Points in target coordinate system
            
        Notes
        -----
        Uses homography transformation via `cv2.perspectiveTransform()`.
        Forward transform converts pixel coordinates to real-world lat/lon coordinates.
        Backward transform reverts lat/lon coords to pixel coords.
        """
        if len(pts) == 0:
            return pts
        
        pts = np.array([pts], dtype=np.float32)
        if pts.ndim == 2:
            pts = pts.reshape(1, -1, 2)
        
        m = self.H if direction == "forward" else self.H_I

        pts = cv2.perspectiveTransform(pts, m)
        return pts.reshape(-1, 2).astype(np.int32)
    
    def _calc_H_mat(self, src_pts:NDArray, dst_pts:NDArray):
        """
        Compute homography matrix using Direct Linear Transformation (DLT).
        
        Solves for 3x3 homography mapping source image points to real-world destination points. Constructs 9x9 system of linear equations from
        point correspondences.
        
        Returns
        -------
        H : NDArray, shape (3, 3)
            Homography matrix normalized so H[2,2] = 1
            
        Notes
        -----
        Each point correspondence contributes 2 equations to the system.
        With 4 point pairs, we get 8 equations for 8 unknowns (9th fixed to 1).
        """
        A = np.zeros((9, 9), dtype=np.float32)
        A[8, 8] = 1

        ui_vi = src_pts[:, :, :].reshape(-1, 2)
        xi_yi = dst_pts[:, :, :].reshape(-1, 2)
        DOF = list(range(0, 8, 2))

        for dof, (ui, vi), (xi, yi) in zip(DOF, ui_vi, xi_yi):
            A[dof,:] = np.array([-ui, -vi, -1, 0, 0, 0, ui * xi, vi * xi, xi])
            A[dof+1,:] = np.array([0, 0, 0, -ui, -vi, -1, ui * yi, vi * yi, yi])

        b = np.array([0]*8 + [1], dtype=np.float32)

        H = np.linalg.solve(A, b).reshape(3, 3)

        return H / H[2, 2]
    
    def _pts_validation(self, pts:NDArray):
        '''
        Description
        -----------
        Private method called upon during object initialization to validate the shape, enforce a point order, and convert the pts dtype to `np.float32`.

        The point order and shape is as follows:
            [[
            [Bottom Left]
            [Bottom Right]
            [Top Right]
            [Top Left]
            ]]

        Parameters
        ----------
        pts : NDArray
            Four points represented cooridnate pairs.

        Returns
        --------------
        pts : NDArray
            pts validated, reshaped, and type casted.
        '''
        if pts.shape != (1, 4, 2):
            try:
                pts = pts.reshape(1, 4, 2)
            except Exception as e:
                raise ValueError(e)

        points = pts[0]
        
        avg_y = points[:, 1].mean()
        bottom = points[points[:, 1] > avg_y]
        top = points[points[:, 1] <= avg_y]

        bottom_left, bottom_right = bottom[np.argsort(bottom[:, 0])]
        top_left, top_right = top[np.argsort(top[:, 0])]

        return np.array([[bottom_left],
                         [bottom_right],
                         [top_right],
                         [top_left]], 
                         dtype=np.float32).reshape(1, 4, 2)