import numpy as np
import utm

from typing import Literal
from numpy.typing import NDArray

from .click_points import ClickPoints
from .world_projector import WorldProjector

from conflict_detection.utils import get_logger

logger = get_logger(__name__)

class InversePerspectiveMapper:
    
    def __init__(self, frame:NDArray, world_pts:NDArray, val_pts:NDArray):

        self.projector = self._initialize_projector(frame, world_pts)
        self.test_src_pts = self._request_points(frame, val_pts.shape[1], "val_pts")
        self.test_dst_pts = self._latlon_to_utm(val_pts)
        self.eval = self._evaluate_projector(self.test_src_pts, self.test_dst_pts)

    def _initialize_projector(self, frame:NDArray, world_pts:NDArray):
        img_pts = self._request_points(frame, 4, "world_pts")
        dst_pts = self._latlon_to_utm(world_pts)
        src_pts = self._shape_validation(img_pts)

        return WorldProjector(src_pts, dst_pts)
    
    def _request_points(self, frame:NDArray, n_points:int, ref_pts:Literal["world_pts", "val_pts"]):
        logger.info(f"""Select {n_points} points that correspond to the real-world coordinates passed as the argument for '{ref_pts}'.
                Ensure the image points are selected in the same order as the coordinates provided for '{ref_pts}'. Press 'ESC' when complete.""")

        click = ClickPoints(frame, "Point Selection")
        click.draw()

        img_pts = np.array(click.get_pts(), dtype=np.float32)

        logger.debug(f"User selected the following points in image space:\n{img_pts}")
        
        return img_pts

    def _utm_to_latlon(self, pts:NDArray):

        flat = pts.reshape(-1, 2)

        lat, lon = utm.to_latlon(flat[:, 0], flat[:, 1], zone_number=self.zone_num, zone_letter=self.zone_let)

        return np.array([lat, lon], dtype=np.float32).reshape(pts.shape)  

    def _latlon_to_utm(self, dst_pts:NDArray):

        flat = dst_pts.reshape(-1, 2)
        utm_pts = []

        for i, (lat, lon) in enumerate(flat):

            e, n, zn, zl = utm.from_latlon(lat, lon)
            if i == 0:
                self.zone_num = zn
                self.zone_let = zl
            
            utm_pts.append([e, n])

        return np.array(utm_pts, dtype=np.float32).reshape(dst_pts.shape)            

    def _shape_validation(self, pts:NDArray):
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
            
        return pts
    
    def _evaluate_projector(self, test_src_pts:NDArray, test_dst_pts:NDArray):
        rmse = []

        pts = self.projector.project(test_src_pts)

        logger.debug(f"WorldProjector geo-referenced the provided validation points as:\n{pts}")

        Geo1 = test_dst_pts[0]
        Geo2 = pts[0]

        rmse = np.sqrt(
            np.mean(
                [(x2 - x1)**2 + (y2-y1)**2 for (x1, y1), (x2, y2) in zip(Geo1, Geo2)]
            )
        )

        logger.info(f"Homography Matrix achieves a Root Mean Squared Error score of {rmse:.2f}.")
        return rmse
    
    def map_persepective(self, pts:NDArray):
        pts = self.projector.project(pts, "forward")
        return self._utm_to_latlon(pts)