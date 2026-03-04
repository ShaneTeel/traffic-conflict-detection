import numpy as np
import utm

from typing import Literal
from numpy.typing import NDArray

from .click_points import ClickPoints
from .world_projector import WorldProjector

from conflict_detection.visualize import MapMaker
from conflict_detection.utils import get_logger, MAE, RMSE

logger = get_logger(__name__)

class InversePerspectiveMapper:
    
    def __init__(self, frame:NDArray, world_pts:NDArray, img_pts:NDArray=None):

        self.projector = self._initialize_projector(frame, world_pts, img_pts)
        self.control_pts = world_pts
        self.map_maker = MapMaker(world_pts)
        self.rmse_dict = None
        self.mae_dict = None
        self.Geo_true = None
        self.Geo_pred = None

    def _initialize_projector(self, frame:NDArray, world_pts:NDArray, img_pts:NDArray=None):
        if img_pts is None:
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

        lat, lon = utm.to_latlon(easting=flat[:, 0], northing=flat[:, 1], zone_number=self.zone_num, zone_letter=self.zone_let)

        return np.stack([lat, lon], axis=1, dtype=np.float32).reshape(pts.shape)  

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
    
    def evaluate_projector(self, frame:NDArray, world_val_pts:NDArray, img_val_pts:NDArray=None):
        results_dict = {}
        if img_val_pts is None:
            img_val_pts = self._request_points(frame, world_val_pts.shape[1], "val_pts")
    
        Geo_true = self._latlon_to_utm(world_val_pts)

        Geo_pred = self.projector.project(img_val_pts)

        logger.debug(f"WorldProjector geo-referenced the provided validation points as:\n{self._utm_to_latlon(Geo_pred)}")

        rmse_str, rmse_dict = RMSE(Geo_true[0], Geo_pred[0])
        mae_str, mae_dict = MAE(Geo_true[0], Geo_pred[0])

        logger.info(f"""\n
\033[01m\033[04mInverse Persepective Mapping metrics\033[0m:
{rmse_str}
{mae_str}
""")     
        results_dict["rmse"] = rmse_dict
        results_dict["mae"] = mae_dict
        Geo_pred = self._utm_to_latlon(Geo_pred)   
        return Geo_pred, results_dict
    
    def inspect_results(self, Geo_true:NDArray, Geo_pred:NDArray, results:dict[dict], file_out:str):
        rmse_dict, mae_dict = results["rmse"], results["mae"] 

        true_popups = [f"Geo True Pt. {i}" for i in range(len(Geo_true[0]))]
        self.add_layer(Geo_true[0], true_popups, color="blue", name="Ground Truth")

        pred_popups = [
            f"""
<b><u>Geo Pred. Pt. {i}</b></u><br>
<b>East Error^2</b>: {inner_rmse["East Error^2"]:.2f}<br>
<b>East |Error|</b>: {inner_mae["North |Error|"]:.2f}<br>
<b>North Error^2</b>: {inner_rmse["North Error^2"]:.2f}<br>
<b>North |Error|</b>: {inner_mae["North |Error|"]:.2f}<br>
<b>Total Sqrd. Error</b>: {inner_rmse["Total Error"]:.2f}<br>
<b>Total Abs. Error</b>: {inner_mae["Total Error"]:.2f}
""" for i, (inner_rmse, inner_mae) in enumerate(zip(rmse_dict.values(), mae_dict.values()))
        ]

        self.add_layer(Geo_pred[0], pred_popups, color="red", name="Projection")

        self.add_layer(self.control_pts[0], [f"Ground Control Pt. # {i}" for i in range(4)], color="yellow", name="Ground Control Points")
        
        return self.save_map(file_out, view=True)
    
    def add_layer(self, coords:list | NDArray, popups:list[str]=None, color:Literal["red", "blue", "yellow"]="red", name:str=None):
        self.map_maker.generate_overlay(coords, popups, color, name)

    def save_map(self, file_out:str, view:bool=True):
        return self.map_maker.save_map(file_out, view)

    def map_persepective(self, pts:NDArray):
        pts = self.projector.project(pts, "forward")
        return self._utm_to_latlon(pts)
    
    def generate_heatmap(self, coords:NDArray, file_out:str, name:str="TTC Conflicts"):
        self.map_maker.generate_heatmap(coords, name)
        return self.save_map(file_out, view=True)