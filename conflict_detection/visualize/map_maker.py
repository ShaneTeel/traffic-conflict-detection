import folium
from folium.plugins import HeatMap
import webbrowser
import os

from typing import Literal
from numpy.typing import NDArray

class MapMaker:

    def __init__(self, pts:NDArray):

        coords = pts[0]
        cx, cy = coords[:, 0].mean(), coords[:, 1].mean()

        self.m = folium.Map((cx, cy), zoom_start=12, width="100%")

        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr="Esri",
            name="Esri Satellite",
            overlay=False,
            control=True
        ).add_to(self.m)

        self.feature_groups = []

    def generate_heatmap(self, coords:list, name:str="TTC Conflicts"):
        '''
        Description
        -----------
        Generates a heatmap layer as a Folium Feature Group, then adds the Feature Group
        to a Folium Map base-map. 

        Parameters
        ----------
        coords : list (2-dimensional)
            A list of lists containing the lat and lon for each point to be added 
            to the feature group and, subseqeuntly, to the base-map

        name : str, default = "Stay-Points"
            Name of feature group to be added to base-map. 
        
        Returns
        -------
        map_object : folium.Map
            The map object with the heatmap feature group layer added. 
        '''
        heatmap_fg = folium.FeatureGroup(name)

        HeatMap(
            coords, 
            min_opacity=1.0, 
            blur=0, 
            radius=10
            ).add_to(heatmap_fg)

        heatmap_fg.add_to(self.m)

        self.feature_groups.append(name)
        return self.m
    
    def generate_overlay(self, coords:list, popup_info:list[str]=None, color:Literal["red", "blue", "yellow"]="red", name:str="Projection Overaly"):
        '''
        Description
        -----------
        Generates a Folium Feature Group, then adds the Feature Group
        to a Folium Map base-map.

        Parameters
        ----------
        coords : list (2-dimensional)
            A list of lists containing the lat and lon for each point to be added 
            to the feature group and, subseqeuntly, to the base-map
        
        popup_info : list[str], default=None
            List of str objects that will serve as the pop-up info for the icons drawn. 

        name : str, default="Projection Overlay"
            Name of feature group to be added to base-map. 
        
        Returns
        -------
        map_object : folium.Map
            The map object with the feature group layer added. 
        '''
        fg = folium.FeatureGroup(name)

        for pt, popup in zip(coords, popup_info):
            popup = folium.Popup(popup, max_width=300)
            folium.CircleMarker(
                location=pt,
                popup=popup,
                radius=3,
                color=color,
                fill_color=color,
                fill=True,
                ).add_to(fg)

        fg.add_to(self.m)

        self.feature_groups.append(name)
        return self.m

    def add_layer_control(self):
        for key in list(self.m._children.keys()):
            if key.startswith("layer_control"):
                del self.m._children[key]
    
        folium.LayerControl(collapsed=False).add_to(self.m)

    def save_map(self, file_out:str, view:bool=True):
        self.add_layer_control()
        self.m.save(file_out)

        if view:
            webbrowser.open_new_tab(f"file://{os.path.realpath(file_out)}")

        return self.m