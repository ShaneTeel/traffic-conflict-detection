import numpy as np
import folium

class MapMaker:

    def __init__(self, world_pts:np.ndarray):

        coords = world_pts[0]
        cx, cy = coords[:, 0].mean(), coords[:, 1].mean()

        self.m = folium.Map((cx, cy), zoom_start=5, width="100%")

        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr="Esri",
            name="Esri Satellite",
            overlay=False,
            control=True
        ).add_to(self.m)

        self.feature_groups = []
    
    def generate_overlay(self, coords:list, popup_info:list[str], name:str):
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
        
        popup_info : list[str]
            List of str objects that will serve as the pop-up info for the icons drawn. 

        name : str
            Name of feature group to be added to base-map. 
        
        Returns
        -------
        map_object : folium.Map
            The map object with the feature group layer added. 
        '''
        fg = folium.FeatureGroup(name)

        for pt, popup in zip(coords, popup_info):

            folium.CircleMarker(
                pt,
                popup=popup,
                radius=1
                ).add_to(fg)

        fg.add_to(self.m)

        self.feature_groups.append(name)
        return self.m

    def add_layer_control(self):
        for key in list(self.m._children.keys()):
            if key.startswith("layer_control"):
                del self.m._children[key]
    
        folium.LayerControl(collapsed=False).add_to(self.m)