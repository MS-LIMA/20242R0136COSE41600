import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from utils.o3d_utils import *
from typing import List

class Visualizer3D():
    def __init__(self,
                 point_size:float=2.0,
                 window_name:str='Default Visualizer'):
        vis = o3d.visualization.Visualizer()
        
        self.vis = vis
        vis.create_window(window_name=window_name)
        vis.get_render_option().point_size = point_size
        vis.get_render_option().background_color = [0, 0, 0]
        
    def show(self):
        vis = self.vis
        
        vis.run()
        vis.destroy_window()
    
    def set_points(self,
                   points:np.ndarray,
                   point_color:List[float]=[1,1,1]):
        vis = self.vis

        pcd = points_to_pcd(points, point_color)
        vis.add_geometry(pcd)
    
    def add_axis(self, 
                 origin:List[float]=[0, 0, 0]):
        vis = self.vis
        
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=origin)
        vis.add_geometry(axis)
        
    def add_bboxes_3d(self,
                      bboxes:np.ndarray):
        for bbox in bounding_boxes:
            vis.add_geometry(bbox)
    
    