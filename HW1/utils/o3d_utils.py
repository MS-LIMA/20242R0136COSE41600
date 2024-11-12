import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from typing import List

v3d = o3d.utility.Vector3dVector

def points_to_pcd(points:np.ndarray,
                  point_color:List[float]=[1,1,1]):
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = v3d(points)
    pcd.colors= v3d(np.array([point_color for _ in range(len(pcd.points))]))
    
    return pcd

def load_pcd(pcd_path:str):
    return o3d.io.read_point_cloud(pcd_path)

def pcd_to_numpy(pcd):
    return np.asarray(pcd.points)