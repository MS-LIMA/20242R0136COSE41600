import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from tqdm import tqdm

v3d = o3d.utility.Vector3dVector

def points_to_pcd(points:np.ndarray,
                  point_color:List[float]=[1,1,1]):
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = v3d(points)
    pcd.colors= v3d(np.array([point_color for _ in range(len(pcd.points))]))
    
    return pcd

def load_pcd(pcd_path:str):
    return o3d.io.read_point_cloud(pcd_path)

def load_pcds_from_paths(pcd_paths:List[str],
                              down_sample=True,
                              voxel_size=0.1):
    print("Loading PCD files from paths")
    pcd_list = [load_pcd(file) for file in pcd_paths]
    for i, pcd in enumerate(tqdm(pcd_list)):
        points = pcd_to_numpy(pcd)
        non_zero_points = points[~np.all(points == [0, 0, 0], axis=1)]
        pcd.points = v3d(non_zero_points)
        pcd_list[i] = pcd
    print("Loading PCD files completed!")
    return pcd_list

def pcd_to_numpy(pcd):
    return np.asarray(pcd.points)

def numpy_to_v3v(points:np.ndarray):
    return v3d(points)