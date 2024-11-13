import os
import random
from copy import deepcopy
from utils.o3d_utils import *
from utils.visualizer import Visualizer3D
from scipy.spatial import cKDTree

# I/O
# -------------------------------------------
def load_pcd_paths(index:int):
    data_list = ['01_straight_walk',
                '02_straight_duck_walk',
                '03_straight_crawl',
                '04_zigzag_walk',
                '05_straight_duck_walk',
                '06_straight_crawl',
                '07_straight_walk']

    data_root = os.path.join('data', data_list[index], 'pcd')
    pcd_files = os.listdir(data_root)
    pcd_files.sort()
    pcd_files = [os.path.join(data_root, x) for x in pcd_files]
    return pcd_files

def split_pcd_paths(pcd_files:List[str],
                    split_ratio:float=0.5,
                    shuffle=True):
    pcd_files = deepcopy(pcd_files)
    if shuffle:
        random.shuffle(pcd_files)
    split_index = int(len(pcd_files) * split_ratio)
    return pcd_files[:split_index], pcd_files[split_index:]

def load_pcds_from_paths(pcd_paths:List[str],
                              down_sample=True,
                              voxel_size=0.1):
    pcd_list = [load_pcd(file) for file in pcd_paths]
    for pcd in pcd_list:
        points = pcd_to_numpy(pcd)
        non_zero_points = points[~np.all(points == [0, 0, 0], axis=1)]
        pcd.points = o3d.utility.Vector3dVector(non_zero_points)
        
        if down_sample:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            
    return pcd_list

def create_multi_scale_voxels(pcd,
                              max_voxel_size=0.5,
                              count=5):
    queries = np.asarray(pcd.points)
    output = voxel_grid.check_if_included(o3d.utility.Vector3dVector(queries))
    
    
def create_buffer_tree(pcds):
    points_list = []
    for pcd in pcds:
        points = np.asarray(pcd.points)
        points_list.append(points)
    points = np.vstack(points_list)
    tree = cKDTree(points)
    return points, tree

# Preprocess
# -------------------------------------------
def remove_static_points(pcd_target, 
                         points_buffer_tree, 
                         distance_threshold=0.1,
                         match_count=2):
    
    target_points = np.asarray(pcd_target.points)
    static_indices = []
    for i, point in enumerate(target_points):
        indices = points_buffer_tree.query_ball_point(point, distance_threshold)
        if len(indices) >= match_count:
            static_indices.append(i)    

    filtered_pcd = pcd_target.select_by_index(static_indices, invert=True)
    return filtered_pcd

def visualize_with_bounding_boxes(pcd, bounding_boxes, window_name="Filtered Clusters and Bounding Boxes", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    vis.get_render_option().point_size = 4
    vis.get_render_option().background_color = [0, 0, 0]
    vis.run()
    vis.destroy_window()

def filter_outliers(pcd, 
                    nb_points=3, 
                    radius=1.6):
    cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    ror_pcd = pcd.select_by_index(ind)
    return ror_pcd

def remove_road_plane(pcd, 
                      distance_threshold=0.1,
                      ransac_n=3,
                      num_iterations=1000):
    
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    ransac_pcd = pcd.select_by_index(inliers, invert=True)
    return ransac_pcd

def find_3d_bboxes(pcd,
                   eps=0.35,
                   min_points=10):
    
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

    # 필터링 기준 설정
    min_points_in_cluster = 2   # 클러스터 내 최소 포인트 수
    max_points_in_cluster = 999  # 클러스터 내 최대 포인트 수
    min_z_value = -993.5          # 클러스터 내 최소 Z값
    max_z_value = 993.5           # 클러스터 내 최대 Z값
    min_height = -99            # Z값 차이의 최소값
    max_height = 99.0            # Z값 차이의 최대값
    max_distance = 1000.0         # 원점으로부터의 최대 거리

    # 1번, 2번, 3번 조건을 모두 만족하는 클러스터 필터링 및 바운딩 박스 생성
    bboxes_1234 = []
    for i in range(labels.max() + 1):
        cluster_indices = np.where(labels == i)[0]
        if min_points_in_cluster <= len(cluster_indices) <= max_points_in_cluster:
            cluster_pcd = pcd.select_by_index(cluster_indices)
            points = np.asarray(cluster_pcd.points)
            z_values = points[:, 2]
            z_min = z_values.min()
            z_max = z_values.max()
            if min_z_value <= z_min and z_max <= max_z_value:
                height_diff = z_max - z_min
                if min_height <= height_diff <= max_height:
                    distances = np.linalg.norm(points, axis=1)
                    if distances.max() <= max_distance:
                        bbox = cluster_pcd.get_axis_aligned_bounding_box()
                        bbox.color = (1, 0, 0) 
                        bboxes_1234.append(bbox)

    return bboxes_1234

pcd_paths = load_pcd_paths(0)
pcd_buffer, _ = split_pcd_paths(pcd_files=pcd_paths,
                                split_ratio=0.1,
                                shuffle=True)    

buffer_size = len(pcd_buffer)
pcd_buffer = load_pcds_from_paths(pcd_buffer)

for pcd in pcd_buffer:
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name='')
    # #vis.get_render_option().point_size = 4
    # vis.get_render_option().background_color = [0, 0, 0]
    
    # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
    #                                                         voxel_size=0.05)
    # vis.add_geometry([voxel_grid])
    # vis.run()
    # vis.destroy_window()
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.1)

    # Voxel Grid 시각화
    o3d.visualization.draw_geometries([voxel_grid], window_name="Voxelized Point Cloud")



pcd_targets = load_pcds_from_paths(pcd_paths)
_, pcd_buffer_tree = create_buffer_tree(pcd_buffer)
for pcd_target in pcd_targets:
    target_pcd = remove_static_points(pcd_target, 
                                    pcd_buffer_tree,
                                    distance_threshold=0.1,
                                    match_count=int(buffer_size * 0.1))
    target_pcd = filter_outliers(target_pcd)
    target_pcd = remove_road_plane(target_pcd)

    bboxes_3d = find_3d_bboxes(target_pcd,
                            eps=0.8,
                            min_points=5)

    target_pcd.colors = o3d.utility.Vector3dVector(np.array([[1, 1, 1] for _ in range(len(target_pcd.points))]))
    visualize_with_bounding_boxes(pcd_target, bboxes_3d, point_size=2.0)

    



    
    
# for pcd_file in pcd_files:
#     pcd = load_pcd(os.path.join(data_root, pcd_file))
#     points = pcd_to_numpy(pcd)
    
#     vis = Visualizer3D()
#     vis.set_points(points, [1, 1, 1])
#     vis.show()