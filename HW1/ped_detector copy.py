import os
import random
from copy import deepcopy
from utils.o3d_utils import *
from utils.visualizer import Visualizer3D
from scipy.spatial import cKDTree
from tqdm import tqdm

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
        
    n = len(pcd_files)
    sample_size = int(n * split_ratio)
    interval = n // sample_size  
    sampled_list = pcd_files[::interval]

    if len(sampled_list) > sample_size:
        sampled_list = sampled_list[:sample_size]
            
    return sampled_list

def compute_static_voxels(pcd_buffers,
                          voxel_sizes:List[float]=[0.15, 0.2, 0.3],
                          distance_thresholds:List[tuple[float]]=[(0,40),
                                                                  (40,80),
                                                                  (80,999)],
                          matching_counts:List[int]=[100, 50, 30]):
    print("Computing Static Voxels")
    
    p2v = o3d.geometry.VoxelGrid.create_from_point_cloud
    
    def get_voxel_centers(pcd, voxel_size):
        voxel_grid = p2v(pcd, voxel_size)
        centers = np.array([voxel_grid.get_voxel_center_coordinate(voxel.grid_index) for voxel in voxel_grid.get_voxels()])
        centers = np.round(centers, 2)
        return centers

    static_voxel_grids = []  
    for voxel_size, dist_thres, matching_count in tqdm(zip(voxel_sizes, distance_thresholds, matching_counts)):
        voxel_centers_list = [get_voxel_centers(x, voxel_size) for x in pcd_buffers]
        voxel_centers = np.vstack(voxel_centers_list)
        tree = cKDTree(voxel_centers)
        
        dist = np.linalg.norm(voxel_centers, axis=-1)
        dist_mask = (dist >= dist_thres[0]) & (dist < dist_thres[1])
        voxel_centers_mask = voxel_centers[dist_mask]
    
        distances, counts = tree.query(voxel_centers_mask,
                                       k=matching_count * 5,
                                       distance_upper_bound=voxel_size)
        mask = np.sum(distances < voxel_size, axis=1) >= int(matching_count)
        temp = voxel_centers_mask[mask]
        static_voxel_grid = p2v(points_to_pcd(temp, [0, 0, 0]), voxel_size * 1.15)
        static_voxel_grids.append(static_voxel_grid)      
        
        # o3d.visualization.draw_geometries([static_voxel_grid], window_name="Voxelized Point Cloud")
    
    print("Computing Static Voxels Completed!")
    return static_voxel_grids 


# Preprocess
# -------------------------------------------
def downsample_pcd(pcd,
                   voxel_size=0.2):
    downsample_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return downsample_pcd

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
                   eps=0.6,
                   min_points=5):
    
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
pcd_buffers = split_pcd_paths(pcd_files=pcd_paths,
                              split_ratio=0.2,
                              shuffle=False)    

buffer_size = len(pcd_buffers)
pcd_buffers = load_pcds_from_paths(pcd_buffers)
pcd_targets = load_pcds_from_paths(pcd_paths)

print("Preprocess loaded PCD files")
pcd_buffers = [downsample_pcd(x, 0.15) for x in tqdm(pcd_buffers)]
# pcd_buffers = [filter_outliers(x) for x in tqdm(pcd_buffers)]
# spcd_buffers = [remove_road_plane(x) for x in tqdm(pcd_buffers)]
print("Preprocessing loaded PCD files completed!")

distance_thresholds = [(0, 40), (40, 9999)] 
eps_thresholds = [0.2, 0.5, 0.4]                        
min_points_thresholds = [10, 3, 3]   

voxel_buffers_list = compute_static_voxels(pcd_buffers=pcd_buffers,
                                           voxel_sizes=[0.5, 0.55,],
                                           distance_thresholds=distance_thresholds,
                                           matching_counts=[int(buffer_size * 0.25), int(buffer_size * 0.15)])

for pcd_target in pcd_targets:
    
    pcd_filtered = pcd_target
    
    #pcd_filtered = filter_outliers(pcd_filtered)
    #pcd_filtered = remove_road_plane(pcd_filtered)
    points = pcd_to_numpy(pcd_filtered)
    
    points_isolated = []
    points_static = []
    
    dist = np.linalg.norm(points, axis=-1)
    mask = np.zeros((points.shape[0], ))
    bboxes_3d = []
    
    for voxel_grid, distance_threshold, eps, min_points in zip(voxel_buffers_list, 
                                                               distance_thresholds, 
                                                               eps_thresholds, 
                                                               min_points_thresholds):
        mask_dist = (dist >= distance_threshold[0]) & (dist < distance_threshold[1])
        points_dist = points[mask_dist]
        
        mask_voxel = np.array(voxel_grid.check_if_included(numpy_to_v3v(points_dist)))
        points_isolated.append(points_dist[~mask_voxel])
        points_static.append(points_dist[mask_voxel])
        
        bboxes = find_3d_bboxes(points_to_pcd(points_isolated[-1]),
                                eps=eps,
                                min_points=min_points)
        bboxes_3d.extend(bboxes)
        points_masked = points_dist[~mask_voxel]
        # vis = Visualizer3D()
        # vis.set_points(points_masked)
        # vis.show()
    
    print(bboxes_3d)
    visualize_with_bounding_boxes(pcd_filtered, bboxes_3d, point_size=2.0)