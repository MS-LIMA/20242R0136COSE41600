import os
import random
from copy import deepcopy
from utils.o3d_utils import *
from utils.visualizer import Visualizer3D, create_video
from scipy.spatial import cKDTree
from tqdm import tqdm
from typing import List, Tuple

# I/O
# ===================================================== #
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


# I/O
# ===================================================== #




def compute_static_voxels(pcd_buffers,
                          voxel_sizes:List[float]=[0.15, 0.2, 0.3],
                          distance_thresholds:List[Tuple[float]]=[(0,40),
                                                                  (40,80),
                                                                  (80,999)],
                          matching_counts:List[int]=[100, 50, 30]):
    print("Computing Static Voxels")
    
    points_list = [pcd_to_numpy(x) for x in pcd_buffers]
    points = np.vstack(points_list)
    dist = np.linalg.norm(points, axis=-1)
    
    static_voxel_grids = []  
    for voxel_size, dist_thres, matching_count in tqdm(zip(voxel_sizes, distance_thresholds, matching_counts)):
        
        dist_mask = (dist >= dist_thres[0]) & (dist < dist_thres[1])
        points_in_dist = points[dist_mask]
        # voxel_grid = pcd_to_voxel(points_to_pcd(points_in_dist), voxel_size)
        # voxels = voxel_grid.get_voxels()
        # voxel_centers = get_voxel_centers(voxel_grid)
        voxel_indices = np.floor(points_in_dist / voxel_size).astype(int)
        unique_voxels, counts = np.unique(voxel_indices, axis=0, return_counts=True)
        
        filtered_voxels = unique_voxels[counts >= matching_count]
        filtered_voxel_centers = filtered_voxels * voxel_size + (voxel_size / 2)
        
        voxel_grid = pcd_to_voxel(points_to_pcd(filtered_voxel_centers), voxel_size)
        static_voxel_grids.append(voxel_grid)
        
        print(f"Total Voxels: {len(unique_voxels)}")
        print(f"Filtered Voxels (>=5 points): {len(filtered_voxels)}")

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
                      distance_threshold=0.2,
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
        
    bboxes_1234 = []
    for i in range(labels.max() + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster_pcd = pcd.select_by_index(cluster_indices)
        points = np.asarray(cluster_pcd.points)
        z_values = points[:, 2]
        z_min = z_values.min()
        z_max = z_values.max()
        bbox = cluster_pcd.get_axis_aligned_bounding_box()
        bbox.color = (1, 0, 0) 
        bboxes_1234.append(bbox)

    return bboxes_1234, labels

def filter_3d_bboxes(bboxes, labels):
    filtered_bboxes = []
    for i, bbox in enumerate(bboxes):
        cluster_indices = np.where(labels == i)[0]
        
        include = False
        if len(cluster_indices) >= 3:
            include = True
            
        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()
        center = (min_bound + max_bound) / 2
        filtered_bboxes.append(bbox)
        
    return filtered_bboxes

def refine_3d_bboxes(bboxes):
    refined_boxes = []
    for bbox in bboxes:
        # 기존 바운딩 박스 정보
        center = bbox.get_center()
        extent = bbox.get_extent()  # Bounding Box 크기 (dx, dy, dz)

        # 사람 크기 추정: 상체 높이와 위치를 기준으로 전체 키 추정
        upper_height = extent[2]  # y 방향 높이 (상체 높이)
        full_height = max(1.6, upper_height * 2.0)  # 전체 사람 키 (1.6m 이상으로 가정)
        full_width = max(0.7, extent[0])  # x 방향 폭 (최소 0.5m 이상)
        full_depth = max(0.7, extent[1])  # y 방향 깊이 (최소 0.4m 이상)
    
        # 바운딩 박스 확장: 아래쪽으로 높이를 추가
        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()

        # y축 (높이) 조정: 아래로 확장
        min_bound[2] -= full_height - upper_height  # 아래로 확장
        max_bound[2] += 0.1  # 상체 위쪽 약간 여유 추가
        x_center = (min_bound[0] + max_bound[0]) / 2
        y_center = (min_bound[1] + max_bound[1]) / 2
        min_bound[0] = x_center - full_width / 2  # x축 좌측 확장
        max_bound[0] = x_center + full_width / 2  # x축 우측 확장
        min_bound[1] = y_center - full_depth / 2  # y축 앞 확장
        max_bound[1] = y_center + full_depth / 2  # y축 뒤 확장

        # 새 바운딩 박스 생성
        refined_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        refined_bbox.color = (1, 0, 0) 
        refined_boxes.append(refined_bbox)
    return refined_boxes

index = 0
pcd_paths = load_pcd_paths(index)
pcd_buffers = split_pcd_paths(pcd_files=pcd_paths,
                              split_ratio=1.0,
                              shuffle=False)    

buffer_size = len(pcd_buffers)
pcd_buffers = load_pcds_from_paths(pcd_buffers)
pcd_targets = load_pcds_from_paths(pcd_paths)

print("Preprocess loaded PCD files")
# pcd_buffers = [downsample_pcd(x, 0.05) for x in tqdm(pcd_buffers)]
# pcd_buffers = [filter_outliers(x) for x in tqdm(pcd_buffers)]
# pcd_buffers = [remove_road_plane(x) for x in tqdm(pcd_buffers)]
print("Preprocessing loaded PCD files completed!")

voxel_sizes = [0.25, 0.35]
distance_thresholds = [(0, 40), (40, 9999)] 
matching_counts=[int(buffer_size * 0.1), int(buffer_size * 0.1)]
eps_thresholds = [0.3, 0.8]                        
min_points_thresholds = [10, 3]   

voxel_buffers_list = compute_static_voxels(pcd_buffers=pcd_buffers,
                                           voxel_sizes=voxel_sizes,
                                           distance_thresholds=distance_thresholds,
                                           matching_counts=matching_counts)

print("Inferencing!")   
for i, pcd_target in tqdm(enumerate(pcd_targets)):
    
    pcd_filtered = pcd_target
    
    #pcd_filtered = filter_outliers(pcd_filtered)
    pcd_filtered = remove_road_plane(pcd_filtered)
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
        
        bboxes, labels = find_3d_bboxes(points_to_pcd(points_isolated[-1]),
                                eps=eps,
                                min_points=min_points)
        # bboxes = refine_3d_bboxes(bboxes)
        bboxes = filter_3d_bboxes(bboxes, labels)
        bboxes_3d.extend(bboxes)
        points_masked = points_dist[~mask_voxel]
        
        # vis = Visualizer3D()
        # vis.set_points(points_masked)
        # vis.show()
    
    #points_isolated = np.vstack(points_isolated)
    #points_static = np.vstack(points_static)
    
    vis = Visualizer3D()
    vis.set_points(points)
    vis.add_bboxes_3d(bboxes_3d)
    vis.set_camera_params('cam.json')
    vis.show()
    # vis.save_to_image('01', i)

create_video('01', '01.avi', 30)