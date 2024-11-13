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
    split_index = int(len(pcd_files) * split_ratio)
    return pcd_files[:split_index], pcd_files[split_index:]

def compute_static_voxels(pcd_buffers,
                          voxel_sizes:List[float]=[0.15],
                          distance_thresholds:List[tuple[float]]=[(0,20),
                                                                  (20,40),
                                                                  (40,999)],
                          distance_multipliers:List[float]=[1.0, 1.25, 1.5],
                          matching_counts:List[int]=[10]):
    print("Computing Static Voxels")
    
    p2v = o3d.geometry.VoxelGrid.create_from_point_cloud
    def get_voxel_centers(pcd, voxel_size):
        voxel_grid = p2v(pcd, voxel_size)
        centers = np.array([voxel_grid.get_voxel_center_coordinate(voxel.grid_index) for voxel in voxel_grid.get_voxels()])
        centers = np.round(centers, 2)
        return centers

    static_voxel_grids = []  
    
    for voxel_size, matching_count in zip(voxel_sizes, matching_counts):
        
        print("Computing Voxel Size {}".format(voxel_size))
        
        voxel_centers_list = [get_voxel_centers(x, voxel_size) for x in pcd_buffers]
        voxel_centers = np.vstack(voxel_centers_list)
        tree = cKDTree(voxel_centers)
        
        static_voxel_centers = []
        for distance_threshold, distance_multiplier in zip(distance_thresholds, distance_multipliers):
            dist = np.linalg.norm(voxel_centers)
            mask = (dist >= distance_threshold[0]) & (dist < distance_threshold[1])
            voxel_centers_mask = voxel_centers[mask]
        
            distances, counts = tree.query(voxel_centers_mask,
                                           k=matching_count,
                                           distance_upper_bound=voxel_size * distance_multiplier)
            mask = np.sum(distances < voxel_size, axis=1) >= matching_count
            temp = voxel_centers_mask[mask]
            static_voxel_centers.append(temp)
        static_voxel_centers = np.vstack(static_voxel_centers)
        static_voxel_grid = p2v(points_to_pcd(static_voxel_centers, [0, 0, 0]), voxel_size)
        static_voxel_grids.append(static_voxel_grid)        
        
        # static_voxel_centers = []
        # for voxel_center in tqdm(voxel_centers):
        #     indices = tree.query_ball_point(voxel_center, voxel_size)
        #     if len(indices) >= matching_count:
        #         static_voxel_centers.append(voxel_center)
        # static_voxel_centers = np.vstack(static_voxel_centers)
        # voxel_grid = p2v(points_to_pcd(static_voxel_centers, [0,0,0]), voxel_size)
        # static_voxel_grids.append(voxel_grid)
        
        # distances, counts = tree.query(voxel_centers, k=matching_count, distance_upper_bound=voxel_size)
        # mask = np.sum(distances < voxel_size, axis=1) >= matching_count
        # static_voxel_centers = voxel_centers[mask]
        # static_voxel_grid = p2v(points_to_pcd(static_voxel_centers, [0, 0, 0]), voxel_size)
        # static_voxel_grids.append(static_voxel_grid)
    
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


# x1, y1, z1 = 0.0, 0.0, 0.0
# x2, y2, z2 = 51.0, 51.0, 51.0
# interval = 1
# x = np.linspace(x1, x2, int(abs(x2-x1)/interval) + 1)
# y = np.linspace(y1, y2, int(abs(y2-y1)/interval) + 1)
# z = np.linspace(z1, z2, int(abs(z2-z1)/interval) + 1)

# # meshgrid를 사용하여 모든 가능한 (x, y, z) 조합 생성
# grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')

# # 생성된 포인트들을 1차원 배열로 변환
# points = np.vstack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel())).T
# print("BB")
# print(points.shape)
# print("AA")
#vis = Visualizer3D()
#vis.set_points(points)
#vis.show()

pcd_paths = load_pcd_paths(0)
pcd_buffers, pcd_targets = split_pcd_paths(pcd_files=pcd_paths,
                                           split_ratio=0.3,
                                           shuffle=True)    

buffer_size = len(pcd_buffers)
pcd_buffers = load_pcds_from_paths(pcd_buffers)
pcd_targets = load_pcds_from_paths(pcd_paths)

print("Preprocess loaded PCD files")
pcd_buffers = [downsample_pcd(x, 0.1) for x in tqdm(pcd_buffers)]
# pcd_buffers = [filter_outliers(x) for x in tqdm(pcd_buffers)]
print("Preprocessing loaded PCD files completed!")

voxel_buffers_list = compute_static_voxels(pcd_buffers=pcd_buffers,
                                           voxel_sizes=[0.15,],
                                           matching_counts=[10,])

for pcd_target in pcd_targets:
    
    # o3d.visualization.draw_geometries([voxel_buffers_list[0]], window_name="Voxelized Point Cloud")
    
    pcd_filtered = filter_outliers(pcd_target)
    pcd_filtered = remove_road_plane(pcd_filtered, 0.2, 3, 2000)
    # pcd_filtered = pcd_target
    points = pcd_to_numpy(pcd_filtered)
    points_filtered = np.array(voxel_buffers_list[0].check_if_included(numpy_to_v3v(points)))
    
    # for i in [1, 2]:
    #     b = np.array(voxel_buffers_list[i].check_if_included(numpy_to_v3v(points)))
    #     points_filtered = points_filtered | b
    
    points_org = points[np.array(points_filtered)]
    points_filtered = points[~np.array(points_filtered)]
    pcd_filtered.points = numpy_to_v3v(points_filtered)
    
    vis = Visualizer3D()
    vis.set_points(points_filtered, [1, 0, 0])
    vis.set_points(points_org)
    vis.show()
    # bboxes_3d = find_3d_bboxes(pcd_filtered,
    #                         eps=0.3,
    #                         min_points=5)

    # pcd_filtered.colors = o3d.utility.Vector3dVector(np.array([[1, 1, 1] for _ in range(len(pcd_filtered.points))]))
    # visualize_with_bounding_boxes(pcd_filtered, bboxes_3d, point_size=2.0)

# for voxel_grid in voxel_buffers_list:
#     # pcd = remove_road_plane(pcd)
#     o3d.visualization.draw_geometries([voxel_grid], window_name="Voxelized Point Cloud")

# for pcd in pcd_buffers:
    
#     # print(pcd.get_max_bound(), pcd.get_min_bound(), pcd.get_center())
#     voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)
#     print(voxel_grid.origin)
    # pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
    #       center=pcd.get_center())
    
    # points = pcd_to_numpy(pcd)
    # vis = Visualizer3D()
    # vis.set_points(points, [1, 1, 1])
    # vis.add_axis()
    # vis.add_axis(voxel_grid.origin)
    # vis.show()

# voxel_buffers = compute_static_voxels(pcd_buffers,
#                                       voxel_sizes=[0.05])


# for pcd in pcd_buffer:
#     pcd = remove_road_plane(pcd)
#     pcd = filter_outliers(pcd)
#     voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)
#     o3d.visualization.draw_geometries([voxel_grid], window_name="Voxelized Point Cloud")

# pcd_targets = load_pcds_from_paths(pcd_paths)
# _, pcd_buffer_tree = create_buffer_tree(pcd_buffer)
# for pcd_target in pcd_targets:
#     target_pcd = remove_static_points(pcd_target, 
#                                     pcd_buffer_tree,
#                                     distance_threshold=0.1,
#                                     match_count=int(buffer_size * 0.1))
#     target_pcd = filter_outliers(target_pcd)
#     target_pcd = remove_road_plane(target_pcd)

#     bboxes_3d = find_3d_bboxes(target_pcd,
#                             eps=0.8,
#                             min_points=5)

#     target_pcd.colors = o3d.utility.Vector3dVector(np.array([[1, 1, 1] for _ in range(len(target_pcd.points))]))
#     visualize_with_bounding_boxes(pcd_target, bboxes_3d, point_size=2.0)

# for pcd_file in pcd_files:
#     pcd = load_pcd(os.path.join(data_root, pcd_file))
#     points = pcd_to_numpy(pcd)
    
#     vis = Visualizer3D()
#     vis.set_points(points, [1, 1, 1])
#     vis.show()