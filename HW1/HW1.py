from fileinput import filename
import os
import random
from copy import deepcopy
from utils.utils import Timer
from utils.o3d_utils import *
from utils.visualizer import Visualizer3D, create_video, create_gif
from scipy.spatial import cKDTree
from tqdm import tqdm
from typing import List, Tuple
import multiprocessing
import parmap

timer = Timer()

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

    file_name = data_list[index]
    data_root = os.path.join('data', file_name, 'pcd')
    pcd_files = os.listdir(data_root)
    pcd_files.sort()
    pcd_files = [os.path.join(data_root, x) for x in pcd_files]
    return pcd_files, file_name

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

def get_pcd_buffers(file_infos:dict, 
                    index:int,
                    use_all_related_files=False,
                    split_ratio=1.0):
    pcd_paths, file_name = file_infos[index]['pcd_path'], file_infos[index]['file_name']
    pcd_paths_all = pcd_paths

    if use_all_related_files == True:
        pcd_paths_all = []
        indices = []
        if index in [0, 1, 2]:
            indices = [0, 1, 2]
        else:
            indices = [3, 4, 5, 6]
        for i in indices:
            pcd_paths_all.extend(file_infos[i]['pcd_path'])

    pcd_buffers = split_pcd_paths(pcd_files=pcd_paths_all,
                                split_ratio=split_ratio,
                                shuffle=False)  
    
    return pcd_buffers, pcd_paths, file_name
    
# Pre-process
# ===================================================== #
def downsample_pcd(pcd,
                   voxel_size=0.2):
    downsample_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return downsample_pcd

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


# Inference
# ===================================================== #
def compute_static_voxels(pcd_buffers,
                          voxel_sizes:List[float],
                          distance_thresholds:List[Tuple[float]],
                          matching_counts:List[int],
                          downsample:bool=False,
                          downsample_voxel_size:float=0.1):
    
    points_list = [pcd_to_numpy(x) for x in pcd_buffers]
    points = np.vstack(points_list)
    
    if downsample:
        pcd_downsampled = numpy_to_pcd(points)
        pcd_downsampled = downsample_pcd(pcd_downsampled, downsample_voxel_size)
        points = pcd_to_numpy(pcd_downsampled)
        
    dist = np.linalg.norm(points, axis=-1)
    static_voxel_grids = []  
    for voxel_size, distance_threshold, matching_count in tqdm(zip(voxel_sizes, distance_thresholds, matching_counts)):
        
        points_in_dist = points[((dist >= distance_threshold[0]) & (dist < distance_threshold[1]))]
        voxel_indices = np.floor(points_in_dist / voxel_size).astype(int)
        unique_voxels, counts = np.unique(voxel_indices, axis=0, return_counts=True)
        
        filtered_voxels = unique_voxels[counts >= matching_count]
        filtered_voxel_centers = filtered_voxels * voxel_size + (voxel_size / 2)
        
        voxel_grid = pcd_to_voxel(numpy_to_pcd(filtered_voxel_centers), voxel_size)
        static_voxel_grids.append(voxel_grid)
        
        print(f"Total Voxels (voxel size {voxel_size}): {len(unique_voxels)}")
        print(f"Static Voxels (voxel size {voxel_size}, >= {matching_count} points): {len(filtered_voxels)}")

    return static_voxel_grids 

def remove_static_points(points : np.ndarray, 
                         distance_thresholds:List[Tuple[float]],
                         voxel_buffers_list,
                         matching_count = 2):
    
    dist = np.linalg.norm(points, axis=-1)
    points_list = []
    for distance_threshold, voxel_grid in zip(distance_thresholds, voxel_buffers_list):
        points_in_dist = points[((dist >= distance_threshold[0]) & (dist < distance_threshold[1]))]
        mask_static = np.array(voxel_grid.check_if_included(numpy_to_v3v(points_in_dist)))
        points_list.append(points_in_dist[~mask_static])
    
    return np.vstack(points_list)

    # points = points.copy()  
    
    # voxel_grid = voxel_buffers_list[0]
    # mask_static = np.array(voxel_grid.check_if_included(numpy_to_v3v(points)))
    
    # points = points[~mask_static]
    # mask_static = np.zeros((points.shape[0], len(voxel_buffers_list)-1))
    
    # for i, voxel_grid in enumerate(voxel_buffers_list[1:]):
    #     mask_static_single = np.array(voxel_grid.check_if_included(numpy_to_v3v(points)))
    #     mask_static[..., i] = mask_static_single
    
    # mask_static = mask_static.sum(axis=1)
    # mask_static = mask_static >= matching_count
    
    # points = points[~mask_static]
    
    return points

def find_3d_bboxes(pcd,
                   eps=0.6,
                   min_points=5):
    
    #with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        # labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    
    min_points_in_cluster = 3
    max_points_in_cluster = 999
    min_z_value = -999
    max_z_value = 999
    height_diff = 0.12
    
    bboxes_1234 = []
    max_label = labels.max()
    
    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster_pcd = pcd.select_by_index(cluster_indices)
        points = np.asarray(cluster_pcd.points)
        
        z_values = points[:, 2] 
        z_min = z_values.min()
        z_max = z_values.max()
        
        #print(min_points_in_cluster, len(cluster_indices), max_points_in_cluster, min_z_value, z_min, z_max, max_z_value)
        if min_points_in_cluster <= len(cluster_indices) <= max_points_in_cluster:
            if min_z_value <= z_min and z_max <= max_z_value:
                bbox = cluster_pcd.get_axis_aligned_bounding_box()
                
                min_bound = bbox.get_min_bound()
                max_bound = bbox.get_max_bound()
                
                w, l = abs(min_bound[0] - max_bound[0]), abs(min_bound[1] - max_bound[1])
                xy_volume = w * l
                
                # print(xy_volume, w, l, z_min, z_max, z_max - z_min, bbox.get_center())
        
                if ((w >= 0.7 or l >= 0.7) or abs(z_max - z_min) < height_diff) and z_max >= 4.6:
                    continue

                if xy_volume >= 1.0:
                    continue
                
                bbox.color = (1, 0, 0) 
                bboxes_1234.append(bbox)

    return bboxes_1234, labels

# Post-process
# ===================================================== #
def refine_3d_bboxes(bboxes):
    refined_boxes = []
    for bbox in bboxes:
        # 기존 바운딩 박스 정보
        center = bbox.get_center()
        extent = bbox.get_extent()  

        upper_height = extent[2]  
        full_height = max(1.6, upper_height * 2.0)  
        full_width = max(0.7, extent[0])  
        full_depth = max(0.7, extent[1]) 
    
        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()

        min_bound[2] -= full_height - upper_height  
        max_bound[2] += 0.1  
        x_center = (min_bound[0] + max_bound[0]) / 2
        y_center = (min_bound[1] + max_bound[1]) / 2
        min_bound[0] = x_center - full_width / 2 
        max_bound[0] = x_center + full_width / 2  
        min_bound[1] = y_center - full_depth / 2 
        max_bound[1] = y_center + full_depth / 2  

        refined_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        refined_bbox.color = (1, 0, 0) 
        refined_boxes.append(refined_bbox)
    return refined_boxes


# Multi-processing
# ===================================================== #
def split_list(data, num_splits):
    chunk_size = len(data) // num_splits
    return [data[i * chunk_size: (i + 1) * chunk_size] for i in range(num_splits)] + [data[num_splits * chunk_size:]]

def inference_mp(items, shared_dict):
    result = []
    
    voxel_buffers_list = shared_dict['voxel_buffers_list']
    distance_thresholds = shared_dict['distance_thresholds']
    eps_thresholds = shared_dict['eps_thresholds']
    min_points_thresholds = shared_dict['min_points_thresholds']
    
    for points in items:
        pcd_target = numpy_to_pcd(points)
        pcd_filtered = pcd_target
        pcd_filtered = remove_road_plane(pcd_filtered)
        points = pcd_to_numpy(pcd_filtered)
        
        points_org = points.copy()
        points = remove_static_points(points, 
                                      voxel_buffers_list)
        
        dist = np.linalg.norm(points, axis=-1)
        bboxes_3d = []
        
        for distance_threshold, eps, min_points in zip(distance_thresholds, eps_thresholds, min_points_thresholds):
            
            mask_dist = (dist >= distance_threshold[0]) & (dist < distance_threshold[1])
            points_dist = points[mask_dist]
                
            bboxes, labels = find_3d_bboxes(numpy_to_pcd(points_dist),
                                            min_points=min_points)
            bboxes = refine_3d_bboxes(bboxes)
            bboxes_3d.extend(bboxes)
        
        result.append({'points_org' : points_org, 'bboxes_3d' : bboxes_3d})
    return result

# Main
# ===================================================== #

# 0: '01_straight_walk',
# 1: '02_straight_duck_walk',
# 2: '03_straight_crawl',
# 3: '04_zigzag_walk',
# 4: '05_straight_duck_walk',
# 5: '06_straight_crawl',
# 6: '07_straight_walk'

# (0 ~ 6)
index = 3

print("* Loading PCD files. *")
file_infos = []
for i in range(7):
    pcd_path, file_name = load_pcd_paths(i)
    file_infos.append({'pcd_path' : pcd_path, 'file_name' : file_name})
pcd_buffers, pcd_paths, file_name = get_pcd_buffers(file_infos=file_infos, 
                                                    index=index, 
                                                    use_all_related_files=True,
                                                    split_ratio=0.5)

buffer_size = len(pcd_buffers)
pcd_buffers = load_pcds_from_paths(pcd_buffers)
pcd_targets = load_pcds_from_paths(pcd_paths)
print("* Loading PCD files completed! - {}s *".format(timer.get_passed_time()))
print("=================================")

# print("* Preprocess loaded PCD files. *")
# # pcd_buffers = [downsample_pcd(x, 0.05) for x in tqdm(pcd_buffers)]
# # pcd_buffers = [filter_outliers(x) for x in tqdm(pcd_buffers)]
# # pcd_buffers = [remove_road_plane(x) for x in tqdm(pcd_buffers)]
# print("* Preprocessing loaded PCD files completed! - {}s *".format(timer.get_passed_time()))
# print("=================================")

print("* Computing static voxels. *")
voxel_sizes = [0.2, 0.25, 0.3]
matching_counts=[int(buffer_size * 0.05), 
                 int(buffer_size * 0.05),
                 int(buffer_size * 0.05)]
distance_thresholds = [(0, 20), (20, 50), (50, 9999)]

voxel_buffers_list = compute_static_voxels(pcd_buffers=pcd_buffers,
                                           voxel_sizes=voxel_sizes,
                                           distance_thresholds=distance_thresholds,
                                           matching_counts=matching_counts,
                                           downsample=False,
                                           downsample_voxel_size=0.05)
print("* Computing static voxels completed! - {}s *".format(timer.get_passed_time()))
print("=================================")

print("* Inferencing! *")   
use_mp = False

if use_mp:
    cpu_count = multiprocessing.cpu_count()  # CPU 개수 확인
    points_list = [pcd_to_numpy(x) for x in pcd_targets]
    split_data_list = split_list(points_list, cpu_count)  # 데이터를 CPU 개수에 맞게 분할

    shared_info_dict = {
        'distance_thresholds' : distance_thresholds,
        'eps_thresholds' : eps_thresholds,
        'min_points_thresholds' : min_points_thresholds,
        'voxel_buffers_list' : voxel_buffers_list
    }
    
    results = parmap.map(inference_mp, 
                        split_data_list, 
                        shared_info_dict, 
                        pm_pbar=True, 
                        pm_processes=cpu_count)

    for i, res in enumerate(results):
        points_org = res['points_org']
        bboxes_3d = res['bboxes_3d']
        
        vis = Visualizer3D()
        vis.set_points(points_org)
        vis.add_bboxes_3d(bboxes_3d)
        vis.set_camera_params('cam.json')
        # vis.show()
        vis.save_to_image('./output/{}/raw'.format(index), i)
else: 
    
    inference_count = 0
    inference_time_cum = 0
    
    eps_thresholds = [0.3, 2.5, 2]                        
    min_points_thresholds = [8, 5, 4]   
    
    for i, pcd_target in (enumerate(pcd_targets)):
        
        timer.reset()
        
        pcd_filtered = pcd_target
        pcd_filtered = remove_road_plane(pcd_filtered)
        points = pcd_to_numpy(pcd_filtered)
        
        points_org = points.copy()
        points = remove_static_points(points,
                                      distance_thresholds,
                                      voxel_buffers_list,
                                      matching_count=1)

        # vis = Visualizer3D()
        # vis.set_points(points)
        # vis.set_camera_params('cam.json')
        # vis.show()

        dist = np.linalg.norm(points, axis=-1)
        bboxes_3d = []
        
        for distance_threshold, eps, min_points in zip(distance_thresholds, eps_thresholds, min_points_thresholds):
            
            mask_dist = (dist >= distance_threshold[0]) & (dist < distance_threshold[1])
            points_dist = points[mask_dist]
                
            bboxes, labels = find_3d_bboxes(numpy_to_pcd(points_dist),
                                            min_points=min_points)
            bboxes = refine_3d_bboxes(bboxes)
            bboxes_3d.extend(bboxes)
            
        inference_count += 1
        inference_time = timer.get_passed_time()
        inference_time_cum += inference_time
        
        print("Scene Index {} / {} - {}s / avg {}".format(i + 1, 
                                                          len(pcd_targets), 
                                                          round(inference_time, 3),
                                                          round(inference_time_cum / inference_count, 3)))
        
        vis = Visualizer3D()
        vis.set_points(points_org)
        vis.add_bboxes_3d(bboxes_3d)
        vis.set_camera_params('cam.json')
        # vis.show()
        vis.save_to_image('./output/{}/raw'.format(file_name), i)

print("* Inferencing finished! *")   
print("=================================")

print("* Creating output video. *")   
# create_gif('./output/{}/raw'.format(index), 
#            './output/{}'.format(index), 0.01)
create_video('./output/{}/raw'.format(file_name),
             './output/{}'.format(file_name), 
             file_name,
             30)
print("* Creating output video finished! *")   
print("=================================")