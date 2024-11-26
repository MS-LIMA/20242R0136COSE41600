import os
import random
from copy import deepcopy
from utils.o3d_utils import *
from utils.visualizer import *
from scipy.spatial import cKDTree
import cv2
from tqdm import tqdm

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

def load_pcds_from_paths(pcd_paths:List[str],
                              down_sample=True,
                              voxel_size=0.1):
    pcd_list = [load_pcd(file) for file in pcd_paths]
    for i, pcd in enumerate(pcd_list):
        points = pcd_to_numpy(pcd)
        non_zero_points = points[~np.all(points == [0, 0, 0], axis=1)]
        pcd.points = o3d.utility.Vector3dVector(non_zero_points)
        
        if down_sample:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        
        pcd_list[i] = pcd
        
    return pcd_list

def fixed_camera_visualization(pcd_files, output_folder, camera_params):
    # 시각화 환경 설정
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters('./param.json')
    ctr.convert_from_pinhole_camera_parameters(param)

    for i, pcd_file in enumerate(pcd_files):
        # 포인트 클라우드 로드
        pcd = o3d.io.read_point_cloud(pcd_file)
        
        # 시각화 환경에 포인트 클라우드 추가
        vis.add_geometry(pcd)
        
        # 뷰 업데이트 및 렌더링
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # 프레임 캡처
        image = vis.capture_screen_float_buffer(False)
        plt_image = np.asarray(image) * 255
        plt_image = plt_image.astype(np.uint8)
        cv2.imwrite(f"{output_folder}/frame_{i:04d}.png", plt_image)
        vis.run()
        # 다음 프레임을 위해 기존 포인트 클라우드 제거
        # vis.remove_geometry(pcd)

    vis.destroy_window()

camera_params = {
    "lookat": [0, 0, 0],  # 보는 지점
    "front": [0, 0, -1],  # 카메라의 보기 방향
    "up": [0, -1, 0],     # 업 벡터
    "zoom": 0.5           # 줌 레벨
}
def downsample_pcd(pcd,
                   voxel_size=0.2):
    downsample_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return downsample_pcd

def lerp(a, b, t):
    return a * (1 - t) + b * t


index = 6
pcd_paths = load_pcd_paths(index)
pcd_targets = load_pcds_from_paths(pcd_paths)
# pcd_targets = [downsample_pcd(x, 0.1) for x in pcd_targets]

# target = pcd_targets[0]
# pcd_targets = pcd_targets[1:]
# trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
#                          [-0.139, 0.967, -0.215, 0.7],
#                          [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])

# dist_target = np.linalg.norm(pcd_to_numpy(target), axis=-1)
# mask_dist = (dist_target >= 0) & (dist_target < 100)
# target_points = pcd_to_numpy(target)[mask_dist]

# for i, pcd in enumerate(pcd_targets):
#     points = pcd_to_numpy(pcd)
#     dist = np.linalg.norm(points, axis=-1)
#     mask_dist = (dist >= 0) & (dist < 100)
#     points_dist = points[mask_dist]
#     reg_p2p = o3d.pipelines.registration.registration_icp(
#     points_to_pcd(points_dist), points_to_pcd(target_points), 
#     0.02, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
#     pcd.transform(reg_p2p.transformation)
#     pcd_targets[i] = pcd
#     # vis = Visualizer3D()
#     # vis.set_points(pcd_to_numpy(pcd))
#     # vis.show()


# create_video('./visualize_sequence/{}/raw'.format(index), 
#              './visualize_sequence/{}'.format(index), 
#              'result',
#              30)

points_cum = np.zeros((0, 3))
for i, pcd in (enumerate(pcd_targets)):
    
    pcd_down = downsample_pcd(pcd)
    points = pcd_to_numpy(pcd_down)
    points_cum = np.concatenate([points_cum, points], axis=0)
    
    # color = lerp(np.array([0.1, 0.1, 0.1]), np.array([1, 1, 1]), i / len(pcd_targets))
    color = [1, 1, 1]
    vis = Visualizer3D()
    vis.set_points(points_cum, color)
    vis.set_camera_params('cam.json')
    vis.show()
    # vis.save_to_image('./visualize_sequence/{}/raw'.format(index), i)
    
    print('{} / {}'.format(i, len(pcd_targets)))

# create_gif('./visualize_sequence/{}/raw'.format(index), './visualize_sequence/{}'.format(index))
create_video('./visualize_sequence/{}/raw'.format(index), 
             './visualize_sequence/{}'.format(index), 
             'result',
             30)
vis.show()

# fixed_camera_visualization(pcd_paths, './output_frames', camera_params)

# create_video('./output_frames', 'output_video.avi')
