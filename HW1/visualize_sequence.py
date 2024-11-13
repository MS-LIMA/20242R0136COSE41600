import os
import random
from copy import deepcopy
from utils.o3d_utils import *
from utils.visualizer import Visualizer3D
from scipy.spatial import cKDTree
import cv2

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

def create_video(input_folder, output_file, fps=10):
    # 이미지 파일 로드
    images = [img for img in os.listdir(input_folder) if img.endswith(".png")]
    images.sort()

    # 첫 번째 이미지로부터 프레임 크기 추출
    frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, layers = frame.shape

    # 비디오 라이터 설정
    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(input_folder, image)))

    cv2.destroyAllWindows()
    video.release()

camera_params = {
    "lookat": [0, 0, 0],  # 보는 지점
    "front": [0, 0, -1],  # 카메라의 보기 방향
    "up": [0, -1, 0],     # 업 벡터
    "zoom": 0.5           # 줌 레벨
}

pcd_paths = load_pcd_paths(0)

fixed_camera_visualization(pcd_paths, './output_frames', camera_params)

create_video('./output_frames', 'output_video.avi')