import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from utils.o3d_utils import *
from typing import List
import cv2
import imageio

class Visualizer3D():
    def __init__(self,
                 point_size:float=2.0,
                 window_name:str='Default Visualizer'):
        vis = o3d.visualization.Visualizer()
        
        self.vis = vis
        vis.create_window(window_name=window_name,
                          width=1920,
                          height=1080)
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

        pcd = numpy_to_pcd(points, point_color)
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
            
        vis.poll_events()
        vis.update_renderer()
    
    def add_axis(self, 
                 origin:List[float]=[0, 0, 0]):
        vis = self.vis
        
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=origin)
        vis.add_geometry(axis)
        
    def add_bboxes_3d(self,
                      bboxes):
        vis = self.vis
        for bbox in bboxes:
            vis.add_geometry(bbox)
            vis.update_geometry(bbox)
            
        vis.poll_events()
        vis.update_renderer()
    
    def set_camera_params(self,
                          param_path:str):
        vis = self.vis
        view_control = vis.get_view_control()
        parameters = o3d.io.read_pinhole_camera_parameters(param_path)
        view_control.convert_from_pinhole_camera_parameters(parameters, 
                                                            allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()

    def save_to_image(self,
                      output_path:str,
                      index:int):
        vis = self.vis
        image = vis.capture_screen_float_buffer(False)
        plt_image = np.asarray(image) * 255
        plt_image = plt_image.astype(np.uint8)
        plt_image = plt_image[:, :, ::-1]
        
        if os.path.exists(output_path) == False:
            os.makedirs(output_path)
            
        cv2.imwrite(f"{output_path}/frame_{index:04d}.png", plt_image)
        
        vis.destroy_window()

def create_video(input_folder, output_path, fps=10):
    images = [img for img in os.listdir(input_folder) if img.endswith(".png")]
    images.sort()

    # 첫 번째 이미지로부터 프레임 크기 추출
    frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, layers = frame.shape

    # 비디오 라이터 설정
    video = cv2.VideoWriter('result.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(output_path, image)))

    cv2.destroyAllWindows()
    video.release()
    
def create_gif(input_folder, output_path, interval=0.05):
    images = [img for img in os.listdir(input_folder) if img.endswith(".png")]
    images.sort()
    
    frames = [cv2.imread(os.path.join(input_folder, x)) for x in images]
    frames = [x[...,[2,1,0]] for x in frames]
    
    gif_config = {
        'loop':0, 
        'duration':interval 
    }
    if os.path.exists(output_path) == False:
        os.makedirs(output_path)
            
    imageio.mimwrite(os.path.join(output_path, 'result.gif'), ## 저장 경로
                 frames, 
                 format='gif', 
                 **gif_config 
    )