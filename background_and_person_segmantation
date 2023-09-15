import open3d as o3d
import logging as log
import sys
import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import trimesh
from time import perf_counter
from argparse import ArgumentParser
from pathlib import Path
from matplotlib import pyplot as plt
import datetime
from scipy.spatial import Delaunay
from numpy import linalg as LA
from itertools import combinations

import pdb
import keyboard

from instance_segmentation_demo.tracker import StaticIOUTracker

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/openvino/model_zoo'))

from model_api.models import MaskRCNNModel, YolactModel, OutputTransform
from model_api.adapters import create_core, OpenvinoAdapter, OVMSAdapter
from model_api.pipelines import get_user_config, AsyncPipeline
from model_api.performance_metrics import PerformanceMetrics

import monitors
from images_capture import open_images_capture
from helpers import resolution, log_latency_per_stage
from visualizers import InstanceSegmentationVisualizer

global  mesh_bg, pcd_bg, should_continue, RSpipeline

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def get_model(model_adapter, configuration):
    inputs = model_adapter.get_input_layers()
    outputs = model_adapter.get_output_layers()
    if len(inputs) == 1 and len(outputs) == 4 and 'proto' in outputs.keys():
        return YolactModel(model_adapter, configuration)
    return MaskRCNNModel(model_adapter, configuration)


def print_raw_results(boxes, classes, scores, frame_id):
    log.debug('  -------------------------- Frame # {} --------------------------  '.format(frame_id))
    log.debug('  Class ID | Confidence |     XMIN |     YMIN |     XMAX |     YMAX ')
    for box, cls, score in zip(boxes, classes, scores):
        log.debug('{:>10} | {:>10f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} '.format(cls, score, *box))



def build_argparser():
    parser = ArgumentParser()
    args = parser.add_argument_group('Options')
    args.add_argument('-m', '--model', required=True,
                      help='Required. Path to an .xml file with a trained model '
                           'or address of model inference service if using ovms adapter.')
    args.add_argument('--adapter', default='openvino', choices=('openvino', 'ovms'),
                      help='Optional. Specify the model adapter. Default is openvino.')
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('-d', '--device', default='CPU',
                      help='Optional. Specify the target device to infer on; CPU or GPU is '
                           'acceptable. The demo will look for a suitable plugin for device specified. '
                           'Default value is CPU.')

    common_model_args = parser.add_argument_group('Common model options')
    common_model_args.add_argument('--labels', required=True,
                                   help='Required. Path to a text file with class labels.')
    common_model_args.add_argument('-t', '--prob_threshold', default=0.5, type=float,
                                   help='Optional. Probability threshold for detections filtering.')
    common_model_args.add_argument('--no_track', action='store_true',
                                   help='Optional. Disable object tracking for video/camera input.')
    common_model_args.add_argument('--show_scores', action='store_true',
                                   help='Optional. Show detection scores.')
    common_model_args.add_argument('--show_boxes', action='store_true',
                                   help='Optional. Show bounding boxes.')
    common_model_args.add_argument('--layout', type=str, default=None,
                                   help='Optional. Model inputs layouts. '
                                        'Format "[<layout>]" or "<input1>[<layout1>],<input2>[<layout2>]" in case of more than one input. '
                                        'To define layout you should use only capital letters')

    infer_args = parser.add_argument_group('Inference options')
    infer_args.add_argument('-nireq', '--num_infer_requests', default=0, type=int,
                            help='Optional. Number of infer requests')
    infer_args.add_argument('-nstreams', '--num_streams', default='',
                            help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput '
                                 'mode (for HETERO and MULTI device cases use format '
                                 '<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).')
    infer_args.add_argument('-nthreads', '--num_threads', default=None, type=int,
                            help='Optional. Number of threads to use for inference on CPU (including HETERO cases).')

    io_args = parser.add_argument_group('Input/output options')
    io_args.add_argument('--loop', action='store_true',
                         help='Optional. Enable reading the input in a loop.')
    io_args.add_argument('-o', '--output',
                         help='Optional. Name of the output file(s) to save.')
    io_args.add_argument('-limit', '--output_limit', default=1000, type=int,
                         help='Optional. Number of frames to store in output. '
                              'If 0 is set, all frames are stored.')
    io_args.add_argument('--no_show', action='store_true',
                         help="Optional. Don't show output.")
    io_args.add_argument('--output_resolution', default=None, type=resolution,
                         help='Optional. Specify the maximum output window resolution '
                              'in (width x height) format. Example: 1280x720. '
                              'Input frame size used by default.')
    io_args.add_argument('-u', '--utilization_monitors',
                         help='Optional. List of monitors to show initially.')

    debug_args = parser.add_argument_group('Debug options')
    debug_args.add_argument('-r', '--raw_output_message', action='store_true',
                             help='Optional. Output inference results raw values showing.')
    return parser

        

class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

#args = build_argparser().parse_args()
    
state = AppState()

def mesh_the_pcd(pcd, method='bpa', depth=8, num_faces=None):
    if method == 'bpa':
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = avg_dist

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, 
            o3d.utility.DoubleVector([radius, radius * 2, radius * 4])
        )
    elif method == 'poisson':
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    elif method == 'delaunay':
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        tri = Delaunay(points[:, :2])  

        new_simplices = []
        LL = np.zeros(3)
        for smpl in tri.simplices:
            ii = 0
            for edge in combinations(smpl, 2):
                LL[ii] = LA.norm(tri.points[edge[0]] - tri.points[edge[1]])
                ii += 1
            if np.max(LL) < 1.5:  
                new_simplices.append(smpl)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.triangles = o3d.utility.Vector3iVector(new_simplices)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    else:
        raise ValueError(f"Invalid method: {method}. ")
    if num_faces:
        mesh = mesh.simplify_quadric_decimation(num_faces)

    return mesh


def save_callback(vis):
    """
    Callback function to get for the 'S' key to save the current view
    """
    global should_continue
    o3d.io.write_point_cloud('./background_pcd.ply', pcd_bg)
    print('Saved pcd')
    o3d.io.write_triangle_mesh('./background_mesh.obj', mesh_bg)
    print("Saved mesh")
    should_continue = False


# vis = o3d.visualization.Visualizer()
# vis.create_window(height=480, width=640)
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.register_key_callback(83, save_callback)  # 83 is the ASCII value for 'S'
vis.create_window(height=1280, width=720)


vis2 = o3d.visualization.VisualizerWithKeyCallback()
vis2.register_key_callback(83, save_callback)  # 83 is the ASCII value for 'S'
vis2.create_window(height=1280, width=720)

RSpipeline = None
config = None
is_pipeline_started = False

def initialize_camera(args):
    global RSpipeline, config, is_pipeline_started
    RSpipeline = rs.pipeline()
    config = rs.config()
    
    RSpipeline_wrapper = rs.pipeline_wrapper(RSpipeline)
    RSpipeline_profile = config.resolve(RSpipeline_wrapper)
    device = RSpipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    RSpipeline.start(config)
    
    # AI model setup
    plugin_config = get_user_config(args.device, args.num_streams, args.num_threads)
    model_adapter = OpenvinoAdapter(create_core(), args.model, device=args.device, plugin_config=plugin_config,
                                    max_num_requests=args.num_infer_requests,
                                    model_parameters={'input_layouts': args.layout})

    configuration = {
        'confidence_threshold': args.prob_threshold,
        'path_to_labels': args.labels,
    }
    model = get_model(model_adapter, configuration)
    model.log_layers_info()

    pipeline = AsyncPipeline(model)
    next_frame_id = 0
    next_frame_id_to_show = 0

    tracker = None
    if not args.no_track:  # and cap.get_type() in {'VIDEO', 'CAMERA'}:
        tracker = StaticIOUTracker()
    visualizer = InstanceSegmentationVisualizer(model.labels, args.show_boxes, args.show_scores)

    metrics = PerformanceMetrics()
    render_metrics = PerformanceMetrics()
    presenter = None
    output_transform = None
    video_writer = cv2.VideoWriter()

    # Get stream profile and camera intrinsics
    profile = RSpipeline.get_active_profile()
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height

    # Processing blocks
    pc = rs.pointcloud()
    decimate = rs.decimation_filter()
    decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
    colorizer = rs.colorizer()

    return  pipeline, next_frame_id, next_frame_id_to_show, tracker, visualizer, metrics, render_metrics, video_writer

def start_pipeline_safely():
    global RSpipeline, config
    try:
        print("Attempting to start the pipeline...")
        RSpipeline.start(config)
        print("Pipeline started!")
    except RuntimeError as e:
        if "start() cannot be called before stop()" in str(e):
            print("Pipeline already started!")
        else:
            raise e



def main():
    global RSpipeline
    args = build_argparser().parse_args()
    pipeline, next_frame_id, next_frame_id_to_show, tracker, visualizer, metrics, render_metrics, video_writer = initialize_camera(args)
    background_loop()
    #time.sleep(5)
    segmentation_loop(args, pipeline, next_frame_id, next_frame_id_to_show, tracker, visualizer, metrics, render_metrics, video_writer)

def background_loop(): 
    global mesh_bg, pcd_bg, RSpipeline, should_continue
    should_continue = True
    while should_continue: 
        start_time = time.time() # start time of the loop
        frames = RSpipeline.wait_for_frames()
        
        background_depth_frame = frames.get_depth_frame()
        background_color_frame = frames.get_color_frame()

        background_color_img = np.asanyarray(background_color_frame.get_data())
        background_color_image = np.copy(background_color_img[:,:,::-1])
        background_depth_image = np.asanyarray(background_depth_frame.get_data())

        depth_image_uint16 = (background_depth_image).astype(np.uint16)
        depth_o3d = o3d.geometry.Image(depth_image_uint16)
        color_o3d = o3d.geometry.Image(background_color_image)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d, depth_o3d, convert_rgb_to_intensity=False)

        intrinsics = rs.video_stream_profile(background_color_frame.profile).get_intrinsics()
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        camera_intrinsic.set_intrinsics(
            width=intrinsics.width,
            height=intrinsics.height,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.ppx,
            cy=intrinsics.ppy)
        camera_extrinsic = \
            [[1, 0, 0, 0], \
                [0, -1, 0, 0], \
                [0, 0, -1, 2], \
                [0, 0, 0, 1]]

        if not 'pcd_bg' in globals():
            pcd_bg = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic, camera_extrinsic)
            print ('adding geometry')
            vis.add_geometry(pcd_bg)
        else:
            pcd_bg_new = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic, camera_extrinsic)
            pcd_bg.points = pcd_bg_new.points
            pcd_bg.colors = pcd_bg_new.colors
            print ('updating geometry')
            vis.update_geometry(pcd_bg)
            
        print (vis.get_view_control().convert_to_pinhole_camera_parameters().extrinsic)                
        vis.poll_events()

        print('estimating normals')
        pcd_bg.estimate_normals()
        print ('downsampling pcd')
        pcd_bg_ss = pcd_bg.voxel_down_sample(voxel_size=0.05)
        print ('meshing pcd')

        if not 'mesh_bg' in globals():
            mesh_bg = mesh_the_pcd (pcd_bg_ss, method='delaunay')
            vis2.add_geometry(mesh_bg)
        else:
            mesh_bg_new = mesh_the_pcd (pcd_bg_ss, method='delaunay')
            mesh_bg.triangles = mesh_bg_new.triangles
            mesh_bg.vertices = mesh_bg_new.vertices
            mesh_bg.vertex_colors = mesh_bg_new.vertex_colors
            mesh_bg.textures = mesh_bg_new.textures
            vis2.update_geometry(mesh_bg)
            print (vis2.get_view_control().convert_to_pinhole_camera_parameters().extrinsic)
        
        print(np.array(mesh_bg.vertices).shape)
        print(np.array(mesh_bg.triangles).shape)

        vis2.poll_events()
        vis2.update_renderer() 
        
        print("running FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop

def segmentation_loop(args, pipeline, next_frame_id, next_frame_id_to_show, tracker, visualizer, metrics, render_metrics, video_writer):
    global mesh_bg, RSpipeline

    pcd_bg = o3d.io.read_point_cloud('./background_pcd.ply')
    mesh_bg = o3d.io.read_triangle_mesh('./background_mesh.obj')

    vis.add_geometry(pcd_bg)
    vis2.add_geometry(mesh_bg)

    pcds = []
    meshes = []

    start_time = time.time()
    while True: 
        start_pipeline_safely()
        frames = RSpipeline.wait_for_frames()
        
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        color_image_bgr = np.asanyarray(color_frame.get_data())
        color_image = np.copy(color_image_bgr[:,:,::-1])
        
        pipeline.submit_data(color_image, next_frame_id, {'frame': color_image, 'depth': depth_frame, 'start_time': start_time})
        next_frame_id += 1

        results = pipeline.get_result(next_frame_id_to_show)
        if results:
            (scores, classes, boxes, masks), frame_meta = results

            # Update the pcds and meshes lists based on detected masks
            while len(pcds) > len(masks):
                pcd_to_remove = pcds.pop()
                vis.remove_geometry(pcd_to_remove)

            while len(pcds) < len(masks):
                new_pcd = o3d.geometry.PointCloud()
                pcds.append(new_pcd)
                vis.add_geometry(new_pcd)

            while len(meshes) > len(masks):
                mesh_to_remove = meshes.pop()
                vis2.remove_geometry(mesh_to_remove)

            while len(meshes) < len(masks):
                new_mesh = o3d.geometry.TriangleMesh()
                meshes.append(new_mesh)
                vis2.add_geometry(new_mesh)

            # Now process each detected mask
            for idx, mask in enumerate(masks):
                bkg_mask_rgb = np.dstack((mask, mask, mask))
                color_image_masked = np.multiply(bkg_mask_rgb, color_image)

                depth_image_np = np.asanyarray(depth_frame.get_data())
                bkg_mask_depth = np.maximum(np.zeros(depth_image_np.shape), mask.astype(float))

                depth_image_masked = np.multiply(bkg_mask_depth, np.asanyarray(depth_frame.get_data()))

                depth_image_uint16 = (depth_image_masked).astype(np.uint16)
                depth_o3d = o3d.geometry.Image(depth_image_uint16)
                color_o3d = o3d.geometry.Image(color_image_masked)
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d, depth_o3d, convert_rgb_to_intensity=False)

                intrinsics = rs.video_stream_profile(color_frame.profile).get_intrinsics()
                camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
                camera_intrinsic.set_intrinsics(
                    width=intrinsics.width,
                    height=intrinsics.height,
                    fx=intrinsics.fx,
                    fy=intrinsics.fy,
                    cx=intrinsics.ppx,
                    cy=intrinsics.ppy
                )
                camera_extrinsic = \
                            [[1, 0, 0, 0], \
                            [0, -1, 0, 0], \
                            [0, 0, -1, 2], \
                            [0, 0, 0, 1]]
                # Process the point cloud
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic, camera_extrinsic)
                pcds[idx].points = pcd.points
                pcds[idx].colors = pcd.colors
                vis.update_geometry(pcds[idx])

                # Process the mesh
                pcd_ss = pcd.voxel_down_sample(voxel_size=0.05)
                if len(pcd_ss.points) > 0:
                    mesh_new = mesh_the_pcd(pcd_ss, method='delaunay')
                    
                    # Assigning the point cloud colors to the mesh vertices
                    colors = np.asarray(pcd_ss.colors)
                    if len(colors) == len(mesh_new.vertices):
                        mesh_new.vertex_colors = o3d.utility.Vector3dVector(colors)
                    
                    meshes[idx].triangles = mesh_new.triangles
                    meshes[idx].vertices = mesh_new.vertices
                    meshes[idx].vertex_colors = mesh_new.vertex_colors  # Updating the color of the mesh vertices
                    vis2.update_geometry(meshes[idx])


            vis.poll_events()
            vis2.poll_events()
            vis2.update_renderer()
                
            print("Running FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop



            if args.raw_output_message:
                print_raw_results(boxes, classes, scores, next_frame_id_to_show)

            rendering_start_time = perf_counter()
            masks_tracks_ids = tracker(masks, classes) if tracker else None
            color_image = visualizer(color_image, boxes, classes, scores, masks, masks_tracks_ids)
            render_metrics.update(rendering_start_time)

            #presenter.drawGraphs(color_image)
            metrics.update(start_time, color_image)

            if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit - 1):
                video_writer.write(color_image)
            next_frame_id_to_show += 1

            if not args.no_show:
                #cv2.imshow('Instance Segmentation results', color_image)
                #cv2.imshow('depth image', (0.17*np.minimum(depth_image,15000)).astype(np.uint8))
                key = cv2.waitKey(1)
                if key == 27 or key == 113 or key == 225: # end if Esc, q or Q key pressed
                    break
                #presenter.handleKey(key)
    #vis.destroy_window()
    #RSpipeline.stop()

if __name__ == "__main__":
    main()
