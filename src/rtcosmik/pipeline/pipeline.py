from collections import deque
import torch
import numpy as np
import pinocchio as pin
import example_robot_data as robex
from datetime import datetime
from multiprocessing import Process, Array, Lock, Value, Event, Queue
from typing import List
import time

from rtcosmik.nlf.nlf import NLFEstimator
from rtcosmik.triangulation.triangulation import triangulate_points
from rtcosmik.filtering.iir import IIR
from rtcosmik.human_model.model_utils import scale_human_model, mks_registration, recalibrate_marker_frames_in_joint_space
from rtcosmik.ik.ik import RT_IK, RT_SWIKA
from rtcosmik.camera.cam_utils import load_camera_parameters,load_world_transformation

import logging

LOGGER = logging.getLogger(__name__)

class PipelineProcess(Process):

    """
    Main real-time asynchronous estimation pipeline running as a dedicated system Process.
    
    This process acts as the core engine of the system. It synchronizes multi-camera 
    shared memory inputs, infers 2D keypoints via a batched NLF Estimator, triangulates 
    them into 3D world space, applies an IIR multi-channel filter 
    to remove noise, do marker registration, and Inverse Kinematics (IK).
    """

    def __init__(self, 
                 settings,
                 frame_counters,
                 camera_buffers, 
                 camera_locks, 
                 timestamp_buffers,
                 results_queues: List[Queue],
                 stop_event: Event,
                 mtxs,
                 dists,
                 projections,
                 world_R1_cam,
                 world_T1_cam,
                 frame_shape: tuple = (720, 1280, 3),
                 num_cameras: int = 2,
                 logger=None,
                 ):
        
        """
        Initializes the pipeline process with multi-process communication queues and calibration objects.

        Parameters:
            settings (Object): Global configuration settings container parsing parameters
            frame_counters (List[int]): tracks individual camera lane frame counts.
            camera_buffers (List[Array]): Raw shared memory buffers containing raw sequential image payloads.
            camera_locks (List[Lock]): Process-safe mutual exclusion locks protecting shared frame buffers from race conditions.
            timestamp_buffers (List[Array]): Shared character byte streams documenting hardware trigger timestamps.
            results_queues (List[Queue]): Inter-process communication output conduits 
                - Index 0: Emits filtered 3D world marker position dictionaries.
                - Index 1: Emits the calculated generalized joint configuration vector 'q'
            stop_event (Event): Inter-process event flag triggering complete termination of the process execution.
            mtxs (List[np.ndarray]): List of 3x3 array matrices containing individual camera coefficients.
            dists (List[np.ndarray]): Lens distortion coefficients utilized to correct the images.
            projections (List[np.ndarray]): 3x4 projection matrices mapping points to the main camera.
            world_R1_cam (np.ndarray): 3x3 rotation matrix mapping primary camera spatial layout into global coordinates.
            world_T1_cam (np.ndarray): 3x1 translation offset mapping primary camera position into global coordinates.
            frame_shape (tuple, optional): Expected dimensions of camera arrays as (Height, Width, Channels). Defaults to (720, 1280, 3).
            num_cameras (int, optional): Total active multi-cam input lanes to parse. Defaults to 2.
            logger (logging.Logger, optional): Custom logging handler. If None, defaults to the system global instance. Defaults to None.
        """

        super().__init__()
        # MP
        self.camera_buffers = camera_buffers
        self.camera_locks = camera_locks
        self.timestamp_buffers = timestamp_buffers
        self.frame_shape = frame_shape  # (height, width, channels)
        self.num_cameras = num_cameras
        self.stop_event = stop_event
        self.results_queues = results_queues

        self.last_frame_counters = [0] * self.num_cameras
        self.frame_counters = frame_counters

        # Settings related parameters
        self.settings=settings

        # Others, cam parameters
        self.first_sample = True

        self.p3d_buffer=deque(maxlen=self.settings.N)

        self.mtxs=mtxs
        self.dists=dists
        self.projections=projections
        self.world_R1_cam=world_R1_cam
        self.world_T1_cam=world_T1_cam
        
        self.logger = logger or LOGGER

    def run(self):

        """
        The main infinite runtime processing loop for the execution block.

        Performs the following synchronous pipeline stages per frame step:
          1. Polls and locks atomic shared memory blocks for fresh multi-camera image sets.
          2. Runs batched deep 2D human keypoint bounding regressions via the NLF estimator.
          3. Triangulates multi-view inputs and re-projects coordinates into world coordinates.
          4. Smooths coordinates using a multi-channel digital IIR filter.
          5. Calibration Phase (First Sample): Automatically builds, scales, and registers
             a Pinocchio `HumanLoader` robot model to the subject, recalibrates joint marker 
             frame translations, and boots cold-start optimization trajectories.
          6. Tracking Phase: Resolves kinematics frame-by-frame via either:
                - Sample-by-Sample (sbs): Fast localized quadratic programming (QuadProg).
                - Moving Horizon Estimation (mhe): Comprehensive temporal non-linear window estimation (SWIKA).
          7. Pushes outputs directly into analytical consumer queues for system viewing.
        """

        est = NLFEstimator(
            yolo_path=self.settings.yolo_path,
            nlf_path=self.settings.nlf_path,
            cano_path=self.settings.cano_path,
            image_size=(self.frame_shape[1], self.frame_shape[0]),
            cam_Ks=self.mtxs,
            indices=self.settings.nlf_indices,
            conf=self.settings.yolo_conf,
            imgsz=self.settings.yolo_imgsz,
            device=self.settings.device,
        )

        num_channel = 3*len(self.settings.marker_names)
        iir_filter = IIR(
            num_channel=num_channel,
            sampling_frequency=self.settings.fs
        )
        iir_filter.add_filter(order=self.settings.order, cutoff=self.settings.cutoff_freq, filter_type=self.settings.filter_type)

        try:
            while not self.stop_event.is_set():
                    frames = []
                    new_counters = []
                    for i, (lock, buffer, cam_ts, frame_counter) in enumerate(zip(self.camera_locks, self.camera_buffers, self.timestamp_buffers, self.frame_counters)):
                        with lock:
                            #  Only accept data if this camera has produced a new frame
                            if frame_counter.value > self.last_frame_counters[i]:
                                # Read and copy shared data atomically
                                arr = np.frombuffer(buffer, dtype=np.uint8)
                                frame = arr.reshape(self.frame_shape).copy()
                                # Get current timestamp
                                timestamp = bytes(cam_ts[:]).decode().strip('\x00')

                                if timestamp == '': # empty data
                                    continue
                                else:
                                    frames.append(frame)
                                new_counters.append(frame_counter.value)
                    
                    if len(frames)!=self.num_cameras:
                        continue

                    self.last_frame_counters = new_counters.copy()

                    nlf_out, infer_ms, yres, boxes = est.estimate_from_frames(frames)

                    nlf_out_2d = nlf_out["poses2d"]

                    if nlf_out_2d is None or len(nlf_out_2d) < self.num_cameras:
                        continue

                    keypoints_list = [None] * self.num_cameras
                    valid_cam_ids = []

                    for ii in range(self.num_cameras):
                        poses2d = nlf_out_2d[ii]
                        
                        if poses2d is None or len(poses2d) == 0 or poses2d[0] is None:
                            continue

                        keypoints_list[ii] = poses2d[0].detach().float().cpu().numpy()
                        valid_cam_ids.append(ii)

                    if len(valid_cam_ids) < 2:
                        continue
                    
                    p3d = triangulate_points(
                        keypoints_list=keypoints_list,
                        mtxs=self.mtxs,
                        dists=self.dists,
                        projections=self.projections,
                    )

                    p3d_np = torch.from_numpy(p3d).to(dtype=torch.float32)

                    p3d_in_world=np.array([np.dot(self.world_R1_cam,point) + self.world_T1_cam for point in p3d_np])

                    if self.first_sample:
                        for k in range(self.settings.N):
                            self.p3d_buffer.append(p3d_in_world)  # add the 1st frame 30 times
                    else:
                        self.p3d_buffer.append(p3d_in_world) # add the keypoints to the buffer normally

                    if len(self.p3d_buffer) == self.settings.N:
                        p3d_buffer_array = np.array(self.p3d_buffer)

                        # Filter keypoints in world to remove noisy artefacts 
                        filtered_p3d_buffer = iir_filter.filter(np.reshape(p3d_buffer_array,(self.settings.N, 3*len(self.settings.marker_names))))
                        filtered_p3d_buffer = np.reshape(filtered_p3d_buffer,(self.settings.N, len(self.settings.marker_names), 3))

                        augmented_markers=filtered_p3d_buffer[-1]

                        if self.first_sample:
                            mks_dict = dict(zip(self.settings.marker_names, augmented_markers))

                            human = robex.human.HumanLoader(height=self.settings.human_height, weight=self.settings.human_weight, gender=self.settings.human_gender).robot
                            human_model = human.model

                            #scale the model to data
                            human_model = scale_human_model(human_model, mks_dict, gender=self.settings.human_gender, subject_height=self.settings.human_height)
                            human_model= mks_registration(human_model, mks_dict, gender=self.settings.human_gender, subject_height=self.settings.human_height)

                            # IK
                            if self.settings.ik_type == 'sbs':
                                omega = {}
                                for key in self.settings.keys_to_track_list:
                                    omega[key] = 1
                                q = pin.neutral(human_model)
                                ik_class = RT_IK(human_model, mks_dict, q, self.settings.keys_to_track_list, self.settings.dt, omega)

                                q = ik_class.solve_ik_sample_casadi()
                                ik_class._q0 = q

                                # Recalibrate briefly the markers translation in joint frames
                                human_model=recalibrate_marker_frames_in_joint_space(human_model,q,mks_dict,self.settings.marker_names)

                                ik_class = RT_IK(human_model, mks_dict, q, self.settings.keys_to_track_list, self.settings.dt, omega)
                                self.logger.info("[INFO] Model calibration finished, ready to process...")

                            elif self.settings.ik_type == 'mhe':
                                ik_class = RT_SWIKA(human_model, self.settings.keys_to_track_list, self.settings.N, code = self.settings.ik_code)

                                x_array = np.zeros((human_model.nq+human_model.nv, self.settings.N))
                                x_array[6,:]=1
                                u_array = np.zeros((human_model.nv, self.settings.N))
                                deque_lstm_dict = deque(maxlen=self.settings.N)
                                for k in range(self.settings.N):
                                    deque_lstm_dict.append(mks_dict)

                                array_data = np.array([np.hstack([d[marker] for marker in self.settings.keys_to_track_list]) for d in deque_lstm_dict]).T

                                x_array, u_array = ik_class.solve(x_array, u_array, array_data, x_array[:,-1], self.settings.cost_weights, self.settings.dt)

                                q = pin.neutral(human_model)
                                q[:] = np.array(x_array[:human_model.nq,-1]).flatten()
                                viz_human.display(q)

                                # Recalibrate briefly the markers translation in joint frames
                                human_model=recalibrate_marker_frames_in_joint_space(human_model,q,mks_dict,self.settings.marker_names)

                                ik_class = RT_SWIKA(human_model, self.settings.keys_to_track_list, self.settings.N, code = self.settings.ik_code)
                                self.logger.info("[INFO] Model calibration finished, ready to process...")
                            else : 
                                raise ValueError("Invalid ik type, should be sbs (sample by sample) or mhe (moving horizon estimation)")

                            self.first_sample = False

                        else: # Init phase finished
                            mks_dict = dict(zip(self.settings.marker_names, augmented_markers))
                            self.results_queues[0].put((new_counters, mks_dict))

                            # IK directly 
                            if self.settings.ik_type == 'sbs':
                                ik_class._dict_m = mks_dict
                                q = ik_class.solve_ik_sample_quadprog() 
                                ik_class._q0 = q
                                self.results_queues[1].put((new_counters, q))

                            elif settings.ik_type == 'mhe':
                                deque_lstm_dict.append(mks_dict)
                                array_data = np.array([np.hstack([d[marker] for marker in self.settings.keys_to_track_list]) for d in deque_lstm_dict]).T
                                
                                x_array, u_array = ik_class.solve(x_array, u_array, array_data, x_array[:,-1], self.settings.cost_weights, self.settings.dt)

                                q = pin.neutral(human_model)
                                q[:] = np.array(x_array[:human_model.nq,-1]).flatten()
                                self.results_queues[1].put((new_counters, q))
                            else : 
                                raise ValueError("Invalid ik type, should be sbs (sample by sample) or mhe (moving horizon estimation)")
        finally:        
            self.logger.info("[INFO] Pipeline Process terminated")
