#!/usr/bin/env python3
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import argparse

import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import viser
import cv2

from rtcosmik.utils.videoReader import OfflineVideoSource, list_videos
import numpy as np
import torch
import pinocchio as pin

from rtcosmik.viewer.viewer import ViserRobotVisualizer

from rtcosmik.config_loader import settings
from rtcosmik.nlf.nlf import NLFEstimator, DisplayConsumerNLF
from rtcosmik.triangulation.triangulation import triangulate_points
from rtcosmik.filtering.iir import IIR
from rtcosmik.human_model.model_utils import scale_human_model, mks_registration, recalibrate_marker_frames_in_joint_space
from rtcosmik.ik.ik import RT_IK, RT_SWIKA_FATROP, RT_SWIKA_ACADOS
from rtcosmik.camera.cam_utils import list_cameras, load_camera_parameters, load_world_transformation
from rtcosmik.camera.camera import Camera
from rtcosmik.utils.mp_utils import create_camera_shared_ressources, create_pipeline_shared_ressources
from rtcosmik.pipeline.pipeline import PipelineProcess
from rtcosmik.viewer.viewer import ViewerProcess

from multiprocessing import set_start_method
from collections import deque
import example_robot_data as robex

import logging

import subprocess
import json
import re

import threading
import queue

import csv

JOINT_ANGLES_NAMES = [
    'FF_X', 'FF_Y', 'FF_Z', 'FF_quatx', 'FF_quaty', 'FF_quatz', 'FF_quatw',
    'Lhip_flex_ext', 'Lhip_abd_add', 'Lhip_int_ext_rot', 'Lknee_flex_ext', 'Lankle_flex_ext', 'Lankle_abd_add',
    'Lumbar_flex_ext', 'Lumbar_lateral_flex',
    'Thoracic_flex_ext', 'Thoracic_lateral_flex', 'Thoracic_rot_int_ext',
    'Lcalvicule_x',
    'Lshoulder_flex_ext', 'Lshoulder_abd_add', 'Lshoulder_int_ext_rot', 'Lelbow_flex_ext', 'Lelbow_pron_supi', 'Lwrist_flex_ext', 'Lwrist_x',
    'Cervical_flex_ext', 'Cervical_lat_bend', 'Cervical_int_ext_rot',
    'rcalvicule_x',
    'Rshoulder_flex_ext', 'Rshoulder_abd_add', 'Rshoulder_int_ext_rot', 'Relbow_flex_ext', 'Relbow_pron_supi', 'Rwrist_flex_ext', 'Rwrist_x',
    'Rhip_flex_ext', 'Rhip_abd_add', 'Rhip_int_ext_rot',
    'Rknee_flex_ext', 'Rankle_flex_ext', 'Rankle_abd_add'
]

LOGGER = logging.getLogger(__name__)


def _extract_cam_id(name) -> Optional[int]:
    """Extract the integer camera ID encoded in `name`, matching the
    'camera_N' naming scheme used in video filenames (e.g. 'camera_2.mp4').
    Used only for offline mode, where the ID is real, deterministic data
    burned into the filename -- unlike online mode, where bus enumeration
    order is arbitrary and --camera-index is a fixed positional convention
    instead (see the online branch in main()).

    Returns None if no matching pattern is found.
    """
    s = str(name)
    m = re.search(r'camera_(\d+)', s, re.IGNORECASE)
    return int(m[1]) if m else None


def save_joint_angles_csv(saved_data, save_dir, settings, ocp=None, benchmark_stats=None):
    """Saves joint angles to joint_angles_pre.csv with benchmark stats and OCP solver options in the header."""
    if not saved_data:
        LOGGER.warning("[WARN] No joint angle data recorded to save.")
        return

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    csv_file = save_path / "joint_angles_pre.csv"

    # Match joint names to vector length
    sample_q = saved_data[0][1]
    joint_names = getattr(settings, "joint_angles_names", None)
    if not joint_names or len(joint_names) != len(sample_q):
        if "JOINT_ANGLES_NAMES" in globals() and len(sample_q) == len(JOINT_ANGLES_NAMES):
            joint_names = JOINT_ANGLES_NAMES
        else:
            joint_names = [f"q_{i}" for i in range(len(sample_q))]

    headers = ["frame"] + list(joint_names)

    # Debug log to verify what object is actually being passed in
    if ocp is None:
        LOGGER.warning("[WARN] `ocp` argument passed to `save_joint_angles_csv` is None! Check your function call.")
    else:
        LOGGER.info(f"[DEBUG] `ocp` object passed: {type(ocp).__name__}")

    # Recursive dynamic lookup across ocp, settings, and internal sub-objects
    def find_param(key, default="N/A", max_depth=3):
        roots = [r for r in [ocp, settings] if r is not None]
        visited = set()

        def _search(obj, depth):
            if depth > max_depth or obj is None or id(obj) in visited:
                return None
            visited.add(id(obj))

            # 1. Direct attribute check (wrapped: some solver objects raise
            # non-AttributeError exceptions from property getters, e.g.
            # acados/fatrop wrappers touching a compiled/CasADi backend)
            try:
                if hasattr(obj, key):
                    val = getattr(obj, key, None)
                    if val is not None and not callable(val):
                        return val
            except Exception:
                pass

            # 2. Dictionary check
            if isinstance(obj, dict) and key in obj and obj[key] is not None:
                return obj[key]

            # 3. Recurse into child attributes / properties (including private ones)
            attrs = []
            if hasattr(obj, "__dict__"):
                attrs.extend(obj.__dict__.keys())
            else:
                attrs.extend([a for a in dir(obj) if not a.startswith("__")])

            for attr in attrs:
                if attr.startswith("__") or attr in ("saved_data", "data", "parent"):
                    continue
                try:
                    child = getattr(obj, attr, None)
                    if child is not None and not callable(child) and not isinstance(child, (str, int, float, list, tuple)):
                        res = _search(child, depth + 1)
                        if res is not None:
                            return res
                except Exception:
                    pass
            return None

        for root in roots:
            res = _search(root, 0)
            if res is not None:
                return res
        return default

    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        # --- Benchmark Timing Stats Metadata ---
        if benchmark_stats:
            for key, val in benchmark_stats.items():
                f.write(f"# {key}: {val}\n")

        # --- Solver & OCP Options Metadata ---
        ik_type = getattr(settings, "ik_type", "mhe")
        mhe_backend = getattr(settings, "mhe_backend", "acados")
        f.write(f"# IK Type: {ik_type}\n")
        f.write(f"# Solver: {mhe_backend}\n")
        f.write(f"# nlp_solver_type: {find_param('nlp_solver_type')}\n")
        f.write(f"# qp_solver: {find_param('qp_solver')}\n")
        f.write(f"# hessian_approx: {find_param('hessian_approx')}\n")
        f.write(f"# integrator_type: {find_param('integrator_type')}\n")
        f.write(f"# qp_solver_warm_start: {find_param('qp_solver_warm_start')}\n")
        f.write(f"# nlp_solver_max_iter: {find_param('nlp_solver_max_iter', find_param('mhe_max_iter'))}\n")
        f.write(f"# qp_solver_iter_max: {find_param('qp_solver_iter_max')}\n")
        f.write(f"# tol: {find_param('tol')}\n")
        f.write(f"# globalization: {find_param('globalization')}\n")
        f.write(f"# N: {settings.N}\n")

        # --- CSV Header & Data Rows ---
        writer = csv.writer(f)
        writer.writerow(headers)
        for frame_idx, q_vec, _ in saved_data:
            writer.writerow([frame_idx] + [float(val) for val in q_vec])

    LOGGER.info(
        f"[INFO] Successfully saved {len(saved_data)} frames with benchmarks and parameters to {csv_file.resolve()}"
    )


# -----------------------
# Viser debug helpers
# -----------------------


def _pin_se3_to_viser_pose(M: pin.SE3):
    """Convert a pinocchio SE3 to (wxyz, xyz) as viser scene handles expect."""
    quat = pin.Quaternion(M.rotation)
    wxyz = np.array([quat.w, quat.x, quat.y, quat.z], dtype=np.float64)
    xyz = np.asarray(M.translation, dtype=np.float64)
    return wxyz, xyz


def setup_debug_visuals(
    server: "viser.ViserServer",
    model: pin.Model,
    marker_names,
    triad_length=0.08,
    triad_radius=0.003,
    root="debug",
    clear_root=True,
):
    """Sets up per-joint and per-marker-frame coordinate triads plus a live
    point cloud showing the model's own marker frame positions.

    Unlike meshcat (path-addressable, no handle needed), viser scene nodes
    are mutated through the handle object returned at creation time, so we
    keep those handles around in `dbg` instead of just paths.
    """
    if clear_root:
        # viser has no per-subtree "delete everything under this path"; the
        # handles below get replaced in-place on every setup call instead.
        pass

    dbg = {
        "root": root,
        "joint_handles": {},     # jid -> frame handle
        "marker_handles": {},    # fid -> frame handle
        "model_marker_path": f"/{root}/model_markers",
        "model_marker_handle": None,
        "missing_marker_frames": [],
    }

    for jid in range(1, model.njoints):
        jname = model.names[jid]
        path = f"/{root}/joints/{jid:04d}_{jname}"
        handle = server.scene.add_frame(
            path,
            axes_length=triad_length,
            axes_radius=max(triad_length * 0.03, 0.001),
            show_axes=True,
        )
        dbg["joint_handles"][jid] = handle

    for mk in marker_names:
        try:
            fid = model.getFrameId(mk)
        except Exception:
            fid = None

        if fid is None or fid < 0 or fid >= len(model.frames):
            dbg["missing_marker_frames"].append(mk)
            continue

        path = f"/{root}/marker_frames/{fid:04d}_{mk}"
        handle = server.scene.add_frame(
            path,
            axes_length=triad_length,
            axes_radius=max(triad_length * 0.03, 0.001),
            show_axes=True,
        )
        dbg["marker_handles"][fid] = handle

    dbg["model_marker_handle"] = server.scene.add_point_cloud(
        dbg["model_marker_path"],
        points=np.zeros((0, 3), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.uint8),
        point_size=0.01,
    )

    if dbg["missing_marker_frames"]:
        print("[DEBUG] marker frames missing in model (not registered / not added):")
        print("        ", dbg["missing_marker_frames"])

    return dbg


def update_debug_visuals(server: "viser.ViserServer", model: pin.Model, data: pin.Data, q, dbg):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    for jid, handle in dbg.get("joint_handles", {}).items():
        wxyz, xyz = _pin_se3_to_viser_pose(data.oMi[jid])
        handle.wxyz = wxyz
        handle.position = xyz

    marker_points = []
    for fid, handle in dbg.get("marker_handles", {}).items():
        oMf = data.oMf[fid]
        wxyz, xyz = _pin_se3_to_viser_pose(oMf)
        handle.wxyz = wxyz
        handle.position = xyz
        marker_points.append(oMf.translation)

    if marker_points and dbg.get("model_marker_handle") is not None:
        P = np.stack(marker_points, axis=0).astype(np.float32)
        C = np.tile(np.array([0, 255, 0], dtype=np.uint8), (P.shape[0], 1))
        # Re-adding under the same path replaces the point cloud in place.
        dbg["model_marker_handle"] = server.scene.add_point_cloud(
            dbg["model_marker_path"], points=P, colors=C, point_size=0.01,
        )

# -----------------------
# Named measured markers (debug)
# -----------------------

def setup_measured_markers(server: "viser.ViserServer", marker_names: Sequence[str], radius: float = 0.010, color: int = 0xff0000):
    """Creates one small sphere per named marker and returns a dict of
    name -> handle so positions can be updated cheaply every frame."""
    handles = {}
    for name in marker_names:
        handles[name] = server.scene.add_icosphere(
            f"/markers/measured/{name}",
            radius=radius,
            color=color,
            position=(0.0, 0.0, 0.0),
        )
    return handles


def update_measured_markers(marker_handles: dict, mks_dict: dict):
    for name, p in mks_dict.items():
        handle = marker_handles.get(name)
        if handle is None:
            continue
        try:
            handle.position = np.asarray(p, dtype=float).reshape(3)
        except Exception:
            continue


def run_pipelined(src, est, server, marker_path, mtxs, dists, projections,
                   world_R1_cam, world_T1_cam, settings, total_frames, stop_event,
                   show_nlf=False, visualizer="viser"):
    """`server` is the shared display backend instance -- a viser.ViserServer
    if `visualizer == "viser"`, or a meshcat.Visualizer if
    `visualizer == "meshcat"`. `marker_path` is the scene path the live
    measured-marker point cloud is published under (e.g. "/markers").

    `show_nlf`: if True, pops up a live cv2 window in gpu_worker showing
    YOLO boxes + NLF 2D keypoints overlaid per camera, side by side.
    """
    frame_q = queue.Queue(maxsize=3)
    infer_q = queue.Queue(maxsize=3)
    SENTINEL = None

    read_times, nlf_times, ik_times = [], [], []
    tri_filt_times = []           # triangulation + IIR filter time per frame
    marker_viz_times = []         # marker point-cloud publish time per frame
    solve_times = []              # ik_class.solve(...) time per frame (steady-state only)
    display_viz_times = []        # viz_human.display(q) time per frame (steady-state only)
    frame_latencies = []          # end-to-end: read-start -> ik-done, per frame
    frame_latency_idx = []        # matching frame idx for each entry in frame_latencies
    frame_start_ts = {}           # idx -> perf_counter() when reader started that frame
    frame_start_lock = threading.Lock()

    first_read_ts = [None]        # perf_counter() when the very first frame started reading
    last_done_ts = [None]         # perf_counter() when the most recent frame finished ik
    calib_done_ts = [None]        # perf_counter() when the calibration frame finished (1st completed frame)
    warm_done_ts = [None]         # perf_counter() when the first-call compile-spike frame finished (2nd completed frame)
    span_lock = threading.Lock()

    ik_class_out = [None]         # holds the constructed ik_class after calibration, for post-run diagnostics

    # In-memory storage for offline joint angle & marker logging (avoids disk blocking in ik_worker)
    saved_data = []

    def reader():
        idx = 0
        while not stop_event.is_set() and idx < total_frames:
            t0 = time.perf_counter()
            frames = src.read()
            if frames is None:
                break
            read_times.append((time.perf_counter() - t0) * 1000.0)
            with frame_start_lock:
                frame_start_ts[idx] = t0
            with span_lock:
                if first_read_ts[0] is None:
                    first_read_ts[0] = t0
            frame_q.put((idx, frames))
            idx += 1
        frame_q.put(SENTINEL)

    def gpu_worker():
        while True:
            item = frame_q.get()
            if item is SENTINEL:
                infer_q.put(SENTINEL)
                break
            idx, frames = item
            t0 = time.perf_counter()
            nlf_out, infer_ms, yres, boxes = est.estimate_from_frames(frames)
            nlf_times.append((time.perf_counter() - t0) * 1000.0)

            if show_nlf:
                vis_frames = est.visualize_frames(
                    frames,
                    nlf_out,
                    boxes=boxes,
                    draw_boxes=True,
                    put_text=True,
                    text_prefix="cam",
                )
                cv2.imshow("NLF Output", np.hstack(vis_frames))
                cv2.waitKey(1)

            infer_q.put((idx, frames, nlf_out, boxes))

    def ik_worker():
        first_sample = True
        p3d_buffer = deque(maxlen=settings.N)
        num_channel = 3 * len(settings.marker_names)
        iir_filter = IIR(num_channel=num_channel, sampling_frequency=settings.fs)
        iir_filter.add_filter(order=settings.order, cutoff=settings.cutoff_freq,
                               filter_type=settings.filter_type)

        # Resolved once here instead of re-imported every frame in the hot
        # loop below (module lookup + attribute bind on every iteration adds
        # up at tens-of-Hz).
        mc_geometry = None
        if visualizer == "meshcat":
            import meshcat.geometry as mc_geometry

        human_model = None
        human_data = None
        viz_human = None
        ik_class = None
        x_array = u_array = None
        deque_lstm_dict = None
        markers_handle = None  # persistent viser point-cloud handle, created once below

        while True:
            item = infer_q.get()
            if item is SENTINEL:
                break
            idx, frames, nlf_out, boxes = item
            t0 = time.perf_counter()

            nlf_out_2d = nlf_out["poses2d"]
            NUM_CAMERAS = len(frames)
            if nlf_out_2d is None or len(nlf_out_2d) < NUM_CAMERAS:
                continue

            keypoints_list = [None] * NUM_CAMERAS
            valid_cam_ids = []
            for ii in range(NUM_CAMERAS):
                poses2d = nlf_out_2d[ii]
                if poses2d is None or len(poses2d) == 0 or poses2d[0] is None:
                    continue
                keypoints_list[ii] = poses2d[0].detach().float().cpu().numpy()
                valid_cam_ids.append(ii)

            if len(valid_cam_ids) < 2:
                continue

            t_tri0 = time.perf_counter()
            p3d = triangulate_points(keypoints_list=keypoints_list, mtxs=mtxs,
                                      dists=dists, projections=projections)
            p3d_np = torch.from_numpy(p3d).to(dtype=torch.float32)
            p3d_in_world = np.array([np.dot(world_R1_cam, pt) + world_T1_cam for pt in p3d_np])

            if first_sample:
                for _ in range(settings.N):
                    p3d_buffer.append(p3d_in_world)
            else:
                p3d_buffer.append(p3d_in_world)

            if len(p3d_buffer) != settings.N:
                continue

            p3d_buffer_array = np.array(p3d_buffer)
            filtered = iir_filter.filter(
                np.reshape(p3d_buffer_array, (settings.N, 3 * len(settings.marker_names)))
            )
            filtered = np.reshape(filtered, (settings.N, len(settings.marker_names), 3))
            augmented_markers = filtered[-1]
            t_tri1 = time.perf_counter()
            tri_filt_times.append((t_tri1 - t_tri0) * 1000.0)

            colors = np.zeros((augmented_markers.shape[0], 3), dtype=np.uint8)
            colors[:, 0] = 255  # R
            colors[:, 1] = 0    # G
            colors[:, 2] = 0    # B
            t_mviz0 = time.perf_counter()
            if visualizer == "meshcat":
                # meshcat has no lightweight "mutate existing node" call like
                # viser's handle.points/.colors -- set_object every frame is
                # the normal meshcat update path (same as Viewer.display_markers).
                mc_colors = np.zeros((3, augmented_markers.shape[0]), dtype=np.float32)
                mc_colors[0, :] = 1.0  # R, meshcat wants (3,N) floats in [0,1]
                server[marker_path].set_object(
                    mc_geometry.PointCloud(position=augmented_markers.T.astype(np.float32),
                                            color=mc_colors, size=0.02)
                )
            elif markers_handle is None:
                # First frame only: this actually creates the scene node
                # (geometry, material, transform) -- comparable in cost to
                # meshcat's set_object.
                markers_handle = server.scene.add_point_cloud(
                    marker_path,
                    points=augmented_markers.astype(np.float32),
                    colors=colors,
                    point_size=0.02,
                )
            else:
                # Every subsequent frame: mutate the existing node's buffers
                # in place. This is viser's actual fast path -- it sends a
                # smaller "data changed" message instead of recreating the
                # whole node, unlike meshcat's set_object which has no
                # equivalent lightweight update call.
                markers_handle.points = augmented_markers.astype(np.float32)
                markers_handle.colors = colors
            marker_viz_times.append((time.perf_counter() - t_mviz0) * 1000.0)

            mks_dict = dict(zip(settings.marker_names, augmented_markers))

            if first_sample:
                human = robex.human.HumanLoader(
                    height=settings.human_height,
                    weight=settings.human_weight,
                    gender=settings.human_gender
                ).robot
                human_model = human.model
                human_collision_model = human.collision_model
                human_visual_model = human.visual_model

                human_model = scale_human_model(
                    human_model, mks_dict, gender=settings.human_gender,
                    subject_height=settings.human_height
                )
                human_model = mks_registration(
                    human_model, mks_dict, gender=settings.human_gender,
                    subject_height=settings.human_height
                )

                if visualizer == "meshcat":
                    from pinocchio.visualize import MeshcatVisualizer
                    viz_human = MeshcatVisualizer(human_model, human_collision_model, human_visual_model)
                    viz_human.initViewer(server, open=False)
                    viz_human.loadViewerModel("ref")
                else:
                    viz_human = ViserRobotVisualizer(human_model, human_collision_model, human_visual_model)
                    viz_human.initViewer(viewer=server)
                    viz_human.loadViewerModel(rootNodeName="ref")

                    # loadViewerModel() loads BOTH the collision capsules and the
                    # visual mesh, but only one is actually shown -- and
                    # ViserRobotVisualizer's on/off defaults differ from
                    # MeshcatVisualizer's (which shows visuals, hides collisions,
                    # out of the box). Set explicitly so you get the mesh, not
                    # the capsule mannequin.
                    viz_human.displayCollisions(False)
                    viz_human.displayVisuals(True)

                # viser has no meshcat-style top/bottom gradient background
                # property; drop or replace with viser's environment/lighting
                # controls if you want a custom scene backdrop.

                if settings.ik_type == 'sbs':
                    omega = {key: 1 for key in settings.keys_to_track_list}
                    q = pin.neutral(human_model)
                    ik_class = RT_IK(human_model, mks_dict, q, settings.keys_to_track_list, settings.dt, omega)

                    q = ik_class.solve_ik_sample_casadi()
                    ik_class._q0 = q
                    viz_human.display(q)

                    human_model = recalibrate_marker_frames_in_joint_space(
                        human_model, q, mks_dict, settings.marker_names
                    )
                    human_data = human_model.createData()

                    ik_class = RT_IK(human_model, mks_dict, q, settings.keys_to_track_list, settings.dt, omega)
                    LOGGER.info("[INFO] Model calibration finished, ready to process...")

                elif settings.ik_type == 'mhe':
                    x_array = np.zeros((human_model.nq + human_model.nv, settings.N))
                    x_array[6, :] = 1
                    u_array = np.zeros((human_model.nv, settings.N))
                    deque_lstm_dict = deque(maxlen=settings.N)
                    for _ in range(settings.N):
                        deque_lstm_dict.append(mks_dict)

                    omega = {key: 1 for key in settings.keys_to_track_list}
                    q = pin.neutral(human_model)
                    ik_class = RT_IK(human_model, mks_dict, q, settings.keys_to_track_list, settings.dt, omega)

                    q = ik_class.solve_ik_sample_casadi()
                    ik_class._q0 = q
                    viz_human.display(q)

                    human_model = recalibrate_marker_frames_in_joint_space(
                        human_model, q, mks_dict, settings.marker_names
                    )
                    human_data = human_model.createData()

                    LOGGER.info("[INFO] Model calibration finished, ready to process...")

                    if settings.mhe_backend == 'acados':
                        ik_class = RT_SWIKA_ACADOS(
                            human_model, settings.keys_to_track_list, settings.N, settings.dt,
                            export_dir=settings.acados_export_dir,
                            acados_source_dir=settings.acados_source_dir,
                            max_iter=settings.mhe_max_iter
                        )
                    else:
                        ik_class = RT_SWIKA_FATROP(
                            human_model, settings.keys_to_track_list, settings.N, code=settings.ik_code,
                            max_iter=settings.mhe_max_iter
                        )
                    LOGGER.info("[INFO] Model calibration finished, ready to process...")
                else:
                    raise ValueError("Invalid ik type, should be sbs (sample by sample) or mhe (moving horizon estimation)")

                ik_class_out[0] = ik_class
                first_sample = False

            else:
                if settings.ik_type == 'sbs':
                    t_solve0 = time.perf_counter()
                    ik_class._dict_m = mks_dict
                    q = ik_class.solve_ik_sample_quadprog()
                    ik_class._q0 = q
                    t_solve1 = time.perf_counter()
                    viz_human.display(q)
                    display_viz_times.append((time.perf_counter() - t_solve1) * 1000.0)
                    solve_times.append((t_solve1 - t_solve0) * 1000.0)
                elif settings.ik_type == 'mhe':
                    deque_lstm_dict.append(mks_dict)
                    array_data = np.array([np.hstack([d[marker] for marker in settings.keys_to_track_list])
                                            for d in deque_lstm_dict]).T

                    t_solve0 = time.perf_counter()
                    x_array, u_array = ik_class.solve(x_array, u_array, array_data,
                                                        x_array[:, -1], settings.cost_weights, settings.dt)
                    t_solve1 = time.perf_counter()

                    q = pin.neutral(human_model)
                    q[:] = np.array(x_array[:human_model.nq, -1]).flatten()
                    viz_human.display(q)
                    display_viz_times.append((time.perf_counter() - t_solve1) * 1000.0)
                    solve_times.append((t_solve1 - t_solve0) * 1000.0)
                else:
                    raise ValueError("Invalid ik type, should be sbs (sample by sample) or mhe (moving horizon estimation)")

            # Record frame data in-memory if CSV saving is enabled
            if getattr(settings, "SAVE_CSV", False):
                saved_data.append((idx, np.asarray(q).flatten().copy(), mks_dict.copy()))

            ik_times.append((time.perf_counter() - t0) * 1000.0)

            with frame_start_lock:
                frame_t0 = frame_start_ts.pop(idx, None)
            done_ts = time.perf_counter()
            if frame_t0 is not None:
                frame_latencies.append((done_ts - frame_t0) * 1000.0)
                frame_latency_idx.append(idx)
            with span_lock:
                if calib_done_ts[0] is None:
                    calib_done_ts[0] = done_ts
                elif warm_done_ts[0] is None:
                    warm_done_ts[0] = done_ts
                last_done_ts[0] = done_ts

    threads = [threading.Thread(target=fn, daemon=True) for fn in (reader, gpu_worker, ik_worker)]
    for t in threads: t.start()
    for t in threads: t.join()

    if show_nlf:
        cv2.destroyAllWindows()

    total_first_to_last_s = None
    if first_read_ts[0] is not None and last_done_ts[0] is not None:
        total_first_to_last_s = last_done_ts[0] - first_read_ts[0]

    steady_state_s = None
    if calib_done_ts[0] is not None and last_done_ts[0] is not None:
        steady_state_s = last_done_ts[0] - calib_done_ts[0]

    warm_steady_state_s = None
    if warm_done_ts[0] is not None and last_done_ts[0] is not None:
        warm_steady_state_s = last_done_ts[0] - warm_done_ts[0]

    return (read_times, nlf_times, ik_times, frame_latencies, frame_latency_idx,
            total_first_to_last_s, steady_state_s, warm_steady_state_s,
            tri_filt_times, marker_viz_times, solve_times, display_viz_times,
            ik_class_out[0], saved_data)


def main(args):
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    W = settings.width
    H = settings.height

    if args.online:
        cameras = list_cameras()

        if args.camera_index is not None:
            # Online: enumeration order on the bus is out of our control and
            # tells us nothing about physical identity. --camera-index here
            # is a FIXED convention, not a lookup into device data: label 0
            # always means "1st detected camera", 2 means "2nd", 4 means
            # "3rd", 6 means "4th", i.e. position = label // 2. This is
            # independent of whatever list_cameras() actually returns.
            cam_keys = list(cameras.keys())
            selected = {}
            for label in args.camera_index:
                if label < 0 or label % 2 != 0:
                    raise RuntimeError(
                        f"--camera-index {label}: online camera labels follow the fixed "
                        f"0, 2, 4, 6, ... convention (position = label // 2); "
                        f"got a negative or odd value."
                    )
                pos = label // 2
                if pos >= len(cam_keys):
                    raise RuntimeError(
                        f"--camera-index {label} maps to position {pos} (0-indexed), but "
                        f"only {len(cam_keys)} camera(s) were detected."
                    )
                key = cam_keys[pos]
                selected[key] = cameras[key]
            cameras = selected

        NUM_CAMERAS = len(cameras)
        FRAME_SHAPE = (H, W, 3)
        mtxs, dists, projections, rotations, translations = load_camera_parameters(settings.cam_calib_path, NUM_CAMERAS)
        world_R1_cam, world_T1_cam = load_world_transformation(settings.cam_calib_path)
        camera_buffers, camera_timestamps, camera_locks, frame_counters, camera_barrier, stop_event = create_camera_shared_ressources(NUM_CAMERAS, FRAME_SHAPE)
        results_queues = create_pipeline_shared_ressources()

        camera_processes = [
            Camera(list(cameras.keys())[i],
                camera_buffers[i],
                camera_timestamps[i],
                camera_locks[i],
                frame_counters[i],
                camera_barrier,
                stop_event,
                FRAME_SHAPE,
                settings.fs,
                settings.fourcc,)
            for i in range(NUM_CAMERAS)
        ]

        pipeline = PipelineProcess(
            settings=settings,
            frame_counters=frame_counters,
            camera_buffers=camera_buffers,
            camera_locks=camera_locks,
            timestamp_buffers=camera_timestamps,
            results_queues=results_queues,
            stop_event=stop_event,
            mtxs=mtxs,
            dists=dists,
            projections=projections,
            world_R1_cam=world_R1_cam,
            world_T1_cam=world_T1_cam,
            frame_shape=FRAME_SHAPE,
            num_cameras=NUM_CAMERAS,
            show_nlf=args.show_nlf,
        )

        viewer = ViewerProcess(
            settings=settings,
            results_queues=results_queues,
            stop_event=stop_event,
            num_cameras=NUM_CAMERAS,
            backend=args.visualizer,
        )

        processes = camera_processes + [pipeline, viewer]

        for p in processes:
            p.start()

        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            stop_event.set()
            for process in processes:
                process.stop() if hasattr(process, 'stop') else None
                process.join(timeout=2)

    else: # offline mode

        if args.visualizer == "meshcat":
            import meshcat
            server = meshcat.Visualizer()
            LOGGER.info(f"[INFO] Meshcat visualizer available here: {server.url()}")
            marker_path = "markers"
        else:
            server = viser.ViserServer()
            LOGGER.info(f"[INFO] Viser visualizer available here: http://{server.get_host()}:{server.get_port()}")

            # Ground grid, matching the floor grid MeshcatVisualizer/pinocchio
            # shows by default.
            server.scene.add_grid(
                "/grid",
                width=10.0,
                height=10.0,
                position=(0.0, 0.0, 0.0),
            )

            marker_path = "/markers"

        if args.videos and len(args.videos) > 0:
            paths = [Path(v) for v in args.videos]
        else:
            paths = list_videos(Path(args.data_dir))
        if len(paths) == 0:
            raise RuntimeError(f"No videos found in {args.data_dir}")

        if args.camera_index is not None:
            cam_id_to_path = {}
            for path in paths:
                cid = _extract_cam_id(path.stem)
                if cid is not None:
                    cam_id_to_path[cid] = path

            selected_paths = []
            for idx in args.camera_index:
                if idx not in cam_id_to_path:
                    raise RuntimeError(
                        f"--camera-index {idx} not found: detected camera IDs are "
                        f"{sorted(cam_id_to_path.keys())} (from files {[p.name for p in paths]})"
                    )
                selected_paths.append(cam_id_to_path[idx])
            paths = selected_paths

        NUM_CAMERAS = len(paths)
        mtxs, dists, projections, rotations, translations = load_camera_parameters(settings.cam_calib_path, NUM_CAMERAS)
        world_R1_cam, world_T1_cam = load_world_transformation(settings.cam_calib_path)

        src = OfflineVideoSource(paths=paths, size_wh=(W, H))

        est = NLFEstimator(
            yolo_path=settings.yolo_path,
            nlf_path=settings.nlf_path,
            cano_path=settings.cano_path,
            image_size=(W, H),
            cam_Ks=mtxs,
            indices=settings.nlf_indices,
            conf=settings.yolo_conf,
            imgsz=settings.yolo_imgsz,
            device=settings.device,
        )

        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_frames',
            '-of', 'json', str(paths[0])
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        total_frames = int(data['streams'][0]['nb_frames'])
        LOGGER.info(f"[INFO] Total frames determined from ffprobe: {total_frames}")

        stop_event = threading.Event()

        (read_times, nlf_times, ik_times, frame_latencies, frame_latency_idx,
         total_first_to_last_s, steady_state_s, warm_steady_state_s,
         tri_filt_times, marker_viz_times, solve_times, display_viz_times,
         ik_class, saved_data) = run_pipelined(
            src, est, server, marker_path, mtxs, dists, projections,
            world_R1_cam, world_T1_cam, settings, total_frames, stop_event,
            show_nlf=args.show_nlf, visualizer=args.visualizer
        )

        # Drop ONE-TIME-COST frames (calibration + JIT/first-call spike)
        n_dropped = min(2, len(frame_latencies))
        dropped_idxs = frame_latency_idx[:n_dropped]
        del frame_latencies[:n_dropped]
        del frame_latency_idx[:n_dropped]
        del ik_times[:n_dropped]
        del tri_filt_times[:n_dropped]
        del marker_viz_times[:n_dropped]

        if solve_times:
            solve_times.pop(0)
        if display_viz_times:
            display_viz_times.pop(0)
        if dropped_idxs:
            LOGGER.info(f"[INFO] Dropped one-time-cost frames idx={dropped_idxs} from latency/IK stats")

        n_processed = len(frame_latencies)

        # ==============================================================================
        # 1. BUILD BENCHMARK STATS DICTIONARY (Calculated Once)
        # ==============================================================================
        benchmark_stats = {
            "Total Video Frames": str(total_frames),
            "Frames fully processed": str(n_processed),
        }

        # Pipeline stages
        if read_times:
            benchmark_stats["Read"] = f"mean {np.mean(read_times):.1f} ms | median {np.median(read_times):.1f} ms | max {np.max(read_times):.1f} ms"
        if nlf_times:
            benchmark_stats["NLF"] = f"mean {np.mean(nlf_times):.1f} ms | median {np.median(nlf_times):.1f} ms | max {np.max(nlf_times):.1f} ms"
        if ik_times:
            benchmark_stats["IK"] = f"mean {np.mean(ik_times):.1f} ms | median {np.median(ik_times):.1f} ms | max {np.max(ik_times):.1f} ms"

        # IK Sub-stage breakdown
        if tri_filt_times:
            benchmark_stats["Triangulate+Filter"] = f"mean {np.mean(tri_filt_times):.1f} ms | median {np.median(tri_filt_times):.1f} ms | max {np.max(tri_filt_times):.1f} ms"
        if marker_viz_times:
            benchmark_stats["Marker viz (viser point cloud)"] = f"mean {np.mean(marker_viz_times):.1f} ms | median {np.median(marker_viz_times):.1f} ms | max {np.max(marker_viz_times):.1f} ms"
        if solve_times:
            benchmark_stats["Solve (ik_class.solve/quadprog)"] = f"mean {np.mean(solve_times):.1f} ms | median {np.median(solve_times):.1f} ms | max {np.max(solve_times):.1f} ms"
        if display_viz_times:
            benchmark_stats["Display viz (viser viz_human.display)"] = f"mean {np.mean(display_viz_times):.1f} ms | median {np.median(display_viz_times):.1f} ms | max {np.max(display_viz_times):.1f} ms"

        if frame_latencies:
            benchmark_stats["Full Pipeline (per frame, end-to-end)"] = (
                f"mean {np.mean(frame_latencies):.1f} ms | "
                f"median {np.median(frame_latencies):.1f} ms | "
                f"max {np.max(frame_latencies):.1f} ms"
            )

        # Spans & Throughput
        if steady_state_s is not None:
            benchmark_stats["Post-calibration span (includes first-call compile spike)"] = f"{steady_state_s:.2f} s"
        if warm_steady_state_s is not None and n_processed > 0:
            benchmark_stats["Steady-state time (init, compilation and first frame excluded)"] = f"{warm_steady_state_s:.2f} s"
            benchmark_stats["Steady-state throughput"] = f"{n_processed / warm_steady_state_s:.2f} FPS"

        # ==============================================================================
        # 2. CONSOLE PRINTS (Reading Directly from benchmark_stats)
        # ==============================================================================
        print("\n--- BENCHMARK RESULTS ---")
        print(f"Total Video Frames        : {benchmark_stats['Total Video Frames']}")
        print(f"Frames fully processed    : {benchmark_stats['Frames fully processed']}")
        if "Read" in benchmark_stats:
            print(f"Read:      {benchmark_stats['Read']}")
        if "NLF" in benchmark_stats:
            print(f"NLF:       {benchmark_stats['NLF']}")
        if "IK" in benchmark_stats:
            print(f"IK:        {benchmark_stats['IK']}")

        if "Full Pipeline (per frame, end-to-end)" in benchmark_stats:
            print(f"Full Pipeline (per frame): {benchmark_stats['Full Pipeline (per frame, end-to-end)']}")

        print("\n--- IK sub-stage breakdown (steady-state frames) ---")
        ik_substages = [
            "Triangulate+Filter",
            "Marker viz (viser point cloud)",
            "Solve (ik_class.solve/quadprog)",
            "Display viz (viser viz_human.display)"
        ]
        for key in ik_substages:
            if key in benchmark_stats:
                print(f"{key}: {benchmark_stats[key]}")

        print()
        span_keys = [
            "Post-calibration span (includes first-call compile spike)",
            "Steady-state time (init, compilation and first frame excluded)",
            "Steady-state throughput"
        ]
        for key in span_keys:
            if key in benchmark_stats:
                print(f"{key}: {benchmark_stats[key]}")

        # ==============================================================================
        # 3. SAVE CSV WITH HEADER METADATA
        # ==============================================================================
        if getattr(settings, "SAVE_CSV", False):
            save_dir = getattr(settings, "SAVE_DIR", "./output")
            save_joint_angles_csv(saved_data, save_dir, settings, ocp=ik_class, benchmark_stats=benchmark_stats)

        src.release()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--online", action="store_true")
    p.add_argument("--data-dir", type=str, default="data", help="Folder containing input videos")
    p.add_argument("--videos", nargs="*", default=None, help="Optional explicit list of input videos")

    p.add_argument("--camera-index", type=int, nargs="*", default=None,
                   help="Camera selector -- meaning differs by mode. Online: a FIXED "
                        "positional convention (bus order is arbitrary and uncontrollable), "
                        "label 0/2/4/6 = 1st/2nd/3rd/4th detected camera regardless of "
                        "actual device numbering, i.e. position = label // 2. Offline: "
                        "matches the real 'camera_N' ID burned into video filenames. "
                        "Defaults to all detected, in order.")

    p.add_argument("--show-nlf", action="store_true",
                   help="Show a live cv2 window with YOLO boxes + NLF 2D keypoints "
                        "overlaid per camera, in both online and offline modes.")

    p.add_argument("--visualizer", type=str, choices=["viser", "meshcat"], default="viser",
                   help="3D display backend for the human model + marker point cloud, "
                        "in both online (ViewerProcess) and offline (ik_worker) modes. "
                        "Defaults to viser.")

    args = p.parse_args()

    if args.online:
        set_start_method('spawn')

    main(args)