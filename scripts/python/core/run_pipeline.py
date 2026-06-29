#!/usr/bin/env python3
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import json
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
import pinocchio as pin
import torch
import example_robot_data as robex
from pinocchio.visualize import MeshcatVisualizer
from multiprocessing import set_start_method

from rtcosmik.config_loader import settings
from rtcosmik.filtering.iir import IIR
from rtcosmik.human_model.model_utils import (
    scale_human_model,
    mks_registration,
    recalibrate_marker_frames_in_joint_space,
)
from rtcosmik.ik.ik import RT_IK, RT_SWIKA_FATROP, RT_SWIKA_ACADOS
from rtcosmik.camera.cam_utils import (
    list_cameras,
    load_camera_parameters,
    load_world_transformation,
)
from rtcosmik.camera.camera import Camera
from rtcosmik.utils.mp_utils import (
    create_camera_shared_ressources,
    create_pipeline_shared_ressources,
)
from rtcosmik.pipeline.pipeline import PipelineProcess
from rtcosmik.viewer.viewer import ViewerProcess
from rtcosmik.nlf.PoseEstimationAPI import InstantHMREstimatorWrapper

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True,
)
LOGGER = logging.getLogger(__name__)


# =============================================================================
# Synthetic biomarker generation  (70 joints → Direct Structural Mapping)
# =============================================================================

def _safe_unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-4 else fallback


def generate_synthetic_biomarkers(joints: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Génère les marqueurs de surface avec des axes orthogonaux alignés 
    sur la convention Pinocchio standard (Z-Up, X-Forward, Y-Left).
    """
    L_hip, R_hip = joints[9],  joints[10]
    L_sho, R_sho = joints[5],  joints[6]
    neck          = joints[69]

    hip_mid = (L_hip + R_hip) / 2.0
    sho_mid = (L_sho + R_sho) / 2.0

    # --- Alignement des axes (Règle de la main droite - Standard Z-Up) ---
    u_up    = _safe_unit(sho_mid - hip_mid,       np.array([0.0, 0.0, 1.0])) # Vertical (+Z)
    u_fwd   = _safe_unit(np.cross(L_hip - R_hip, u_up), np.array([1.0, 0.0, 0.0])) # Avant (+X)
    u_left  = _safe_unit(np.cross(u_up, u_fwd),   np.array([0.0, 1.0, 0.0])) # Gauche (+Y)
    u_right = -u_left                                                        # Droite (-Y)

    mks: Dict[str, np.ndarray] = {}

    # ---- Bassin (Positionnement anatomique réel) ----
    mks["RASI"] = R_hip + 0.08 * u_fwd
    mks["LASI"] = L_hip + 0.08 * u_fwd
    mks["RPSI"] = R_hip - 0.08 * u_fwd
    mks["LPSI"] = L_hip - 0.08 * u_fwd
    mks["SACR"] = hip_mid - 0.09 * u_fwd

    # ---- Tronc & Colonne ----
    mks["C7"]  = neck + 0.03 * u_up - 0.05 * u_fwd
    mks["T11"] = hip_mid + 0.33 * (sho_mid - hip_mid) - 0.05 * u_fwd
    mks["T6"]  = hip_mid + 0.66 * (sho_mid - hip_mid) - 0.05 * u_fwd

    # ---- Épaules ----
    mks["RSHO"] = R_sho + 0.02 * u_up
    mks["LSHO"] = L_sho + 0.02 * u_up

    # ---- Bras (Latéral vs Médial) ----
    mks["RELB"]  = joints[8] + 0.03 * u_right
    mks["RMELB"] = joints[8] + 0.03 * u_left
    mks["LELB"]  = joints[7] + 0.03 * u_left
    mks["LMELB"] = joints[7] + 0.03 * u_right
    
    mks["RWRI"]  = joints[41] + 0.02 * u_fwd
    mks["RMWRI"] = joints[41] - 0.02 * u_fwd
    mks["LWRI"]  = joints[62] + 0.02 * u_fwd
    mks["LMWRI"] = joints[62] - 0.02 * u_fwd

    # ---- Mains ----
    mks["RTHU"] = joints[21]
    mks["LTHU"] = joints[42]
    mks["RMID"] = joints[29]
    mks["LMID"] = joints[50]
    mks["RPIN"] = joints[37]
    mks["LPIN"] = joints[58]

    # ---- Jambes (Latéral vs Médial) ----
    mks["RKNE"]  = joints[12] + 0.04 * u_right
    mks["RMKNE"] = joints[12] + 0.04 * u_left
    mks["LKNE"]  = joints[11] + 0.04 * u_left
    mks["LMKNE"] = joints[11] + 0.04 * u_right

    mks["RANK"]  = joints[14] + 0.03 * u_right
    mks["RMANK"] = joints[14] + 0.03 * u_left
    mks["LANK"]  = joints[13] + 0.03 * u_left
    mks["LMANK"] = joints[13] + 0.03 * u_right

    # ---- Pieds ----
    mks["RTOE"]  = joints[18]
    mks["LTOE"]  = joints[15]
    mks["R5MHD"] = joints[19]
    mks["L5MHD"] = joints[16]
    mks["RHEE"]  = joints[20] - 0.05 * u_fwd
    mks["LHEE"]  = joints[17] - 0.05 * u_fwd

    # ---- Visage & Tête ----
    mks["Nose"] = joints[0]
    mks["LEye"] = joints[1]
    mks["REye"] = joints[2]
    mks["LEar"] = joints[3]
    mks["REar"] = joints[4]
    
    head_mid = (joints[1] + joints[2]) / 2.0
    mks["Head"] = neck + 1.25 * (head_mid - neck)

    mks["RFHD"] = joints[2] + 0.02 * u_up + 0.02 * u_fwd
    mks["LFHD"] = joints[1] + 0.02 * u_up + 0.02 * u_fwd
    mks["RBHD"] = joints[4] + 0.02 * u_up - 0.02 * u_fwd
    mks["LBHD"] = joints[3] + 0.02 * u_up - 0.02 * u_fwd

    lowercase_mks = {k.lower(): v for k, v in mks.items()}
    uppercase_mks = {k.upper(): v for k, v in mks.items()}
    mks.update(lowercase_mks)
    mks.update(uppercase_mks)

    return mks


# =============================================================================
# Video helpers
# =============================================================================

def list_videos(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"data dir does not exist: {data_dir}")
    return [p for p in sorted(data_dir.iterdir()) if p.suffix.lower() == ".mp4"]


def ffprobe_frame_count(video_path: Path) -> int:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return int(data["streams"][0]["nb_frames"])


@dataclass
class OfflineVideoSource:
    paths: List[Path]
    size_wh: Tuple[int, int]
    caps: List[cv2.VideoCapture] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.caps = [cv2.VideoCapture(str(p)) for p in self.paths]
        for p, cap in zip(self.paths, self.caps):
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {p}")

    def read(self) -> Optional[List[np.ndarray]]:
        frames: List[np.ndarray] = []
        W, H = self.size_wh
        for cap in self.caps:
            ok, frame = cap.read()
            if not ok:
                return None
            if frame.shape[1] != W or frame.shape[0] != H:
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        return frames

    def release(self):
        for cap in self.caps:
            cap.release()


# =============================================================================
# Meshcat visualisation helpers
# =============================================================================

def _pin_se3_to_tf(M: pin.SE3) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = M.rotation
    T[:3, 3]  = M.translation
    return T


def setup_measured_markers(
    vis: meshcat.Visualizer,
    marker_names: Sequence[str],
    radius: float = 0.010,
    color: int = 0xFF0000,
):
    sphere = g.Sphere(radius)
    mat = g.MeshPhongMaterial(color=color, opacity=0.9)
    for name in marker_names:
        vis[f"markers/measured/{name}"].set_object(sphere, mat)


def update_measured_markers(vis: meshcat.Visualizer, mks_dict: Dict[str, np.ndarray]):
    for name, p in mks_dict.items():
        try:
            T = tf.translation_matrix(np.asarray(p, dtype=float).reshape(3))
        except Exception:
            continue
        vis[f"markers/measured/{name}"].set_transform(T)


# =============================================================================
# Main
# =============================================================================

def main(args):
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    W, H = settings.width, settings.height
    world_R1_cam, world_T1_cam = load_world_transformation(settings.cam_calib_path)
    world_T1_cam_flat = world_T1_cam.flatten()

    mtxs, dists_cal, projections_cal, rotations, translations = load_camera_parameters(
        settings.cam_calib_path
    )

    if args.online:
        cameras = list_cameras()
        NUM_CAMERAS = len(cameras)
        FRAME_SHAPE = (H, W, 3)

        camera_buffers, camera_timestamps, camera_locks, frame_counters, camera_barrier, stop_event = \
            create_camera_shared_ressources(NUM_CAMERAS, FRAME_SHAPE)
        results_queues = create_pipeline_shared_ressources()

        camera_processes = [
            Camera(
                list(cameras.keys())[i],
                camera_buffers[i], camera_timestamps[i], camera_locks[i],
                frame_counters[i], camera_barrier, stop_event,
                FRAME_SHAPE, settings.fs, settings.fourcc,
            )
            for i in range(NUM_CAMERAS)
        ]

        pipeline = PipelineProcess(
            settings=settings, frame_counters=frame_counters, camera_buffers=camera_buffers,
            camera_locks=camera_locks, timestamp_buffers=camera_timestamps, results_queues=results_queues,
            stop_event=stop_event, mtxs=mtxs, dists=dists_cal, projections=projections_cal,
            world_R1_cam=world_R1_cam, world_T1_cam=world_T1_cam, frame_shape=FRAME_SHAPE, num_cameras=NUM_CAMERAS,
        )
        viewer = ViewerProcess(
            settings=settings, results_queues=results_queues, stop_event=stop_event, num_cameras=NUM_CAMERAS,
        )

        processes = camera_processes + [pipeline, viewer]
        for p in processes:
            p.start()

        try:
            while True: time.sleep(0.1)
        except KeyboardInterrupt:
            stop_event.set()
            for process in processes:
                if hasattr(process, "stop"): process.stop()
                process.join(timeout=2)
        return

    # OFFLINE mode
    vis = meshcat.Visualizer()
    LOGGER.info(f"[INFO] Meshcat URL: {vis.url()}")
    vis_markers = vis["markers"]

    if args.videos:
        paths = [Path(v) for v in args.videos]
    else:
        paths = list_videos(Path(args.data_dir))
    if not paths:
        raise RuntimeError(f"No videos found in {args.data_dir}")

    NUM_CAMERAS = len(paths)
    total_frames = ffprobe_frame_count(paths[0])
    LOGGER.info(f"[INFO] Total frames: {total_frames}")

    src = OfflineVideoSource(paths=paths, size_wh=(W, H))

    est = InstantHMREstimatorWrapper(
        model_path=getattr(
            settings, "model_path",
            "/root/workspace/RT-COSMIK/src/InstantHMR/models/instanthmr.onnx",
        ),
        device=settings.device, yolo_path=settings.yolo_path, config={"conf": settings.yolo_conf, "imgsz": 640},
    )

    first_sample = True
    frame_counter = 0
    p3d_buffer = deque(maxlen=settings.N)

    num_channel = 3 * len(settings.marker_names)
    iir_filter = IIR(num_channel=num_channel, sampling_frequency=settings.fs)
    iir_filter.add_filter(order=settings.order, cutoff=settings.cutoff_freq, filter_type=settings.filter_type)

    total_history = []
    ik_history = []

    human_model = human_data = viz_human = ik_class = None
    x_array = u_array = deque_lstm_dict = None
    
    # Conservation stricte de l'état de configuration d'une frame à l'autre
    q = None 

    try:
        while frame_counter < total_frames:
            t0 = time.perf_counter()

            frames = src.read()
            if frames is None: break

            hmr_predictions, timings, detections, bboxes, images_rgb = est.estimate(frames)

            if not hmr_predictions or hmr_predictions[0] is None:
                frame_counter += 1
                continue

            pred0 = hmr_predictions[0]
            joints_3d_cam = np.asarray(pred0.joints_3d_cam, dtype=np.float64)
            if joints_3d_cam.ndim == 3:
                joints_3d_cam = joints_3d_cam[0]

            p3d_world = (world_R1_cam @ joints_3d_cam.T).T + world_T1_cam_flat
            synthetic_mks = generate_synthetic_biomarkers(p3d_world)

            try:
                ordered_markers = np.array(
                    [synthetic_mks[name] for name in settings.marker_names],
                    dtype=np.float64,
                )
            except KeyError as e:
                LOGGER.warning(f"Missing marker in synthetic dict: {e} – skipping frame")
                frame_counter += 1
                continue

            if first_sample:
                for _ in range(settings.N):
                    p3d_buffer.append(ordered_markers)
            else:
                p3d_buffer.append(ordered_markers)

            if len(p3d_buffer) < settings.N:
                frame_counter += 1
                continue

            p3d_buf_arr = np.array(p3d_buffer)
            filtered = iir_filter.filter(np.reshape(p3d_buf_arr, (settings.N, num_channel)))
            filtered = np.reshape(filtered, (settings.N, len(settings.marker_names), 3))
            augmented_markers = filtered[-1]

            colors = np.zeros((3, len(settings.marker_names)), dtype=np.float64)
            colors[0, :] = 1.0
            vis_markers.set_object(g.PointCloud(position=augmented_markers.T, color=colors, size=0.02))

            mks_dict = dict(zip(settings.marker_names, augmented_markers))

            if first_sample:
                LOGGER.info("[INFO] Running model calibration on first frame …")

                human = robex.human.HumanLoader(
                    height=settings.human_height, weight=settings.human_weight, gender=settings.human_gender,
                ).robot
                human_model, human_collision_model, human_visual_model = human.model, human.collision_model, human.visual_model

                human_model = mks_registration(
                    human_model, mks_dict, gender=settings.human_gender, subject_height=settings.human_height,
                )

                human_model = scale_human_model(
                    human_model, mks_dict,
                    gender=settings.human_gender,
                    subject_height=settings.human_height,
                )
                human_model = mks_registration(
                    human_model, mks_dict,
                    gender=settings.human_gender,
                    subject_height=settings.human_height,
                )

                viz_human = MeshcatVisualizer(human_model, human_collision_model, human_visual_model)
                viz_human.initViewer(vis, open=True)
                try: vis["ref"].delete()
                except Exception: pass
                
                viz_human.loadViewerModel("ref")
                viz_human.viewer["/Background"].set_property("top_color",    [1, 1, 1])
                viz_human.viewer["/Background"].set_property("bottom_color", [0.65, 0.65, 0.65])

                # 1. Initialisation du vecteur de configuration
                q = pin.neutral(human_model)

                # 2. ALIGNEMENT RACINE INITIAL (Téléporte le modèle au centre du nuage)
                pelvis_keys = ["RASI", "LASI", "RPSI", "LPSI", "rasi", "lasi", "rpsi", "lpsi"]
                pelvis_pts = [mks_dict[k] for k in pelvis_keys if k in mks_dict]
                if len(pelvis_pts) >= 2:
                    root_translation = np.mean(pelvis_pts, axis=0)
                else:
                    root_translation = np.mean(list(mks_dict.values()), axis=0)
                
                q[:3] = root_translation  # Positionne la base au cœur des marqueurs

                # Calcul de l'orientation globale de la base (X=avant, Y=gauche, Z=haut)
                try:
                    r_hip_m = (mks_dict.get("RASI") + mks_dict.get("RPSI")) / 2.0
                    l_hip_m = (mks_dict.get("LASI") + mks_dict.get("LPSI")) / 2.0
                    sho_mid_m = (mks_dict.get("RSHO") + mks_dict.get("LSHO")) / 2.0
                    hip_mid_m = (l_hip_m + r_hip_m) / 2.0
                    
                    u_up_m = _safe_unit(sho_mid_m - hip_mid_m, np.array([0.0, 0.0, 1.0]))
                    u_fwd_m = _safe_unit(np.cross(l_hip_m - r_hip_m, u_up_m), np.array([1.0, 0.0, 0.0]))
                    u_left_m = _safe_unit(np.cross(u_up_m, u_fwd_m), np.array([0.0, 1.0, 0.0]))
                    
                    R_matrix = np.column_stack((u_fwd_m, u_left_m, u_up_m))
                    q[3:7] = pin.Quaternion(R_matrix).coeffs()  # [qx, qy, qz, qw]
                except Exception:
                    q[3:7] = np.array([0.0, 0.0, 0.0, 1.0])

                # 3. Calibration locale des marqueurs basée sur la position globale correcte
                human_model = recalibrate_marker_frames_in_joint_space(human_model, q, mks_dict, settings.marker_names)
                viz_human.display(q)

                q = pin.neutral(human_model)

                if settings.ik_type == "sbs":
                    omega = {key: 1 for key in settings.keys_to_track_list}
                    ik_class = RT_IK(human_model, mks_dict, q, settings.keys_to_track_list, settings.dt, omega)
                    q = ik_class.solve_ik_sample_casadi()
                    ik_class._q0 = q
                    viz_human.display(q)
                    human_model = recalibrate_marker_frames_in_joint_space(human_model, q, mks_dict, settings.marker_names)
                    human_data = human_model.createData()
                    ik_class = RT_IK(human_model, mks_dict, q, settings.keys_to_track_list, settings.dt, omega)

                elif settings.ik_type == "mhe":
                    omega = {key: 1 for key in settings.keys_to_track_list}
                    ik_bootstrap = RT_IK(human_model, mks_dict, q, settings.keys_to_track_list, settings.dt, omega)
                    q = ik_bootstrap.solve_ik_sample_casadi()
                    viz_human.display(q)
                    human_model = recalibrate_marker_frames_in_joint_space(human_model, q, mks_dict, settings.marker_names)
                    human_data = human_model.createData()

                    nq, nv = human_model.nq, human_model.nv
                    x_array = np.zeros((nq + nv, settings.N))
                    x_array[6, :] = 1.0
                    u_array = np.zeros((nv, settings.N))
                    deque_lstm_dict = deque(maxlen=settings.N)
                    for _ in range(settings.N):
                        deque_lstm_dict.append(mks_dict)
                    array_data = np.array([np.hstack([d[m] for m in settings.keys_to_track_list]) for d in deque_lstm_dict]).T

                    if settings.mhe_backend == "acados":
                        ik_class = RT_SWIKA_ACADOS(human_model, settings.keys_to_track_list, settings.N, settings.dt,
                            export_dir=settings.acados_export_dir, acados_source_dir=settings.acados_source_dir)
                    else:
                        ik_class = RT_SWIKA_FATROP(human_model, settings.keys_to_track_list, settings.N, code=settings.ik_code)

                    x_array, u_array = ik_class.solve(x_array, u_array, array_data, x_array[:, -1], settings.cost_weights, settings.dt)
                    q = pin.neutral(human_model)
                    q[:] = x_array[:nq, -1]
                    viz_human.display(q)
                else:
                    raise ValueError(f"Invalid ik_type '{settings.ik_type}'. Expected 'sbs' or 'mhe'.")

                LOGGER.info("[INFO] Model calibration finished – ready to process.")
                first_sample = False
            else:
                t_ik = time.perf_counter()

                if settings.ik_type == "sbs":
                    ik_class._dict_m = mks_dict
                    q = ik_class.solve_ik_sample_quadprog()
                    ik_class._q0 = q
                    viz_human.display(q)

                elif settings.ik_type == "mhe":
                    deque_lstm_dict.append(mks_dict)
                    array_data = np.array([np.hstack([d[m] for m in settings.keys_to_track_list]) for d in deque_lstm_dict]).T

                    # --- CORRECTION DE L'ACTUALISATION DE L'HORIZON MHE ---
                    # 1. On extrait la configuration actuelle estimée lors du cycle précédent
                    #    pour l'imposer comme contrainte d'état initial (x0) de notre problème courant.
                    x0_current = x_array[:, -1].copy()

                    try:
                        # 2. On laisse le solveur gérer son Warm-Start interne (x_array, u_array intacts)
                        #    en lui fournissant simplement les nouvelles cibles glissantes (array_data)
                        x_sol, u_sol = ik_class.solve(
                            x_array, u_array, array_data, x0_current, settings.cost_weights, settings.dt
                        )
                        
                        # --- FILTRE DE SÉCURITÉ ANTI-DIVÈRGENCE / NAN ---
                        if not np.isnan(x_sol).any() and np.linalg.norm(x_sol[:3, -1] - x0_current[:3]) < 1.5:
                            x_array, u_array = x_sol, u_sol
                            q[:] = x_array[:human_model.nq, -1] # Extraction de l'état final stabilisé
                            viz_human.display(q)
                        else:
                            LOGGER.warning(f"[WARNING] Solver output invalid or jumped too far at frame {frame_counter}. Posture locked.")
                            # On réinjecte la dernière configuration valide pour réinitialiser le warm-start
                            x_array[:human_model.nq, :] = q[:, None]
                            x_array[human_model.nq:, :] = 0.0
                    except Exception as e:
                        LOGGER.error(f"[ERROR] Solver crashed at frame {frame_counter}: {e}")
                        x_array[:human_model.nq, :] = q[:, None]
                        x_array[human_model.nq:, :] = 0.0

                ik_history.append((time.perf_counter() - t_ik) * 1000.0)

            frame_counter += 1
            total_history.append((time.perf_counter() - t0) * 1000.0)

    finally:
        src.release()

    if len(ik_history)    > 1: ik_history.pop(0)
    if len(total_history) > 1: total_history.pop(0)

    print("\n--- BENCHMARK RESULTS ---")
    print(f"Total frames processed : {frame_counter}")
    if total_history:
        print(f"Total pipeline – mean  : {np.mean(total_history):.1f} ms  |  median : {np.median(total_history):.1f} ms  |  FPS : {1000.0 / np.mean(total_history):.1f}")
    if ik_history:
        print(f"IK solver      – mean  : {np.mean(ik_history):.1f} ms  |  median : {np.median(ik_history):.1f} ms")
    print("-" * 40)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RT-COSMIK offline pipeline – InstantHMR backend")
    p.add_argument("--online", action="store_true", help="Use live cameras instead of recorded videos")
    p.add_argument("--data-dir", type=str, default="data", help="Folder containing .mp4 video files")
    p.add_argument("--videos", nargs="*", default=None, help="Explicit list of video paths (overrides --data-dir)")
    args = p.parse_args()

    if args.online:
        set_start_method("spawn")

    main(args)