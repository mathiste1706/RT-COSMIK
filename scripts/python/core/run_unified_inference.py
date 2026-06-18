#!/usr/bin/env python3
import sys
import argparse
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

import cv2
import torch
from multiprocessing import set_start_method

# Workspace Path Appends
SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rtcosmik.nlf.nlf import check_yolo_engine, DisplayConsumerNLF
from rtcosmik.config_loader import settings
from rtcosmik.camera.cam_utils import list_cameras, load_camera_parameters
from rtcosmik.camera.camera import Camera
from rtcosmik.utils.mp_utils import create_camera_shared_ressources
from rtcosmik.utils.videoReader import OfflineVideoSource

from rtcosmik.nlf.PoseEstimationAPI import PoseEstimationAPI
from instanthmr.visualizer import RerunVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True
)
LOGGER = logging.getLogger("UnifiedInference")


def list_videos(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir.resolve()}")
    return [p for p in sorted(data_dir.iterdir()) if p.suffix.lower() in [".mp4"]]


def main(args):
    # Performance Fix: Activate kernel benchmark optimizations for runtime workloads
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    W, H = settings.width, settings.height

    # 1. Resolve Path Tracks
    if args.online:
        cameras = list_cameras()
        NUM_CAMERAS = len(cameras)
        paths = list(cameras.keys())
    else:
        paths = [Path(v) for v in args.videos] if args.videos else list_videos(Path(args.data_dir))
        if not paths:
            raise RuntimeError(f"No matching video tracks identified: {args.data_dir}")
        NUM_CAMERAS = len(paths)

    check_yolo_engine(NUM_CAMERAS)
    
    # Extract physical transformation calibration configurations
    mtxs, dists, projections, rotations, translations = load_camera_parameters(
        settings.cam_calib_path, NUM_CAMERAS
    )

    # 2. Map Multi-View Parameter Matrices Safely Into Configuration Dictionary
    pipeline_config = {
        "yolo_path": settings.yolo_path,
        "model_path": settings.nlf_path if args.mode == "nlf" else args.hmr_model,
        "smplx_path": settings.cano_path,
        "img_size": (W, H),
        "indices": settings.nlf_indices,
        "conf": settings.yolo_conf,
        "imgsz": settings.yolo_imgsz,
        "device": settings.device,
        
        # Geometrical properties passed to tracking coordinators
        "cam_Ks": mtxs,
        "cam_Rs": rotations,
        "cam_Ts": translations,
        "dists": dists,
        "projections": projections,
        
        # Triangulation activation flags (Fixed: Restored missing comma separation)
        "nlf_triangulation": args.nlf_triangulation,
        "hmr_triangulation_mode": args.hmr_triangulation
    }

    # 3. Instantiate Architecture Coordinator Engine
    LOGGER.info(f"Initializing Tracking Framework on target engine: {args.mode.upper()}")
    api = PoseEstimationAPI(mode=args.mode, config=pipeline_config)

    # 4. Route Operational Executions
    if args.online:
        FRAME_SHAPE = (H, W, 3)
        camera_buffers, camera_timestamps, camera_locks, frame_counters, camera_barrier, stop_event = \
            create_camera_shared_ressources(NUM_CAMERAS, FRAME_SHAPE)

        camera_processes = [
            Camera(paths[i], camera_buffers[i], camera_timestamps[i], camera_locks[i], 
                   frame_counters[i], camera_barrier, stop_event, FRAME_SHAPE, settings.fs, settings.fourcc)
            for i in range(NUM_CAMERAS)
        ]

        display = DisplayConsumerNLF(
            settings=settings, frame_counters=frame_counters, camera_buffers=camera_buffers,
            camera_locks=camera_locks, timestamp_buffers=camera_timestamps, stop_event=stop_event,
            mtxs=mtxs, frame_shape=FRAME_SHAPE, num_cameras=NUM_CAMERAS,
        )

        processes = camera_processes + [display]
        for p in processes:
            p.start()

        try:
            while not stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            stop_event.set()
            for p in processes:
                p.stop() if hasattr(p, 'stop') else None
                p.join(timeout=2)

    else:  # --- Offline Batch Processing Loop Track ---
        src = OfflineVideoSource(paths=paths, size_wh=(W, H))
        frame_idx = 0
        
        while True:
            frames = src.read()
            if frames is None:
                break

            output = api.process_frames(frames)
            print(f"[{api.mode.upper()}] Timings Diagnostics Layout -> {output['timings']}")

            frame_idx += 1

        src.release()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Unified Tracking and Coordinate Verification Engine")
    p.add_argument("--mode", type=str, default="instant_hmr", choices=["nlf", "instant_hmr"],
                   help="Select active estimation architecture module target strategy.")
    p.add_argument("--hmr-model", type=str, default=f"{SRC_ROOT}/InstantHMR/models/instanthmr.onnx",
                   help="Explicit path configuration pointing to a deployed InstantHMR engine model.")
    p.add_argument("--hmr-triangulation", type=str, default="dlt", choices=["dlt", "native"],
                   help="Geometrical track selection mode used during InstantHMR processing.")
    p.add_argument("--nlf-triangulation", action="store_true", default=True,
                   help="Flag to activate algebraic point-cloud triangulation configurations inside NLF loops.")
    p.add_argument("--online", action="store_true", help="Initialize real-time hardware execution camera pipelines.")
    p.add_argument("--data-dir", type=str, default="data", help="Target path tracking data location directory.")
    p.add_argument("--videos", nargs="*", default=None, help="Explicit list definitions for batch media targets.")
    
    args = p.parse_args()
    if args.online:
        set_start_method('spawn', force=True)

    main(args)