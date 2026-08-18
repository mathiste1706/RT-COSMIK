#!/usr/bin/env python3
import re
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import time
from typing import List, Optional, Tuple
import logging
from multiprocessing import Process, Queue, set_start_method

import cv2
import numpy as np
import torch

from rtcosmik.nlf.nlf import NLFEstimator, DisplayConsumerNLF
from rtcosmik.config_loader import settings
from rtcosmik.camera.cam_utils import list_cameras, load_camera_parameters
from rtcosmik.camera.camera import Camera
from rtcosmik.utils.mp_utils import create_camera_shared_ressources
from rtcosmik.utils.videoReader import OfflineVideoSource, list_videos

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True
)

class VideoWriterProcess(Process):
    """Asynchronous video writer process for BGR streams from OfflineVideoSource."""
    def __init__(self, output_paths: List[str], fps: float, frame_size: Tuple[int, int]):
        super().__init__()
        self.output_paths = [str(p) for p in output_paths]
        self.fps = fps
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.queue = Queue(maxsize=300)

    def run(self):
        codecs = ['mp4v', 'avc1', 'XVID', 'MJPG']
        writers = []
        
        for path_str in self.output_paths:
            writer = None
            for codec in codecs:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                w = cv2.VideoWriter(path_str, fourcc, self.fps, self.frame_size)
                if w.isOpened():
                    writer = w
                    break
                w.release()
            writers.append(writer)

        try:
            while True:
                item = self.queue.get()
                if item is None:  # Shutdown signal
                    break
                
                frames = item
                for idx, frame in enumerate(frames):
                    w = writers[idx]
                    if w is not None and w.isOpened():
                        h, w_dim = frame.shape[:2]
                        if (w_dim, h) != self.frame_size:
                            frame = cv2.resize(frame, self.frame_size)

                        if not frame.flags['C_CONTIGUOUS']:
                            frame = np.ascontiguousarray(frame)

                        w.write(frame)
        finally:
            for w in writers:
                if w is not None:
                    w.release()

    def write(self, frames: List[np.ndarray]):
        self.queue.put([f.copy() for f in frames], block=True)

    def stop(self):
        self.queue.put(None)

def main(args):

    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    W = settings.width
    H = settings.height

    if args.online:
        cameras = list_cameras()

        if args.camera_index is not None:
            # label 0: 1st camera 
            # label 2: 2nd camera
            # label 4: 3rd camera
            # label 6: 4th camera
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
        elif args.num_cameras is not None:
            if args.num_cameras > len(cameras):
                raise RuntimeError(
                    f"--num-cameras={args.num_cameras} but only {len(cameras)} "
                    f"camera(s) detected: {list(cameras.keys())}"
                )
            cameras = dict(list(cameras.items())[:args.num_cameras])


        NUM_CAMERAS = len(cameras)
        FRAME_SHAPE = (H, W, 3)
        mtxs, dists, projections, rotations, translations = load_camera_parameters(settings.cam_calib_path, NUM_CAMERAS)
        camera_buffers, camera_timestamps, camera_locks, frame_counters, camera_barrier, stop_event = create_camera_shared_ressources(NUM_CAMERAS, FRAME_SHAPE)

        camera_processes = [
            Camera(
                list(cameras.keys())[i], 
                camera_buffers[i], 
                camera_timestamps[i], 
                camera_locks[i], 
                frame_counters[i], 
                camera_barrier, 
                stop_event,
                FRAME_SHAPE, 
                settings.fs, 
                settings.fourcc,
            )
            for i in range(NUM_CAMERAS)
        ]

        display = DisplayConsumerNLF(
            settings=settings,
            frame_counters=frame_counters,
            camera_buffers=camera_buffers,
            camera_locks=camera_locks,
            timestamp_buffers=camera_timestamps,
            stop_event=stop_event,
            mtxs=mtxs,
            frame_shape=FRAME_SHAPE,
            num_cameras=NUM_CAMERAS,
        )

        processes = camera_processes + [display]

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
        if args.videos and len(args.videos) > 0:
            paths = [Path(v) for v in args.videos]
        else:
            paths = list_videos(Path(args.data_dir))
        if len(paths) == 0:
            raise RuntimeError(f"No videos found in {args.data_dir}")

        if args.camera_index is not None:
            cam_id_to_path = {}
            for path in paths:
                s = str(path.stem)
                m = re.search(r'camera_(\d+)', s, re.IGNORECASE)
                if m:
                    cid=int(m[1])
                else:
                    cid=None
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
        
        writer_proc = None
        if args.save_sync_check:
            output_files = [f"sync_check_video{i+1}.mp4" for i in range(NUM_CAMERAS)]
            writer_proc = VideoWriterProcess(output_paths=output_files, fps=40.0, frame_size=(W, H))
            writer_proc.start()

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

        WINDOW_NAME = "Visualization"
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        try:
            while True:
                frames = src.read()
                if frames is None:
                    logging.info("[INFO] Reached end of video stream.")
                    break

                nlf_out, infer_ms, yres, boxes = est.estimate_from_frames(frames)
                print(f"Timings to perform inference = {infer_ms}")

                vis_frames = est.visualize_frames(
                    frames,
                    nlf_out,
                    boxes=boxes,
                    draw_boxes=True,
                    put_text=True,
                    text_prefix="cam",
                )

                if writer_proc is not None:
                    writer_proc.write(vis_frames)

                vis = np.hstack(vis_frames)
                cv2.imshow(WINDOW_NAME, vis)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    logging.info("[INFO] User requested exit.")
                    break
        except KeyboardInterrupt:
            logging.info("[INFO] Interrupted by user. Finalizing video files...")
        finally:
            src.release()
            
            if writer_proc is not None:
                logging.info("[INFO] Flushing queued frames to disk...")
                writer_proc.stop()
                writer_proc.join()
                logging.info("[INFO] Videos saved successfully.")

            cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--online", action="store_true")
    p.add_argument("--save-sync-check", action="store_true", help="Save stream outputs to disk asynchronously")
    p.add_argument("--data-dir", type=str, default="data", help="Folder containing input videos")
    p.add_argument("--videos", nargs="*", default=None, help="Optional explicit list of input videos")
    p.add_argument("--camera-index", type=int, nargs="*", default=None,
               help="Explicit camera indices/IDs to use (e.g. --camera-index 0 3 5). "
                    "Defaults to all detected, in order.")
    p.add_argument("--show-nlf", action="store_true",
                   help="Show a live cv2 window with YOLO boxes + NLF 2D keypoints "
                        "overlaid per camera, in both online and offline modes.")
 
    p.add_argument("--visualizer", type=str, choices=["viser", "meshcat"], default="viser",
                   help="3D display backend for the human model + marker point cloud, "
                        "in both online (ViewerProcess) and offline (ik_worker) modes. "
                        "Defaults to viser.")

    args = p.parse_args()

    set_start_method('spawn', force=True)
    main(args)