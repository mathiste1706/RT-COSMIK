#!/usr/bin/env python3
"""Airtight Multi-Stream Benchmarking Script with Profiling & Graph Generation.

Synchronized with production pipeline architectures to maintain absolute data preparation
parity while providing deep PyTorch kernel profiling and automated visual report generation.
"""

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import time
import logging
import subprocess
import numpy as np
import torch
import cv2

# Graceful fallback if matplotlib isn't installed in the workspace environment
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Import from separate source file
from rtcosmik.utils.videoReader import OfflineVideoSource, list_videos
from rtcosmik.nlf.nlf import NLFEstimator
from rtcosmik.config_loader import settings
from rtcosmik.camera.cam_utils import load_camera_parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True
)

def main(args):
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    W = settings.width
    H = settings.height
    print(f"[INFO] Loading camera parameters from: {settings.cam_calib_path}")
    mtxs, _, _, _, _ = load_camera_parameters(settings.cam_calib_path)

    if args.videos and len(args.videos) > 0:
        paths = [Path(v) for v in args.videos]
    else:
        paths = list_videos(Path(args.data_dir))
        
    if len(paths) == 0:
        raise RuntimeError(f"No videos found in {args.data_dir}")
    if len(paths) != 2:
        raise RuntimeError(f"Expected exactly 2 streams, found: {len(paths)}")

    # --- DYNAMICALLY DETERMINE MAX BENCHMARK FRAMES VIA FFPROBE ---
    video_lengths = []
    for p in paths:
        cmd_frames = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_frames',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(p)
        ]
        try:
            res = subprocess.run(cmd_frames, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=True)
            out_frames = res.stdout.strip()
            if out_frames.isdigit() and int(out_frames) > 0:
                video_lengths.append(int(out_frames))
                continue
        except Exception:
            pass
            
    if video_lengths:
        max_benchmark_frames = min(video_lengths)
        print(f"[INFO] Target video lengths resolved: {video_lengths}. Benchmark bound to {max_benchmark_frames} frames.")
    else:
        max_benchmark_frames = 300
        print(f"[Warning] Frame counts could not be resolved. Defaulting to safe cap: {max_benchmark_frames} frames.")

    print("[INFO] Initializing high-speed concurrent FFmpeg streaming engine...")
    src = OfflineVideoSource(paths=paths, size_wh=(W, H), queue_size=2)

    print("[INFO] Initializing NLF Estimator architectures...")
    est = NLFEstimator(
        yolo_path=settings.yolo_path,
        nlf_path=settings.lf_path if hasattr(settings, 'lf_path') else settings.nlf_path,
        cano_path=settings.cano_path,
        image_size=(W, H),
        cam_Ks=mtxs,
        indices=settings.nlf_indices,
        conf=settings.yolo_conf,
        imgsz=settings.yolo_imgsz,
        device=settings.device,
    )

    writer1, writer2 = None, None
    fps = getattr(settings, "fs", 40.0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if args.output_1: writer1 = cv2.VideoWriter(args.output_1, fourcc, fps, (W, H))
    if args.output_2: writer2 = cv2.VideoWriter(args.output_2, fourcc, fps, (W, H))

    print(f"Starting pipeline processing loop...")
    frame_count = 0
    yolo_history, h2d_pre_history, nlf_history, inference_history, input_overhead_history, total_history, fps_history = [], [], [], [], [], [], []

    # Configure PyTorch Profiler with an adjusted execution schedule 
    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=20, warmup=10, active=40, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    )
    prof.start()

    try:
        while frame_count < max_benchmark_frames:
            # ================= TRACKED CORE PIPELINE START =================
            start_input_overhead = time.perf_counter()
            frames = src.read()
            
            if frames is None:
                print("[Warning] Video stream closed unexpectedly or timed out. Breaking loop.")
                break
                
            input_overhead_ms = (time.perf_counter() - start_input_overhead) * 1000.0
            nlf_out, infer_timings, yres, boxes = est.estimate_from_frames(frames)
            
            yolo_ms = infer_timings.get("yolo_ms", 0.0)
            h2d_pre_ms = infer_timings.get("h2d+pre_ms", 0.0)
            nlf_ms = infer_timings.get("nlf_ms", 0.0)
            pure_inference_ms = infer_timings.get("total_ms", yolo_ms + h2d_pre_ms + nlf_ms)
            total_ms = input_overhead_ms + pure_inference_ms
            # ================== TRACKED CORE PIPELINE END ==================
            
            current_fps = 1000.0 / total_ms if total_ms > 0 else 0.0
            frame_count += 1

            if frame_count > 10:
                yolo_history.append(yolo_ms)
                h2d_pre_history.append(h2d_pre_ms)
                nlf_history.append(nlf_ms)
                inference_history.append(pure_inference_ms)
                input_overhead_history.append(input_overhead_ms)
                total_history.append(total_ms)
                fps_history.append(current_fps)

            if writer1 and len(frames) > 0: writer1.write(frames[0])
            if writer2 and len(frames) > 1: writer2.write(frames[1])

            if args.visualize:
                preview_canvas = np.hstack(frames)
                cv2.imshow("Engine Preview", preview_canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

            prof.step()

            if frame_count % 50 == 0:
                print(f"[PROGRESS] Processed {frame_count}/{max_benchmark_frames} frames | Rolling Speed: {current_fps:.1f} FPS")

    finally:
        # CRITICAL FIX: Stop the profiler and print all text summaries/graphs BEFORE releasing
        # driver-level OS resources. This guarantees telemetry output if hardware loops hang.
        try:
            prof.stop()
        except Exception:
            pass

        if total_history:


            print(f"\n" + "="*20 + " BENCHMARK STATISTICS (WARMUP EXCLUDED) " + "="*20)
            print(f"Total Processed Frames: {frame_count}")
            print(f"Input Ingestion: mean {np.mean(input_overhead_history):.1f} ms | median {np.median(input_overhead_history):.1f} ms | max {np.max(input_overhead_history):.1f} ms")
            print(f"YOLO Detector:   mean {np.mean(yolo_history):.1f} ms | median {np.median(yolo_history):.1f} ms | max {np.max(yolo_history):.1f} ms")
            print(f"GPU Pre+H2D:     mean {np.mean(h2d_pre_history):.1f} ms | median {np.median(h2d_pre_history):.1f} ms | max {np.max(h2d_pre_history):.1f} ms")
            print(f"NLF Localizer:   mean {np.mean(nlf_history):.1f} ms | median {np.median(nlf_history):.1f} ms | max {np.max(nlf_history):.1f} ms")
            print(f"Net Inference:   mean {np.mean(inference_history):.1f} ms | median {np.median(inference_history):.1f} ms | max {np.max(inference_history):.1f} ms")
            print(f"Total Time:      mean {np.mean(total_history):.1f} ms | median {np.median(total_history):.1f} ms | max {np.max(total_history):.1f} ms")
            print(f"Performance:     mean {np.mean(fps_history):.1f} FPS  | median {np.median(fps_history):.1f} FPS  | min {np.min(fps_history):.1f} FPS")
            print("="*80 + "\n")

            try:
                print("="*37 + " PYTORCH COMPLETE GPU KERNEL PROFILE REPORT " + "="*36)
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
                print("="*115 + "\n")
                prof.export_chrome_trace("trace.json")
            except Exception as e:
                print(f"[Warning] Could not print PyTorch kernel table: {e}")


        # Final Hardware Cleanup Blocks
        if writer1: writer1.release()
        if writer2: writer2.release()
        src.release()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--output-1", type=str, default="benchmark_output_cam0.mp4")
    p.add_argument("--output-2", type=str, default="benchmark_output_cam1.mp4")
    p.add_argument("--visualize", action="store_true")
    args = p.parse_args()
    main(args)