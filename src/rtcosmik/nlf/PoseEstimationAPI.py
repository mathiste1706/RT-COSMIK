#!/usr/bin/env python3
from __future__ import annotations

import math
import time
import logging
import cv2
import numpy as np
import torch
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import onnxruntime as ort
import rerun as rr  


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import argparse

from rtcosmik.triangulation.triangulation import triangulate_points

from InstantHMR.instanthmr.detector import DualCameraPersonDetector
from InstantHMR.instanthmr.skeleton import edges_for 
from InstantHMR.instanthmr.visualizer import RerunVisualizer

# --- Optimized Pre-calculated Normalization Inverses ---
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

INPUT_SIZE = 224
CROP_EXPAND = 1.2 


@dataclass
class HMRPrediction:
    """Per-person outputs from one forward pass, expected by RerunVisualizer."""
    bbox: np.ndarray
    confidence: float
    joints_3d_local: np.ndarray
    joints_3d_cam: np.ndarray
    joints_2d: np.ndarray
    cam_trans: np.ndarray
    focal_length: np.ndarray
    principal_point: np.ndarray
    image_shape: tuple[int, int]
    mhr_params: np.ndarray
    shape_params: np.ndarray


# ==============================================================================
# YOUR CUSTOM SPECIFIED NLF VISUALIZER BLUEPRINT LAYOUT
# ==============================================================================

# ==============================================================================
# UNIFIED VISUALIZER BLUEPRINT LAYOUT (Stripped Triangulation for NLF)
# ==============================================================================

class NLFVisualizer:
    def __init__(
        self, 
        application_id: str = "rt_cosmik_nlf_viewer", 
        spawn_viewer: bool = True,
        world_R1_cam: Optional[np.ndarray] = None,
        world_T1_cam: Optional[np.ndarray] = None
    ):
        self._rr = rr
        # Clear out any stale cached layout configurations from previous runs
        rr.init(application_id, spawn=spawn_viewer, default_blueprint=None)
        
        self.world_R1_cam = world_R1_cam
        self.world_T1_cam = world_T1_cam
        
        rr.log("stream_0/world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        rr.log("stream_1/world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        
        self._prev_num_persons = [0, 0]
        self._blueprint_sent = False

    def log_frame(
        self, 
        images_bgr: List[np.ndarray], 
        nlf_output: Dict[str, Any], 
        bboxes: List[Optional[Any]],
        frame_idx: int,
        timings: Dict[str, float],
        p3d_skeleton: Optional[np.ndarray] = None, # Retained in signature for routing API compatibility
        timestamp: float | None = None,
        predictions: Optional[List[Optional[HMRPrediction]]] = None
    ) -> None:
        """Logs 2D feeds, performance metrics, and independent InstantHMR 3D predictions."""
        rr = self._rr
        rr.set_time("frame", sequence=frame_idx)
        if timestamp is not None:
            rr.set_time("timestamp", duration=timestamp)
        
        # Dynamic telemetry streaming loop using engine dictionary keys
        for metric, ms_val in timings.items():
            if ms_val is not None:
                rr.log(f"timing/{metric}", rr.Scalars(float(ms_val)))

        poses2d = nlf_output.get("poses2d", [None, None])

        for cam_id in range(2):
            stream_path = f"stream_{cam_id}"
            img = cv2.cvtColor(images_bgr[cam_id], cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            image_path = f"{stream_path}/camera/image"
            rr.log(image_path, rr.Image(img))
            
            # ─── INSTANTHMR CAMERA-SPACE 3D SKELETON LOGGING ───
            pred = predictions[cam_id] if predictions and cam_id < len(predictions) else None
            if pred is not None:
                path = f"{stream_path}/world/persons/person_0"
                joints = pred.joints_3d_cam
                
                # Render independent 3D joints matching original spec
                rr.log(f"{path}/joints", rr.Points3D(positions=joints, radii=0.012, colors=[0, 230, 0]))
                
                edges = edges_for(joints.shape[0])
                if edges:
                    lines = [[joints[i].tolist(), joints[j].tolist()] for i, j in edges]
                    rr.log(f"{path}/skeleton", rr.LineStrips3D(lines, colors=[255, 230, 0], radii=0.004))
                
                # Render original PinHole projection values per-stream
                fx, fy = float(pred.focal_length[0]), float(pred.focal_length[1])
                depth = float(pred.joints_3d_cam[:, 2].mean())
                rr.log(f"{stream_path}/camera", rr.Pinhole(
                    width=w, height=h, focal_length=[fx, fy],
                    principal_point=[w / 2.0, h / 2.0],
                    image_plane_distance=max(depth * 0.25, 0.2)
                ))

            # ─── 2D OVERLAYS & KEYPOINTS (NLF Mode) ───
            j2d = poses2d[cam_id] if poses2d and cam_id < len(poses2d) else None
            has_person = (j2d is not None and len(j2d) > 0 and j2d[0] is not None)
            
            current_count = 1 if has_person else 0
            for stale in range(current_count, self._prev_num_persons[cam_id]):
                rr.log(f"{image_path}/persons/person_{stale}", rr.Clear(recursive=True))
            self._prev_num_persons[cam_id] = current_count

            if has_person and pred is None:  # Only output 2D markers if InstantHMR isn't active
                pts = j2d[0].detach().cpu().numpy() if hasattr(j2d[0], "detach") else np.array(j2d[0])
                pts = pts[0] if pts.ndim == 3 else pts
                
                rr.log(
                    f"{image_path}/persons/person_0/keypoints",
                    rr.Points2D(positions=pts, radii=3.0, colors=[0, 255, 255])
                )

                # --- Overlap Bounding Box Detection Selection ---
                bbox = bboxes[cam_id] if bboxes and cam_id < len(bboxes) else None
                best_box = None
                
                if bbox is not None:
                    if hasattr(bbox, "xyxy"):
                        b = bbox.xyxy.cpu().numpy() if hasattr(bbox.xyxy, "cpu") else np.array(bbox.xyxy)
                    else:
                        b = bbox.detach().cpu().numpy() if hasattr(bbox, "detach") else np.array(bbox)
                    
                    if b.ndim == 2 and len(b) > 0:
                        best_count = -1
                        for row in b:
                            if len(row) >= 4:
                                bx1, by1, bx2, by2 = row[:4]
                                count = np.sum((pts[:, 0] >= bx1) & (pts[:, 0] <= bx2) & (pts[:, 1] >= by1) & (pts[:, 1] <= by2))
                                if count > best_count:
                                    best_count = count
                                    best_box = row[:4]
                        if best_count <= 0:
                            best_box = None
                    elif b.ndim == 1 and len(b) >= 4:
                        best_box = b[:4]

                if best_box is None:
                    p_x1, p_y1 = np.min(pts[:, 0]), np.min(pts[:, 1])
                    p_x2, p_y2 = np.max(pts[:, 0]), np.max(pts[:, 1])
                    pad_x = (p_x2 - p_x1) * 0.12
                    pad_y = (p_y2 - p_y1) * 0.12
                    best_box = [p_x1 - pad_x, p_y1 - pad_y, p_x2 + pad_x, p_y2 + pad_y]

                x1, y1, x2, y2 = map(int, best_box)
                rr.log(
                    f"{image_path}/persons/person_0/bbox",
                    rr.Boxes2D(array=[x1, y1, x2, y2], array_format=rr.Box2DFormat.XYXY, colors=[0, 255, 0])
                )

        if not self._blueprint_sent:
            self._send_blueprint(has_hmr=(predictions is not None))
            self._blueprint_sent = True

    def _send_blueprint(self, has_hmr: bool = False):
        import rerun.blueprint as rrb
        
        if has_hmr:
            # InstantHMR layout: Independent stream viewports side-by-side with 3D spaces
            eye_ctrls = rrb.EyeControls3D(position=[0, 0, -2], look_target=[0, 0, 1], eye_up=[0, -1, 0], kind=rrb.Eye3DKind.Orbital)
            blueprint = rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Vertical(
                        rrb.Spatial2DView(origin="stream_0/camera/image", name="Left Cam Feed"), 
                        rrb.Spatial3DView(origin="stream_0/world", name="Left Stream Space", eye_controls=eye_ctrls)
                    ),
                    rrb.Vertical(
                        rrb.Spatial2DView(origin="stream_1/camera/image", name="Right Cam Feed"), 
                        rrb.Spatial3DView(origin="stream_1/world", name="Right Stream Space", eye_controls=eye_ctrls)
                    ),
                    rrb.TimeSeriesView(origin="timing", name="Execution Profiles"),
                    column_shares=[3, 3, 2]
                ),
                collapse_panels=True
            )
        else:
            # NLF layout: Purely 2D video feeds and telemetry timelines (No 3D view canvas clutter)
            blueprint = rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Vertical(
                        rrb.Spatial2DView(origin="stream_0/camera/image", name="Left Cam Feed"),
                        rrb.Spatial2DView(origin="stream_1/camera/image", name="Right Cam Feed"),
                        name="2D Overlays"
                    ),
                    rrb.TimeSeriesView(
                        contents=[
                            "timing/yolo_ms", "timing/h2d+pre_ms", "timing/nlf_ms",
                            "timing/cpu_overhead_ms", "timing/total_ms"
                        ], 
                        name="System Performance Profiles"
                    ),
                    column_shares=[2, 1]
                ),
                collapse_panels=True
            )
        self._rr.send_blueprint(blueprint)

# ==============================================================================
# ONNX ENGINE IMPLEMENTATION (InstantHMR)
# ==============================================================================

class InstantHMR:
    def __init__(self, onnx_path: str | Path, device: str = "cuda:0", providers: Optional[list[Any]] = None):
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        if hasattr(ort, "preload_dlls"):
            try: ort.preload_dlls()
            except Exception: pass

        if providers is None:
            providers = self._default_providers(device)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=providers)
        self.active_provider = self.session.get_providers()[0]

        in_names = [i.name for i in self.session.get_inputs()]
        self._in_image = in_names[0]   
        self._in_cliff = in_names[1]   
        self._out_names = [o.name for o in self.session.get_outputs()]

    def warmup(self) -> None:
        dummy_image = np.zeros((2, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        dummy_cliff = np.zeros((2, 3), dtype=np.float32)
        try:
            io_binding = self.session.io_binding()
            io_binding.bind_cpu_input(self._in_image, dummy_image)
            io_binding.bind_cpu_input(self._in_cliff, dummy_cliff)
            is_gpu = "CPUExecutionProvider" not in self.active_provider
            for name in self._out_names:
                io_binding.bind_output(name, device_type="cuda" if is_gpu else "cpu", device_id=0)
            self.session.run_with_iobinding(io_binding)
        except Exception:
            pass

    def predict_dual_camera(self, images_rgb: list[np.ndarray], detections: list[dict | None]) -> tuple[list[HMRPrediction | None], float, float]:
        t_prep_start = time.perf_counter()
        crops = np.zeros((2, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        cliffs = np.zeros((2, 3), dtype=np.float32)
        sq_meta = [None, None]

        for i in range(2):
            if i >= len(images_rgb): continue
            det = detections[i] if i < len(detections) else None
            img = np.ascontiguousarray(images_rgb[i])
            h, w = img.shape[:2]
            if det is None: continue
            
            bbox_arr = np.asarray(det["bbox"], dtype=np.float32).reshape(4)
            crop, sq_x1, sq_y1, sq_size, cliff = self._preprocess(img, bbox_arr, h, w)
            crops[i] = crop
            cliffs[i] = cliff
            sq_meta[i] = (sq_x1, sq_y1, sq_size, bbox_arr, h, w, float(det.get("confidence", 1.0)))
            
        t_prep_ms = (time.perf_counter() - t_prep_start) * 1000.0

        t_infer_start = time.perf_counter()
        io_binding = self.session.io_binding()
        io_binding.bind_cpu_input(self._in_image, crops)
        io_binding.bind_cpu_input(self._in_cliff, cliffs)
        is_gpu = "CPUExecutionProvider" not in self.active_provider
        for name in self._out_names:
            io_binding.bind_output(name, device_type="cuda" if is_gpu else "cpu", device_id=0)

        self.session.run_with_iobinding(io_binding)
        outs = io_binding.copy_outputs_to_cpu()
        t_infer_ms = (time.perf_counter() - t_infer_start) * 1000.0

        results: list[HMRPrediction | None] = [None, None]
        for i in range(2):
            meta = sq_meta[i]
            if meta is None: continue
            sq_x1, sq_y1, sq_size, bbox_arr, h, w, confidence = meta
            
            crop_px = (outs[3][i] + 1.0) * 0.5 * INPUT_SIZE
            scale = sq_size / INPUT_SIZE
            joints_2d = np.stack([crop_px[:, 0] * scale + sq_x1, crop_px[:, 1] * scale + sq_y1], axis=-1).astype(np.float32)
            f = math.sqrt(h * h + w * w)

            results[i] = HMRPrediction(
                bbox=bbox_arr, confidence=confidence, joints_3d_local=outs[4][i],
                joints_3d_cam=outs[4][i] + outs[2][i], joints_2d=joints_2d, cam_trans=outs[2][i],
                focal_length=np.array([f, f], dtype=np.float32), principal_point=np.array([w / 2.0, h / 2.0], dtype=np.float32),
                image_shape=(h, w), mhr_params=outs[0][i], shape_params=outs[1][i]
            )
        return results, t_prep_ms, t_infer_ms

    @staticmethod
    def _preprocess(image_rgb: np.ndarray, bbox: np.ndarray, h: int, w: int) -> tuple[np.ndarray, float, float, float, np.ndarray]:
        x1, y1, x2, y2 = bbox.astype(float)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        cliff_cond = np.array([2.0 * (cx / w) - 1.0, 2.0 * (cy / h) - 1.0, max(x2 - x1, y2 - y1) / max(w, h)], dtype=np.float32)

        sq_size = max(x2 - x1, y2 - y1) * CROP_EXPAND
        sq_x1, sq_y1 = cx - (sq_size / 2.0), cy - (sq_size / 2.0)

        ix1, iy1 = int(math.floor(sq_x1)), int(math.floor(sq_y1))
        ix2, iy2 = int(math.ceil(sq_x1 + sq_size)), int(math.ceil(sq_y1 + sq_size))  

        patch = image_rgb[max(0, iy1):min(h, iy2), max(0, ix1):min(w, ix2)]
        if iy1 < 0 or ix1 < 0 or ix2 > w or iy2 > h:
            patch = cv2.copyMakeBorder(patch, max(0, -iy1), max(0, iy2 - h), max(0, -ix1), max(0, ix2 - w), cv2.BORDER_CONSTANT, value=(0,0,0))

        crop = cv2.resize(patch, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32, copy=False) * (1.0 / 255.0)
        crop = np.transpose((crop - IMAGENET_MEAN) / IMAGENET_STD, (2, 0, 1))
        return np.ascontiguousarray(crop, dtype=np.float32), float(sq_x1), float(sq_y1), float(sq_size), cliff_cond

    @staticmethod
    def _default_providers(device: str) -> list[Any]:
        available = set(ort.get_available_providers())
        wanted = []
        dev_id = 0
        if ":" in device:
            try: dev_id = int(device.split(":")[-1])
            except ValueError: pass

        if "cuda" in device.lower():
            cuda_opts = {"device_id": dev_id, "arena_extend_strategy": "kNextPowerOfTwo"}
            if "CUDAExecutionProvider" in available: wanted.append(("CUDAExecutionProvider", cuda_opts))
        wanted.append("CPUExecutionProvider")
        return wanted


class InstantHMREstimatorWrapper:
    def __init__(self, model_path: str, device: str, yolo_path: str, config: Dict[str, Any]):
        self.detector = DualCameraPersonDetector(variant="nano", confidence=config.get("conf", 0.75), device=device)
        if hasattr(self.detector, "warmup"): self.detector.warmup()
        self.hmr = InstantHMR(onnx_path=model_path, device=device)
        self.hmr.warmup()

    def estimate(self, frames_bgr: List[np.ndarray]) -> Tuple[List[Optional[HMRPrediction]], Dict[str, float], List[Optional[Dict]], List[Optional[np.ndarray]], List[np.ndarray]]:
        t0 = time.perf_counter()
        images_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
        t1 = time.perf_counter()
        
        detections = self.detector.detect(images_rgb)
        t2 = time.perf_counter()
        
        hmr_predictions, t_prep, t_infer = self.hmr.predict_dual_camera(images_rgb, detections)
        t3 = time.perf_counter()
        
        return hmr_predictions, {
            "rgb_conv_ms": (t1 - t0) * 1000.0, "yolo_ms": (t2 - t1) * 1000.0,
            "hmr_prep_ms": t_prep, "hmr_infer_pure_ms": t_infer,
            "hmr_ms": (t3 - t2) * 1000.0, "total_ms": (t3 - t0) * 1000.0
        }, detections, [d["bbox"] if d else None for d in detections], images_rgb

# ==============================================================================
# UNIFIED POSE ESTIMATION ROUTING API LAYER
# ==============================================================================

# ==============================================================================
# UNIFIED POSE ESTIMATION ROUTING API LAYER
# ==============================================================================

class PoseEstimationAPI:
    def __init__(self, mode: str = "instant_hmr", config: Dict[str, Any] = None):
        self.mode = mode.lower()
        self.config = config or {}
        self.frame_counter = 0
        self.triangulation_mode = self.config.get("triangulation_mode", "rt_cosmik").lower()
        self.yolo_path = self.config.get("yolo_path", "/root/workspace/RT-COSMIK/weights/yolo/yolov10n.engine")
        self.device = self.config.get("device", "cuda:0")
        
        # This will store our execution pathway function pointer resolved at init
        self._processor_callable = None 
        self.initialize_engine()
        
    def initialize_engine(self) -> None:
        shared_app_id = "rt_cosmik_nlf_viewer"
        world_R1_cam = self.config.get("world_R1_cam", None)
        world_T1_cam = self.config.get("world_T1_cam", None)

        if self.mode == "nlf":
            import rtcosmik.nlf.nlf as nlf_mod
            target_class = getattr(nlf_mod, "NLFEstimator")
            
            nlf_path   = self.config.get("nlf_path", self.config.get("model_path", "/root/workspace/RT-COSMIK/weights/nlf/nlf_s_multi_0.2.2.torchscript"))
            cano_path  = self.config.get("cano_path", "/root/workspace/RT-COSMIK/weights/canonical_verts/smplx.npy")
            image_size = self.config.get("img_size", (640, 480))
            
            self.engine = target_class(
                yolo_path=self.yolo_path, nlf_path=nlf_path, cano_path=cano_path,
                image_size=image_size, cam_Ks=self.config.get("cam_Ks"), indices=self.config.get("indices", np.arange(6890)),
                conf=self.config.get("conf", 0.75), device=self.device
            )
            self.visualizer = NLFVisualizer(application_id=shared_app_id, spawn_viewer=True, world_R1_cam=world_R1_cam, world_T1_cam=world_T1_cam)
            
            # Direct link to the NLF channel method reference
            self._processor_callable = self._process_nlf
            
        elif self.mode == "instant_hmr":
            self.engine = InstantHMREstimatorWrapper(
                model_path=self.config.get("model_path", "/root/workspace/RT-COSMIK/weights/instant_hmr.onnx"),
                device=self.device,
                yolo_path=self.yolo_path,
                config=self.config
            )
            self.visualizer = NLFVisualizer(application_id=shared_app_id, spawn_viewer=True, world_R1_cam=world_R1_cam, world_T1_cam=world_T1_cam)
            
            # Direct link to the InstantHMR channel method reference
            self._processor_callable = self._process_instant_hmr
        else:
            raise ValueError(f"Unsupported processing engine mode profile: {self.mode}")

    def process_frames(self, frames_bgr: List[np.ndarray]) -> Dict[str, Any]:
        """
        No more runtime conditional branch checks here. 
        Straight, clean execution targeting the pre-bound pipeline handler.
        """
        t_start = time.perf_counter()
        return self._processor_callable(frames_bgr, t_start)

    def _process_nlf(self, frames_bgr: List[np.ndarray], t_start: float) -> Dict[str, Any]:
        """Isolated processing channel for the PyTorch-based NLF mesh engine wrapper."""
        expected_cameras = 2
        adjusted_frames = list(frames_bgr)
        if len(adjusted_frames) < expected_cameras:
            adjusted_frames = adjusted_frames + [adjusted_frames[-1]] * (expected_cameras - len(adjusted_frames))
        elif len(adjusted_frames) > expected_cameras:
            adjusted_frames = adjusted_frames[:expected_cameras]

        try:
            predictions, timings, yres, boxes = self.engine.estimate_from_frames(adjusted_frames)
        except IndexError:
            predictions = [None] * expected_cameras
            boxes = [None] * expected_cameras
            yres = [None] * expected_cameras
            timings = {"yolo_ms": 0.0, "h2d+pre_ms": 0.0, "nlf_ms": 0.0, "triangulation_processing_ms": 0.0}

        timings["total_ms"] = (time.perf_counter() - t_start) * 1000.0
        
        if self.visualizer is not None:
            nlf_payload = predictions if isinstance(predictions, dict) else {"poses2d": predictions}
            
         
            self.visualizer.log_frame(
                images_bgr=adjusted_frames, nlf_output=nlf_payload, bboxes=boxes,
                frame_idx=self.frame_counter, timings=timings, p3d_skeleton=None
            )
            self.frame_counter += 1
            
        return {"prediction": predictions, "timings": timings, "yolo_results": yres, "bboxes": boxes}

    def _process_instant_hmr(self, frames_bgr: List[np.ndarray], t_start: float) -> Dict[str, Any]:
        """Isolated processing channel for the ONNX-backed multi-camera InstantHMR pipeline."""
        predictions_list, timings, yres, boxes, images_rgb = self.engine.estimate(frames_bgr)
        
        t_triang_start = time.perf_counter()
        p3d_skeleton = None
        
        poses2d_list = []
        for pred in predictions_list:
            if pred is not None:
                poses2d_list.append([pred.joints_2d])
            else:
                poses2d_list.append(None)
        nlf_payload = {"poses2d": poses2d_list}

        # --- STRATEGY 1: NATIVE RECONSTRUCTION ---
        if self.triangulation_mode == "native":
            if len(predictions_list) > 0 and predictions_list[0] is not None:
                p3d_skeleton = predictions_list[0].joints_3d_cam

        # --- STRATEGY 2: RT-COSMIK GEOMETRIC TRIANGULATION ---
        elif self.triangulation_mode == "rt_cosmik":
            if len(predictions_list) >= 2 and predictions_list[0] is not None and predictions_list[1] is not None:
                pts1 = predictions_list[0].joints_2d
                pts2 = predictions_list[1].joints_2d
                keypoints_list = [pts1, pts2]
                
                h, w = frames_bgr[0].shape[:2]
                f = max(h, w)
                standard_K = np.array([[f, 0, w/2.0], [0, f, h/2.0], [0, 0, 1]], dtype=np.float64)
                
                mtxs = self.config.get("cam_Ks", [standard_K, standard_K])
                dists = self.config.get("cam_dists", [np.zeros(5, dtype=np.float64), np.zeros(5, dtype=np.float64)])
                
                rel_R = self.config.get("world_R1_cam", np.eye(3, dtype=np.float64))
                rel_T = self.config.get("world_T1_cam", np.array([[1.0], [0.0], [0.0]], dtype=np.float64))
                if rel_T.ndim == 1:
                    rel_T = rel_T.reshape(3, 1)
                    
                projections = [
                    np.hstack([np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)]),
                    np.hstack([rel_R, rel_T])
                ]
                
                p3d_skeleton = triangulate_points(keypoints_list, mtxs, dists, projections)

        timings["triangulation_processing_ms"] = (time.perf_counter() - t_triang_start) * 1000.0
        timings["total_ms"] = (time.perf_counter() - t_start) * 1000.0
        
        if self.visualizer is not None:
            self.visualizer.log_frame(
                images_bgr=frames_bgr, nlf_output=nlf_payload, bboxes=boxes,
                frame_idx=self.frame_counter, timings=timings, p3d_skeleton=p3d_skeleton
            )
            self.frame_counter += 1

        return {
            "prediction": [p.joints_3d_cam if p is not None else None for p in predictions_list], 
            "p3d_skeleton": p3d_skeleton, "timings": timings, "yolo_results": yres, "bboxes": boxes
        }