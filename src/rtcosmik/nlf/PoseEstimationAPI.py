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
import rerun as rr  # Required backend for your custom NLFVisualizer layout

# Resolve InstantHMR source paths if executing relative to the library root
SRC_ROOT = Path(__file__).resolve().parents[2] / "InstantHMR"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from instanthmr.detector import DualCameraPersonDetector
from instanthmr.skeleton import edges_for 
from instanthmr.visualizer import RerunVisualizer

# --- Optimized Pre-calculated Normalization Inverses ---
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INV_STD = 1.0 / IMAGENET_STD
MEAN_DIV_STD = IMAGENET_MEAN / IMAGENET_STD

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

class NLFVisualizer:
    def __init__(
        self, 
        application_id: str = "rt_cosmik_nlf_viewer", 
        spawn_viewer: bool = True,
        world_R1_cam: Optional[np.ndarray] = None,
        world_T1_cam: Optional[np.ndarray] = None
    ):
        self._rr = rr
        rr.init(application_id, spawn=spawn_viewer, default_blueprint=None)
        
        self.world_R1_cam = world_R1_cam
        self.world_T1_cam = world_T1_cam
        
        rr.log("stream_0/world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        rr.log("stream_1/world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        
        self._prev_num_persons = [0, 0]
        self._blueprint_sent = False
        
        self.marker_names = [
            "PELV", "LHIP", "RHIP", "SPIN", "LKNE", "RKNE", "THOR", "LANK", "RANK", "NECK",
            "LTOE", "RTOE", "CLAV", "LSHO", "RSHO", "LELB", "RELB", "LWRI", "RWRI", "LHEA",
            "RHEA", "LMELB", "RMELB", "LMWRI", "RMWRI"
        ]

    def log_frame(
        self, 
        images_bgr: List[np.ndarray], 
        nlf_output: Dict[str, Any], 
        bboxes: List[Optional[Any]],
        frame_idx: int,
        timings: Dict[str, float],
        p3d_skeleton: Optional[np.ndarray] = None,
        timestamp: float | None = None
    ) -> None:
        """Logs 2D video feeds, 3D world streams, and streams unified profiling metrics."""
        rr = self._rr
        rr.set_time("frame", sequence=frame_idx)
        if timestamp is not None:
            rr.set_time("timestamp", duration=timestamp)
        
        for metric, ms_val in timings.items():
            if ms_val is not None:
                rr.log(f"timing/{metric}", rr.Scalars(float(ms_val)))

        poses2d = nlf_output.get("poses2d", [None, None])

        for cam_id in range(2):
            stream_path = f"stream_{cam_id}"
            img = cv2.cvtColor(images_bgr[cam_id], cv2.COLOR_BGR2RGB)
            
            image_path = f"{stream_path}/camera/image"
            rr.log(image_path, rr.Image(img))
            
            j2d = poses2d[cam_id] if poses2d and cam_id < len(poses2d) else None
            has_person = (j2d is not None and len(j2d) > 0 and j2d[0] is not None)
            
            current_count = 1 if has_person else 0
            for stale in range(current_count, self._prev_num_persons[cam_id]):
                rr.log(f"{image_path}/persons/person_{stale}", rr.Clear(recursive=True))
            self._prev_num_persons[cam_id] = current_count

            if has_person:
                pts = j2d[0].detach().cpu().numpy() if hasattr(j2d[0], "detach") else np.array(j2d[0])
                pts = pts[0] if pts.ndim == 3 else pts
                
                rr.log(
                    f"{image_path}/persons/person_0/keypoints",
                    rr.Points2D(positions=pts, radii=3.0, colors=[0, 255, 255])
                )

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

        if p3d_skeleton is not None and p3d_skeleton.shape[0] > 0:
            points_all = p3d_skeleton.reshape(-1, 3)
            
            if self.world_R1_cam is not None and self.world_T1_cam is not None:
                points_all = (self.world_R1_cam @ points_all.T).T + self.world_T1_cam

            N = points_all.shape[0]
            J = len(self.marker_names)
            joint_ids = np.arange(N) % J
            colors = np.tile(np.array([0, 0, 255], dtype=np.uint8), (N, 1)) 

            for idx, name in enumerate(self.marker_names):
                mask = (joint_ids == idx)
                if name.startswith("R") or name.startswith("right_"):
                    colors[mask] = [0, 255, 0] 
                elif name.startswith("L") or name.startswith("left_"):
                    colors[mask] = [255, 0, 0] 
                if name in ["RMELB", "RMWRI", "LMELB", "LMWRI"]:
                    colors[mask] = [0, 0, 0] 

            rr.log(
                "stream_0/world/persons/person_0/joints_3d",
                rr.Points3D(positions=points_all, radii=0.02, colors=colors)
            )

        if not self._blueprint_sent:
            self._send_blueprint()
            self._blueprint_sent = True

    def _send_blueprint(self):
        import rerun.blueprint as rrb
        eye_ctrls = rrb.EyeControls3D(position=[0, 0, -2], look_target=[0, 0, 1], eye_up=[0, -1, 0], kind=rrb.Eye3DKind.Orbital)
        
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial2DView(origin="stream_0/camera/image", name="Left Cam Feed"),
                    rrb.Spatial2DView(origin="stream_1/camera/image", name="Right Cam Feed"),
                    name="2D Overlays"
                ),
                rrb.Spatial3DView(origin="stream_0/world", name="World Triangulation", eye_controls=eye_ctrls),
                rrb.TimeSeriesView(
                    contents=[
                        "timing/yolo_ms",
                        "timing/h2d+pre_ms",
                        "timing/nlf_ms",
                        "timing/triangulation_processing_ms",
                        "timing/cpu_overhead_ms",
                        "timing/total_ms"
                    ], 
                    name="System Performance Profiles"
                ),
                column_shares=[3, 4, 3]
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
        
        # Optimize internal execution execution context threads
        sess_options.intra_op_num_threads = 4
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=providers)
        self.active_provider = self.session.get_providers()[0]
        
        # LOUD TELEMETRY DIAGNOSTICS AT INITIALIZATION
        print(f"\n[InstantHMR] INITIALIZED SESSION WITH PROVIDER: {self.active_provider}")
        if self.active_provider == "CPUExecutionProvider" and "cuda" in device.lower():
            print("===========================================================================")
            print("[CRITICAL WARNING] ONNX RUNTIME FAILED TO INITIALIZE GPU PATHS!")
            print("FALLING BACK TO SLOW CPU LOOP. CHECK DRIVER / ONNXRUNTIME-GPU CONDA LINKING.")
            print("===========================================================================\n")

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
            
            # Map explicit device target targets
            device_id = 0
            for name in self._out_names:
                io_binding.bind_output(name, device_type="cuda" if is_gpu else "cpu", device_id=device_id)
            self.session.run_with_iobinding(io_binding)
        except Exception:
            pass

    def predict_dual_camera(self, images_rgb: list[np.ndarray], detections: list[dict | None]) -> tuple[list[HMRPrediction | None], float, float]:
        t_prep_start = time.perf_counter()
        crops = np.zeros((2, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        cliffs = np.zeros((2, 3), dtype=np.float32)
        sq_meta: list[tuple[float, float, float, np.ndarray, int, int, float] | None] = [None, None]

        for i in range(2):
            det = detections[i]
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
        
        # Explicit target mapping for multi-GPU indices (e.g. "cuda:0" -> 0)
        dev_id = 0
        if ":" in device:
            try: dev_id = int(device.split(":")[-1])
            except ValueError: pass

        if "cuda" in device.lower():
            # Build explicit structured option dictionaries to enforce initialization on GPU
            cuda_opts = {
                "device_id": dev_id,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": "1"
            }
            if "TensorRTExecutionProvider" in available: 
                wanted.append(("TensorRTExecutionProvider", {"device_id": dev_id, "trt_fp16_enable": "1"}))
            if "CUDAExecutionProvider" in available: 
                wanted.append(("CUDAExecutionProvider", cuda_opts))
                
        wanted.append("CPUExecutionProvider")
        return wanted


class InstantHMREstimatorWrapper:
    def __init__(self, model_path: str, device: str, yolo_path: str, config: Dict[str, Any]):
        self.detector = DualCameraPersonDetector(variant="nano", confidence=config.get("conf", 0.5), device=device)
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

class PoseEstimationAPI:
    def __init__(self, mode: str = "instant_hmr", config: Dict[str, Any] = None, visualizer: Optional[Any] = None):
        self.mode = mode.lower()
        self.config = config or {}
        self.visualizer = visualizer
        self.frame_counter = 0
        self.initialize_engine()
        
    def initialize_engine(self):
        shared_app_id = "rt_cosmik_nlf_viewer"

        if self.mode == "nlf":
            import rtcosmik.nlf.nlf as nlf_mod
            target_class = getattr(nlf_mod, "NLFEstimator")
            
            yolo_path  = self.config.get("yolo_path", "/root/workspace/RT-COSMIK/weights/yolo/yolov10n.engine")
            nlf_path   = self.config.get("nlf_path", self.config.get("model_path", "/root/workspace/RT-COSMIK/weights/nlf/nlf_s_multi_0.2.2.torchscript"))
            cano_path  = self.config.get("cano_path", "/root/workspace/RT-COSMIK/weights/canonical_verts/smplx.npy")
            image_size = self.config.get("img_size", (640, 480))
            
            world_R1_cam = self.config.get("world_R1_cam", None)
            world_T1_cam = self.config.get("world_T1_cam", None)

            cam_Ks = self.config.get("cam_Ks")
            if cam_Ks is None:
                num_cams = self.config.get("num_cameras", 2)
                f = max(image_size)
                standard_K = np.array([[f, 0, image_size[0]/2.0], [0, f, image_size[1]/2.0], [0, 0, 1]], dtype=np.float32)
                cam_Ks = [standard_K] * num_cams

            indices = self.config.get("indices", np.arange(6890))

            self.engine = target_class(
                yolo_path=yolo_path, nlf_path=nlf_path, cano_path=cano_path,
                image_size=image_size, cam_Ks=cam_Ks, indices=indices,
                conf=self.config.get("conf", 0.75), device=self.config.get("device", "cuda:0")
            )
            
            self.visualizer = NLFVisualizer(
                application_id=shared_app_id, spawn_viewer=True,
                world_R1_cam=world_R1_cam, world_T1_cam=world_T1_cam
            )
            
        elif self.mode == "instant_hmr":
            self.engine = InstantHMREstimatorWrapper(
                model_path=self.config.get("model_path", "/root/workspace/RT-COSMIK/weights/instant_hmr.onnx"),
                device=self.config.get("device", "cuda:0"),
                yolo_path=self.config.get("yolo_path", "/root/workspace/RT-COSMIK/weights/yolo/yolov10n.engine"),
                config=self.config
            )
            self.visualizer = RerunVisualizer(
                application_id=shared_app_id, spawn_viewer=True,
                mhr_renderer=getattr(self.engine.hmr, "mhr_renderer", None)
            )

    def process_frames(self, frames_bgr: List[np.ndarray]) -> Dict[str, Any]:
        t_start = time.perf_counter()

        if self.mode == "nlf":
            expected_cameras = 2
            if hasattr(self.engine, "num_cameras"):
                expected_cameras = self.engine.num_cameras
            elif hasattr(self.engine, "cam_Ks") and self.engine.cam_Ks is not None:
                expected_cameras = len(self.engine.cam_Ks)

            adjusted_frames = list(frames_bgr)
            if len(adjusted_frames) < expected_cameras:
                adjusted_frames = adjusted_frames + [adjusted_frames[-1]] * (expected_cameras - len(adjusted_frames))
            elif len(adjusted_frames) > expected_cameras:
                adjusted_frames = adjusted_frames[:expected_cameras]

            try:
                predictions, timings, yres, boxes = self.engine.estimate_from_frames(adjusted_frames)
            except IndexError as e:
                predictions = [None] * expected_cameras
                boxes = [None] * expected_cameras
                yres = [None] * expected_cameras
                timings = {
                    "yolo_ms": 0.0, "h2d+pre_ms": 0.0, "nlf_ms": 0.0,
                    "triangulation_processing_ms": 0.0, "cpu_overhead_ms": 0.0
                }

            timings["total_ms"] = (time.perf_counter() - t_start) * 1000.0
            
            if self.visualizer is not None:
                try:
                    nlf_payload = predictions if isinstance(predictions, dict) else {"poses2d": predictions}
                    p3d_skeleton = None
                    if isinstance(predictions, dict):
                        p3d_skeleton = predictions.get("poses3d", predictions.get("p3d_skeleton", None))
                    
                    self.visualizer.log_frame(
                        images_bgr=adjusted_frames, nlf_output=nlf_payload, bboxes=boxes,
                        frame_idx=self.frame_counter, timings=timings, p3d_skeleton=p3d_skeleton
                    )
                    self.frame_counter += 1
                except Exception:
                    traceback.print_exc(file=sys.stderr)
                
            return {"prediction": predictions, "timings": timings, "yolo_results": yres, "bboxes": boxes}

        elif self.mode == "instant_hmr":
            predictions_list, timings, yres, boxes, images_rgb = self.engine.estimate(frames_bgr)
            timings["total_ms"] = (time.perf_counter() - t_start) * 1000.0
            
            if self.visualizer is not None:
                self.visualizer.log_frame(
                    images_rgb=images_rgb,
                    predictions=predictions_list, frame_idx=self.frame_counter,
                    detector_ms=timings.get("yolo_ms"), hmr_ms=timings.get("hmr_ms"), total_ms=timings.get("total_ms")
                )
                self.frame_counter += 1

            return {
                "prediction": [p.joints_3d_cam if p is not None else None for p in predictions_list], 
                "timings": timings, "yolo_results": yres, "bboxes": boxes
            }