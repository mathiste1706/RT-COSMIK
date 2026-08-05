# import cv2
# import numpy as np
# from datetime import datetime
# import multiprocessing as mp
# from multiprocessing import Process, Array, Value, Lock, Barrier, Event, Queue
# import logging

# LOGGER = logging.getLogger(__name__)

# class Camera(Process):
#     def __init__(self, 
#                  cam_id: int,
#                  shared_buffer: Array,
#                  timestamp_buffer: Array, # Character array for timestamp
#                  lock: Lock,
#                  frame_counter: Value,
#                  barrier: Barrier,
#                  stop_event: Event,
#                  frame_shape: tuple = (720, 1280, 3),
#                  cam_fps: int = 40,
#                  cam_fourcc: str = "MJPG",
#                  logger=None,
#                  ):
        
#         super().__init__()
#         self.cam_id = cam_id
#         self.shared_buffer = shared_buffer
#         self.timestamp_buffer = timestamp_buffer  # For timestamp string
#         self.lock = lock
#         self.frame_counter = frame_counter
#         self.barrier = barrier
#         self.stop_event = stop_event
        
#         # Video capture parameters
#         self.frame_shape = frame_shape  # (height, width, channels)
#         self.cam_fps = cam_fps
#         self.cam_fourcc = cam_fourcc

#         self.logger=logger or LOGGER

#         # Validate timestamp buffer size (need 26 chars for format)
#         if len(timestamp_buffer) != 26:
#             raise ValueError("Timestamp buffer must be exactly 26 characters")

#     def run(self):
#         cap = cv2.VideoCapture(self.cam_id, cv2.CAP_V4L2)
#         if not cap.isOpened():
#             raise Exception(f"Camera {self.cam_id} could not be opened.")
        
#         # Set camera properties once if specified
#         if self.cam_fourcc:
#             cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.cam_fourcc))
#         if self.frame_shape:
#             cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_shape[0])
#             cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_shape[1])
#         if self.cam_fps:
#             cap.set(cv2.CAP_PROP_FPS, self.cam_fps)

#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

#         # reshape shared buffer once
#         arr          = np.frombuffer(self.shared_buffer, dtype=np.uint8)
#         frame_buffer = arr.reshape(self.frame_shape)

#         # let everyone get to this point
#         self.logger.info(f"[INFO] Camera {self.cam_id} is ready to acquire images ...")
#         self.barrier.wait()

#         try:
#             while not self.stop_event.is_set():
#                 # --- 1) all processes synchronize before grabbing next frame
#                 self.barrier.wait()

#                 # --- 2) tell the driver to queue the next frame
#                 cap.grab()

#                 # --- 3) wait here until everyone has grabbed
#                 self.barrier.wait()

#                 # --- 4) pull the actual image out of the buffer
#                 ret, frame = cap.retrieve()
#                 if not ret:
#                     continue

#                 # --- 5) timestamp right away
#                 now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

#                 # --- 6) resize/check, then write under lock
#                 resized = cv2.resize(frame, (self.frame_shape[1], self.frame_shape[0]))
#                 with self.lock:
#                     np.copyto(frame_buffer, resized)
#                     self.timestamp_buffer[:26] = now_str.ljust(26, "\0").encode("utf-8")
#                     self.frame_counter.value += 1

#         finally:
#             cap.release()
#             self.logger.info(f"[INFO] Camera process for camera {self.cam_id} terminated...")

# class DisplayConsumer(Process):
#     def __init__(self, 
#                  frame_counters,
#                  camera_buffers, 
#                  camera_locks, 
#                  timestamp_buffers, 
#                  stop_event, 
#                  frame_shape, 
#                  num_cameras):
#         super().__init__()
#         self.camera_buffers = camera_buffers
#         self.camera_locks = camera_locks
#         self.timestamp_buffers = timestamp_buffers
#         self.frame_shape = frame_shape  # (height, width, channels)
#         self.num_cameras = num_cameras
#         self.stop_event = stop_event

#         self.last_frame_counters = [0] * self.num_cameras
#         self.frame_counters = frame_counters
        
#     def run(self):
#         window_names = [f'Camera {i}' for i in range(self.num_cameras)]
        
#         # Optimization 1: Create a single window for all cameras
#         combined_window = "Multi-Camera View"
        
#         try: 
#             while not self.stop_event.is_set():
#                 frames = []
#                 keypoints_list = []
#                 new_counters = []
#                 for i, (lock, buffer, cam_ts, frame_counter) in enumerate(zip(self.camera_locks, self.camera_buffers, self.timestamp_buffers, self.frame_counters)):
#                     with lock:
#                         #  Only accept data if this camera has produced a new frame
#                         if frame_counter.value > self.last_frame_counters[i]:
#                             # Read and copy shared data atomically
#                             arr = np.frombuffer(buffer, dtype=np.uint8)
#                             frame = arr.reshape(self.frame_shape).copy()
#                             # Get current timestamp
#                             timestamp = bytes(cam_ts[:]).decode().strip('\x00')

#                             if timestamp == '': # empty data
#                                 continue
#                             else:
#                                 frames.append(frame)
#                             new_counters.append(frame_counter.value)
#                 if len(frames)!=self.num_cameras:
#                     continue
#                 self.last_frame_counters = new_counters.copy()
            
#                 print(new_counters)
#                 # Optimization 1: Combine all frames into single view
#                 ########################################
#                 # Create a horizontal stack of frames
#                 combined_frame = np.hstack(frames)
                
#                 # Show combined view
#                 cv2.imshow(combined_window,  combined_frame)
#                 ########################################
                
#                 # Original individual windows display (comment out when using combined view)
#                 # for i, frame in enumerate(frames):
#                 #     if frame.shape[2] == 3:
#                 #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 #     cv2.imshow(window_names[i], frame)

#                 # Break on 'q' key press
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#         finally:        
#             cv2.destroyAllWindows()
#             print("Display process terminated.")

import cv2
import numpy as np
import os
import subprocess
import threading
import time
from datetime import datetime
import multiprocessing as mp
from multiprocessing import Process, Array, Value, Lock, Barrier, Event, Queue
import logging

LOGGER = logging.getLogger(__name__)

# Maps the fourcc codes you already use for cv2 to ffmpeg's -input_format
# names for the v4l2 demuxer. Extend as needed for other sensors.
FOURCC_TO_FFMPEG_INPUT_FORMAT = {
    "MJPG": "mjpeg",
    "YUYV": "yuyv422",
    "YUY2": "yuyv422",
    "H264": "h264",
    "RGB3": "rgb24",
}


class Camera(Process):
    def __init__(self,
                 cam_id: int,
                 shared_buffer: Array,
                 timestamp_buffer: Array,  # Character array for timestamp
                 lock: Lock,
                 frame_counter: Value,
                 barrier: Barrier,
                 stop_event: Event,
                 frame_shape: tuple = (720, 1280, 3),
                 cam_fps: int = 40,
                 cam_fourcc: str = "MJPG",
                 device_path: str = None,
                 logger=None,
                 ):

        super().__init__()
        self.cam_id = cam_id
        self.shared_buffer = shared_buffer
        self.timestamp_buffer = timestamp_buffer
        self.lock = lock
        self.frame_counter = frame_counter
        self.barrier = barrier
        self.stop_event = stop_event

        self.frame_shape = frame_shape  # (height, width, channels)
        self.cam_fps = cam_fps
        self.cam_fourcc = cam_fourcc

        # cam_id used to be an OpenCV device index; ffmpeg's v4l2 demuxer
        # wants a /dev/videoN path. Override with device_path if your
        # enumeration (see list_cameras() in cam_utils.py) doesn't map
        # 1:1 to /dev/video{cam_id}.
        self.device_path = device_path or f"/dev/video{cam_id}"

        self.logger = logger or LOGGER

        # Only touched inside run(), i.e. in the child process.
        self._proc = None
        self._latest_lock = None
        self._latest_frame = None
        self._latest_ts = None

        if len(timestamp_buffer) != 26:
            raise ValueError("Timestamp buffer must be exactly 26 characters")

    def _build_ffmpeg_cmd(self, frame_size: int):
        h, w, _ = self.frame_shape
        input_format = FOURCC_TO_FFMPEG_INPUT_FORMAT.get(self.cam_fourcc.upper(), "mjpeg")
        # Output-side flags (vcodec/pix_fmt/blocksize/threads/image2pipe)
        # deliberately mirror OfflineVideoSource's ffmpeg command so both
        # capture paths decode identically downstream.
        return [
            "ffmpeg",
            "-loglevel", "error",
            "-f", "v4l2",
            "-input_format", input_format,
            "-video_size", f"{w}x{h}",
            "-framerate", str(self.cam_fps),
            "-i", self.device_path,
            "-vf", f"scale={w}:{h}",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-blocksize", str(frame_size),
            "-threads", "2",
            "-f", "image2pipe",
            "-",
        ]

    def _clean_env(self):
        # Same rationale as OfflineVideoSource: strip VSCODE_* vars that can
        # otherwise leak debugger/IPC settings into the ffmpeg child.
        clean_env = os.environ.copy()
        for key in list(clean_env.keys()):
            if "VSCODE" in key:
                clean_env.pop(key)
        return clean_env

    def _drain_stderr(self, proc):
        """ffmpeg's stderr pipe blocks the encoder if nobody reads it, even
        at -loglevel error (init/warning lines still land there). This
        process runs indefinitely (unlike the offline reader, which only
        drains stderr once at release() since its runtime is bounded), so
        stderr needs continuous draining for the process lifetime."""
        for line in iter(proc.stderr.readline, b""):
            if not line:
                break
            msg = line.decode(errors="replace").rstrip()
            if msg:
                self.logger.warning(f"[ffmpeg cam {self.cam_id}] {msg}")

    def _reader_loop(self, proc, frame_bytes, h, w):
        """Continuously drains proc.stdout and keeps only the most recently
        decoded frame + its timestamp, using the same readinto-based
        zero-copy read pattern as OfflineVideoSource._pipe_reader_worker.
        Unlike the offline reader (which pushes onto a bounded Queue to
        preserve order), this keeps a single "latest" slot -- ffmpeg has no
        equivalent to CAP_PROP_BUFFERSIZE=1, so "always the newest frame,
        never a stale queued one" is enforced here."""
        while not self.stop_event.is_set() and proc.poll() is None:
            frame_buffer = np.empty((h, w, 3), dtype=np.uint8)
            try:
                bytes_read = proc.stdout.readinto(frame_buffer)
            except (ValueError, OSError):
                break  # pipe closed under us during shutdown

            if bytes_read == 0 or bytes_read is None:
                continue

            while bytes_read < frame_bytes and not self.stop_event.is_set():
                remaining_view = memoryview(frame_buffer.reshape(-1))[bytes_read:]
                extra_bytes = proc.stdout.readinto(remaining_view)
                if extra_bytes == 0 or extra_bytes is None:
                    break
                bytes_read += extra_bytes

            if bytes_read != frame_bytes:
                continue

            # Timestamp taken right after the frame is fully decoded off the
            # pipe -- as close as we get to "capture time" without ffmpeg
            # exposing V4L2 buffer timestamps through image2pipe.
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

            with self._latest_lock:
                self._latest_frame = frame_buffer
                self._latest_ts = now_str

        if proc.poll() is not None and not self.stop_event.is_set():
            self.logger.error(
                f"[ERROR] ffmpeg for camera {self.cam_id} exited "
                f"unexpectedly (code={proc.returncode})"
            )

    def run(self):
        h, w, c = self.frame_shape
        frame_bytes = h * w * c

        cmd = self._build_ffmpeg_cmd(frame_bytes)
        self.logger.info(f"[INFO] Camera {self.cam_id} starting ffmpeg: {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=frame_bytes,
            env=self._clean_env(),
        )
        self._proc = proc

        self._latest_lock = threading.Lock()
        self._latest_frame = None
        self._latest_ts = None

        stderr_thread = threading.Thread(target=self._drain_stderr, args=(proc,), daemon=True)
        stderr_thread.start()

        reader_thread = threading.Thread(
            target=self._reader_loop, args=(proc, frame_bytes, h, w), daemon=True
        )
        reader_thread.start()

        arr = np.frombuffer(self.shared_buffer, dtype=np.uint8)
        frame_buffer_shared = arr.reshape(self.frame_shape)

        # Don't let this camera join the barrier until ffmpeg has actually
        # produced a first frame -- device open + driver negotiation time
        # varies a lot per camera/USB bus, and joining the sync loop before
        # a camera is ready would stall every other camera's barrier.wait().
        startup_deadline = time.time() + 10.0
        while self._latest_frame is None and not self.stop_event.is_set():
            if proc.poll() is not None:
                raise RuntimeError(
                    f"Camera {self.cam_id}: ffmpeg exited during startup (code={proc.returncode})"
                )
            if time.time() > startup_deadline:
                raise RuntimeError(f"Camera {self.cam_id}: timed out waiting for first frame from ffmpeg")
            time.sleep(0.005)

        self.logger.info(f"[INFO] Camera {self.cam_id} (ffmpeg) is ready to acquire images ...")
        self.barrier.wait()

        try:
            while not self.stop_event.is_set():
                # --- 1) all processes synchronize before sampling the
                # latest decoded frame. This bounds SOFTWARE skew between
                # cameras (process scheduling jitter); it is not a
                # hardware-triggered acquisition the way cap.grab() was --
                # ffmpeg streams frames out continuously as the driver
                # produces them, with no per-call trigger to synchronize on.
                self.barrier.wait()

                with self._latest_lock:
                    frame = self._latest_frame
                    now_str = self._latest_ts

                # --- 2) second barrier kept for symmetry with the original
                # grab/retrieve structure, and so any downstream timing
                # assumptions about inter-barrier duration still hold.
                self.barrier.wait()

                if frame is None or now_str is None:
                    continue

                # --- 3) resize/check, then write under lock
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h))
                with self.lock:
                    np.copyto(frame_buffer_shared, frame)
                    self.timestamp_buffer[:26] = now_str.ljust(26, "\0").encode("utf-8")
                    self.frame_counter.value += 1

        finally:
            self.stop_event.set()
            try:
                if proc.poll() is None:
                    proc.terminate()
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    pass
            except Exception:
                pass
            reader_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            try:
                if proc.stdout:
                    proc.stdout.close()
                if proc.stderr:
                    proc.stderr.close()
            except Exception:
                pass
            self.logger.info(f"[INFO] Camera process for camera {self.cam_id} (ffmpeg) terminated...")


class DisplayConsumer(Process):
    def __init__(self,
                 frame_counters,
                 camera_buffers,
                 camera_locks,
                 timestamp_buffers,
                 stop_event,
                 frame_shape,
                 num_cameras):
        super().__init__()
        self.camera_buffers = camera_buffers
        self.camera_locks = camera_locks
        self.timestamp_buffers = timestamp_buffers
        self.frame_shape = frame_shape  # (height, width, channels)
        self.num_cameras = num_cameras
        self.stop_event = stop_event

        self.last_frame_counters = [0] * self.num_cameras
        self.frame_counters = frame_counters

    def run(self):
        window_names = [f'Camera {i}' for i in range(self.num_cameras)]
        combined_window = "Multi-Camera View"

        try:
            while not self.stop_event.is_set():
                frames = []
                keypoints_list = []
                new_counters = []
                for i, (lock, buffer, cam_ts, frame_counter) in enumerate(zip(self.camera_locks, self.camera_buffers, self.timestamp_buffers, self.frame_counters)):
                    with lock:
                        if frame_counter.value > self.last_frame_counters[i]:
                            arr = np.frombuffer(buffer, dtype=np.uint8)
                            frame = arr.reshape(self.frame_shape).copy()
                            timestamp = bytes(cam_ts[:]).decode().strip('\x00')

                            if timestamp == '':
                                continue
                            else:
                                frames.append(frame)
                            new_counters.append(frame_counter.value)
                if len(frames) != self.num_cameras:
                    continue
                self.last_frame_counters = new_counters.copy()

                print(new_counters)
                combined_frame = np.hstack(frames)
                cv2.imshow(combined_window, combined_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cv2.destroyAllWindows()
            print("Display process terminated.")