"""
viewer.py -- ViewerProcess for the online (multi-camera, multiprocessing)
pipeline, using viser instead of meshcat.

Includes ManualViserRobotVisualizer, a Pinocchio -> viser geometry loader
that works around a ViserVisualizer.loadViewerModel() bug where multiple
geometries (e.g. mesh-backed body segments sharing/near the same joint) were
being dropped or mis-attached, instead of one node per geometry.

ManualViserRobotVisualizer mirrors the small subset of MeshcatVisualizer's
API this pipeline actually uses (initViewer / loadViewerModel / display), so
it's a close to drop-in replacement for pinocchio's ViserVisualizer.

Requires: pip install trimesh
"""
import numpy as np
import pinocchio as pin

try:
    import trimesh
except ImportError as exc:
    raise ImportError(
        "ManualViserRobotVisualizer needs trimesh: "
        "pip install trimesh --break-system-packages"
    ) from exc


def _geom_to_trimesh(geom_obj) -> "trimesh.Trimesh":
    """Convert one pinocchio GeometryObject's hppfcl shape into a trimesh.Trimesh.

    NOTE: hppfcl class names below (Capsule/Sphere/Cylinder/Box) match recent
    pinocchio/hppfcl releases, but shape-class naming has shifted across
    versions in the past -- if you hit an "Unsupported geometry type" error,
    print type(geom_obj.geometry).__name__ and add a branch for it here.
    """
    geom = geom_obj.geometry
    gtype = type(geom).__name__

    if gtype == "Capsule":
        mesh = trimesh.creation.capsule(
            radius=geom.radius, height=2.0 * geom.halfLength, count=(8, 8)
        )
    elif gtype == "Sphere":
        mesh = trimesh.creation.icosphere(radius=geom.radius, subdivisions=2)
    elif gtype == "Cylinder":
        mesh = trimesh.creation.cylinder(
            radius=geom.radius, height=2.0 * geom.halfLength, sections=16
        )
    elif gtype == "Box":
        mesh = trimesh.creation.box(extents=2.0 * np.array(geom.halfSide))
    else:
        # Mesh-backed geometry (BVHModel/Convex/etc.) -- load from the
        # original mesh file instead of trying to reconstruct the shape.
        if getattr(geom_obj, "meshPath", None):
            mesh = trimesh.load(geom_obj.meshPath, force="mesh")
        else:
            raise ValueError(
                f"Unsupported geometry type '{gtype}' for '{geom_obj.name}' "
                f"and no meshPath to fall back on."
            )

    color = getattr(geom_obj, "meshColor", None)
    if color is not None and len(color) >= 3:
        rgba = np.array(
            [color[0], color[1], color[2], color[3] if len(color) > 3 else 1.0]
        )
        mesh.visual.vertex_colors = np.tile((rgba * 255).astype(np.uint8), (len(mesh.vertices), 1))

    # Critical: GeometryObject.meshScale is what scale_human_model uses to
    # fit generic template meshes to this subject's anthropometry. Skipping
    # this renders every segment at its raw template size -- MeshcatVisualizer
    # applies it internally, which is why meshcat looked right and this
    # manual loader (before this fix) rendered an oversized, overlapping blob.
    scale = np.asarray(getattr(geom_obj, "meshScale", [1.0, 1.0, 1.0]), dtype=float).flatten()
    if not np.allclose(scale, 1.0):
        S = np.eye(4)
        S[0, 0], S[1, 1], S[2, 2] = scale
        mesh.apply_transform(S)

    return mesh


class ManualViserRobotVisualizer:
    """Drop-in-ish replacement for pinocchio's ViserVisualizer for this
    pipeline's usage pattern: initViewer(viewer=server) -> loadViewerModel()
    -> display(q) every frame."""

    def __init__(self, model: pin.Model, collision_model: pin.GeometryModel,
                 visual_model: pin.GeometryModel):
        self.model = model
        # Match ViserVisualizer's constructor signature; we only actually
        # render the visual model (collision capsules stay hidden, same as
        # MeshcatVisualizer's default).
        self.collision_model = collision_model
        self.visual_model = visual_model
        self.data = model.createData()
        self.visual_data = visual_model.createData()
        self.server = None
        self.root = "ref"
        self._handles = []  # index-aligned with visual_model.geometryObjects

    def initViewer(self, viewer):
        self.server = viewer

    def loadViewerModel(self, rootNodeName="ref"):
        self.root = rootNodeName
        self._handles = []
        for i, geom_obj in enumerate(self.visual_model.geometryObjects):
            try:
                mesh = _geom_to_trimesh(geom_obj)
            except Exception as exc:
                print(f"[WARN] ManualViserRobotVisualizer: skipping "
                      f"'{geom_obj.name}' ({exc})")
                self._handles.append(None)
                continue
            # Index-prefixed path guarantees uniqueness even if multiple
            # geometries share a name or parent joint 
            path = f"/{self.root}/{i:03d}_{geom_obj.name}"
            handle = self.server.scene.add_mesh_trimesh(path, mesh)
            self._handles.append(handle)

    def display(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateGeometryPlacements(self.model, self.data, self.visual_model, self.visual_data)
        for i, handle in enumerate(self._handles):
            if handle is None:
                continue
            oMg = self.visual_data.oMg[i]
            quat = pin.Quaternion(oMg.rotation)
            handle.wxyz = np.array([quat.w, quat.x, quat.y, quat.z], dtype=np.float64)
            handle.position = np.asarray(oMg.translation, dtype=np.float64)

    # No-ops kept for API parity with ViserVisualizer/MeshcatVisualizer call
    # sites in run_pipeline.py / pipeline.py -- collisions are never loaded
    # here so there's nothing to toggle.
    def displayCollisions(self, flag: bool):
        pass

    def displayVisuals(self, flag: bool):
        pass
import viser
# ManualViserRobotVisualizer (defined above) replaces pinocchio.visualize's
# ViserVisualizer, which drops/mis-attaches multi-geometry bodies.
ViserVisualizer = ManualViserRobotVisualizer

from multiprocessing import Process, Queue, Event
import pinocchio as pin
import example_robot_data as robex
from rtcosmik.saver.csv_saver import CSVSaver
from rtcosmik.human_model.model_utils import scale_human_model
from typing import List
import numpy as np
from collections import OrderedDict

from pynput import keyboard
import logging

LOGGER = logging.getLogger(__name__)

class Viewer:
    def __init__(self, model, collision_model, visual_model, marker_names, freeflyer=True):
        self.model = model
        self.collision_model = collision_model
        self.visual_model = visual_model

        self.marker_names = marker_names
        self.freeflyer = freeflyer

        # viser starts its own web server; the URL is printed automatically,
        # but we also grab host/port to log it the same way the meshcat
        # version did.
        self.server = viser.ViserServer()
        LOGGER.info(f"[INFO] Viser visualizer available here: http://{self.server.get_host()}:{self.server.get_port()}")

        # Ground grid, matching the floor grid MeshcatVisualizer/pinocchio
        # shows by default. Assumes z-up with the model standing on z=0
        # (consistent with the joint placements we've been seeing: feet at
        # the most negative z, head at the most positive z).
        self.server.scene.add_grid(
            "/grid",
            width=10.0,
            height=10.0,
            position=(0.0, 0.0, 0.0),
        )

        self.marker_colors = np.zeros((len(self.marker_names), 3), dtype=np.uint8)
        self.marker_colors[:, 0] = 255  # R
        self.marker_colors[:, 1] = 0    # G
        self.marker_colors[:, 2] = 0    # B

        self._markers_handle = None

        # Init viser visualizer for human
        self.viz_human = ViserVisualizer(self.model, self.collision_model, self.visual_model)
        self.viz_human.initViewer(viewer=self.server)
        self.viz_human.loadViewerModel(rootNodeName="ref")

        # Background styling: viser doesn't expose a meshcat-style
        # top/bottom gradient background property. A flat scene background
        # color is the closest equivalent; comment out if unsupported in
        # your viser version.
        try:
            self.server.scene.set_background_image(None)  # clear any default env image
        except Exception:
            pass

    def display_q(self, q):
        self.viz_human.display(q)

    def display_markers(self, pos_markers_dict):
        pts = np.stack(list(pos_markers_dict.values()), axis=0).astype(np.float32)
        # Re-adding a point cloud under the same name replaces it in place,
        # same effect as meshcat's set_object on a stable path.
        self._markers_handle = self.server.scene.add_point_cloud(
            name="/markers",
            points=pts,
            colors=self.marker_colors,
            point_size=0.02,
        )


class ViewerProcess(Process):
    def __init__(self,
                 settings,
                 results_queues: List[Queue],
                 stop_event: Event,
                 num_cameras: int,
                 freeflyer=True,
                 saving_flag=None,
                 logger=None,
                 ):
        super().__init__()
        self.settings=settings

        self.saving_flag = saving_flag

        self.results_queues = results_queues
        self.stop_event = stop_event
        self.num_cameras = num_cameras
        self.marker_names = self.settings.marker_names
        self.joint_angles_names = self.settings.joint_angles_names

        self.freeflyer = freeflyer
        self.first_sample=True
        self.logger = logger or LOGGER

        self.SAVE_CSV = self.settings.SAVE_CSV
        if self.SAVE_CSV:
            self.SAVE_DIR = self.settings.SAVE_DIR
            self.frame_counters = []
            for i in range(self.num_cameras):
                self.frame_counters.append('Frame_'+str(i))

            self.markers_header = self.frame_counters+self.marker_names
            self.joint_angles_header = self.frame_counters+self.joint_angles_names
            self.logger.info("[INFO] Saving data as csv enabled ...")

    def run(self):

        #load model from AT
        self.human = robex.human.HumanLoader(height=self.settings.human_height, weight=self.settings.human_weight, gender=self.settings.human_gender).robot
        self.model= self.human.model
        self.collision_model = self.human.collision_model
        self.visual_model  = self.human.visual_model
        self.logger.info("[INFO] Human model successfully loaded in viewer process...")

        self.saving_enabled = False

        if self.SAVE_CSV:
            self.csv_saver = CSVSaver(
                self.SAVE_DIR,
                self.markers_header,
                self.joint_angles_header
            ) 

        self.viewer = Viewer(
                             self.model, 
                             self.collision_model, 
                             self.visual_model, 
                             self.marker_names, 
                             self.freeflyer
                             )
        
        def on_press(key):
            try:
                if key.char == 's':
                    self.logger.info("[INFO] Start saving data in viewer process ...")
                    self.saving_enabled = True
                    if self.saving_flag is not None:
                        self.saving_flag.value = True  
                elif key.char == 'q':
                    self.logger.info("[INFO] Stop saving data in viewer process ...")
                    self.saving_enabled = False
                    if self.saving_flag is not None:
                        self.saving_flag.value = False  
            except AttributeError:
                pass

        # Start keyboard listener in background
        listener = keyboard.Listener(on_press=on_press)
        listener.start()

        try: 
            while not self.stop_event.is_set():
                cam_counters, mks_dict = self.results_queues[0].get()
                _, q = self.results_queues[1].get()

                #scale the model to data
                if self.first_sample:
                    self.model = scale_human_model(self.model, mks_dict, gender=self.settings.human_gender, subject_height=self.settings.human_height)
                    self.first_sample=False
                
                self.viewer.display_markers(mks_dict)
                self.viewer.display_q(q)

                if self.SAVE_CSV and self.saving_enabled:
                    mks_dict_to_save = mks_dict
                    q_dict_to_save = {}
                    for i in range(len(cam_counters)):
                        mks_dict_to_save['Frame_'+str(i)]=cam_counters[i]
                        q_dict_to_save['Frame_'+str(i)]=cam_counters[i]
                    
                    for i in range(len(self.joint_angles_names)):
                        q_dict_to_save[self.joint_angles_names[i]]=q[i]
                    
                    ordered_markers = OrderedDict()
                    for header_key in self.markers_header:
                        if header_key.startswith("Frame_"):
                            ordered_markers[header_key] = mks_dict_to_save.get(header_key, None)
                        else:
                            if header_key.endswith('_x') or header_key.endswith('_y') or header_key.endswith('_z'):
                                base, comp = header_key.rsplit('_', 1)
                                arr = mks_dict_to_save.get(base)
                                if arr is not None and hasattr(arr, '__getitem__') and len(arr) >= 3:
                                    if comp == "x":
                                        ordered_markers[header_key] = float(arr[0])
                                    elif comp == "y":
                                        ordered_markers[header_key] = float(arr[1])
                                    elif comp == "z":
                                        ordered_markers[header_key] = float(arr[2])
                                else:
                                    ordered_markers[header_key] = None
                            else:
                                base = header_key
                                arr = mks_dict_to_save.get(base)
                                if arr is not None and hasattr(arr, '__getitem__') and len(arr) >= 3:
                                    ordered_markers[base + '_x'] = float(arr[0])
                                    ordered_markers[base + '_y'] = float(arr[1])
                                    ordered_markers[base + '_z'] = float(arr[2])
                                else:
                                    ordered_markers[base + '_x'] = None
                                    ordered_markers[base + '_y'] = None
                                    ordered_markers[base + '_z'] = None

                    # Joint angles are assumed to be scalars.
                    ordered_joint_angles = OrderedDict(
                        (key, q_dict_to_save.get(key, None)) for key in self.joint_angles_header
                    )

                    self.csv_saver.save_markers(ordered_markers)
                    self.csv_saver.save_joint_angles(ordered_joint_angles)

        finally:
            listener.stop()
            self.logger.info("[INFO] Viewer process terminated")