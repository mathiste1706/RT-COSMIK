"""
Isolation test: displays the SAME neutral configuration through both
MeshcatVisualizer and ViserVisualizer, back to back, so you can compare.

If meshcat shows a normal standing pose and viser shows the smashed-together
mess, the bug is confirmed to be in ViserVisualizer's per-geometry placement,
not in your model loading / scaling / FK code.
"""
import pinocchio as pin
import example_robot_data as robex

human = robex.human.HumanLoader(height=1.75, weight=70.0, gender="male").robot
model = human.model
collision_model = human.collision_model
visual_model = human.visual_model

q = pin.neutral(model)

# --- Sanity check the FK itself, independent of any visualizer ---
data = model.createData()
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)
print("=== Joint placements at q = pin.neutral(model) ===")
for jid in range(1, model.njoints):
    t = data.oMi[jid].translation
    print(f"{model.names[jid]:20s}  xyz = ({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f})")
print()
print("If these translations already look wrong (e.g. everything near")
print("(0,0,0), or legs above the torso) the bug is upstream of any")
print("visualizer -- likely scale_human_model / mks_registration -- and")
print("has nothing to do with meshcat vs viser.")
print()

# --- Meshcat side ---
try:
    import meshcat
    from pinocchio.visualize import MeshcatVisualizer

    vis = meshcat.Visualizer()
    print(f"[meshcat] {vis.url()}")
    viz_mc = MeshcatVisualizer(model, collision_model, visual_model)
    viz_mc.initViewer(vis, open=True)
    viz_mc.loadViewerModel("mc_ref")
    viz_mc.display(q)
    print("[meshcat] displayed neutral pose -- check the browser tab")
except ImportError:
    print("[meshcat] not installed, skipping")

# --- Viser side ---
import viser

"""
Manual Pinocchio -> viser geometry loader.

Works around a ViserVisualizer.loadViewerModel() bug where multiple
geometries (e.g. capsule body segments sharing/near the same joint) were
being dropped or mis-attached, instead of one node per geometry.

This mirrors the small subset of MeshcatVisualizer's API this pipeline
actually uses (initViewer / loadViewerModel / display), so it's close to a
drop-in replacement for ViserVisualizer in run_pipeline.py / pipeline.py.

Requires: pip install trimesh --break-system-packages   (if not already installed)
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
            # geometries share a name or parent joint -- this is exactly
            # what ViserVisualizer's loadViewerModel was getting wrong.
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

print("=== visual_model geometry inventory ===")
print(f"Total geometries: {len(visual_model.geometryObjects)}")
for i, geom_obj in enumerate(visual_model.geometryObjects):
    gtype = type(geom_obj.geometry).__name__
    has_mesh = bool(getattr(geom_obj, "meshPath", None))
    print(f"  [{i:02d}] name={geom_obj.name!r:30s} shape_type={gtype!r:15s} meshPath_set={has_mesh}")
print()

server = viser.ViserServer()
print(f"[viser] http://{server.get_host()}:{server.get_port()}")
viz_vs = ManualViserRobotVisualizer(model, collision_model, visual_model)
viz_vs.initViewer(viewer=server)
viz_vs.loadViewerModel(rootNodeName="viser_ref")
viz_vs.display(q)
print("[viser] displayed neutral pose (via ManualViserRobotVisualizer) -- check the browser tab")

input("\nCompare the two tabs, then press Enter to exit...")