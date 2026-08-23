"""
Camera Position Setter Plugin for MoleditPy.

Provides precise 3D camera control, standard axis views (+X, -X, +Y, -Y, +Z, -Z,
Isometric), exact numerical coordinate setters, target centering (molecule /
selected atoms), orbital step adjustments (nudge), viewpoint bookmarks with
session and project persistence, clipboard copy/paste, and smooth animated transitions.

Source code, README, and full license (GNU GPL):
    https://github.com/HiroYokoyama/moleditpy_camera_position_setter
Copyright (c) HiroYokoyama. Licensed under the GNU General Public License;
see the LICENSE file in the repository above for the full terms.
"""

# pylint: disable=too-many-lines,too-many-instance-attributes,too-many-public-methods
# pylint: disable=too-many-locals,too-many-statements,attribute-defined-outside-init
# pylint: disable=no-name-in-module,broad-exception-caught

import ast
import json
import math
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PyQt6.QtCore import QCoreApplication, QTimer
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QDoubleSpinBox,
    QComboBox,
    QPushButton,
    QCheckBox,
    QListWidget,
    QListWidgetItem,
    QInputDialog,
    QMessageBox,
    QWidget,
    QTabWidget,
)

logger = logging.getLogger(__name__)

# --- Plugin Metadata ---
PLUGIN_NAME = "Camera Position Setter"
PLUGIN_VERSION = "1.0.0"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = (
    "Precise 3D camera controller, axis view aligner, and viewpoint manager for MoleditPy."
)
PLUGIN_CATEGORY = "View"
PLUGIN_TAGS = ["Visualization", "Utility"]
PLUGIN_DEPENDENCIES = ["pyvista", "PyQt6", "numpy"]
PLUGIN_SUPPORTED_MOLEDITPY_VERSION = ">=3.0.0, <5.0.0"
PLUGIN_SUPPORTED_PYTHON_VERSION = ">=3.9, <3.15"
PLUGIN_SUPPORTED_OS = ["Windows", "macOS", "Linux", "WSL"]

# Global cache for session presets and project data
_PROJECT_PRESETS: List[Dict[str, Any]] = []


def initialize(context) -> None:
    """Register the tool in the View menu and set up persistence handlers."""
    context.add_menu_action(
        "View/Camera Position Setter...",
        lambda: show_camera_dialog(context),
    )

    if hasattr(context, "register_save_handler"):
        context.register_save_handler(lambda: _on_save_project(context))
    if hasattr(context, "register_load_handler"):
        context.register_load_handler(lambda data: _on_load_project(context, data))
    if hasattr(context, "register_document_reset_handler"):
        context.register_document_reset_handler(lambda: _on_document_reset(context))


def show_camera_dialog(context) -> None:
    """Singleton: reuse the existing window if it is already open."""
    win = context.get_window("camera_position_setter_dialog")
    if win:
        win.show()
        win.raise_()
        win.read_camera_from_view()
        return

    mw = context.get_main_window()
    dialog = CameraPositionDialog(context, mw)
    context.register_window("camera_position_setter_dialog", dialog)
    dialog.show()
    dialog.read_camera_from_view()


def _on_save_project(context) -> Dict[str, Any]:
    """Save project-specific camera bookmarks into the project file."""
    win = context.get_window("camera_position_setter_dialog")
    if win and hasattr(win, "get_all_presets"):
        presets = win.get_all_presets()
    else:
        presets = list(_PROJECT_PRESETS)
    return {"bookmarks": presets}


def _on_load_project(context, data: Any) -> None:
    """Restore project-specific camera bookmarks when opening a project."""
    global _PROJECT_PRESETS  # pylint: disable=global-statement
    if isinstance(data, dict) and "bookmarks" in data and isinstance(data["bookmarks"], list):
        _PROJECT_PRESETS = list(data["bookmarks"])
    else:
        _PROJECT_PRESETS = []

    win = context.get_window("camera_position_setter_dialog")
    if win and hasattr(win, "load_presets"):
        win.load_presets(_PROJECT_PRESETS)


def _on_document_reset(context) -> None:
    """Clear project presets on File > New."""
    global _PROJECT_PRESETS  # pylint: disable=global-statement
    _PROJECT_PRESETS = []
    win = context.get_window("camera_position_setter_dialog")
    if win and hasattr(win, "load_presets"):
        win.load_presets([])


# ---------------------------------------------------------------------------
# Mathematical / Calculation Helpers
# ---------------------------------------------------------------------------


def compute_standard_view(
    view_name: str,
    focal_point: Tuple[float, float, float],
    distance: float,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Compute camera position and view_up vector for a standard Cartesian or isometric view.

    Returns:
        (position, view_up)
    """
    fx, fy, fz = focal_point
    d = max(float(distance), 1e-4)

    views = {
        "+X": ((fx + d, fy, fz), (0.0, 0.0, 1.0)),
        "-X": ((fx - d, fy, fz), (0.0, 0.0, 1.0)),
        "+Y": ((fx, fy + d, fz), (0.0, 0.0, 1.0)),
        "-Y": ((fx, fy - d, fz), (0.0, 0.0, 1.0)),
        "+Z": ((fx, fy, fz + d), (0.0, 1.0, 0.0)),
        "-Z": ((fx, fy, fz - d), (0.0, -1.0, 0.0)),
        "Iso (+X+Y+Z)": (
            (fx + d / math.sqrt(3), fy + d / math.sqrt(3), fz + d / math.sqrt(3)),
            (0.0, 0.0, 1.0),
        ),
        "Iso (+X-Y+Z)": (
            (fx + d / math.sqrt(3), fy - d / math.sqrt(3), fz + d / math.sqrt(3)),
            (0.0, 0.0, 1.0),
        ),
        "Iso (-X-Y+Z)": (
            (fx - d / math.sqrt(3), fy - d / math.sqrt(3), fz + d / math.sqrt(3)),
            (0.0, 0.0, 1.0),
        ),
        "Iso (-X+Y+Z)": (
            (fx - d / math.sqrt(3), fy + d / math.sqrt(3), fz + d / math.sqrt(3)),
            (0.0, 0.0, 1.0),
        ),
    }

    if view_name in views:
        return views[view_name]
    raise ValueError(f"Unknown standard view name: {view_name}")


def rodrigues_rotate(v: np.ndarray, k: np.ndarray, theta_rad: float) -> np.ndarray:
    """Rotate 3D vector v around unit axis k by theta_rad radians."""
    k_norm = np.linalg.norm(k)
    if k_norm < 1e-8:
        return v.copy()
    k_unit = k / k_norm
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    return v * cos_t + np.cross(k_unit, v) * sin_t + k_unit * np.dot(k_unit, v) * (1.0 - cos_t)


def orbit_step(
    pos: Tuple[float, float, float],
    fp: Tuple[float, float, float],
    up: Tuple[float, float, float],
    mode: str,
    step_deg: float,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Apply an incremental orbital rotation (azimuth, elevation, roll) to camera.

    Args:
        pos: Camera position
        fp: Camera focal point
        up: Camera view up
        mode: 'azimuth', 'elevation', or 'roll'
        step_deg: Step angle in degrees

    Returns:
        (new_pos, new_up)
    """
    p = np.array(pos, dtype=float)
    f = np.array(fp, dtype=float)
    u = np.array(up, dtype=float)

    view_vec = p - f
    v_norm = np.linalg.norm(view_vec)
    u_norm = np.linalg.norm(u)
    if v_norm < 1e-8 or u_norm < 1e-8:
        return pos, up

    v_dir = view_vec / v_norm
    u_dir = u / u_norm
    right_dir = np.cross(u_dir, v_dir)
    r_norm = np.linalg.norm(right_dir)
    if r_norm > 1e-8:
        right_dir = right_dir / r_norm
    else:
        right_dir = np.array([1.0, 0.0, 0.0])

    theta = math.radians(step_deg)

    if mode == "azimuth":
        new_v = rodrigues_rotate(view_vec, u_dir, -theta)
        new_u = u
    elif mode == "elevation":
        new_v = rodrigues_rotate(view_vec, right_dir, -theta)
        new_u = rodrigues_rotate(u, right_dir, -theta)
    elif mode == "roll":
        new_v = view_vec
        new_u = rodrigues_rotate(u, v_dir, theta)
    else:
        return pos, up

    new_pos = f + new_v
    return (float(new_pos[0]), float(new_pos[1]), float(new_pos[2])), (
        float(new_u[0]),
        float(new_u[1]),
        float(new_u[2]),
    )


def calculate_molecule_centroid(mol) -> Optional[Tuple[float, float, float]]:
    """Calculate the 3D centroid of all atoms in an RDKit molecule."""
    if mol is None:
        return None
    try:
        if mol.GetNumConformers() == 0 or mol.GetNumAtoms() == 0:
            return None
        conf = mol.GetConformer()
        positions = conf.GetPositions()
        if len(positions) == 0:
            return None
        mean = np.mean(positions, axis=0)
        return float(mean[0]), float(mean[1]), float(mean[2])
    except Exception as e:
        logger.debug("Failed to calculate molecule centroid: %s", e)
        return None


def calculate_selected_atoms_centroid(
    mol, indices: List[int]
) -> Optional[Tuple[float, float, float]]:
    """Calculate the 3D centroid of selected atom indices in an RDKit molecule."""
    if mol is None or not indices:
        return None
    try:
        if mol.GetNumConformers() == 0:
            return None
        conf = mol.GetConformer()
        num_atoms = mol.GetNumAtoms()
        pts = []
        for idx in indices:
            if 0 <= idx < num_atoms:
                pos = conf.GetAtomPosition(idx)
                pts.append([pos.x, pos.y, pos.z])
        if not pts:
            return None
        mean = np.mean(pts, axis=0)
        return float(mean[0]), float(mean[1]), float(mean[2])
    except Exception as e:
        logger.debug("Failed to calculate selected atoms centroid: %s", e)
        return None


# ---------------------------------------------------------------------------
# Dialog Implementation
# ---------------------------------------------------------------------------


class CameraPositionDialog(QDialog):
    """Interactive GUI Dialog for precise 3D camera setting and viewpoint management."""

    def __init__(self, context, parent=None):
        super().__init__(parent)
        self.context = context
        self.setWindowTitle("Camera Position Setter")
        self.resize(520, 680)

        self._block_signals = False
        self._presets: List[Dict[str, Any]] = []

        # Smooth animation timer & state
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(20)  # 50 fps
        self._anim_timer.timeout.connect(self._on_anim_step)
        self._anim_step = 0
        self._anim_total_steps = 15  # 300 ms
        self._anim_start_state: Dict[str, Any] = {}
        self._anim_target_state: Dict[str, Any] = {}

        self.setup_ui()
        self._load_saved_presets()

    def setup_ui(self) -> None:
        """Construct all UI tabs and controls."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        tabs = QTabWidget()
        tabs.addTab(self._setup_coords_tab(), "Coordinates & Controls")
        tabs.addTab(self._setup_presets_tab(), "Bookmarks & Presets")
        main_layout.addWidget(tabs)

        self._setup_bottom_bar(main_layout)

    def _setup_coords_tab(self) -> QWidget:
        """Build the main coordinates and viewport control tab."""
        tab_coords = QWidget()
        layout = QVBoxLayout(tab_coords)
        layout.setSpacing(8)

        # Standard views
        group_std = QGroupBox("Standard Views")
        grid_std = QGridLayout(group_std)
        grid_std.setSpacing(6)

        self.btn_x_plus = QPushButton("+X (Right)")
        self.btn_x_minus = QPushButton("-X (Left)")
        self.btn_y_plus = QPushButton("+Y (Back)")
        self.btn_y_minus = QPushButton("-Y (Front)")
        self.btn_z_plus = QPushButton("+Z (Top)")
        self.btn_z_minus = QPushButton("-Z (Bottom)")
        self.btn_iso_1 = QPushButton("Iso (+X+Y+Z)")
        self.btn_iso_2 = QPushButton("Iso (+X-Y+Z)")
        self.btn_reset_view = QPushButton("Fit to Screen")

        grid_std.addWidget(self.btn_x_plus, 0, 0)
        grid_std.addWidget(self.btn_x_minus, 0, 1)
        grid_std.addWidget(self.btn_y_plus, 0, 2)
        grid_std.addWidget(self.btn_y_minus, 1, 0)
        grid_std.addWidget(self.btn_z_plus, 1, 1)
        grid_std.addWidget(self.btn_z_minus, 1, 2)
        grid_std.addWidget(self.btn_iso_1, 2, 0)
        grid_std.addWidget(self.btn_iso_2, 2, 1)
        grid_std.addWidget(self.btn_reset_view, 2, 2)

        self.btn_x_plus.clicked.connect(lambda: self.apply_standard_view("+X"))
        self.btn_x_minus.clicked.connect(lambda: self.apply_standard_view("-X"))
        self.btn_y_plus.clicked.connect(lambda: self.apply_standard_view("+Y"))
        self.btn_y_minus.clicked.connect(lambda: self.apply_standard_view("-Y"))
        self.btn_z_plus.clicked.connect(lambda: self.apply_standard_view("+Z"))
        self.btn_z_minus.clicked.connect(lambda: self.apply_standard_view("-Z"))
        self.btn_iso_1.clicked.connect(lambda: self.apply_standard_view("Iso (+X+Y+Z)"))
        self.btn_iso_2.clicked.connect(lambda: self.apply_standard_view("Iso (+X-Y+Z)"))
        self.btn_reset_view.clicked.connect(self.reset_fit_camera)
        layout.addWidget(group_std)

        # Centering
        group_center = QGroupBox("Target Centering")
        h_center = QHBoxLayout(group_center)
        self.btn_center_mol = QPushButton("Center on Molecule")
        self.btn_center_sel = QPushButton("Center on Selected Atoms")
        self.btn_center_mol.clicked.connect(self.center_on_molecule)
        self.btn_center_sel.clicked.connect(self.center_on_selection)
        h_center.addWidget(self.btn_center_mol)
        h_center.addWidget(self.btn_center_sel)
        layout.addWidget(group_center)

        # Numerical coordinates
        group_coords = QGroupBox("Numerical Coordinates")
        grid_c = QGridLayout(group_coords)
        grid_c.setSpacing(6)

        grid_c.addWidget(QLabel("Camera Position:"), 0, 0)
        self.spin_pos_x = self._create_coord_spinbox()
        self.spin_pos_y = self._create_coord_spinbox()
        self.spin_pos_z = self._create_coord_spinbox()
        h_pos = QHBoxLayout()
        h_pos.addWidget(QLabel("X:"))
        h_pos.addWidget(self.spin_pos_x)
        h_pos.addWidget(QLabel("Y:"))
        h_pos.addWidget(self.spin_pos_y)
        h_pos.addWidget(QLabel("Z:"))
        h_pos.addWidget(self.spin_pos_z)
        grid_c.addLayout(h_pos, 0, 1)

        grid_c.addWidget(QLabel("Focal Point (Target):"), 1, 0)
        self.spin_fp_x = self._create_coord_spinbox()
        self.spin_fp_y = self._create_coord_spinbox()
        self.spin_fp_z = self._create_coord_spinbox()
        h_fp = QHBoxLayout()
        h_fp.addWidget(QLabel("X:"))
        h_fp.addWidget(self.spin_fp_x)
        h_fp.addWidget(QLabel("Y:"))
        h_fp.addWidget(self.spin_fp_y)
        h_fp.addWidget(QLabel("Z:"))
        h_fp.addWidget(self.spin_fp_z)
        grid_c.addLayout(h_fp, 1, 1)

        grid_c.addWidget(QLabel("View-Up Vector:"), 2, 0)
        self.spin_up_x = self._create_coord_spinbox(min_val=-1000.0, max_val=1000.0, default=0.0)
        self.spin_up_y = self._create_coord_spinbox(min_val=-1000.0, max_val=1000.0, default=0.0)
        self.spin_up_z = self._create_coord_spinbox(min_val=-1000.0, max_val=1000.0, default=1.0)
        h_up = QHBoxLayout()
        h_up.addWidget(QLabel("X:"))
        h_up.addWidget(self.spin_up_x)
        h_up.addWidget(QLabel("Y:"))
        h_up.addWidget(self.spin_up_y)
        h_up.addWidget(QLabel("Z:"))
        h_up.addWidget(self.spin_up_z)
        grid_c.addLayout(h_up, 2, 1)

        grid_c.addWidget(QLabel("Distance & FOV:"), 3, 0)
        self.spin_dist = QDoubleSpinBox()
        self.spin_dist.setRange(0.01, 100000.0)
        self.spin_dist.setValue(15.0)
        self.spin_dist.setDecimals(2)
        self.spin_dist.setSuffix(" Å")
        self.spin_dist.valueChanged.connect(self._on_distance_spin_changed)

        self.spin_fov = QDoubleSpinBox()
        self.spin_fov.setRange(1.0, 179.0)
        self.spin_fov.setValue(30.0)
        self.spin_fov.setDecimals(1)
        self.spin_fov.setSuffix(" °")
        self.spin_fov.valueChanged.connect(self._on_coord_changed)

        h_dist_fov = QHBoxLayout()
        h_dist_fov.addWidget(QLabel("Dist:"))
        h_dist_fov.addWidget(self.spin_dist)
        h_dist_fov.addWidget(QLabel("FOV:"))
        h_dist_fov.addWidget(self.spin_fov)
        grid_c.addLayout(h_dist_fov, 3, 1)

        grid_c.addWidget(QLabel("Projection:"), 4, 0)
        self.combo_proj = QComboBox()
        self.combo_proj.addItems(["Perspective", "Parallel (Orthographic)"])
        self.combo_proj.currentIndexChanged.connect(self._on_coord_changed)
        grid_c.addWidget(self.combo_proj, 4, 1)
        layout.addWidget(group_coords)

        # Nudge
        group_nudge = QGroupBox("Orbital Step Nudge")
        grid_n = QGridLayout(group_nudge)
        grid_n.setSpacing(6)
        grid_n.addWidget(QLabel("Step Angle:"), 0, 0)
        self.combo_step_deg = QComboBox()
        self.combo_step_deg.addItems(["5°", "15°", "45°", "90°"])
        self.combo_step_deg.setCurrentIndex(1)
        grid_n.addWidget(self.combo_step_deg, 0, 1)

        self.btn_yaw_minus = QPushButton("Yaw - (Left)")
        self.btn_yaw_plus = QPushButton("Yaw + (Right)")
        self.btn_pitch_minus = QPushButton("Pitch - (Down)")
        self.btn_pitch_plus = QPushButton("Pitch + (Up)")
        self.btn_roll_minus = QPushButton("Roll - (CCW)")
        self.btn_roll_plus = QPushButton("Roll + (CW)")
        self.btn_zoom_in = QPushButton("Zoom In (+10%)")
        self.btn_zoom_out = QPushButton("Zoom Out (-10%)")

        self.btn_yaw_minus.clicked.connect(lambda: self.nudge_orbit("azimuth", -1))
        self.btn_yaw_plus.clicked.connect(lambda: self.nudge_orbit("azimuth", 1))
        self.btn_pitch_minus.clicked.connect(lambda: self.nudge_orbit("elevation", -1))
        self.btn_pitch_plus.clicked.connect(lambda: self.nudge_orbit("elevation", 1))
        self.btn_roll_minus.clicked.connect(lambda: self.nudge_orbit("roll", -1))
        self.btn_roll_plus.clicked.connect(lambda: self.nudge_orbit("roll", 1))
        self.btn_zoom_in.clicked.connect(lambda: self.nudge_zoom(0.9))
        self.btn_zoom_out.clicked.connect(lambda: self.nudge_zoom(1.1))

        grid_n.addWidget(self.btn_yaw_minus, 1, 0)
        grid_n.addWidget(self.btn_yaw_plus, 1, 1)
        grid_n.addWidget(self.btn_pitch_minus, 2, 0)
        grid_n.addWidget(self.btn_pitch_plus, 2, 1)
        grid_n.addWidget(self.btn_roll_minus, 3, 0)
        grid_n.addWidget(self.btn_roll_plus, 3, 1)
        grid_n.addWidget(self.btn_zoom_in, 4, 0)
        grid_n.addWidget(self.btn_zoom_out, 4, 1)
        layout.addWidget(group_nudge)

        return tab_coords

    def _setup_presets_tab(self) -> QWidget:
        """Build the saved presets and bookmarks tab."""
        tab_presets = QWidget()
        layout = QVBoxLayout(tab_presets)
        layout.setSpacing(8)

        self.list_presets = QListWidget()
        self.list_presets.itemDoubleClicked.connect(self._on_preset_double_clicked)
        layout.addWidget(QLabel("Saved Viewpoints / Bookmarks:"))
        layout.addWidget(self.list_presets)

        h_btn1 = QHBoxLayout()
        self.btn_save_preset = QPushButton("Save Current View...")
        self.btn_apply_preset = QPushButton("Apply Selected")
        self.btn_update_preset = QPushButton("Overwrite Selected")
        h_btn1.addWidget(self.btn_save_preset)
        h_btn1.addWidget(self.btn_apply_preset)
        h_btn1.addWidget(self.btn_update_preset)
        layout.addLayout(h_btn1)

        h_btn2 = QHBoxLayout()
        self.btn_rename_preset = QPushButton("Rename...")
        self.btn_delete_preset = QPushButton("Delete")
        self.btn_clear_presets = QPushButton("Clear All")
        h_btn2.addWidget(self.btn_rename_preset)
        h_btn2.addWidget(self.btn_delete_preset)
        h_btn2.addWidget(self.btn_clear_presets)
        layout.addLayout(h_btn2)

        self.btn_save_preset.clicked.connect(self.save_current_preset)
        self.btn_apply_preset.clicked.connect(self.apply_selected_preset)
        self.btn_update_preset.clicked.connect(self.update_selected_preset)
        self.btn_rename_preset.clicked.connect(self.rename_selected_preset)
        self.btn_delete_preset.clicked.connect(self.delete_selected_preset)
        self.btn_clear_presets.clicked.connect(self.clear_all_presets)
        return tab_presets

    def _setup_bottom_bar(self, main_layout: QVBoxLayout) -> None:
        """Build bottom action and clipboard bar."""
        h_opts = QHBoxLayout()
        self.chk_live = QCheckBox("Live Preview")
        self.chk_live.setChecked(True)
        self.chk_smooth = QCheckBox("Smooth Transition")
        self.chk_smooth.setChecked(True)
        self.btn_copy_clip = QPushButton("Copy (JSON)")
        self.btn_paste_clip = QPushButton("Paste")
        self.btn_copy_clip.clicked.connect(self.copy_to_clipboard)
        self.btn_paste_clip.clicked.connect(self.paste_from_clipboard)

        h_opts.addWidget(self.chk_live)
        h_opts.addWidget(self.chk_smooth)
        h_opts.addStretch()
        h_opts.addWidget(self.btn_copy_clip)
        h_opts.addWidget(self.btn_paste_clip)
        main_layout.addLayout(h_opts)

        h_action = QHBoxLayout()
        self.btn_read_view = QPushButton("Read from View")
        self.btn_apply_view = QPushButton("Apply to View")
        self.btn_apply_view.setStyleSheet("font-weight: bold;")
        self.btn_close = QPushButton("Close")

        self.btn_read_view.clicked.connect(self.read_camera_from_view)
        self.btn_apply_view.clicked.connect(lambda: self.apply_to_view(force=True))
        self.btn_close.clicked.connect(self.close)

        h_action.addWidget(self.btn_read_view)
        h_action.addWidget(self.btn_apply_view)
        h_action.addWidget(self.btn_close)
        main_layout.addLayout(h_action)

    def _create_coord_spinbox(
        self,
        min_val: float = -100000.0,
        max_val: float = 100000.0,
        default: float = 0.0,
    ) -> QDoubleSpinBox:
        """Create a standardized QDoubleSpinBox for 3D coordinates."""
        sb = QDoubleSpinBox()
        sb.setRange(min_val, max_val)
        sb.setValue(default)
        sb.setDecimals(3)
        sb.setSingleStep(0.5)
        sb.valueChanged.connect(self._on_coord_changed)
        return sb

    # -----------------------------------------------------------------------
    # Camera Read / Write Logic
    # -----------------------------------------------------------------------

    def get_current_ui_state(self) -> Dict[str, Any]:
        """Collect the current camera settings from UI spinboxes."""
        pos = (
            self.spin_pos_x.value(),
            self.spin_pos_y.value(),
            self.spin_pos_z.value(),
        )
        fp = (
            self.spin_fp_x.value(),
            self.spin_fp_y.value(),
            self.spin_fp_z.value(),
        )
        up = (
            self.spin_up_x.value(),
            self.spin_up_y.value(),
            self.spin_up_z.value(),
        )
        fov = self.spin_fov.value()
        parallel = self.combo_proj.currentIndex() == 1
        dist = math.sqrt(
            (pos[0] - fp[0]) ** 2 + (pos[1] - fp[1]) ** 2 + (pos[2] - fp[2]) ** 2
        )
        return {
            "position": pos,
            "focal_point": fp,
            "view_up": up,
            "view_angle": fov,
            "parallel": parallel,
            "distance": dist,
        }

    def set_ui_state(self, state: Dict[str, Any]) -> None:
        """Populate UI controls from a camera state dictionary."""
        self._block_signals = True
        try:
            if "position" in state:
                p = state["position"]
                self.spin_pos_x.setValue(float(p[0]))
                self.spin_pos_y.setValue(float(p[1]))
                self.spin_pos_z.setValue(float(p[2]))
            if "focal_point" in state:
                f = state["focal_point"]
                self.spin_fp_x.setValue(float(f[0]))
                self.spin_fp_y.setValue(float(f[1]))
                self.spin_fp_z.setValue(float(f[2]))
            if "view_up" in state:
                u = state["view_up"]
                self.spin_up_x.setValue(float(u[0]))
                self.spin_up_y.setValue(float(u[1]))
                self.spin_up_z.setValue(float(u[2]))
            if "view_angle" in state:
                self.spin_fov.setValue(float(state["view_angle"]))
            if "parallel" in state:
                self.combo_proj.setCurrentIndex(1 if state["parallel"] else 0)

            p_curr = (
                self.spin_pos_x.value(),
                self.spin_pos_y.value(),
                self.spin_pos_z.value(),
            )
            f_curr = (
                self.spin_fp_x.value(),
                self.spin_fp_y.value(),
                self.spin_fp_z.value(),
            )
            dist = math.sqrt(
                (p_curr[0] - f_curr[0]) ** 2
                + (p_curr[1] - f_curr[1]) ** 2
                + (p_curr[2] - f_curr[2]) ** 2
            )
            self.spin_dist.setValue(dist)
        finally:
            self._block_signals = False

    def read_camera_from_view(self) -> None:
        """Read the active camera parameters from the PyVista plotter into UI."""
        plotter = getattr(self.context, "plotter", None)
        if plotter is None:
            return

        cam = getattr(plotter, "camera", None)
        if cam is None:
            return

        try:
            pos = tuple(cam.position)
            fp = tuple(cam.focal_point)
            up = tuple(cam.up)

            view_angle = 30.0
            if hasattr(cam, "view_angle"):
                view_angle = float(cam.view_angle)
            elif hasattr(cam, "GetViewAngle"):
                view_angle = float(cam.GetViewAngle())

            parallel = False
            if hasattr(cam, "parallel_projection"):
                parallel = bool(cam.parallel_projection)
            elif hasattr(cam, "GetParallelProjection"):
                parallel = bool(cam.GetParallelProjection())

            self.set_ui_state({
                "position": pos,
                "focal_point": fp,
                "view_up": up,
                "view_angle": view_angle,
                "parallel": parallel,
            })
        except Exception as e:
            logger.debug("Failed to read camera properties: %s", e)

    def apply_to_view(self, force: bool = False, animate: Optional[bool] = None) -> None:
        """Apply current UI state to the PyVista plotter."""
        if self._block_signals and not force:
            return
        if not self.chk_live.isChecked() and not force:
            return

        target_state = self.get_current_ui_state()

        use_anim = self.chk_smooth.isChecked() if animate is None else animate
        if use_anim and not self._anim_timer.isActive() and force:
            self.start_smooth_transition(target_state)
        else:
            self._apply_camera_state_direct(target_state)

    def _apply_camera_state_direct(self, state: Dict[str, Any]) -> None:
        """Directly write state coordinates to the active VTK camera."""
        plotter = getattr(self.context, "plotter", None)
        if plotter is None:
            return
        cam = getattr(plotter, "camera", None)
        if cam is None:
            return

        try:
            pos = state["position"]
            fp = state["focal_point"]
            up = state["view_up"]
            parallel = state.get("parallel", False)
            view_angle = state.get("view_angle", 30.0)

            if hasattr(cam, "SetPosition"):
                cam.SetPosition(*pos)
            if hasattr(cam, "SetFocalPoint"):
                cam.SetFocalPoint(*fp)
            if hasattr(cam, "SetViewUp"):
                cam.SetViewUp(*up)
            if hasattr(cam, "OrthogonalizeViewUp"):
                cam.OrthogonalizeViewUp()

            if hasattr(cam, "SetParallelProjection"):
                cam.SetParallelProjection(parallel)
            if not parallel and hasattr(cam, "SetViewAngle"):
                cam.SetViewAngle(view_angle)

            if hasattr(plotter, "renderer") and hasattr(
                plotter.renderer, "ResetCameraClippingRange"
            ):
                plotter.renderer.ResetCameraClippingRange()

            if hasattr(plotter, "render"):
                plotter.render()
        except Exception as e:
            logger.debug("Failed to apply camera parameters: %s", e)

    # -----------------------------------------------------------------------
    # Animation / Smooth Transition
    # -----------------------------------------------------------------------

    def start_smooth_transition(self, target_state: Dict[str, Any]) -> None:
        """Initiate animated camera interpolation towards target_state."""
        plotter = getattr(self.context, "plotter", None)
        if plotter is None or not hasattr(plotter, "camera"):
            self._apply_camera_state_direct(target_state)
            return

        cam = plotter.camera
        try:
            start_pos = tuple(cam.position)
            start_fp = tuple(cam.focal_point)
            start_up = tuple(cam.up)
            start_fov = (
                float(cam.view_angle)
                if hasattr(cam, "view_angle")
                else float(cam.GetViewAngle())
            )
        except Exception:
            self._apply_camera_state_direct(target_state)
            return

        self._anim_start_state = {
            "position": start_pos,
            "focal_point": start_fp,
            "view_up": start_up,
            "view_angle": start_fov,
        }
        self._anim_target_state = target_state
        self._anim_step = 0
        self._anim_timer.start()

    def _on_anim_step(self) -> None:
        """Advance smooth animation by one frame."""
        self._anim_step += 1
        fraction = float(self._anim_step) / float(self._anim_total_steps)
        if fraction >= 1.0:
            self._anim_timer.stop()
            self._apply_camera_state_direct(self._anim_target_state)
            return

        # Cosine ease-in-out
        t = 0.5 * (1.0 - math.cos(math.pi * fraction))

        s_pos = np.array(self._anim_start_state["position"])
        t_pos = np.array(self._anim_target_state["position"])
        s_fp = np.array(self._anim_start_state["focal_point"])
        t_fp = np.array(self._anim_target_state["focal_point"])
        s_up = np.array(self._anim_start_state["view_up"])
        t_up = np.array(self._anim_target_state["view_up"])
        s_fov = float(self._anim_start_state.get("view_angle", 30.0))
        t_fov = float(self._anim_target_state.get("view_angle", 30.0))

        cur_pos = s_pos * (1.0 - t) + t_pos * t
        cur_fp = s_fp * (1.0 - t) + t_fp * t
        cur_up = s_up * (1.0 - t) + t_up * t
        cur_fov = s_fov * (1.0 - t) + t_fov * t

        step_state = {
            "position": (cur_pos[0], cur_pos[1], cur_pos[2]),
            "focal_point": (cur_fp[0], cur_fp[1], cur_fp[2]),
            "view_up": (cur_up[0], cur_up[1], cur_up[2]),
            "view_angle": cur_fov,
            "parallel": self._anim_target_state.get("parallel", False),
        }
        self._apply_camera_state_direct(step_state)
        QCoreApplication.processEvents()

    # -----------------------------------------------------------------------
    # Interactive Features: Standard Views, Centering, Nudge, Distance
    # -----------------------------------------------------------------------

    def apply_standard_view(self, view_name: str) -> None:
        """Calculate and apply a standard view orientation."""
        fp = (
            self.spin_fp_x.value(),
            self.spin_fp_y.value(),
            self.spin_fp_z.value(),
        )
        dist = max(self.spin_dist.value(), 1.0)
        try:
            pos, up = compute_standard_view(view_name, fp, dist)
            self.set_ui_state({
                "position": pos,
                "focal_point": fp,
                "view_up": up,
            })
            self.apply_to_view(force=True)
        except Exception as e:
            logger.debug("Failed to set standard view %s: %s", view_name, e)

    def reset_fit_camera(self) -> None:
        """Call standard camera reset / fit in MoleditPy."""
        if hasattr(self.context, "reset_3d_camera"):
            self.context.reset_3d_camera()
        elif hasattr(self.context, "plotter") and hasattr(
            self.context.plotter, "reset_camera"
        ):
            self.context.plotter.reset_camera()
        self.read_camera_from_view()

    def center_on_molecule(self) -> None:
        """Center camera focal point on the entire molecule's centroid."""
        mol = getattr(self.context, "current_molecule", None)
        centroid = calculate_molecule_centroid(mol)
        if centroid is None:
            if hasattr(self.context, "show_status_message"):
                self.context.show_status_message(
                    "No 3D molecular structure available to center on."
                )
            return

        self._translate_focal_point(centroid)

    def center_on_selection(self) -> None:
        """Center camera focal point on selected atoms."""
        mol = getattr(self.context, "current_molecule", None)
        sel = (
            self.context.get_selected_atom_indices()
            if hasattr(self.context, "get_selected_atom_indices")
            else []
        )
        centroid = calculate_selected_atoms_centroid(mol, sel)
        if centroid is None:
            if hasattr(self.context, "show_status_message"):
                self.context.show_status_message(
                    "Please select one or more atoms first."
                )
            return

        self._translate_focal_point(centroid)

    def _translate_focal_point(self, new_fp: Tuple[float, float, float]) -> None:
        """Shift focal point to new_fp while maintaining the relative camera offset."""
        old_fp = (
            self.spin_fp_x.value(),
            self.spin_fp_y.value(),
            self.spin_fp_z.value(),
        )
        old_pos = (
            self.spin_pos_x.value(),
            self.spin_pos_y.value(),
            self.spin_pos_z.value(),
        )
        delta = (
            new_fp[0] - old_fp[0],
            new_fp[1] - old_fp[1],
            new_fp[2] - old_fp[2],
        )
        new_pos = (
            old_pos[0] + delta[0],
            old_pos[1] + delta[1],
            old_pos[2] + delta[2],
        )

        self.set_ui_state({
            "position": new_pos,
            "focal_point": new_fp,
        })
        self.apply_to_view(force=True)

    def nudge_orbit(self, mode: str, direction: int) -> None:
        """Step rotation of the camera around azimuth, elevation, or roll."""
        step_text = self.combo_step_deg.currentText().replace("°", "")
        try:
            step_deg = float(step_text) * float(direction)
        except ValueError:
            step_deg = 15.0 * float(direction)

        pos = (
            self.spin_pos_x.value(),
            self.spin_pos_y.value(),
            self.spin_pos_z.value(),
        )
        fp = (
            self.spin_fp_x.value(),
            self.spin_fp_y.value(),
            self.spin_fp_z.value(),
        )
        up = (
            self.spin_up_x.value(),
            self.spin_up_y.value(),
            self.spin_up_z.value(),
        )

        new_pos, new_up = orbit_step(pos, fp, up, mode, step_deg)
        self.set_ui_state({
            "position": new_pos,
            "view_up": new_up,
        })
        self.apply_to_view(force=True, animate=False)

    def nudge_zoom(self, factor: float) -> None:
        """Multiply camera distance by factor (e.g. 0.9 = zoom in, 1.1 = zoom out)."""
        pos = np.array([
            self.spin_pos_x.value(),
            self.spin_pos_y.value(),
            self.spin_pos_z.value(),
        ])
        fp = np.array([
            self.spin_fp_x.value(),
            self.spin_fp_y.value(),
            self.spin_fp_z.value(),
        ])
        vec = pos - fp
        new_pos = fp + vec * factor

        self.set_ui_state({
            "position": (float(new_pos[0]), float(new_pos[1]), float(new_pos[2]))
        })
        self.apply_to_view(force=True, animate=False)

    def _on_distance_spin_changed(self, *_args) -> None:
        """Update position when distance spinbox is directly edited."""
        if self._block_signals:
            return
        new_dist = self.spin_dist.value() if not _args else float(_args[0])
        pos = np.array([
            self.spin_pos_x.value(),
            self.spin_pos_y.value(),
            self.spin_pos_z.value(),
        ])
        fp = np.array([
            self.spin_fp_x.value(),
            self.spin_fp_y.value(),
            self.spin_fp_z.value(),
        ])
        vec = pos - fp
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            vec = np.array([0.0, 0.0, 1.0])
            norm = 1.0
        unit_vec = vec / norm
        new_pos = fp + unit_vec * new_dist

        self._block_signals = True
        try:
            self.spin_pos_x.setValue(float(new_pos[0]))
            self.spin_pos_y.setValue(float(new_pos[1]))
            self.spin_pos_z.setValue(float(new_pos[2]))
        finally:
            self._block_signals = False

        self.apply_to_view(force=False)

    def _on_coord_changed(self, *_args) -> None:
        """Triggered whenever any coordinate spinbox is edited."""
        if self._block_signals:
            return
        p = (
            self.spin_pos_x.value(),
            self.spin_pos_y.value(),
            self.spin_pos_z.value(),
        )
        f = (
            self.spin_fp_x.value(),
            self.spin_fp_y.value(),
            self.spin_fp_z.value(),
        )
        dist = math.sqrt(
            (p[0] - f[0]) ** 2 + (p[1] - f[1]) ** 2 + (p[2] - f[2]) ** 2
        )
        self._block_signals = True
        try:
            self.spin_dist.setValue(dist)
        finally:
            self._block_signals = False

        self.apply_to_view(force=False)

    # -----------------------------------------------------------------------
    # Bookmarks / Preset Management
    # -----------------------------------------------------------------------

    def _load_saved_presets(self) -> None:
        """Load presets from context settings and project cache."""
        presets = []
        if hasattr(self.context, "get_setting"):
            stored = self.context.get_setting("camera_presets", [])
            if isinstance(stored, list):
                presets.extend(stored)

        for p in _PROJECT_PRESETS:
            if p not in presets:
                presets.append(p)

        self.load_presets(presets)

    def load_presets(self, presets: List[Dict[str, Any]]) -> None:
        """Populate preset list UI from a list of preset dictionaries."""
        self._presets = list(presets)
        self.list_presets.clear()
        for p in self._presets:
            name = p.get("name", "Untitled View")
            item = QListWidgetItem(name)
            self.list_presets.addItem(item)

    def get_all_presets(self) -> List[Dict[str, Any]]:
        """Return the current list of presets."""
        return list(self._presets)

    def _persist_presets(self) -> None:
        """Save presets to context settings and project cache."""
        global _PROJECT_PRESETS  # pylint: disable=global-statement
        _PROJECT_PRESETS = list(self._presets)
        if hasattr(self.context, "set_setting"):
            self.context.set_setting("camera_presets", self._presets)
        if hasattr(self.context, "mark_project_modified"):
            self.context.mark_project_modified()

    def save_current_preset(self) -> None:
        """Prompt the user for a name and save the current view state."""
        default_name = f"View {len(self._presets) + 1}"
        name, ok = QInputDialog.getText(
            self, "Save Viewpoint", "Enter preset name:", text=default_name
        )
        if not ok or not name.strip():
            return

        state = self.get_current_ui_state()
        state["name"] = name.strip()
        self._presets.append(state)
        self.list_presets.addItem(QListWidgetItem(state["name"]))
        self._persist_presets()

    def apply_selected_preset(self) -> None:
        """Apply the currently selected viewpoint preset in the list."""
        row = self.list_presets.currentRow()
        if 0 <= row < len(self._presets):
            preset = self._presets[row]
            self.set_ui_state(preset)
            self.apply_to_view(force=True)

    def update_selected_preset(self) -> None:
        """Overwrite the selected preset with current camera values."""
        row = self.list_presets.currentRow()
        if 0 <= row < len(self._presets):
            name = self._presets[row].get("name", f"View {row + 1}")
            state = self.get_current_ui_state()
            state["name"] = name
            self._presets[row] = state
            self._persist_presets()
            if hasattr(self.context, "show_status_message"):
                self.context.show_status_message(f"Updated preset '{name}'.")

    def rename_selected_preset(self) -> None:
        """Rename the currently selected preset."""
        row = self.list_presets.currentRow()
        if 0 <= row < len(self._presets):
            old_name = self._presets[row].get("name", "")
            new_name, ok = QInputDialog.getText(
                self, "Rename Preset", "New preset name:", text=old_name
            )
            if ok and new_name.strip():
                self._presets[row]["name"] = new_name.strip()
                item = self.list_presets.item(row)
                if item:
                    item.setText(new_name.strip())
                self._persist_presets()

    def delete_selected_preset(self) -> None:
        """Delete the currently selected viewpoint preset."""
        row = self.list_presets.currentRow()
        if 0 <= row < len(self._presets):
            self._presets.pop(row)
            self.list_presets.takeItem(row)
            self._persist_presets()

    def clear_all_presets(self) -> None:
        """Clear all stored viewpoint presets after confirmation."""
        if not self._presets:
            return
        ans = QMessageBox.question(
            self,
            "Clear Presets",
            "Are you sure you want to delete all saved camera viewpoints?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans == QMessageBox.StandardButton.Yes:
            self._presets.clear()
            self.list_presets.clear()
            self._persist_presets()

    def _on_preset_double_clicked(self, _item: QListWidgetItem) -> None:
        """Double clicking a preset applies it immediately."""
        self.apply_selected_preset()

    # -----------------------------------------------------------------------
    # Clipboard Integration
    # -----------------------------------------------------------------------

    def copy_to_clipboard(self) -> None:
        """Copy the current camera state formatted as JSON and PyVista tuple."""
        state = self.get_current_ui_state()
        data = {
            "position": [round(x, 4) for x in state["position"]],
            "focal_point": [round(x, 4) for x in state["focal_point"]],
            "view_up": [round(x, 4) for x in state["view_up"]],
            "view_angle": round(state["view_angle"], 2),
            "parallel": state["parallel"],
        }
        text = json.dumps(data, indent=2)

        clipboard = QGuiApplication.clipboard()
        if clipboard:
            clipboard.setText(text)
        if hasattr(self.context, "show_status_message"):
            self.context.show_status_message("Camera position copied to clipboard.")

    def paste_from_clipboard(self) -> None:
        """Parse clipboard text (JSON or PyVista tuple) and populate the UI."""
        clipboard = QGuiApplication.clipboard()
        if not clipboard:
            return
        text = clipboard.text().strip()
        if not text:
            return

        parsed = self._parse_clipboard_camera_text(text)
        if parsed:
            self.set_ui_state(parsed)
            self.apply_to_view(force=True)
            if hasattr(self.context, "show_status_message"):
                self.context.show_status_message("Pasted camera position from clipboard.")
        else:
            QMessageBox.warning(
                self,
                "Paste Failed",
                "Could not parse valid camera coordinates from clipboard.\n"
                "Expected JSON format or PyVista [pos, fp, up] tuple.",
            )

    @staticmethod
    def _parse_clipboard_camera_text(text: str) -> Optional[Dict[str, Any]]:
        """Attempt to parse clipboard text in multiple common camera formats."""
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "position" in data and "focal_point" in data:
                return data
            if isinstance(data, (list, tuple)) and len(data) >= 2:
                pos = tuple(data[0])
                fp = tuple(data[1])
                up = tuple(data[2]) if len(data) >= 3 else (0.0, 0.0, 1.0)
                return {"position": pos, "focal_point": fp, "view_up": up}
        except Exception:
            pass

        try:
            val = ast.literal_eval(text)
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                pos = tuple(val[0])
                fp = tuple(val[1])
                up = tuple(val[2]) if len(val) >= 3 else (0.0, 0.0, 1.0)
                return {"position": pos, "focal_point": fp, "view_up": up}
        except Exception:
            pass

        return None

    def closeEvent(self, event) -> None:  # pylint: disable=invalid-name
        """Stop animations and clean up singleton slot."""
        if self._anim_timer.isActive():
            self._anim_timer.stop()
        self.context.register_window("camera_position_setter_dialog", None)
        super().closeEvent(event)
