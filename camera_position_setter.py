"""
Camera Position Setter Plugin for MoleditPy.

Provides precise 3D camera control, standard axis views (+X, -X, +Y, -Y, +Z, -Z,
Isometric), exact numerical coordinate setters, target centering (molecule /
selected atoms), orbital step adjustments (nudge), viewpoint bookmarks with
separated global presets and project bookmarks (with save checkboxes),
clipboard copy/paste, and smooth animated transitions.

Source code, README, and full license (GNU GPL):
    https://github.com/HiroYokoyama/moleditpy_camera_position_setter
Copyright (c) HiroYokoyama. Licensed under the GNU General Public License;
see the LICENSE file in the repository above for the full terms.
"""

# pylint: disable=too-many-lines,too-many-instance-attributes,too-many-public-methods
# pylint: disable=too-many-locals,too-many-statements,attribute-defined-outside-init
# pylint: disable=too-few-public-methods,no-name-in-module,broad-exception-caught

import ast
import json
import math
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
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
PLUGIN_VERSION = "2.0.0a1"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = (
    "Precise 3D camera controller, axis view aligner, and viewpoint manager for MoleditPy."
)
PLUGIN_CATEGORY = "View"
PLUGIN_TAGS = ["Visualization", "Utility"]
PLUGIN_DEPENDENCIES = ["pyvista", "PySide6-Essentials", "numpy"]
PLUGIN_SUPPORTED_MOLEDITPY_VERSION = ">=3.0.0, <5.0.0"
PLUGIN_SUPPORTED_PYTHON_VERSION = ">=3.10, <3.15"
PLUGIN_SUPPORTED_OS = ["Windows", "macOS", "Linux", "WSL"]

# Global cache for active project bookmarks
_PROJECT_BOOKMARKS: List[Dict[str, Any]] = []


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
    if win and hasattr(win, "get_project_presets"):
        presets = win.get_project_presets()
    else:
        presets = [p for p in _PROJECT_BOOKMARKS if p.get("is_project", True)]
    return {"bookmarks": presets}


def _on_load_project(context, data: Any) -> None:
    """Restore project-specific camera bookmarks when opening a project."""
    global _PROJECT_BOOKMARKS  # pylint: disable=global-statement
    if isinstance(data, dict) and "bookmarks" in data and isinstance(data["bookmarks"], list):
        loaded = []
        for p in data["bookmarks"]:
            p_copy = dict(p)
            p_copy["is_project"] = True
            if "is_global" not in p_copy:
                p_copy["is_global"] = False
            loaded.append(p_copy)
        _PROJECT_BOOKMARKS = loaded
    else:
        _PROJECT_BOOKMARKS = []

    win = context.get_window("camera_position_setter_dialog")
    if win and hasattr(win, "merge_project_presets"):
        win.merge_project_presets(_PROJECT_BOOKMARKS)


def _on_document_reset(context) -> None:
    """Clear project-specific bookmarks on File > New."""
    global _PROJECT_BOOKMARKS  # pylint: disable=global-statement
    _PROJECT_BOOKMARKS = []
    win = context.get_window("camera_position_setter_dialog")
    if win and hasattr(win, "clear_project_presets"):
        win.clear_project_presets()


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


def collect_selected_atom_indices(context) -> List[int]:
    """
    Atoms selected in the 3D viewport.

    This deliberately ignores ``context.get_selected_atom_indices()``, which reports
    only the 2D scene selection (``QGraphicsScene.selectedItems()``) and has no bearing
    on what the user picked in the 3D view.

    MoleditPy's own "3D Select" is ``Edit3DManager.selected_atoms_for_measurement`` --
    the atoms clicked in the 3D viewport, drawn as red numbered labels. That is the
    single source the host's ``DialogManager._get_preselected_atoms_3d()`` reads to
    preselect atoms for every 3D dialog, so it is what this plugin follows.

    ``selected_atoms_3d`` (the yellow translucent highlight spheres) is also unioned in
    for completeness, but nothing in the current app writes to it -- ``Edit3DManager``
    exposes ``toggle_atom_selection_3d()`` yet no caller invokes it.
    """
    mw = context.get_main_window() if hasattr(context, "get_main_window") else None
    e3d = getattr(mw, "edit_3d_manager", None) if mw is not None else None
    if e3d is None:
        return []

    indices: set = set()
    for attr in ("selected_atoms_for_measurement", "selected_atoms_3d"):
        picked = getattr(e3d, attr, None)
        if not isinstance(picked, (set, list, tuple)):
            continue
        try:
            indices.update(int(i) for i in picked)
        except (TypeError, ValueError) as e:
            logger.debug("Failed to read %s: %s", attr, e)

    return sorted(indices)


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
# Save Preset Modal Dialog with Scope Checkboxes
# ---------------------------------------------------------------------------


class SavePresetDialog(QDialog):
    """Modal dialog asking for a preset name and checkboxes for global / project scope."""

    def __init__(
        self,
        default_name: str = "View 1",
        default_global: bool = True,
        default_project: bool = True,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Save Viewpoint")
        self.resize(360, 180)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Viewpoint Preset Name:"))
        self.name_edit = QLineEdit(default_name)
        layout.addWidget(self.name_edit)

        # Checkboxes for separating global presets and project bookmarks
        group_scope = QGroupBox("Save Scope")
        scope_layout = QVBoxLayout(group_scope)
        self.chk_global = QCheckBox("Save to Global Presets (available in all sessions)")
        self.chk_global.setChecked(default_global)
        self.chk_project = QCheckBox("Save to Current Project (saved in .pmeprj file)")
        self.chk_project.setChecked(default_project)
        scope_layout.addWidget(self.chk_global)
        scope_layout.addWidget(self.chk_project)
        layout.addWidget(group_scope)

        btn_layout = QHBoxLayout()
        self.btn_save = QPushButton("Save")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_save.clicked.connect(self._on_save_clicked)
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

    def _on_save_clicked(self) -> None:
        """Validate input before accepting."""
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a name for the preset.")
            return
        if not self.chk_global.isChecked() and not self.chk_project.isChecked():
            QMessageBox.warning(
                self,
                "No Scope Selected",
                "Please check at least one option: Global Presets or Current Project.",
            )
            return
        self.accept()

    def get_result(self) -> Tuple[str, bool, bool]:
        """Return (name, is_global, is_project)."""
        return (
            self.name_edit.text().strip(),
            self.chk_global.isChecked(),
            self.chk_project.isChecked(),
        )


# ---------------------------------------------------------------------------
# Main Camera Dialog Implementation
# ---------------------------------------------------------------------------


class CameraPositionDialog(QDialog):
    """Interactive GUI Dialog for precise 3D camera setting and viewpoint management."""

    def __init__(self, context, parent=None):
        super().__init__(parent)
        self.context = context
        self.setWindowTitle("Camera Position Setter")
        self.resize(520, 700)

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
        init_step = 1
        if hasattr(self.context, "get_setting"):
            init_step = int(self.context.get_setting("nudge_step_deg_index", 1))
        self.combo_step_deg.setCurrentIndex(init_step)
        self.combo_step_deg.currentIndexChanged.connect(self._on_step_deg_changed)
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
        """Build the saved presets and bookmarks tab with category filtering."""
        tab_presets = QWidget()
        layout = QVBoxLayout(tab_presets)
        layout.setSpacing(8)

        # Filter bar
        h_filter = QHBoxLayout()
        h_filter.addWidget(QLabel("Show Viewpoints:"))
        self.combo_filter = QComboBox()
        self.combo_filter.addItems([
            "All Viewpoints",
            "Global Presets (All Files)",
            "Project Bookmarks (This File)",
        ])
        init_filter = 0
        if hasattr(self.context, "get_setting"):
            init_filter = int(self.context.get_setting("filter_index", 0))
        self.combo_filter.setCurrentIndex(init_filter)
        self.combo_filter.currentIndexChanged.connect(self._on_filter_changed)
        h_filter.addWidget(self.combo_filter)
        h_filter.addStretch()
        layout.addLayout(h_filter)

        self.list_presets = QListWidget()
        self.list_presets.itemDoubleClicked.connect(self._on_preset_double_clicked)
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
        self.btn_clear_presets = QPushButton("Clear Filtered")
        h_btn2.addWidget(self.btn_rename_preset)
        h_btn2.addWidget(self.btn_delete_preset)
        h_btn2.addWidget(self.btn_clear_presets)
        layout.addLayout(h_btn2)

        self.btn_save_preset.clicked.connect(self.save_current_preset)
        self.btn_apply_preset.clicked.connect(self.apply_selected_preset)
        self.btn_update_preset.clicked.connect(self.update_selected_preset)
        self.btn_rename_preset.clicked.connect(self.rename_selected_preset)
        self.btn_delete_preset.clicked.connect(self.delete_selected_preset)
        self.btn_clear_presets.clicked.connect(self.clear_filtered_presets)
        return tab_presets

    def _setup_bottom_bar(self, main_layout: QVBoxLayout) -> None:
        """Build bottom action and clipboard bar."""
        h_opts = QHBoxLayout()
        init_live = True
        init_smooth = True
        if hasattr(self.context, "get_setting"):
            init_live = bool(self.context.get_setting("live_preview", True))
            init_smooth = bool(self.context.get_setting("smooth_transition", True))

        self.chk_live = QCheckBox("Live Preview")
        self.chk_live.setChecked(init_live)
        self.chk_live.toggled.connect(self._on_live_toggled)

        self.chk_smooth = QCheckBox("Smooth Transition")
        self.chk_smooth.setChecked(init_smooth)
        self.chk_smooth.toggled.connect(self._on_smooth_toggled)

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
        """Center camera focal point on the atoms selected in the 3D viewport."""
        mol = getattr(self.context, "current_molecule", None)
        sel = collect_selected_atom_indices(self.context)
        centroid = calculate_selected_atoms_centroid(mol, sel)
        if centroid is None:
            if hasattr(self.context, "show_status_message"):
                self.context.show_status_message(
                    "Please select one or more atoms in the 3D view first."
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

    def _on_live_toggled(self, checked: bool) -> None:
        """Persist live preview checkbox toggle."""
        if hasattr(self.context, "set_setting"):
            self.context.set_setting("live_preview", checked)

    def _on_smooth_toggled(self, checked: bool) -> None:
        """Persist smooth transition checkbox toggle."""
        if hasattr(self.context, "set_setting"):
            self.context.set_setting("smooth_transition", checked)

    def _on_step_deg_changed(self, idx: int) -> None:
        """Persist orbital step angle selection."""
        if hasattr(self.context, "set_setting"):
            self.context.set_setting("nudge_step_deg_index", idx)

    def _on_filter_changed(self, idx: int) -> None:
        """Persist viewpoint filter dropdown index and refresh list."""
        if hasattr(self.context, "set_setting"):
            self.context.set_setting("filter_index", idx)
        self._refresh_presets_list_ui()

    # -----------------------------------------------------------------------
    # Bookmarks / Preset Management (Separated Global vs Project Scopes)
    # -----------------------------------------------------------------------

    def _load_saved_presets(self) -> None:
        """Load presets from context settings and active project cache."""
        presets = []
        if hasattr(self.context, "get_setting"):
            stored = self.context.get_setting("camera_presets", [])
            if isinstance(stored, list):
                for p in stored:
                    p_copy = dict(p)
                    p_copy["is_global"] = True
                    if "is_project" not in p_copy:
                        p_copy["is_project"] = False
                    presets.append(p_copy)

        for p in _PROJECT_BOOKMARKS:
            match = next((x for x in presets if x.get("name") == p.get("name")), None)
            if match:
                match["is_project"] = True
            else:
                p_copy = dict(p)
                p_copy["is_project"] = True
                if "is_global" not in p_copy:
                    p_copy["is_global"] = False
                presets.append(p_copy)

        self._presets = presets
        self._refresh_presets_list_ui()

    def merge_project_presets(self, bookmarks: List[Dict[str, Any]]) -> None:
        """Merge loaded project bookmarks into dialog without overriding globals."""
        for p in bookmarks:
            match = next((x for x in self._presets if x.get("name") == p.get("name")), None)
            if match:
                match["is_project"] = True
            else:
                p_copy = dict(p)
                p_copy["is_project"] = True
                if "is_global" not in p_copy:
                    p_copy["is_global"] = False
                self._presets.append(p_copy)
        self._refresh_presets_list_ui()

    def clear_project_presets(self) -> None:
        """Clear project bookmarks on File > New, retaining global presets."""
        remaining = []
        for p in self._presets:
            if p.get("is_global", False):
                p["is_project"] = False
                remaining.append(p)
        self._presets = remaining
        self._refresh_presets_list_ui()

    def get_all_presets(self) -> List[Dict[str, Any]]:
        """Return all viewpoints in memory."""
        return list(self._presets)

    def get_global_presets(self) -> List[Dict[str, Any]]:
        """Return viewpoints marked for global settings."""
        return [p for p in self._presets if p.get("is_global", False)]

    def get_project_presets(self) -> List[Dict[str, Any]]:
        """Return viewpoints marked for project persistence."""
        return [p for p in self._presets if p.get("is_project", False)]

    def _visible_presets(self) -> List[Dict[str, Any]]:
        """Presets shown under the active filter, in list-widget row order."""
        filter_idx = self.combo_filter.currentIndex()
        visible = []
        for p in self._presets:
            if filter_idx == 1 and not p.get("is_global", False):
                continue
            if filter_idx == 2 and not p.get("is_project", False):
                continue
            visible.append(p)
        return visible

    def _refresh_presets_list_ui(self, *_args) -> None:
        """Repaint list items filtered by the category dropdown."""
        self.list_presets.clear()

        for p in self._visible_presets():
            is_glob = p.get("is_global", False)
            is_proj = p.get("is_project", False)

            tag = ""
            if is_glob and is_proj:
                tag = "[Global+Project] "
            elif is_glob:
                tag = "[Global] "
            elif is_proj:
                tag = "[Project] "

            name = p.get("name", "Untitled View")
            item = QListWidgetItem(f"{tag}{name}")
            self.list_presets.addItem(item)

    def _get_filtered_preset_at(self, visual_row: int) -> Optional[Dict[str, Any]]:
        """Map visual row in list widget to actual preset dict."""
        visible = self._visible_presets()
        if 0 <= visual_row < len(visible):
            return visible[visual_row]
        return None

    def _persist_presets(self) -> None:
        """Write separated global presets to settings and project bookmarks to cache."""
        global _PROJECT_BOOKMARKS  # pylint: disable=global-statement
        globals_list = self.get_global_presets()
        projects_list = self.get_project_presets()

        _PROJECT_BOOKMARKS = projects_list

        if hasattr(self.context, "set_setting"):
            self.context.set_setting("camera_presets", globals_list)
        if hasattr(self.context, "mark_project_modified"):
            self.context.mark_project_modified()

    def save_current_preset(self) -> None:
        """Prompt user with scope checkboxes (Global vs Project) and save."""
        default_name = f"View {len(self._presets) + 1}"
        default_global = True
        default_project = True
        if hasattr(self.context, "get_setting"):
            default_global = bool(self.context.get_setting("save_scope_global", True))
            default_project = bool(self.context.get_setting("save_scope_project", True))

        dlg = SavePresetDialog(
            default_name=default_name,
            default_global=default_global,
            default_project=default_project,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        name, is_global, is_project = dlg.get_result()
        if hasattr(self.context, "set_setting"):
            self.context.set_setting("save_scope_global", is_global)
            self.context.set_setting("save_scope_project", is_project)

        state = self.get_current_ui_state()
        state["name"] = name
        state["is_global"] = is_global
        state["is_project"] = is_project

        # Replace existing with same name or append
        existing = next((x for x in self._presets if x.get("name") == name), None)
        if existing:
            existing.update(state)
        else:
            self._presets.append(state)

        self._persist_presets()
        self._refresh_presets_list_ui()

    def apply_selected_preset(self) -> None:
        """Apply the currently selected viewpoint preset in the list."""
        row = self.list_presets.currentRow()
        preset = self._get_filtered_preset_at(row)
        if preset:
            self.set_ui_state(preset)
            self.apply_to_view(force=True)

    def update_selected_preset(self) -> None:
        """Overwrite the selected preset with current camera values."""
        row = self.list_presets.currentRow()
        preset = self._get_filtered_preset_at(row)
        if preset:
            name = preset.get("name", "View")
            is_glob = preset.get("is_global", True)
            is_proj = preset.get("is_project", True)
            state = self.get_current_ui_state()
            state["name"] = name
            state["is_global"] = is_glob
            state["is_project"] = is_proj
            preset.update(state)
            self._persist_presets()
            self._refresh_presets_list_ui()
            if hasattr(self.context, "show_status_message"):
                self.context.show_status_message(f"Updated viewpoint '{name}'.")

    def rename_selected_preset(self) -> None:
        """Rename the currently selected preset."""
        row = self.list_presets.currentRow()
        preset = self._get_filtered_preset_at(row)
        if preset:
            old_name = preset.get("name", "")
            new_name, ok = QInputDialog.getText(
                self, "Rename Preset", "New preset name:", text=old_name
            )
            if ok and new_name.strip():
                preset["name"] = new_name.strip()
                self._persist_presets()
                self._refresh_presets_list_ui()

    def delete_selected_preset(self) -> None:
        """Delete the currently selected viewpoint preset."""
        row = self.list_presets.currentRow()
        preset = self._get_filtered_preset_at(row)
        if preset and preset in self._presets:
            self._presets.remove(preset)
            self._persist_presets()
            self._refresh_presets_list_ui()

    def clear_filtered_presets(self) -> None:
        """Clear viewpoints currently visible in the active filter."""
        filter_idx = self.combo_filter.currentIndex()
        if not self._presets:
            return

        desc = (
            "all viewpoints"
            if filter_idx == 0
            else ("global presets" if filter_idx == 1 else "project bookmarks")
        )
        ans = QMessageBox.question(
            self,
            "Clear Viewpoints",
            f"Are you sure you want to delete {desc}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans == QMessageBox.StandardButton.Yes:
            if filter_idx == 0:
                self._presets.clear()
            elif filter_idx == 1:
                self._presets = [p for p in self._presets if not p.get("is_global", False)]
            elif filter_idx == 2:
                self._presets = [p for p in self._presets if not p.get("is_project", False)]

            self._persist_presets()
            self._refresh_presets_list_ui()

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

    def _release(self) -> None:
        """Stop the animation timer and free the singleton slot."""
        if self._anim_timer.isActive():
            self._anim_timer.stop()
        self.context.register_window("camera_position_setter_dialog", None)

    def done(self, result: int) -> None:
        """Esc closes a QDialog via reject() -> done(), never through closeEvent."""
        self._release()
        super().done(result)

    def closeEvent(self, event) -> None:  # pylint: disable=invalid-name
        """Stop animations and clean up singleton slot."""
        self._release()
        super().closeEvent(event)
