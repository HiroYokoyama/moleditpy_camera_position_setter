"""
tests/test_camera_position_setter.py
Unit tests for the Camera Position Setter plugin.
"""

# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,protected-access,invalid-name

import importlib.util as importlib_util
import json
import math
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch
import numpy as np


# ---------------------------------------------------------------------------
# Install Qt stubs so tests run headlessly in any environment
# ---------------------------------------------------------------------------
def _install_qt_stubs():
    if "PyQt6" in sys.modules and hasattr(sys.modules["PyQt6"], "_is_stub"):
        return

    qt_core = types.ModuleType("PyQt6.QtCore")

    class _QCoreApplication:
        @staticmethod
        def processEvents():
            pass

    class _Qt:
        class ItemDataRole:
            UserRole = 256

    class _QTimer:
        def __init__(self, *args, **kwargs):
            self._active = False
            self._interval = 0
            self.timeout = MagicMock()

        def setInterval(self, ms):
            self._interval = ms

        def start(self):
            self._active = True

        def stop(self):
            self._active = False

        def isActive(self):
            return self._active

    qt_core.QCoreApplication = _QCoreApplication
    qt_core.QTimer = _QTimer
    qt_core.Qt = _Qt

    pyqt6 = types.ModuleType("PyQt6")
    pyqt6._is_stub = True
    pyqt6.QtCore = qt_core
    sys.modules["PyQt6"] = pyqt6
    sys.modules["PyQt6.QtCore"] = qt_core

    qt_gui = types.ModuleType("PyQt6.QtGui")

    class _QClipboard:
        def __init__(self):
            self._text = ""

        def text(self):
            return self._text

        def setText(self, t):
            self._text = t

    class _QGuiApplication:
        _clipboard = _QClipboard()

        @classmethod
        def clipboard(cls):
            return cls._clipboard

    qt_gui.QClipboard = _QClipboard
    qt_gui.QGuiApplication = _QGuiApplication
    pyqt6.QtGui = qt_gui
    sys.modules["PyQt6.QtGui"] = qt_gui

    qt_widgets = types.ModuleType("PyQt6.QtWidgets")

    class _QSignal:
        def __init__(self):
            self._slots = []

        def connect(self, slot):
            self._slots.append(slot)

        def emit(self, *args):
            for slot in self._slots:
                slot(*args)

    class _QDialog:
        def __init__(self, parent=None):
            self._parent = parent
            self._window_title = ""

        def setWindowTitle(self, title):
            self._window_title = title

        def resize(self, w, h):
            pass

        def show(self):
            pass

        def raise_(self):
            pass

        def close(self):
            pass

        def closeEvent(self, event):
            pass

    class _QDoubleSpinBox:
        def __init__(self, *args, **kwargs):
            self._val = 0.0
            self._min = -100000.0
            self._max = 100000.0
            self.valueChanged = _QSignal()

        def setRange(self, mi, ma):
            self._min, self._max = mi, ma

        def setValue(self, v):
            old = self._val
            self._val = float(v)
            if old != self._val:
                self.valueChanged.emit(self._val)

        def value(self):
            return float(self._val)

        def setDecimals(self, d):
            pass

        def setSingleStep(self, s):
            pass

        def setSuffix(self, s):
            pass

    class _QComboBox:
        def __init__(self, *args, **kwargs):
            self._items = []
            self._idx = 0
            self.currentIndexChanged = _QSignal()

        def addItems(self, items):
            self._items.extend(items)

        def setCurrentIndex(self, i):
            old = self._idx
            self._idx = i
            if old != self._idx:
                self.currentIndexChanged.emit(self._idx)

        def currentIndex(self):
            return self._idx

        def currentText(self):
            if 0 <= self._idx < len(self._items):
                return self._items[self._idx]
            return ""

    class _QCheckBox:
        def __init__(self, text=""):
            self._checked = False
            self._text = text

        def setChecked(self, c):
            self._checked = bool(c)

        def isChecked(self):
            return self._checked

    class _QPushButton:
        def __init__(self, text=""):
            self._text = text
            self.clicked = _QSignal()

        def setStyleSheet(self, s):
            pass

    class _QListWidgetItem:
        def __init__(self, text=""):
            self._text = text

        def text(self):
            return self._text

        def setText(self, t):
            self._text = t

    class _QListWidget:
        def __init__(self, *args, **kwargs):
            self._items = []
            self._curr = 0
            self.itemDoubleClicked = _QSignal()

        def addItem(self, item):
            self._items.append(item)

        def clear(self):
            self._items.clear()

        def currentRow(self):
            return self._curr if self._items else -1

        def setCurrentRow(self, r):
            self._curr = r

        def item(self, r):
            if 0 <= r < len(self._items):
                return self._items[r]
            return None

        def takeItem(self, r):
            if 0 <= r < len(self._items):
                return self._items.pop(r)
            return None

    class _QMessageBox:
        class StandardButton:
            Yes = 1
            No = 2

        @classmethod
        def question(cls, *args, **kwargs):
            return cls.StandardButton.Yes

        @classmethod
        def warning(cls, *args, **kwargs):
            pass

    class _QInputDialog:
        @classmethod
        def getText(cls, parent, title, label, text=""):
            return "Test Preset", True

    qt_widgets.QDialog = _QDialog
    qt_widgets.QDoubleSpinBox = _QDoubleSpinBox
    qt_widgets.QComboBox = _QComboBox
    qt_widgets.QCheckBox = _QCheckBox
    qt_widgets.QPushButton = _QPushButton
    qt_widgets.QListWidgetItem = _QListWidgetItem
    qt_widgets.QListWidget = _QListWidget
    qt_widgets.QMessageBox = _QMessageBox
    qt_widgets.QInputDialog = _QInputDialog

    for name in [
        "QVBoxLayout", "QHBoxLayout", "QGridLayout", "QGroupBox",
        "QLabel", "QScrollArea", "QWidget", "QTabWidget"
    ]:
        setattr(qt_widgets, name, lambda *args, **kwargs: MagicMock())

    pyqt6.QtWidgets = qt_widgets
    sys.modules["PyQt6.QtWidgets"] = qt_widgets


_install_qt_stubs()

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
import camera_position_setter as cam_mod  # noqa: E402
from PyQt6.QtGui import QGuiApplication  # noqa: E402


class TestMathFunctions(unittest.TestCase):
    def test_compute_standard_view_cartesian(self):
        fp = (1.0, 2.0, 3.0)
        dist = 10.0

        pos, up = cam_mod.compute_standard_view("+X", fp, dist)
        self.assertEqual(pos, (11.0, 2.0, 3.0))
        self.assertEqual(up, (0.0, 0.0, 1.0))

        pos, up = cam_mod.compute_standard_view("-X", fp, dist)
        self.assertEqual(pos, (-9.0, 2.0, 3.0))
        self.assertEqual(up, (0.0, 0.0, 1.0))

        pos, up = cam_mod.compute_standard_view("+Y", fp, dist)
        self.assertEqual(pos, (1.0, 12.0, 3.0))
        self.assertEqual(up, (0.0, 0.0, 1.0))

        pos, up = cam_mod.compute_standard_view("-Y", fp, dist)
        self.assertEqual(pos, (1.0, -8.0, 3.0))
        self.assertEqual(up, (0.0, 0.0, 1.0))

        pos, up = cam_mod.compute_standard_view("+Z", fp, dist)
        self.assertEqual(pos, (1.0, 2.0, 13.0))
        self.assertEqual(up, (0.0, 1.0, 0.0))

        pos, up = cam_mod.compute_standard_view("-Z", fp, dist)
        self.assertEqual(pos, (1.0, 2.0, -7.0))
        self.assertEqual(up, (0.0, -1.0, 0.0))

    def test_compute_standard_view_isometric(self):
        fp = (0.0, 0.0, 0.0)
        dist = 10.0
        pos, up = cam_mod.compute_standard_view("Iso (+X+Y+Z)", fp, dist)
        d_val = 10.0 / math.sqrt(3)
        self.assertAlmostEqual(pos[0], d_val)
        self.assertAlmostEqual(pos[1], d_val)
        self.assertAlmostEqual(pos[2], d_val)
        self.assertEqual(up, (0.0, 0.0, 1.0))

    def test_compute_standard_view_invalid(self):
        with self.assertRaises(ValueError):
            cam_mod.compute_standard_view("InvalidView", (0, 0, 0), 10.0)

    def test_rodrigues_rotate(self):
        v = np.array([1.0, 0.0, 0.0])
        k = np.array([0.0, 0.0, 1.0])
        # 90 deg rotation around Z
        res = cam_mod.rodrigues_rotate(v, k, math.pi / 2)
        self.assertTrue(np.allclose(res, [0.0, 1.0, 0.0]))

    def test_orbit_step_azimuth(self):
        pos = (10.0, 0.0, 0.0)
        fp = (0.0, 0.0, 0.0)
        up = (0.0, 0.0, 1.0)
        # Azimuth 90 deg step: orbits camera around up vector
        new_pos, new_up = cam_mod.orbit_step(pos, fp, up, "azimuth", 90.0)
        self.assertAlmostEqual(new_pos[0], 0.0)
        self.assertAlmostEqual(new_pos[1], -10.0)
        self.assertAlmostEqual(new_pos[2], 0.0)

    def test_orbit_step_elevation(self):
        pos = (0.0, -10.0, 0.0)
        fp = (0.0, 0.0, 0.0)
        up = (0.0, 0.0, 1.0)
        # Elevation 90 deg step
        new_pos, new_up = cam_mod.orbit_step(pos, fp, up, "elevation", 90.0)
        self.assertAlmostEqual(new_pos[0], 0.0)
        self.assertAlmostEqual(new_pos[1], 0.0)
        self.assertAlmostEqual(new_pos[2], 10.0)

    def test_orbit_step_roll(self):
        pos = (0.0, -10.0, 0.0)
        fp = (0.0, 0.0, 0.0)
        up = (0.0, 0.0, 1.0)
        # Roll 90 deg step
        new_pos, new_up = cam_mod.orbit_step(pos, fp, up, "roll", 90.0)
        self.assertEqual(new_pos, pos)
        self.assertAlmostEqual(new_up[0], -1.0)
        self.assertAlmostEqual(new_up[1], 0.0)
        self.assertAlmostEqual(new_up[2], 0.0)


class TestCentroidCalculations(unittest.TestCase):
    def test_molecule_centroid(self):
        mock_mol = MagicMock()
        mock_mol.GetNumConformers.return_value = 1
        mock_mol.GetNumAtoms.return_value = 2
        mock_conf = MagicMock()
        mock_conf.GetPositions.return_value = np.array([
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0]
        ])
        mock_mol.GetConformer.return_value = mock_conf

        c = cam_mod.calculate_molecule_centroid(mock_mol)
        self.assertEqual(c, (2.0, 3.0, 4.0))

    def test_molecule_centroid_none_or_empty(self):
        self.assertIsNone(cam_mod.calculate_molecule_centroid(None))
        mock_mol = MagicMock()
        mock_mol.GetNumConformers.return_value = 0
        self.assertIsNone(cam_mod.calculate_molecule_centroid(mock_mol))

    def test_selected_atoms_centroid(self):
        mock_mol = MagicMock()
        mock_mol.GetNumConformers.return_value = 1
        mock_mol.GetNumAtoms.return_value = 4
        mock_conf = MagicMock()
        
        class _Pt:
            def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z

        mock_conf.GetAtomPosition.side_effect = lambda i: [
            _Pt(0, 0, 0), _Pt(10, 0, 0), _Pt(0, 10, 0), _Pt(0, 0, 10)
        ][i]
        mock_mol.GetConformer.return_value = mock_conf

        c = cam_mod.calculate_selected_atoms_centroid(mock_mol, [0, 1])
        self.assertEqual(c, (5.0, 0.0, 0.0))

    def test_selected_atoms_centroid_empty(self):
        self.assertIsNone(cam_mod.calculate_selected_atoms_centroid(None, []))


class TestDialogFeatures(unittest.TestCase):
    def setUp(self):
        self.context = MagicMock()
        self.plotter = MagicMock()
        self.camera = MagicMock()
        self.renderer = MagicMock()
        self.plotter.camera = self.camera
        self.plotter.renderer = self.renderer
        self.context.plotter = self.plotter

        # Camera defaults
        self.camera.position = (0.0, -10.0, 0.0)
        self.camera.focal_point = (0.0, 0.0, 0.0)
        self.camera.up = (0.0, 0.0, 1.0)
        self.camera.view_angle = 30.0
        self.camera.parallel_projection = False

        self.dialog = cam_mod.CameraPositionDialog(self.context)

    def test_read_camera_from_view(self):
        self.camera.position = (5.0, 6.0, 7.0)
        self.camera.focal_point = (1.0, 2.0, 3.0)
        self.camera.up = (0.0, 1.0, 0.0)
        self.camera.view_angle = 45.0
        self.camera.parallel_projection = True

        self.dialog.read_camera_from_view()
        state = self.dialog.get_current_ui_state()

        self.assertEqual(state["position"], (5.0, 6.0, 7.0))
        self.assertEqual(state["focal_point"], (1.0, 2.0, 3.0))
        self.assertEqual(state["view_up"], (0.0, 1.0, 0.0))
        self.assertEqual(state["view_angle"], 45.0)
        self.assertTrue(state["parallel"])

    def test_apply_standard_view(self):
        self.dialog.spin_fp_x.setValue(0.0)
        self.dialog.spin_fp_y.setValue(0.0)
        self.dialog.spin_fp_z.setValue(0.0)
        self.dialog.spin_dist.setValue(10.0)

        self.dialog.apply_standard_view("+Z")
        state = self.dialog.get_current_ui_state()

        self.assertEqual(state["position"], (0.0, 0.0, 10.0))
        self.assertEqual(state["view_up"], (0.0, 1.0, 0.0))

    def test_nudge_orbit_and_zoom(self):
        self.dialog.set_ui_state({
            "position": (0.0, -10.0, 0.0),
            "focal_point": (0.0, 0.0, 0.0),
            "view_up": (0.0, 0.0, 1.0),
        })
        self.dialog.combo_step_deg.setCurrentIndex(1)  # 15 deg
        self.dialog.nudge_orbit("azimuth", 1)
        state = self.dialog.get_current_ui_state()
        self.assertNotEqual(state["position"], (0.0, -10.0, 0.0))

        # Zoom in
        self.dialog.nudge_zoom(0.5)
        state2 = self.dialog.get_current_ui_state()
        self.assertAlmostEqual(state2["distance"], state["distance"] * 0.5, places=3)

    def test_presets_crud(self):
        self.dialog.set_ui_state({
            "position": (1.0, 2.0, 3.0),
            "focal_point": (0.0, 0.0, 0.0),
            "view_up": (0.0, 0.0, 1.0),
        })
        self.dialog.save_current_preset()
        presets = self.dialog.get_all_presets()
        self.assertEqual(len(presets), 1)
        self.assertEqual(presets[0]["name"], "Test Preset")

        # Update preset
        self.dialog.spin_pos_x.setValue(9.0)
        self.dialog.list_presets.setCurrentRow(0)
        self.dialog.update_selected_preset()
        self.assertEqual(self.dialog.get_all_presets()[0]["position"][0], 9.0)

        # Apply preset
        self.dialog.apply_selected_preset()
        self.assertEqual(self.dialog.spin_pos_x.value(), 9.0)

        # Delete preset
        self.dialog.delete_selected_preset()
        self.assertEqual(len(self.dialog.get_all_presets()), 0)

    def test_clipboard_copy_and_paste(self):
        self.dialog.set_ui_state({
            "position": (10.0, 20.0, 30.0),
            "focal_point": (1.0, 2.0, 3.0),
            "view_up": (0.0, 0.0, 1.0),
            "view_angle": 35.0,
            "parallel": False,
        })
        self.dialog.copy_to_clipboard()
        clip_text = QGuiApplication.clipboard().text()
        self.assertIn('"position"', clip_text)

        # Reset UI then paste
        self.dialog.spin_pos_x.setValue(0.0)
        self.dialog.paste_from_clipboard()
        self.assertEqual(self.dialog.spin_pos_x.value(), 10.0)

    def test_smooth_transition_steps(self):
        self.dialog.set_ui_state({
            "position": (0.0, 0.0, 0.0),
            "focal_point": (0.0, 0.0, 0.0),
            "view_up": (0.0, 0.0, 1.0),
        })
        target = {
            "position": (10.0, 10.0, 10.0),
            "focal_point": (0.0, 0.0, 0.0),
            "view_up": (0.0, 0.0, 1.0),
            "view_angle": 30.0,
            "parallel": False,
        }
        self.dialog.start_smooth_transition(target)
        self.assertTrue(self.dialog._anim_timer.isActive())

        # Advance frame
        self.dialog._on_anim_step()
        self.assertGreater(self.dialog._anim_step, 0)

        # Complete animation
        for _ in range(self.dialog._anim_total_steps + 1):
            self.dialog._on_anim_step()
        self.assertFalse(self.dialog._anim_timer.isActive())


class TestPersistenceHandlers(unittest.TestCase):
    def test_save_load_reset_cycle(self):
        context = MagicMock()
        fake_win = MagicMock()
        fake_win.get_all_presets.return_value = [{"name": "P1", "position": (1, 2, 3)}]
        context.get_window.return_value = fake_win

        saved = cam_mod._on_save_project(context)
        self.assertEqual(len(saved["bookmarks"]), 1)

        # Load
        cam_mod._on_load_project(context, saved)
        fake_win.load_presets.assert_called_with(saved["bookmarks"])

        # Reset
        cam_mod._on_document_reset(context)
        fake_win.load_presets.assert_called_with([])


if __name__ == "__main__":
    unittest.main()
