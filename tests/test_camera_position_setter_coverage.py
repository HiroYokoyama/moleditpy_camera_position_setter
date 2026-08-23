"""
tests/test_camera_position_setter_coverage.py
Extended unit & edge-case coverage tests for camera_position_setter.py.
"""

# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,protected-access,invalid-name

import importlib.util as importlib_util
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
import camera_position_setter as cam_mod


class TestCoverageEdgeCases(unittest.TestCase):
    def setUp(self):
        self.context = MagicMock()
        self.plotter = MagicMock()
        self.camera = MagicMock()
        self.renderer = MagicMock()
        self.plotter.camera = self.camera
        self.plotter.renderer = self.renderer
        self.context.plotter = self.plotter

        self.camera.position = (0.0, -10.0, 0.0)
        self.camera.focal_point = (0.0, 0.0, 0.0)
        self.camera.up = (0.0, 0.0, 1.0)
        self.camera.view_angle = 30.0
        self.camera.parallel_projection = False

        self.dialog = cam_mod.CameraPositionDialog(self.context)

    def test_rodrigues_degenerate_axis(self):
        v = np.array([1.0, 2.0, 3.0])
        zero_k = np.array([0.0, 0.0, 0.0])
        res = cam_mod.rodrigues_rotate(v, zero_k, 1.0)
        self.assertTrue(np.allclose(res, v))

    def test_orbit_step_degenerate_vector(self):
        pos = (1.0, 1.0, 1.0)
        fp = (1.0, 1.0, 1.0)
        up = (0.0, 0.0, 1.0)
        p_out, u_out = cam_mod.orbit_step(pos, fp, up, "azimuth", 10.0)
        self.assertEqual(p_out, pos)
        self.assertEqual(u_out, up)

    def test_orbit_step_unknown_mode(self):
        pos = (0.0, 10.0, 0.0)
        fp = (0.0, 0.0, 0.0)
        up = (0.0, 0.0, 1.0)
        p_out, u_out = cam_mod.orbit_step(pos, fp, up, "unknown", 10.0)
        self.assertEqual(p_out, pos)

    def test_all_isometric_views(self):
        for vname in ["Iso (+X+Y+Z)", "Iso (+X-Y+Z)", "Iso (-X-Y+Z)", "Iso (-X+Y+Z)"]:
            pos, up = cam_mod.compute_standard_view(vname, (0, 0, 0), 10.0)
            self.assertEqual(len(pos), 3)
            self.assertEqual(up, (0.0, 0.0, 1.0))

    def test_parse_clipboard_formats(self):
        j_text = '{"position": [1, 2, 3], "focal_point": [0, 0, 0]}'
        p1 = cam_mod.CameraPositionDialog._parse_clipboard_camera_text(j_text)
        self.assertIsNotNone(p1)
        self.assertEqual(p1["position"], [1, 2, 3])

        pv_text = '[(1, 2, 3), (0, 0, 0), (0, 0, 1)]'
        p2 = cam_mod.CameraPositionDialog._parse_clipboard_camera_text(pv_text)
        self.assertIsNotNone(p2)
        self.assertEqual(p2["position"], (1, 2, 3))
        self.assertEqual(p2["focal_point"], (0, 0, 0))

        self.assertIsNone(cam_mod.CameraPositionDialog._parse_clipboard_camera_text("not a camera string"))

    def test_center_on_molecule_no_mol(self):
        self.context.current_molecule = None
        self.dialog.center_on_molecule()
        self.context.show_status_message.assert_called_with("No 3D molecular structure available to center on.")

    def test_center_on_selection_no_selection(self):
        self.context.current_molecule = MagicMock()
        self.context.get_selected_atom_indices.return_value = []
        self.dialog.center_on_selection()
        self.context.show_status_message.assert_called_with("Please select one or more atoms first.")

    def test_merge_and_clear_project_presets(self):
        self.dialog._presets = [
            {"name": "P1", "position": (1, 1, 1), "focal_point": (0, 0, 0), "view_up": (0, 0, 1), "is_global": True, "is_project": False},
        ]
        self.dialog.merge_project_presets([
            {"name": "P1", "position": (1, 1, 1), "focal_point": (0, 0, 0), "view_up": (0, 0, 1), "is_project": True},
            {"name": "P2", "position": (2, 2, 2), "focal_point": (0, 0, 0), "view_up": (0, 0, 1), "is_project": True},
        ])
        self.assertEqual(len(self.dialog.get_all_presets()), 2)
        self.assertEqual(len(self.dialog.get_global_presets()), 1)
        self.assertEqual(len(self.dialog.get_project_presets()), 2)

        # Clear project bookmarks
        self.dialog.clear_project_presets()
        self.assertEqual(len(self.dialog.get_all_presets()), 1)
        self.assertTrue(self.dialog.get_all_presets()[0]["is_global"])

    def test_clear_filtered_presets(self):
        self.dialog._presets = [
            {"name": "P1", "position": (1, 1, 1), "focal_point": (0, 0, 0), "view_up": (0, 0, 1), "is_global": True, "is_project": False},
            {"name": "P2", "position": (2, 2, 2), "focal_point": (0, 0, 0), "view_up": (0, 0, 1), "is_global": False, "is_project": True},
        ]
        self.dialog.combo_filter.setCurrentIndex(1)  # Clear global only
        self.dialog.clear_filtered_presets()
        self.assertEqual(len(self.dialog.get_global_presets()), 0)
        self.assertEqual(len(self.dialog.get_project_presets()), 1)

        # Clear all
        self.dialog.combo_filter.setCurrentIndex(0)
        self.dialog.clear_filtered_presets()
        self.assertEqual(len(self.dialog.get_all_presets()), 0)

    def test_distance_spinbox_manual_edit(self):
        self.dialog.set_ui_state({
            "position": (0.0, -10.0, 0.0),
            "focal_point": (0.0, 0.0, 0.0),
            "view_up": (0.0, 0.0, 1.0),
        })
        self.dialog._on_distance_spin_changed(25.0)
        state = self.dialog.get_current_ui_state()
        self.assertAlmostEqual(state["distance"], 25.0, places=2)

    def test_ui_settings_persistence(self):
        self.dialog._on_live_toggled(False)
        self.context.set_setting.assert_called_with("live_preview", False)

        self.dialog._on_smooth_toggled(False)
        self.context.set_setting.assert_called_with("smooth_transition", False)

        self.dialog._on_step_deg_changed(2)
        self.context.set_setting.assert_called_with("nudge_step_deg_index", 2)

        self.dialog._on_filter_changed(1)
        self.context.set_setting.assert_called_with("filter_index", 1)

    def test_close_event(self):
        event = MagicMock()
        self.dialog._anim_timer.start()
        self.dialog.closeEvent(event)
        self.assertFalse(self.dialog._anim_timer.isActive())
        self.context.register_window.assert_called_with("camera_position_setter_dialog", None)


if __name__ == "__main__":
    unittest.main()
