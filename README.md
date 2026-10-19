# Camera Position Setter Plugin

[![CI](https://github.com/HiroYokoyama/moleditpy_camera_position_setter/actions/workflows/ci.yml/badge.svg)](https://github.com/HiroYokoyama/moleditpy_camera_position_setter/actions/workflows/ci.yml)
![Test Coverage](https://img.shields.io/badge/coverage->90%25-green)
[![GitHub tag](https://img.shields.io/github/v/tag/HiroYokoyama/moleditpy_camera_position_setter?label=version)](https://github.com/HiroYokoyama/moleditpy_camera_position_setter/tags)
[![GitHub Downloads](https://img.shields.io/github/downloads/HiroYokoyama/moleditpy_camera_position_setter/total)](https://github.com/HiroYokoyama/moleditpy_camera_position_setter/releases)

## Overview

**Camera Position Setter** is an advanced 3D viewport control and viewpoint bookmarking plugin for [MoleditPy](https://github.com/HiroYokoyama/python_molecular_editor). It enables computational chemists and molecular modelers to set, align, and manipulate the 3D camera with numerical precision, jump instantly to standard Cartesian and isometric projections, center the view on specific molecules or selected atoms, store viewpoint bookmarks across sessions and project files, copy/paste camera coordinates, and perform smooth animated transitions.

## Key Features

* **Standard View Presets**: One-click alignment to standard Cartesian axes (+X Right, -X Left, +Y Back, -Y Front, +Z Top, -Z Bottom) and Isometric angles (+X+Y+Z, +X-Y+Z, etc.).
* **Exact Numerical Coordinates**: Precise spinbox inputs for Camera Position $(X, Y, Z)$, Focal Point / Target $(X, Y, Z)$, View-Up vector $(X, Y, Z)$, Distance, and Field of View (FOV).
* **Projection Modes**: Seamless toggle between Perspective and Parallel (Orthographic) projection.
* **Target Centering**:
  * **Center on Molecule**: Auto-centers the focal point on the molecular centroid.
  * **Center on Selection**: Centers the focal point on the atoms selected in the **3D viewport** (MoleditPy's 3D Select — the atoms you click in the 3D view, shown with red number labels). The 2D sketch selection is intentionally ignored.
* **Orbital Step Nudge**: Incremental rotational nudges (Yaw/Azimuth, Pitch/Elevation, Roll) by 5°, 15°, 45°, or 90°, plus fine zoom controls.
* **Viewpoint Bookmarks & Presets**: Save, rename, apply, update, and manage named camera presets with persistence across app sessions and `.pmeprj` project files.
* **Clipboard Integration**: Copy camera state to clipboard as formatted JSON or PyVista `[position, focal_point, view_up]` tuples, and paste directly back.
* **Smooth Animated Transitions**: Optional cosine-interpolated camera motion between viewpoints for clear spatial orientation.
* **Live Preview**: Real-time viewport rendering as values are adjusted.

## Requirements

This plugin runs in the MoleditPy environment:
* `PySide6`
* `numpy`
* `pyvista`

## Installation

1. Copy or symlink `camera_position_setter.py` into your MoleditPy plugin directory:
   * **Windows**: `C:\Users\<YourName>\.moleditpy\plugins\`
   * **Linux / macOS**: `~/.moleditpy/plugins/`
2. Start MoleditPy (or click **Plugins > Reload All Plugins**).
3. Access the tool via **View > Camera Position Setter...**.

## Usage

1. Open a 3D molecule in MoleditPy.
2. Open **View > Camera Position Setter...**.
3. Use the **Standard Views** buttons for quick alignment (e.g. `+Z` for top-down view, `+X` for right-side view).
4. Enter custom coordinates in the **Numerical Coordinates** section or click **Read from View** to capture current camera values.
5. In the **Bookmarks & Presets** tab, click **Save Current View...** to store viewpoints for presentations or publication figure preparation.

## Development and Testing

This repository includes a comprehensive test suite covering headless Qt UI logic, 3D math and rotations, clipboard serialization, project persistence handlers, and static API validation:

```bash
# Run tests
python -m pytest tests/ -v

# Run with test coverage
python -m pytest tests/ --cov=camera_position_setter --cov-report=term-missing
```

## License

This project is licensed under the [GNU Lesser General Public License v3.0](LICENSE).

## Licensing scope

First-party code in this migration is LGPL-3.0-or-later. See [LICENSE](LICENSE), [COPYING](COPYING) and [dependency/provenance terms](LICENSING.md). Third-party components and historical releases retain their own terms.
