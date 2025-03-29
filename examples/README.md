# PyOpenSCAD Examples

This directory contains example scripts demonstrating how to use various features of the PyOpenSCAD library.

## Viewer Examples

### Basic Viewer Example

`viewer_example.py` demonstrates how to create and display 3D models using the PyOpenSCAD viewer. It creates several basic shapes and renders them in an interactive 3D viewer.

To run:
```
python viewer_example.py
```

### M3D Renderer Integration Example

`m3d_viewer_example.py` shows how to integrate the viewer with the M3dRenderer class to visualize Manifold3D objects. It demonstrates creating complex shapes using CSG operations and then viewing them in 3D.

To run:
```
python m3d_viewer_example.py
```

## Requirements

The viewer examples require additional dependencies:
- PyOpenGL
- PyOpenGL-accelerate
- PyGLM
- manifold3d (for M3D examples)

Install these dependencies with:
```
pip install PyOpenGL PyOpenGL-accelerate PyGLM manifold3d
```

## Controls

The 3D viewer supports the following controls:
- Left mouse button drag: Rotate camera
- Mouse wheel: Zoom in/out
- R key: Reset view
- ESC key: Close viewer 