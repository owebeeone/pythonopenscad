import numpy as np
from pythonopenscad import M3dRenderer
import pythonopenscad as posc
import sys

try:
    from pythonopenscad.m3dapi import M3dRenderer
    from pythonopenscad.viewer import Model, Viewer, is_opengl_available
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Make sure manifold3d, PyOpenGL, and PyGLM are installed.")
    print("Try: pip install manifold3d PyOpenGL PyOpenGL-accelerate PyGLM")
    sys.exit(1)


def create_primitive_models():
    """Create example 3D models using various OpenSCAD primitives and operations."""
    renderer = M3dRenderer()
    
    text = posc.Text(
        "Hello, World!",
        size=3,
        font="Arial",
        halign="center",
        valign="center",
        spacing=1.0,
        direction="ltr",
        language="en",
        script="latin"
    )

    text_extrusion = posc.Color("darkseagreen")(
        posc.Translate([-8.0, 0.0, 4.5])(posc.Rotate([0, 0, 90])(posc.Linear_Extrude(height=3.0)(text)))
    )
    text_extrusion_manifold = text_extrusion.renderObj(renderer).get_solid_manifold()
    
    print(f"text_extrusion_manifold.num_prop(): {text_extrusion_manifold.num_prop() =}")

    # Create a sphere
    sphere = posc.Translate([-2.0, -2.0, 0.0])(
        posc.Color("darkolivegreen")(posc.Sphere(r=2.8, _fn=32))
    )
    sphere_manifold = sphere.renderObj(renderer).get_solid_manifold()

    # Create a sphere
    cube = posc.Translate([2.0, -2.0, 0.0])(posc.Color("orchid")(posc.Cube([1.0, 1.0, 2.0])))
    cube_manifold = cube.renderObj(renderer).get_solid_manifold()

    # Create a cylinder
    cylinder = posc.translate([-6.0, -2.0, 0.0])(
        posc.Color("peachpuff")(posc.Cylinder(h=1.5, r1=0.5, r2=1.5, _fn=32, center=False))
    )
    cylinder_manifold = cylinder.renderObj(renderer).get_solid_manifold()

    sphere2 = posc.Color("darkolivegreen")(posc.Sphere(r=1, _fn=32))
    cube2 = posc.Color("orchid")(posc.Cube(1.5, center=True))
    difference = posc.Difference()(cube2, sphere2).translate([6.0, 0.0, 0.0])
    difference_manifold = difference.renderObj(renderer).get_solid_manifold()

    intersection = posc.Intersection()(cube2, sphere2).translate([6.0, -2.0, 0.0])
    intersection_manifold = intersection.renderObj(renderer).get_solid_manifold()

    union = posc.Union()(cube2, sphere2).translate([6.0, -4.0, 0.0])
    union_manifold = union.renderObj(renderer).get_solid_manifold()

    linear_extrusion = posc.Color("darkseagreen")(
        posc.Translate([0.0, 0.0, 4.5])(
            posc.Linear_Extrude(height=3.0, twist=45, slices=16, scale=(2.5, 0.5))(
                posc.Translate([0.0, 0.0, 0.0])(posc.Square([1.0, 1.0]))
            )
        )
    )
    linear_extrusion_manifold = linear_extrusion.renderObj(renderer).get_solid_manifold()

    rotate_extrusion = posc.Color("darkseagreen")(
        posc.Translate([-3.0, 0.0, 4.5])(
            posc.Rotate_Extrude(angle=360, _fn=32)(
                posc.Translate([1.0, 0.0, 0.0])(posc.Circle(r=0.5, _fn=16))
            )
        )
    )
    rotate_extrusion_manifold = rotate_extrusion.renderObj(renderer).get_solid_manifold()

    # Create a polygon as a triangle with an inner triangle hole.
    polygon = posc.Polygon(
        [[2, 0], [0, 2], [-2, 0], [1, 0.5], [0, 1], [-1, 0.5]],
        paths=[[0, 1, 2], [3, 5, 4]],
        convexity=2,
    )
    polygon_extrusion = posc.Color("darkseagreen")(
        posc.Translate([-6.0, 0.0, 4.5])(posc.Linear_Extrude(height=3.0)(polygon))
    )
    polygon_extrusion_manifold = polygon_extrusion.renderObj(renderer).get_solid_manifold()


    # Convert to viewer models
    models = [
        Model.from_manifold(text_extrusion_manifold),
        Model.from_manifold(sphere_manifold),
        Model.from_manifold(cube_manifold),
        Model.from_manifold(cylinder_manifold),
        Model.from_manifold(difference_manifold),
        Model.from_manifold(intersection_manifold),
        Model.from_manifold(union_manifold),
        Model.from_manifold(linear_extrusion_manifold),
        Model.from_manifold(rotate_extrusion_manifold),
        Model.from_manifold(polygon_extrusion_manifold),
    ]

    return models


def main():
    # Create the models
    models = create_primitive_models()

    # Create and run the viewer
    viewer = Viewer(models)
    viewer.run()


if __name__ == "__main__":
    main()
