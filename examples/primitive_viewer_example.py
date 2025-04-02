import numpy as np
from pythonopenscad import M3dRenderer
import pythonopenscad as posc
import sys

from pythonopenscad.modifier import DISABLE, SHOW_ONLY

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

    sphere6 = posc.Translate([6, 0, 0])(posc.Color("sienna")(posc.Sphere(r=6)))
    sphere3 = posc.Translate([-2, 0, 0])(posc.Color("orchid")(posc.Sphere(r=3)))
    hull3d = posc.Translate([0, -14, 0])(
        posc.Color("sienna")(posc.Union()(posc.Hull()(sphere6, sphere3)))
    )
    hull3d_manifold = hull3d.renderObj(renderer).get_solid_manifold()

    circle6 = posc.Translate([6, 0, 0])(posc.Circle(r=6))
    circle3 = posc.Translate([-2, 0, 0])(posc.Circle(r=3))
    hull2d = posc.Hull()(circle6, circle3)
    hull_extrusion = posc.Translate([0, 10, 0])(posc.Linear_Extrude(height=3.0)(hull2d))
    hull_extrusion_manifold = hull_extrusion.renderObj(renderer).get_solid_manifold()

    text = posc.Text(
        "Hello, World!",
        size=3,
        font="Arial",
        halign="center",
        valign="center",
        spacing=1.0,
        direction="ltr",
        language="en",
        script="latin",
    )

    text_extrusion = posc.Color("darkseagreen")(
        posc.Translate([-8.0, 0.0, 4.5])(
            posc.Rotate([0, 0, 90])(posc.Linear_Extrude(height=3.0)(text))
        )
    )
    text_extrusion_manifold = text_extrusion.renderObj(renderer).get_solid_manifold()

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
    difference = posc.Rotate([0, 0, 45])(posc.Rotate([45, 45, 0])(difference))
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

    projection = posc.Projection()(difference)
    projection_extrusion = posc.Color("darkseagreen")(
        posc.Translate([16.0, 0.0, 4.5])(posc.Linear_Extrude(height=1.0)(projection))
    )
    projection_extrusion_manifold = projection_extrusion.renderObj(renderer).get_solid_manifold()

    xmin, ymin, zmin, xmax, ymax, zmax = difference_manifold.bounding_box()
    cut = posc.Projection(cut=True)(posc.Translate([0, 0, -(zmin + zmax) / 2])(difference))
    cut_extrusion = posc.Color("darkseagreen")(
        posc.Translate([16.0, 5.0, 4.5])(posc.Linear_Extrude(height=1.0)(cut))
    )
    cut_extrusion_manifold = cut_extrusion.renderObj(renderer).get_solid_manifold()

    offset = posc.Offset(r=0.5)(cut)
    offset_extrusion = posc.Color("brown")(
        posc.Translate([16.0, 10.0, 4.5])(posc.Linear_Extrude(height=1.0)(offset))
    )
    offset_extrusion_manifold = offset_extrusion.renderObj(renderer).get_solid_manifold()

    filled_text = posc.Fill()(
        posc.Text(
            "A",
            size=3,
            font="Arial",
            halign="center",
            valign="center",
        )
    )
    filled_text_extrusion = posc.Color("royalblue")(
        posc.Translate([16.0, 15.0, 4.5])(posc.Linear_Extrude(height=1.0)(filled_text))
    )
    filled_text_extrusion_manifold = filled_text_extrusion.renderObj(renderer).get_solid_manifold()

    union_of_solids = posc.Translate([0, -40, 0])(
        hull_extrusion,
        hull3d.add_modifier(DISABLE),
        text_extrusion.transparent(),
        sphere,
        cube,
        cylinder,
        difference,
        intersection,
        union,
        linear_extrusion,
        rotate_extrusion,
        polygon_extrusion,
        projection_extrusion,
        cut_extrusion,
        offset_extrusion,
        filled_text_extrusion,
    )

    union_of_solids_manifold = union_of_solids.renderObj(renderer).get_solid_manifold()
    # Convert to viewer models
    models = [
        Model.from_manifold(hull_extrusion_manifold),
        Model.from_manifold(hull3d_manifold),
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
        Model.from_manifold(projection_extrusion_manifold),
        Model.from_manifold(cut_extrusion_manifold),
        Model.from_manifold(offset_extrusion_manifold),
        Model.from_manifold(filled_text_extrusion_manifold),
        Model.from_manifold(union_of_solids_manifold),
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
