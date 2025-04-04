import numpy as np
from pythonopenscad import M3dRenderer
import pythonopenscad as posc
import sys
import manifold3d as m3d
from dataclasses import dataclass, field

from pythonopenscad.modifier import DEBUG, DISABLE, SHOW_ONLY, PoscRendererBase

try:
    from pythonopenscad.m3dapi import M3dRenderer
    from pythonopenscad.viewer.viewer import Model, Viewer
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Make sure manifold3d, PyOpenGL, and PyGLM are installed.")
    print("Try: pip install manifold3d PyOpenGL PyOpenGL-accelerate PyGLM")
    sys.exit(1)


@dataclass
class PrimitiveCreatorBase:
    contexts: list[posc.RenderContext] = field(default_factory=list)
    
    def render(self, obj: PoscRendererBase):
        context = obj.renderObj(M3dRenderer())
        self.contexts.append(context)
        return context
    
    def get_solid_manifolds(self) -> list[m3d.Manifold]:
        return [context.get_solid_manifold() for context in self.contexts]
    
    def get_shell_manifolds(self) -> list[m3d.Manifold]:
        return [context.get_shell_manifold() for context in self.contexts]
    
    def get_solid_model(self) -> list[Model]:
        return [Model.from_manifold(mfd) for mfd in self.get_solid_manifolds()]
    
    def get_shell_model(self) -> list[Model]:
        return [Model.from_manifold(mfd, has_alpha_lt1=True) for mfd in self.get_shell_manifolds()]
    

@dataclass
class PrimitiveCreator(PrimitiveCreatorBase):

    def create_primitive_models(self):
        """Create example 3D models using various OpenSCAD primitives and operations."""
        
        spherefn = 32

        sphere6 = posc.Translate([6, 0, 0])(posc.Color("sienna")(posc.Sphere(r=6, _fn=spherefn )))
        sphere3 = posc.Translate([-2, 0, 0])(posc.Color("orchid")(posc.Sphere(r=3, _fn=spherefn)))
        hull3d = posc.Translate([0, -14, 0])(
            posc.Color("sienna")(posc.Union()(posc.Hull()(sphere6, sphere3)))
        )
        self.render(hull3d)

        circle6 = posc.Translate([6, 0, 0])(posc.Circle(r=6))
        circle3 = posc.Translate([-2, 0, 0])(posc.Circle(r=3))
        hull2d = posc.Hull()(circle6, circle3)
        hull_extrusion = posc.Translate([0, 10, 0])(posc.Linear_Extrude(height=3.0)(hull2d))
        self.render(hull_extrusion)

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
        self.render(text_extrusion)

        # Create a sphere
        sphere = posc.Translate([-2.0, -2.0, 0.0])(
            posc.Color("darkolivegreen")(posc.Sphere(r=2.8, _fn=spherefn))
        )
        self.render(sphere)

        # Create a sphere
        cube = posc.Translate([2.0, -2.0, 0.0])(posc.Color("orchid")(posc.Cube([1.0, 1.0, 2.0])))
        self.render(cube)

        # Create a cylinder
        cylinder = posc.translate([-6.0, -2.0, 0.0])(
            posc.Color("peachpuff")(posc.Cylinder(h=1.5, r1=0.5, r2=1.5, _fn=32, center=False))
        )
        self.render(cylinder)

        sphere2 = posc.Color("darkolivegreen")(posc.Sphere(r=1, _fn=spherefn))
        cube2 = posc.Color("orchid")(posc.Cube(1.5, center=True))
        difference = posc.Difference()(cube2, sphere2).translate([6.0, 0.0, 0.0])
        difference = posc.Rotate([0, 0, 45])(posc.Rotate([45, 45, 0])(difference))
        ctxt = self.render(difference)
        difference_manifold = ctxt.get_solid_manifold()

        intersection = posc.Intersection()(cube2, sphere2).translate([6.0, -2.0, 0.0])
        self.render(intersection)

        union = posc.Union()(cube2, sphere2).translate([6.0, -4.0, 0.0])
        self.render(union)

        linear_extrusion = posc.Color("darkseagreen")(
            posc.Translate([0.0, 0.0, 4.5])(
                posc.Linear_Extrude(height=3.0, twist=45, slices=16, scale=(2.5, 0.5))(
                    posc.Translate([0.0, 0.0, 0.0])(posc.Square([1.0, 1.0]))
                )
            )
        )
        self.render(linear_extrusion)

        rotate_extrusion = posc.Color("darkseagreen")(
            posc.Translate([-3.0, 0.0, 4.5])(
                posc.Rotate_Extrude(angle=360, _fn=32)(
                    posc.Translate([1.0, 0.0, 0.0])(posc.Circle(r=0.5, _fn=16))
                )
            )
        )
        self.render(rotate_extrusion)

        # Create a polygon as a triangle with an inner triangle hole.
        polygon = posc.Polygon(
            [[2, 0], [0, 2], [-2, 0], [1, 0.5], [0, 1], [-1, 0.5]],
            paths=[[0, 1, 2], [3, 5, 4]],
            convexity=2,
        )
        polygon_extrusion = posc.Color("darkseagreen")(
            posc.Translate([-6.0, 0.0, 4.5])(posc.Linear_Extrude(height=3.0)(polygon))
        )
        self.render(polygon_extrusion)

        projection = posc.Projection()(difference)
        projection_extrusion = posc.Color("darkseagreen")(
            posc.Translate([16.0, 0.0, 4.5])(posc.Linear_Extrude(height=1.0)(projection))
        )
        self.render(projection_extrusion)

        xmin, ymin, zmin, xmax, ymax, zmax = difference_manifold.bounding_box()
        cut = posc.Projection(cut=True)(posc.Translate([0, 0, -(zmin + zmax) / 2])(difference))
        cut_extrusion = posc.Color("darkseagreen")(
            posc.Translate([16.0, 5.0, 4.5])(posc.Linear_Extrude(height=1.0)(cut))
        )
        self.render(cut_extrusion)

        offset = posc.Offset(r=0.5)(cut)
        offset_extrusion = posc.Color("brown")(
            posc.Translate([16.0, 10.0, 4.5])(posc.Linear_Extrude(height=1.0)(offset))
        )
        self.render(offset_extrusion)

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
        self.render(filled_text_extrusion)

        union_of_solids = posc.Translate([0, -40, 0])(
            hull_extrusion,
            hull3d.add_modifier(DEBUG),
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

        self.render(union_of_solids)

        return self.get_solid_model() + self.get_shell_model()


def main():    
    #Viewer._init_glut()

    # Create the models
    models = PrimitiveCreator().create_primitive_models()

    # Create and run the viewer
    viewer = Viewer(models)
    
    print(Viewer.VIEWER_HELP_TEXT)
    print(f"Total triangles={viewer.num_triangles()}")
    viewer.run()


if __name__ == "__main__":
    main()
