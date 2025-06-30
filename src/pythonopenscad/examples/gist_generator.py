"""
gist_generator.py

A utility to programmatically generate PythonOpenScad example "gists"
from a set of specifications. This allows for the rapid creation of
consistent, high-quality example files that can be run as Python modules.
"""
import os
from dataclasses import dataclass
from pathlib import Path
import pythonopenscad as posc
from pythonopenscad.posc_main import posc_main
from pythonopenscad.viewer.viewer import Viewer

# --- Templates for the generated gist files ---

GIST_TEMPLATE_EXPLANATION = '''\
This script demonstrates the creation of a `{class_name}` model to
generate the following OpenSCAD code:\
'''

# This template is based on the clean, minimal example structure.
# It will be populated with data from each GistSpec object.
GIST_TEMPLATE = '''
"""
{filename}: A PythonOpenScad example gist.

Note: This file was automatically generated.

{explain_text}

{code_expression}

{class_pydoc}
{init_pydoc}
---
How to run this example:

- To view in the interactive viewer:
  python -m {module_path} --view

- To generate a .scad file (the default action for this script):
  python -m {module_path} --no-view --scad

- To generate a .stl file:
  python -m {module_path} --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import {posc_imports}
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
{code}

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)
'''


class PoscCaptureDict(dict):
    """
    A special dictionary that captures which pythonopenscad classes are
    accessed. When a key is accessed that doesn't exist in the dict,
    it loads it from the `posc` module, stores it, and records the name.
    """
    def __init__(self, module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._module = module
        self.captured_names = set()

    def __getitem__(self, key):
        # This allows `exec` to find names like 'Circle' or 'translate'.
        if key not in self:
            if hasattr(self._module, key):
                value = getattr(self._module, key)
                self[key] = value
                # If it's a PoscBase class, add its name to our import list.
                if isinstance(value, type) and issubclass(value, posc.PoscBase):
                     self.captured_names.add(key)
                # Also capture modifier functions like translate, rotate, etc.
                elif callable(value) and not isinstance(value, type):
                     self.captured_names.add(key)
                return value
            raise KeyError(f"'{key}' not found in the pythonopenscad module.")
        return super().__getitem__(key)


@dataclass
class GistSpec:
    """A dataclass that holds the specification for a single gist file."""
    file_location: str
    file_name: str
    code: str
    posc_class: type | None = None
    explain_text: str | None = GIST_TEMPLATE_EXPLANATION
    
    @property
    def file_path(self):
        return os.path.join(self.file_location, self.file_name)
    
    def make_image(self, posc_obj: posc.PoscBase):
        """
        Makes an image of the posc_obj.
        """
        output_base = Path(os.path.splitext(self.file_path)[0]).as_posix()
        posc_main([posc_obj], 
                  default_view=False, 
                  default_scad=False, 
                  default_stl=False, 
                  default_png=True,
                  output_base=output_base)

    def create(self):
        """
        Generates and writes the gist file based on the spec.
        """
        # Create the module path from the file path
        # e.g., "pythonopenscad/examples/gists/file.py" -> "pythonopenscad.examples.gists.file"
        module_path = os.path.splitext(self.file_path)[0]
        module_path = Path(module_path).as_posix().replace('/', '.')

        # Use the capture dict to safely execute the code string
        capture_dict = PoscCaptureDict(module=posc)
        
        # Execute the code. It will define 'MODEL' within the capture_dict's scope.
        # The globals dict is empty for security.
        exec(self.code, {}, capture_dict)
        
        # Retrieve the created MODEL object from the dict
        MODEL = capture_dict['MODEL']

        # If a primary class wasn't specified, infer it from the result.
        if self.posc_class is None:
            self.posc_class = type(MODEL)

        # Generate the content for the template
        posc_imports_str = ", ".join(sorted(list(capture_dict.captured_names)))
        
        # Ensure the output directory exists
        directory = os.path.dirname(self.file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        filename = os.path.basename(self.file_path)

        # For display in the docstring, get the expression part of the code
        code_expression = str(MODEL)
        
        explain_text = self.explain_text.format(
            class_name=self.posc_class.__name__
        )

        # Populate the template
        content = GIST_TEMPLATE.format(
            filename=filename,
            explain_text=explain_text,
            class_name=self.posc_class.__name__,
            code_expression=code_expression,
            module_path=module_path,
            posc_imports=posc_imports_str,
            code=self.code.strip(),
            class_pydoc=self.posc_class.__doc__ or "",
            init_pydoc=self.posc_class.__init__.__doc__ or ""
        )

        # Write the generated content to the file
        with open(self.file_path, "w") as f:
            f.write(content.strip())
        
        try:
            self.make_image(MODEL)
        except Exception as e:
            print(f" -> Error making image for {self.file_name}: {e}")

@dataclass
class GistFolderSpec:
    """A dataclass that holds the specification for a single gist folder."""
    folder_location: str
    description: str

# --- Gist Directory Paths ---
# These globals define the output directories for each category of gists.
GISTS_2D_PATH = "pythonopenscad/examples/gists_2d"
GISTS_3D_PATH = "pythonopenscad/examples/gists_3d"
GISTS_TRANSFORMS_PATH = "pythonopenscad/examples/gists_transforms"
GISTS_CSG_PATH = "pythonopenscad/examples/gists_csg"
GISTS_OTHER_PATH = "pythonopenscad/examples/gists_other"

MD_FILE_PER_EXAMPLE_TEMPLATE = """\
     
"""

ALL_GIST_FOLDERS = [
    GistFolderSpec(
        folder_location=GISTS_2D_PATH,
        description="""\
2D Shapes

These gists demonstrate the creation of 2D shapes using the pythonopenscad library.
These shapes are 2D and do not have a depth but when rendered they are extruded to a depth of 1.


"""
    ),
    GistFolderSpec(
        folder_location=GISTS_3D_PATH,
        description="3D Shapes"
    ),
    GistFolderSpec(
        folder_location=GISTS_TRANSFORMS_PATH,
        description="Transformations"
    ),
    GistFolderSpec(
        folder_location=GISTS_CSG_PATH,
        description="CSG Modelling"
    ),
    GistFolderSpec(
        folder_location=GISTS_OTHER_PATH,
        description="Other Features"
    ),
]

ALL_GISTS = [
    # --- 2D Shapes ---
    GistSpec(
        file_location=GISTS_2D_PATH,
        file_name="circle_example.py",
        code="MODEL = Circle(d=10, _fn=64)"
    ),
    GistSpec(
        file_location=GISTS_2D_PATH,
        file_name="square_example.py",
        code="MODEL = Square(size=[15, 10], center=True)"
    ),
    GistSpec(
        file_location=GISTS_2D_PATH,
        file_name="polygon_example.py",
        code="MODEL = Polygon(points=[[0,0], [0,10], [10,10]])"
    ),
    GistSpec(
        file_location=GISTS_2D_PATH,
        file_name="text_example.py",
        code='MODEL = Text(text="POSC", size=10, font="Liberation Sans:style=Bold", halign="center")'
    ),

    # --- 3D Primitives ---
    GistSpec(
        file_location=GISTS_3D_PATH,
        file_name="cube_example.py",
        code="MODEL = Cube(size=[10, 15, 5], center=True)"
    ),
    GistSpec(
        file_location=GISTS_3D_PATH,
        file_name="sphere_example.py",
        code="MODEL = Sphere(r=10, _fn=128)"
    ),
    GistSpec(
        file_location=GISTS_3D_PATH,
        file_name="cylinder_example.py",
        code="MODEL = Cylinder(h=20, r1=5, r2=10, center=True, _fn=128)"
    ),
    GistSpec(
        file_location=GISTS_3D_PATH,
        file_name="polyhedron_example.py",
        code="""MODEL = Polyhedron(
    points=[[10,10,0],[10,-10,0],[-10,-10,0],[-10,10,0], [0,0,10]],
    faces=[[0,1,4],[1,2,4],[2,3,4],[3,0,4], [3,2,1,0]]
)"""
    ),

    # --- Transformations (CamelCase Style) ---
    GistSpec(
        file_location=GISTS_TRANSFORMS_PATH,
        file_name="translate_example.py",
        code="MODEL = Translate([10, -10, 5])(Cube(size=5, center=True))"
    ),
    GistSpec(
        file_location=GISTS_TRANSFORMS_PATH,
        file_name="rotate_example.py",
        code="MODEL = Rotate([45, 45, 0])(Cube(size=[10, 15, 5]))"
    ),
    GistSpec(
        file_location=GISTS_TRANSFORMS_PATH,
        file_name="scale_example.py",
        code="MODEL = Scale([1.5, 1, 0.5])(Sphere(r=10))"
    ),
    GistSpec(
        file_location=GISTS_TRANSFORMS_PATH,
        file_name="resize_example.py",
        code="MODEL = Resize(newsize=[30, 10, 5])(Sphere(r=5))"
    ),
    GistSpec(
        file_location=GISTS_TRANSFORMS_PATH,
        file_name="mirror_example.py",
        code="""MODEL = Mirror([1, 1, 0])(Translate([10, 0, 0])(Color('green')(Cube(size=5)))) \\
    + Translate([10, 0, 0])(Color('red')(Cube(size=5))).setMetadataName("not mirrored")
""",
        posc_class=posc.Mirror
    ),
    GistSpec(
        file_location=GISTS_TRANSFORMS_PATH,
        file_name="color_example.py",
        code='MODEL = Color("green")(Sphere(r=10))'
    ),
    GistSpec(
        file_location=GISTS_TRANSFORMS_PATH,
        file_name="multmatrix_example.py",
        code="""MODEL = Multmatrix(m=[
    [1, 0.5, 0, 5],
    [0, 1, 0.5, 10],
    [0.5, 0, 1, 0],
    [0, 0, 0, 1]
])(Cube(size=10))"""
    ),
    GistSpec(
        file_location=GISTS_TRANSFORMS_PATH,
        file_name="projection_example.py",
        code="MODEL = Projection(cut=True)(Cube(7) + Translate([0,0,2.5])(Sphere(r=5)))"
    ),
    GistSpec(
        file_location=GISTS_TRANSFORMS_PATH,
        file_name="linear_extrude_example.py",
        code="MODEL = Linear_Extrude(height=5, center=True, scale=0.5, twist=90)(Square(10))"
    ),
    GistSpec(
        file_location=GISTS_TRANSFORMS_PATH,
        file_name="rotate_extrude_example.py",
        code="MODEL = Rotate_Extrude(angle=270, _fn=128)(Translate([5,0,0])(Circle(r=2)))"
    ),
    GistSpec(
        file_location=GISTS_TRANSFORMS_PATH,
        file_name="offset_example.py",
        code="MODEL = Offset(delta=2)(Square(10)) - Square(10)",
        posc_class=posc.Offset
    ),

    # --- CSG Modelling (CamelCase Style) ---
    GistSpec(
        file_location=GISTS_CSG_PATH,
        file_name="union_functional.py",
        code="""MODEL = Union()(
    Color("red")(Cube(10)),
    Color("blue")(Translate([5,5,5])(Sphere(r=7)))
)"""
    ),
    GistSpec(
        file_location=GISTS_CSG_PATH,
        file_name="union_operator.py",
        code="MODEL = Color('red')(Cube(10)) + Color('blue')(Translate([5,5,5])(Sphere(r=7)))"
    ),
    GistSpec(
        file_location=GISTS_CSG_PATH,
        file_name="difference_functional.py",
        code="""MODEL = Difference()(
    Color("red")(Cube(10, center=True)),
    Color("blue")(Sphere(r=7))
)"""
    ),
    GistSpec(
        file_location=GISTS_CSG_PATH,
        file_name="difference_operator.py",
        code="MODEL = Color('red')(Cube(10, center=True)) - Color('blue')(Sphere(r=7))"
    ),
    GistSpec(
        file_location=GISTS_CSG_PATH,
        file_name="intersection_functional.py",
        code="""MODEL = Intersection()(
    Color("red")(Cube(10, center=True)),
    Color("blue")(Sphere(r=7))
)"""
    ),
    GistSpec(
        file_location=GISTS_CSG_PATH,
        file_name="intersection_operator.py",
        code="MODEL = Color('red')(Cube(10, center=True)) * Color('blue')(Sphere(r=7))"
    ),
    GistSpec(
        file_location=GISTS_CSG_PATH,
        file_name="hull_example.py",
        code="""MODEL = Hull()(
    Color("red")(Translate([-10,0,0])(Sphere(r=5))),
    Color("blue")(Translate([10,0,0])(Cube([1,1,15], center=True)))
)"""
    ),
    GistSpec(
        file_location=GISTS_CSG_PATH,
        file_name="minkowski_example.py",
        code="MODEL = Minkowski()(Square([10, 2]), Circle(r=2))"
    ),

    # --- Other Features ---
    GistSpec(
        file_location=GISTS_OTHER_PATH,
        file_name="render_example.py",
        code="MODEL = Render(convexity=10)(Difference()(Cube(10), Sphere(r=6)))"
    ),
    GistSpec(
        file_location=GISTS_OTHER_PATH,
        file_name="snake_case_style_example.py",
        code="""MODEL = difference()(
    color("red")(cube(10, center=True)),
    color("blue")(sphere(r=7))
)""",
        explain_text="This shows how to use the snake_case style for pythonopenscad classes."
    ),
    # NOTE: Import and Surface require external files and are harder to make
    # self-contained gists for. They are omitted here but could be added if
    # sample files (e.g., 'sample.stl', 'heightmap.png') were also part of the
    # examples directory.
]

def main():
    print("Starting gist generation process...")

    # Define all the gists we want to create.
    # The generator will create the 'generated_gists' directory.
    all_gists = ALL_GISTS

    # Loop through the specs and create each file.
    for spec in all_gists:
        spec.create()

    print("\nGist generation complete.")


# --- Main execution block ---
if __name__ == "__main__":
    main()
