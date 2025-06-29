import argparse
import sys
import os
import inspect
from typing import List, Union, Callable, Optional, Tuple
import manifold3d as m3d

# Assume these imports are valid within the project structure
from datatrees import datatree, Node, dtfield
import pythonopenscad as posc
from pythonopenscad.m3dapi import M3dRenderer, RenderContextCrossSection, RenderContextManifold, manifold_to_stl
from pythonopenscad.viewer.viewer import Viewer, Model
from pythonopenscad.modifier import PoscRendererBase



@datatree
class PoscModel:
    """Wraps a PoscBase object, providing lazy rendering and access to results."""
    item: posc.PoscBase | Callable[[], posc.PoscBase]
    name: str
    _render_context: RenderContextManifold | None = dtfield(default=None, init=False)
    _posc_obj: posc.PoscBase | None = dtfield(default=None, init=False)
    _solid_manifold: m3d.Manifold | None = dtfield(default=None, init=False)
    _shell_manifold: m3d.Manifold | None = dtfield(default=None, init=False)
    
    def __post_init__(self):
        if isinstance(self.item, PoscRendererBase):
            self._posc_obj = self.item

    def get_posc_obj(self) -> posc.PoscBase:
        """Retrieves or computes the PoscBase object."""
        if self._posc_obj is None:
            self._posc_obj = self.item()
        return self._posc_obj

    def render(self) -> RenderContextManifold:
        """Renders the PoscBase object using M3dRenderer if not already cached."""
        if self._render_context is None:
            posc_obj = self.get_posc_obj()
            try:
                renderer = M3dRenderer()
                self._render_context = posc_obj.renderObj(renderer)
                if isinstance(self._render_context, RenderContextCrossSection):
                    # We can only handle manifolds, Linear extrude the cross section 1 unit.
                    self._render_context = renderer._linear_extrude([self._render_context], 1)
            except Exception as e:
                print(f"Error rendering object '{self.name}': {e}", file=sys.stderr)
                raise
        return self._render_context

    def get_solid_manifold(self) -> m3d.Manifold:
        """Returns the solid manifold from the render context."""
        context = self.render()
        self._solid_manifold = context.get_solid_manifold()
        return self._solid_manifold

    def get_shell_manifold(self) -> m3d.Manifold | None:
        """Returns the shell manifold from the render context."""
        context = self.render()
        self._shell_manifold = context.get_shell_manifold()
        return self._shell_manifold

    def get_viewer_models(self, include_shells: bool = False) -> list[Model]:
        """Returns a viewer Model for the solid manifold."""
        manifold = self.get_solid_manifold()
        if not include_shells:
            return [Model.from_manifold(manifold)]
        else:
            shell_manifold = self.get_shell_manifold()
            return [Model.from_manifold(manifold), 
                    Model.from_manifold(shell_manifold, has_alpha_lt1=True)]


def parse_color(color_str: str) -> Optional[Tuple[float, float, float, float]]:
    """Parses a color string (e.g., "1.0,0.5,0.0") into RGBA tuple."""
    try:
        parts = [float(p.strip()) for p in color_str.split(',')]
        if len(parts) == 3:
            return (*parts, 1.0) # Add alpha if only RGB provided
        elif len(parts) == 4:
            return parts
        else:
            raise ValueError("Color must have 3 (RGB) or 4 (RGBA) components.")
    except Exception as e:
        print(f"Error parsing color '{color_str}': {e}", file=sys.stderr)
        return None
    
def add_bool_arg(parser, name, help_text, default=False):
        parser.add_argument(
            f"--{name}",
            action="store_true",
            help=help_text
        )
        
        parser.add_argument(
            f"--no-{name}",
            action="store_false",
            dest=name,
            help=f"Disable: {help_text}"
        )
        parser.set_defaults(**{name: default})

@datatree
class PoscMainRunner:
    """Parses arguments and runs actions for viewing/exporting Posc models."""
    items: list[Callable[[], posc.PoscBase] | posc.PoscBase]
    script_path: str
    _args: argparse.Namespace | None = dtfield(default=None, init=False)
    posc_models: List[PoscModel] = dtfield(default_factory=list, init=False)
    parser: argparse.ArgumentParser | None = dtfield(
        self_default=lambda s: s._make_parser(), init=False)
    output_base: str | None = dtfield(default=None, init=False)
    default_scad: bool = False
    default_png: bool = False
    default_view: bool = True
    default_stl: bool = False
    default_wireframe: bool = False
    default_backface_culling: bool = True
    default_bounding_box_mode: bool = False
    default_zbuffer_occlusion: bool = True
    default_coalesce: bool = True
    default_width: int = 800
    default_height: int = 600
    default_title: str | None = None
    default_projection: str = 'perspective'
    default_bg_color: str = "0.98,0.98,0.85,1.0"
    default_output_base: str | None = None
    
    @property
    def args(self) -> argparse.Namespace:
        if self._args is None:
            self.parse_args()
        return self._args

    def _make_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="View or export PythonOpenSCAD objects.")

        # --- Input/Output Control ---
        parser.add_argument(
            "--output-base",
            type=str,
            default=self.default_output_base,
            help="Base name for output files (STL, PNG). Defaults to the name of the calling script."
        )
        add_bool_arg(parser, "solids", "Only process solid geometry (ignore shells/debug geometry).", default=False)

        # --- Actions ---
        add_bool_arg(parser, "view", "View the models in the OpenGL viewer.", default=self.default_view)
        add_bool_arg(parser, "stl", "Export solid models to STL files.", default=self.default_stl)
        add_bool_arg(parser, "png", "Save a PNG image using the viewer settings (requires viewer modification for non-interactive use).", default=self.default_png)
        add_bool_arg(parser, "scad", "Export to SCAD file (Not implemented).", default=self.default_scad)
        add_bool_arg(parser, "wireframe", "Start viewer in wireframe mode.", default=self.default_wireframe)
        add_bool_arg(parser, "backface-culling", "Disable backface culling in the viewer.", default=self.default_backface_culling)
        add_bool_arg(parser, "bounding-box-mode", "Initial bounding box mode (0=off, 1=wireframe, 2=solid).", default=self.default_bounding_box_mode)
        add_bool_arg(parser, "zbuffer-occlusion", "Disable Z-buffer occlusion for wireframes.", default=self.default_zbuffer_occlusion)
        add_bool_arg(parser, "coalesce", "Model coalescing (may impact transparency rendering).", default=self.default_coalesce)

        # --- Viewer Options ---
        parser.add_argument("--width", type=int, default=self.default_width, help="Viewer window width.")
        parser.add_argument("--height", type=int, default=self.default_height, help="Viewer window height.")
        parser.add_argument("--title", type=str, default=self.default_title, help="Viewer window title.")
        parser.add_argument(
            "--projection",
            type=str,
            choices=['perspective', 'orthographic'],
            default=self.default_projection,
            help="Initial viewer projection mode."
        )
        parser.add_argument(
            "--bg-color",
            type=str,
            default=self.default_bg_color,
            help="Viewer background color as comma-separated RGBA floats (e.g., '0.1,0.1,0.1,1.0')."
        )
        
        return parser

    def parse_args(self):

        self._args = self.parser.parse_args()

    def check_args(self):
        # Determine default output base name
        if self.args.output_base is None:
            self.args.output_base = os.path.splitext(os.path.basename(self.script_path))[0]

        # Set default title if not provided
        if self.args.title is None:
            self.args.title = f"Posc View: {os.path.basename(self.script_path)}"

        # Parse background color
        self.args.parsed_bg_color = parse_color(self.args.bg_color)
        if self.args.parsed_bg_color is None:
             # Fallback to default if parsing fails
             self.args.parsed_bg_color = parse_color("0.98,0.98,0.85,1.0")


    def _prepare_models(self):
        """Instantiate PoscModel objects from the input items."""
        self.posc_models = []
        for i, item in enumerate(self.items):
             # Try to get a name from the item if possible (e.g., function name)
             name = f"item_{i}"

             self.posc_models.append(PoscModel(item, name=name))

    def _prepare_output_base(self):
        """Prepare the output base name."""
        if self.args.output_base is None:
            return self.script_path
        return self.args.output_base

    def run(self):
        self.check_args()
        self._prepare_models()

        if not self.posc_models:
            print("No models were generated or provided.", file=sys.stderr)
            return

        actions_requested = self.args.view or self.args.stl or self.args.png or self.args.scad
        if not actions_requested:
            print("No action specified. Use --view, --stl, --png etc.", file=sys.stderr)
            return

        output_base = self._prepare_output_base()
        # --- SCAD Action ---
        if self.args.scad:
            if len(self.posc_models) == 1:
                model = self.posc_models[0].get_posc_obj()
            else:
                model = posc.LazyUnion()(*[model.get_posc_obj() for model in self.posc_models])
            print(f"Exporting {len(self.posc_models)} models to SCAD: {output_base}.scad")
            filename = f"{output_base}.scad"
            model.write(filename)
            print(f"Exported SCAD: {output_base}.scad")

        # --- STL Action ---
        if self.args.stl:
            exported_count = 0
            for i, model in enumerate(self.posc_models):
                solid_manifold = model.get_solid_manifold()
                # TODO: Handle shell manifolds.
                if solid_manifold:
                    suffix = f"_{i}" if len(self.posc_models) > 1 else ""
                    filename = f"{output_base}{suffix}.stl"
                    manifold_to_stl(solid_manifold, filename=filename, update_normals=False)
                    print(f"Exported STL: {filename}")
                    exported_count += 1

            if exported_count == 0:
                    print("Warning: No solid geometry found to export to STL.", file=sys.stderr)


        # --- Viewer / PNG Actions ---
        if self.args.view or self.args.png:

            viewer_models: List[Model] = []
            for model in self.posc_models:
                viewer_models.extend(model.get_viewer_models(include_shells=not self.args.solids))

            if not viewer_models:
                print("Warning: No viewable geometry was generated.", file=sys.stderr)
                # Don't proceed if there's nothing to view/save
                if not self.args.stl: # Only exit if STL wasn't the primary goal
                     return
                else: # If STL was done, but nothing to view, that's okay
                     print("STL export completed, but no models to view/save as PNG.")


            if self.args.view or self.args.png:
                viewer: Viewer = Viewer(
                    models=viewer_models,
                    width=self.args.width,
                    height=self.args.height,
                    title=self.args.title,
                    background_color=self.args.parsed_bg_color,
                    projection_mode=self.args.projection,
                    wireframe_mode=self.args.wireframe,
                    backface_culling=self.args.backface_culling,
                    bounding_box_mode=self.args.bounding_box_mode,
                    zbuffer_occlusion=self.args.zbuffer_occlusion,
                    use_coalesced_models=self.args.coalesce
                )
                
                if self.args.png:
                    filename = f"{output_base}.png"
                    viewer.offscreen_render(filename)
                    print(f"Saved PNG: {filename}")
                    
                if self.args.view:
                    print(Viewer.VIEWER_HELP_TEXT)
                    print(f"triangle count = {viewer.num_triangles()}")
                    viewer.run() # Enters GLUT main loop

def posc_main(
    items: List[Union[Callable[[], posc.PoscBase], posc.PoscBase]],
    default_scad: bool = False,
    default_png: bool = False,
    default_view: bool = True,
    default_stl: bool = False,
    default_wireframe: bool = False,
    default_backface_culling: bool = True,
    default_bounding_box_mode: bool = False,
    default_zbuffer_occlusion: bool = True,
    default_coalesce: bool = True,
    default_width: int = 800,
    default_height: int = 600,
    default_title: str | None = None,
    default_projection: str = 'perspective',
    default_bg_color: str = "0.98,0.98,0.85,1.0",
    ):
    """
    Main entry point for processing PythonOpenSCAD objects via command line.

    Args:
        items: A list containing PoscBase objects or lambda functions
               that return PoscBase objects.
    """
    # Get the file path of the script that called posc_main
    try:
        calling_frame = inspect.stack()[1]
        script_path = calling_frame.filename
    except IndexError:
        script_path = "unknown_script.py" # Fallback if stack inspection fails

    runner = PoscMainRunner(items, 
                            script_path,
                            default_scad=default_scad,
                            default_png=default_png,
                            default_view=default_view,
                            default_stl=default_stl,
                            default_wireframe=default_wireframe,
                            default_backface_culling=default_backface_culling,
                            default_bounding_box_mode=default_bounding_box_mode,
                            default_zbuffer_occlusion=default_zbuffer_occlusion,
                            default_coalesce=default_coalesce,
                            default_width=default_width,
                            default_height=default_height,
                            default_title=default_title,
                            default_projection=default_projection,
                            default_bg_color=default_bg_color)
    runner.run()


# Example usage (would typically be in a user's script):
if __name__ == "__main__":

    if len(sys.argv) <= 1:
        # Simulate command line arguments for testing
        # In real use, these come from the actual command line
        sys.argv = [
            sys.argv[0],    # Script name
            "--view",       # Action: View the models
            "--scad",       # Action: Export SCAD
            "--stl",        # Action: Export STL
            "--png",        # Action: Save PNG
            "--no-wireframe",  # Viewer option
            "--bg-color", "1,1,1,1.0", # Viewer option
            "--projection", "orthographic", # Viewer option
            # "--output-base", "my_test_output", # Output name override
        ]

    print(f"Running example with simulated args: {' '.join(sys.argv[1:])}")
    
    # Define some example objects/lambdas
    def create_sphere():
        return posc.Color("red")(posc.Sphere(r=5).add_modifier(posc.DEBUG))
    
    def create_cross_section():
        # Cross section is linear extruded by the renderer.
        return posc.Color("green").append(posc.Circle(r=10))
    
    my_cube = posc.Cube(10)

    posc_main([my_cube, create_sphere, create_cross_section])
    print("Example finished.")
