from pythonopenscad.m3dapi import M3dRenderer
from pythonopenscad.posc_main import posc_main, PoscModel
import pythonopenscad as posc
from pythonopenscad import PoscBase
from pythonopenscad.viewer.model import Model

def test_base_viewer():
    def make_model():
        linear_extrusion = posc.Color("darkseagreen")(
            posc.Translate([0.0, 0.0, 4.5])(
                posc.Linear_Extrude(height=3.0, twist=45, slices=16, scale=(2.5, 0.5))(
                    posc.Translate([0.0, 0.0, 0.0])(posc.Square([1.0, 1.0]))
                )
            )
        )
        return linear_extrusion

    #posc_main([make_model])
    
def make_model() -> PoscBase:
	return posc.Color("darkseagreen")(
            posc.Translate([0.0, 0.0, 4.5])(
                posc.Linear_Extrude(height=3.0, twist=45, slices=16, scale=(2.5, 0.5))(
                    posc.Translate([0.0, 0.0, 0.0])(posc.Square([1.0, 1.0]))
                )
            )
        )

def test_base_viewer2():
    model = make_model()
    # Save to OpenSCAD file
    model.write('my_model.scad')

    # Render to STL
    rc = model.renderObj(M3dRenderer())
    #rc.write_solid_stl("mystl.stl")

    # Or, view the result in a 3D viewer.
    #posc_main([make_model])    

if __name__ == '__main__':
    test_base_viewer2()

