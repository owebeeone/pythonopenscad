'''
Created on 28 Sep 2021

@author: gianni
'''

from dataclasses import dataclass
import ParametricSolid.core as core
from ParametricSolid.datatree import datatree, Node
import ParametricSolid.linear as l
from anchorscad.models.basic.pipe import Pipe
from anchorscad.models.screws.dims import holeMetricDims 
from anchorscad.models.grille.round.CurlySpokes import CurlySpokes


@core.shape('anchorscad.models.vent.fan.fan_vent')
@datatree
class FanVent(core.CompositeShape):
    '''
    Creates a fan mount and vent.
    '''
    vent_thickness: float=2
    size: tuple=(30, 30, 7.7)
    screw_hole_size: float=2.6
    screw_hole_tap_dia_scale: float=0.95 # Needs more to tap onto.
    screw_support_dia: float=4.4
    screw_support_depth: float=2.3
    screw_centres: float=(28.4 + 19.33) / 2
    screw_hole_extension: float=1.5
    as_example: bool=False
    r_outer: float=29.0 / 2
    r_inner: float=12.0 / 2
    curl_inner_angle: float=-30
    grille_type: Node=core.ShapeNode(
        CurlySpokes, {'h': 'vent_thickness'}, expose_all=True)
    as_cutout: bool=False
    fn: int=36
    
    EXAMPLE_SHAPE_ARGS=core.args(as_example=True)
    EXAMPLE_ANCHORS=()
    
    def __post_init__(self):
        cage_mode = (core.ModeShapeFrame.SOLID 
                     if self.as_example 
                     else core.ModeShapeFrame.CAGE)
        maker = (core.Box(self.size)
                 .named_shape('fan', cage_mode)
                 .colour([1, 1, 0, 0.5])
                 .transparent(1)
                 .at('face_centre', 1))

        inside_r = (self.screw_hole_tap_dia_scale
            * holeMetricDims(self.screw_hole_size).tap_dia / 2)
            
        screw_mount = Pipe(h=self.screw_support_depth + self.screw_hole_extension,
                           inside_r=inside_r,
                           outside_r=self.screw_support_dia / 2,
                           fn=self.fn)
        screw_cage = core.Box(
            [self.screw_centres, self.screw_centres, self.size[2]])
        maker.add_at(screw_cage.cage('screw_cage').at('centre'),
                     'fan', 'centre')
        for i in range(4):
            maker.add_at(screw_mount.composite(('mount', i)).at('base'),
                                    'screw_cage', 'face_corner', 1, i,
                                    pre=l.tranZ(self.screw_hole_extension))
        grille = self.grille_type()
        
        mode = (core.ModeShapeFrame.HOLE 
                if self.as_cutout 
                else core.ModeShapeFrame.SOLID)
        
        maker.add_at(grille.named_shape('grille', mode).at('base'),
                     'face_centre', 1, post=l.ROTY_180)
        
        self.maker = maker

    @core.anchor('Centre of grille.')
    def grille_centre(self, *args, **kwds):
        return self.maker.at('grille', 'centre', *args, **kwds)
    
    @core.anchor('Centre of grille_base.')
    def grille_base(self, *args, **kwds):
        return self.maker.at('grille', 'base', *args, **kwds)

if __name__ == '__main__':
    core.anchorscad_main(False)
