'''
Created on 28 Sep 2021

@author: gianni
'''

from dataclasses import dataclass
import ParametricSolid.core as core
import ParametricSolid.linear as l
from anchorscad.models.basic.pipe import Pipe
from anchorscad.models.screws.dims import M_HOLE 
from anchorscad.models.grille.round.CurlySpokes import CurlySpokes


@core.shape('anchorscad.models.vent.fan.fan_vent')
@dataclass
class FanVent(core.CompositeShape):
    '''
    Creates a fan mount and vent.
    '''
    vent_thickness: float=2
    size: tuple=(30, 30, 7.4)
    screw_hole_size: float=2.6
    screw_support_dia: float=4.6
    screw_support_depth: float=2.3
    screw_centres: float=(27.3 + 20.6) / 2
    screw_hole_extension: float=1.5
    as_example: bool=False
    r_outer: float=29.0 / 2
    r_inner: float=12.0 / 2
    grille_type: type=CurlySpokes
    grille_as_cutout: bool=False
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
            
        inside_r = M_HOLE[self.screw_hole_size].tap_dia / 2
            
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
        grille = self.grille_type(
            h=self.vent_thickness,
            r_outer=self.r_outer,
            r_inner=self.r_inner,
            as_cutout=self.grille_as_cutout
            )
        
        mode = (core.ModeShapeFrame.HOLE 
                if self.grille_as_cutout 
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
