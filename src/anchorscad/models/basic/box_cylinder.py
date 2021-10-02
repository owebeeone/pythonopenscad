'''
Created on 29 Sep 2021

@author: gianni
'''

from dataclasses import dataclass
import ParametricSolid.core as core
import ParametricSolid.linear as l
from ParametricSolid.extrude import PathBuilder, LinearExtrude


@core.shape('anchorscad.models.basic.box_cylinder')
@dataclass
class BoxCylinder(core.CompositeShape):
    '''
    <description>
    '''
    size: tuple=(10, 20, 30)
    fn: int=36
    
    EXAMPLE_SHAPE_ARGS=core.args()
    EXAMPLE_ANCHORS=(
        core.surface_args('face_corner', 0, 0),
        core.surface_args('cylinder', 'top'),
        core.surface_args('cylinder', 'base'),)
    
    def __post_init__(self):
        r = self.size[1] / 2
        self.r = r
        cage_size = (self.size[0] + r, self.size[1], self.size[2])
        maker = core.Box(cage_size).cage(
            'cage').colour([1, 1, 0, 0.5]).at(
                'face_corner', 0, 0)
            
        path = (PathBuilder()
            .move([0, 0])
            .line([-r, 0], 'edge1')
            .line([-r, self.size[0]], 'edge2')
            .arc_tangent_point([r, self.size[0]], name='arc')
            .line([r, 0], 'edge3')
            .line([0, 0], 'edge4')
            .build())
        
        shape = LinearExtrude(path, self.size[2], fn=self.fn)
        
        maker.add_at(shape.solid('box_cylinder').at('edge4', 1.0),
                     'face_edge', 0, 0, post=l.ROTY_180)
        
        maker.add_at(core.Cylinder(r=r, h=self.size[2])
                     .cage('cylinder').at('surface'),
                     'box_cylinder', 'arc', 0, pre=l.tranX(2 * r))
        
        self.maker = maker

    @core.anchor('Round centre.')
    def round_centre(self, h=0, rh=None):
        if not rh is None:
            h = h + rh * self.size[2]
        return (self.maker.at('face_centre', 4) 
                * l.translate((0, self.size[1] / 2 - self.r , -h)))


if __name__ == '__main__':
    core.anchorscad_main(False)