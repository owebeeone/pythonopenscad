'''
Created on 20 Sep 2021

@author: gianni
'''

from dataclasses import dataclass
import ParametricSolid.core as core
import ParametricSolid.linear as l
from ParametricSolid.extrude import PathBuilder, LinearExtrude


@core.shape('anchorscad.models.fastners.snaps')
@dataclass
class Snap(core.CompositeShape):
    '''
    <description>
    '''
    size: tuple=(10, 8, 2)
    depth_factor: float=0.5
    max_x: float=0.3
    t_size: float=1.0
    tab_protrusion: float=1
    
    
    EXAMPLE_SHAPE_ARGS=core.args()
    EXAMPLE_ANCHORS=()
    
    def __post_init__(self):
        maker = core.Box(self.size).solid(
            'plane1').transparent(1).colour([1, 1, 0, 0.5]).at('centre')
            
        max_x = self.max_x
        t_size = self.t_size
        extentX = self.size[2] * self.depth_factor
        extentY = self.size[1]
        extentX_t = extentX + self.tab_protrusion
        cv_len = (t_size / 2, t_size / 2)
        
        path = (PathBuilder()
            .move([max_x, 0])
            .line([max_x - 0.01, 0.01], 'direction')
            .spline(([-max_x, -t_size], [-max_x, t_size]), cv_len=cv_len, name='lead')
            .spline(([0, 2 * t_size], [0, 2 * t_size]), cv_len=cv_len, name='tail')
            .line([0, extentY], name='draw')
            .line([extentX, extentY], name='top')
            .line([extentX_t, extentY], name='top_protrusion')
            .line([extentX_t, 0], name='side')
            .line([extentX, 0], name='bottom_prot')
            .line([0, 0], name='bottom')
            .build())

        shape = LinearExtrude(path, h=self.size[0])
        
        maker.add_at(shape.solid('tooth').at('top', 0),
                     'face_edge', 0, 3, post=l.ROTY_180)
        
        self.maker = maker

if __name__ == '__main__':
    core.anchorscad_main(False)
