'''
Created on 20 Sep 2021

@author: gianni
'''

from dataclasses import dataclass
import ParametricSolid.core as core
import ParametricSolid.linear as l
from ParametricSolid.extrude import PathBuilder, LinearExtrude
from anchorscad.models.basic.box_side_bevels import BoxSideBevels


@core.shape('anchorscad.models.fastners.snaps')
@dataclass
class Snap(core.CompositeShape):
    '''
    <description>
    '''
    size: tuple=(15, 10, 2)
    depth_factor: float=0.5
    max_x: float=0.55
    t_size: float=1.5
    tab_protrusion: float=1.5
    tab_height: float=4
    epsilon: float=1.e-2
    fn:int = 16
    
    EXAMPLE_SHAPE_ARGS=core.args()
    EXAMPLE_ANCHORS=(
                core.surface_args('snap', 0.5),)
    
    def __post_init__(self):
        box = core.Box(self.size)
        maker = box.cage(
            'cage').transparent(1).colour([1, 1, 0, 0.5]).at('centre')
            
        max_x = self.max_x
        t_size = self.t_size
        extentX = self.size[2] * self.depth_factor
        extentY = self.size[1]
        extentX_t = extentX + self.tab_protrusion
        cv_len = (t_size / 2, t_size / 2)
        
        path = (PathBuilder()
            .move([max_x, 0])
            .line([max_x - 0.01, 0.01], 'direction')
            .spline(([-max_x, -t_size], [-max_x, 1.3 * t_size]), cv_len=cv_len, name='lead')
            .spline(([0, 2 * t_size], [0, 2 * t_size]), cv_len=cv_len, name='tail')
            .line([0, extentY], name='draw')
            .line([extentX, extentY], name='top')
            .line([extentX_t, extentY - self.tab_protrusion], name='top_protrusion')
            .line([extentX, 0], name='side')
            .line([0, 0], name='bottom')
            .build())

        shape = LinearExtrude(path, h=self.size[0])
        
        maker.add_at(shape.solid('tooth').at('top', 0),
                     'face_edge', 0, 3, post=l.ROTY_180)
        
        # Round the clip.
        clip_size = box.size.A[0:3] + 2 * self.epsilon
        clip_size[2] = clip_size[2] + self.tab_protrusion
        clip_cage = (core.Box(clip_size).cage('clip_cage')
                .transparent(1).colour([0, 1, 0, 0.5])
                .at('centre'))
        
        
        th = self.tab_height
        cutter_size = clip_size + self.epsilon / 2
        cutter_size[1] = th
        clip = core.Box(cutter_size)
        clip_cage.add_at(clip.solid('clip').at(
            'face_corner', 1, 1), 'face_corner', 1, 1)
           
        keep_size = clip_size + self.epsilon
        keep = BoxSideBevels(keep_size, th, fn=self.fn)
        clip_cage.add_at(keep.hole('keep').at('shell', 'face_corner', 1, 1),
                    'face_corner', 1, 1)
        
        
        
        maker.add_at(
            clip_cage.hole('clip').at('face_centre', 1), 
            'face_centre', 1, 
            post=l.translate([-self.epsilon / 2, -self.epsilon, 0]))
        
        self.maker = maker
        
    @core.anchor('Snap seam edge.')
    def snap(self, rpos=0.5):
        '''Anchors to the seam line..
        Args:
            rpos: 0.0-1.0, 0.5 is centre.
        '''
        return (self.at('tooth', 'bottom', 1.0, rh=rpos) 
                * l.tranZ(self.tab_height) * l.ROTV111_120 * l.ROTY_180)


if __name__ == '__main__':
    core.anchorscad_main(False)