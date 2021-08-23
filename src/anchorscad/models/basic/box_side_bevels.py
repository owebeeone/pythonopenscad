'''
Created on 25 Jan 2021

@author: gianni
'''

from dataclasses import dataclass

from numpy.core.defchararray import center

import ParametricSolid.core as core
import ParametricSolid.linear as l
import numpy as np


@core.shape('anchorscad/models/basic/box_side_bevels')
@dataclass
class BoxSideBevels(core.CompositeShape):
    '''
    Creates a box with bevels on 4 size (flat top and bottom).
    '''
    size: tuple=(30., 20., 10.)
    bevel_radius: float=2.0
    fn: int=None
    fa: float=None
    fs: float=None


    EXAMPLE_SHAPE_ARGS=core.args([100., 80., 40.], bevel_radius=8, fn=20)
    EXAMPLE_ANCHORS=tuple(
        (core.surface_args('face_corner', f, c)) for f in (0, 3) for c in range(4)
        ) + tuple(core.surface_args('face_edge', f, c) for f in (1, 3) for c in range(4)
        ) + tuple(core.surface_args('face_centre', f) for f in (0, 3)
        ) + (
            core.surface_args('face_edge', 2, 2, 0.1),
            core.surface_args('face_edge', 2, 2, -0.5),
             core.inner_args('centre'),)

    def __post_init__(self):
        size_delta = np.array([2 * self.bevel_radius, 2 * self.bevel_radius, 0])
        inner_size = np.array(self.size) - size_delta
        maker = core.Box(self.size).cage('shell').at('centre')
        maker.add(core.Box(inner_size).cage('hull').at('centre'))
        
        params = core.non_defaults_dict(self, include=('fn', 'fa', 'fs'))
        round = core.Cone(h=self.size[2], r_base=self.bevel_radius, r_top=self.bevel_radius, **params)
        faces = ((0, 1), (2, 1), (3, 3), (5, 1))
        for f, e in faces:
            maker.add_at(round.solid(f).at('centre'), 'hull', 'face_edge', f, e, post=l.ROTY_90)
        
        size_delta + np.array([2 * self.bevel_radius, 2 * self.bevel_radius, 0])
        
        for i in range(2):
            adjust = np.array([0, 0, 0])
            adjust[i] = 2 * self.bevel_radius
            new_size = adjust + inner_size
            maker.add(core.Box(new_size).solid(('box', i)).at('centre'))
        
        self.maker = maker

@core.shape('anchorscad/models/basic/box_shell')
@dataclass
class BoxShell(core.CompositeShape):
    '''
    Creates a box with the same box type hollowed out.
    '''
    size: tuple=(30., 20., 10.)
    bevel_radius: float=2.0
    shell_size: float=1.0
    box_class: type=BoxSideBevels
    fn: int=None
    fa: float=None
    fs: float=None
    
    EXAMPLE_SHAPE_ARGS=core.args(
        [100., 80., 40.], bevel_radius=8, shell_size=1.5, fn=20)
    
    EXAMPLE_ANCHORS=BoxSideBevels.EXAMPLE_ANCHORS
    
    def __post_init__(self):
        size = np.array(self.size)
        inner_size = size - 2 * self.shell_size
        if self.bevel_radius > self.shell_size:
            inner_bevel = self.bevel_radius - self.shell_size
        else:
            inner_bevel= 0
        
        params = core.non_defaults_dict(self, include=('fn', 'fa', 'fs'))
        
        outer_box = self.box_class(size=self.size, bevel_radius=self.bevel_radius, **params)
        inner_box = self.box_class(size=inner_size, bevel_radius=inner_bevel, **params)
        
        maker = outer_box.solid('outer').at('centre')
        maker.add(inner_box.hole('inner').at('centre'))
        
        self.maker = maker


@core.shape('anchorscad/models/basic/box_open_shell')
@dataclass
class BoxOpenShell(core.CompositeShape):
    '''
    Creates a box with the same box type but open at the top.
    '''
    size: tuple=(30., 20., 10.)
    bevel_radius: float=2.0
    shell_size: float=1.0
    box_class: type=BoxSideBevels
    z_adjust: float=0.0
    fn: int=None
    fa: float=None
    fs: float=None
    
    EXAMPLE_SHAPE_ARGS=core.args(
        [100., 80., 40.], bevel_radius=8, shell_size=3, z_adjust=-.01, fn=20)
    
    EXAMPLE_ANCHORS=BoxSideBevels.EXAMPLE_ANCHORS
    
    def __post_init__(self):
        size = np.array(self.size)
        inner_size = size - np.array([2, 2, 1]) * self.shell_size
        if self.bevel_radius > self.shell_size:
            inner_bevel = self.bevel_radius - self.shell_size
        else:
            inner_bevel= 0
        
        params = core.non_defaults_dict(self, include=('fn', 'fa', 'fs'))
        
        outer_box = self.box_class(size=self.size, bevel_radius=self.bevel_radius, **params)
        inner_box = self.box_class(size=inner_size, bevel_radius=inner_bevel, **params)
        
        maker = outer_box.solid('outer').at('centre')
        maker.add_at(inner_box.hole('inner').at(
            'face_centre', 4, pre=l.translate([0, 0, self.z_adjust])), 'face_centre', 4)
        
        self.maker = maker
        


@core.shape('anchorscad/models/basic/box_shell')
@dataclass
class BoxCutter(core.CompositeShape):
    ''''''
    model: core.Shape   # The model to be cut
    cut_size: tuple=(200, 200, 200)
    cut_face: int=1
    post: l.GVector=l.IDENTITY
    
    
    EXAMPLE_SHAPE_ARGS=core.args(
        BoxShell([100., 80., 40.], bevel_radius=8, shell_size=1.5, fn=40), 
        post=l.translate([0, 0, 10]) * l.ROTY_180)
    
    EXAMPLE_ANCHORS=BoxSideBevels.EXAMPLE_ANCHORS
    
    def __post_init__(self):
        maker = self.model.composite('main').at()
        maker.add(core.Box(self.cut_size).hole('cut_box').at(
            'face_centre', self.cut_face, post=self.post))
        
        self.maker = maker

if __name__ == "__main__":
    core.anchorscad_main(False)