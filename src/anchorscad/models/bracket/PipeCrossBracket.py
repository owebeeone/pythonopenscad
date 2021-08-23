'''
Created on 21 Aug 2021

@author: gianni
'''

from dataclasses import dataclass

import ParametricSolid.core as core
import ParametricSolid.extrude as extrude
import ParametricSolid.linear as l
from anchorscad.models.screws.CountersunkScrew import CountersunkScrew
import numpy as np


@core.shape('anchorscad/models/bracket/PipeCrossBracket')
@dataclass
class PipeCrossBracket(core.CompositeShape):
    '''
    Pipe cross bracket.
    '''
    
    radius1: float=24.8 / 2
    #radius2: float=24.8 / 2
    radius2: float=19.4 / 2
    clearance: float=12.0
    thickness: float=6.5
    screw_depth: float=7.0
    screw_depth_hole_delta: float=0.1
    clip_height: float= 10
    clip_height_hole_room: float= 0.5
    clip_wedge_run: float=4
    clip_pre_wedge_run: float=1
    clip_wedge_hight: float=1
    clip_thickness:float=7.0
    nipple_offset: float=5
    nipple_width: float=4.0
    nipple_height: float=0.5
    edge_radius: float=1.0
    screw1_size_name: str='M2.6'
    screw1_len: float=20.0
    fn: int=37
    
    
    NOEXAMPLE_ANCHORS=(
                core.surface_args('bbox', 'face_corner', 0, 0),
                core.surface_args('base', 'face_corner', 3, 1),)
    
    def __post_init__(self):
        
        size_x = self.clearance * 2 + self.thickness * 2 + self.radius1 * 2.0
        size_y = self.thickness + self.radius2 * 2.0
        size_z = size_x
        
        dimens = [size_x, size_y, size_z]
        
        maker = core.Box(dimens).cage('bbox').at(
            'face_corner', 0, 0)
        
        maker.add_at(core.Box([size_x, self.screw_depth, size_z]).cage('base').at(
            'face_corner', 0, 0))
        
        hole_x = size_x / 2.0 - self.radius2
        outer_radius = self.radius2 + self.thickness
        outer_hole_x = hole_x - self.thickness
        rhs_outer_hole_x = outer_radius * 2.0 + outer_hole_x
    
        path = (extrude.PathBuilder()
            .move([hole_x, 0])
            .line([0, 0], 'edge0')
            .line([0, self.screw_depth], 'edge1')
            .line([outer_hole_x, self.screw_depth], 'edge2')
            .line([outer_hole_x, self.radius2], 'edge3')
            .arc_points_radius(
                [rhs_outer_hole_x, self.radius2], 
                self.radius2 + self.thickness, name='edge4', metadata=self)
            .line([rhs_outer_hole_x, self.screw_depth], 'edge5')
            .line([size_x, self.screw_depth], 'edge6')
            .line([size_x, 0], 'edge7')
            .line([size_x - hole_x, 0], 'edge8')
            .line([size_x / 2 + self.radius2, 0], 'edge9')
            .line([size_x / 2 + self.radius2, self.radius2], 'edge10')
            .arc_points_radius(
                [hole_x, self.radius2], 
                self.radius2, direction=True, name='edge11', metadata=self)
            .line([hole_x, 0], 'edge12')
            
            .build())
        
        shape = extrude.LinearExtrude(path, dimens[2])
        
        maker.add_at(shape.solid('bracket').at('edge0', 1.0),
            'face_corner', 0, 0, post=l.rotY(180))
        
        wedge_hole_y = self.screw_depth_hole_delta + self.screw_depth
        nipple_top_y = wedge_hole_y - self.nipple_height
        nipple_top_x = self.nipple_offset + self.nipple_width / 1.2
        nipple_end_x = self.nipple_offset + self.nipple_width
        clip_wedge_start = nipple_end_x + self.clip_pre_wedge_run
        clip_wedge_end = clip_wedge_start + self.clip_wedge_run
        wedge_bottom_y = wedge_hole_y + self.clip_thickness
        
        tag_path = (extrude.PathBuilder()
            .move([0, 0])
            .line([0, -wedge_hole_y], 'edge0')
            .line([self.nipple_offset, -wedge_hole_y], 'edge1')
            .arc_points([nipple_top_x, -nipple_top_y], 
                        [nipple_end_x, -wedge_hole_y], 
                        direction=False, name='nipple')
            .line([clip_wedge_start, -wedge_hole_y], 'edge3')
            .line([clip_wedge_end, -wedge_hole_y - self.clip_wedge_hight], 'edge4')
            .line([clip_wedge_end, -wedge_bottom_y], 'edge5')
            .line([-self.clip_thickness, -wedge_bottom_y], 'edge6')
            .line([-self.clip_thickness, self.screw_depth], 'edge7')
            .line([0, self.screw_depth], 'edge8')
            .line([0, 0], 'edge9')
            .build())
        
        tag_shape = extrude.LinearExtrude(
            tag_path, self.clip_height + self.clip_height_hole_room)
        
        screw1_cap_thickness = 0.3
        screw1_overall_len = wedge_bottom_y + self.screw_depth - screw1_cap_thickness
        screw1_tap_len = self.clip_thickness + wedge_hole_y
        screw1_shape = CountersunkScrew(
            shaft_overall_length=screw1_overall_len,
            shaft_thru_length=screw1_tap_len,
            size_name=self.screw1_size_name,
            include_thru_shaft=False,
            include_tap_shaft=False,
            as_solid=False,
            fn=self.fn
            )
        
        tag_solid_maker = tag_shape.solid('tag').at('edge0', 0)
        tag_hole_maker = tag_shape.hole('tag').at('edge0', 0)
        
        screw_transform = l.rotY(180) * l.translate([0, self.clip_height / 2, 0])
        
        tag_solid_maker.add_at(
            screw1_shape.composite('screw1').at('screw_cage', 'top'),
            'tag', 'edge8', 2.0, post=screw_transform)
        
        tag_hole_maker.add_at(
            screw1_shape.composite('screw1').at('screw_cage', 'top'),
            'tag', 'edge8', 2.0, post=screw_transform)
        
        maker.add_at(tag_solid_maker.composite('tag').at('edge0', 0),
            'face_corner', 0, 0, post=l.rotY(-90))
        
        
        maker.add_at(tag_hole_maker.composite('tag_hole').at('edge1', 0),
            'base', 'face_corner', 3, 1, post=l.rotY(0) )
        
        
        self.maker = maker
        

    
if __name__ == "__main__":
    core.anchorscad_main(False)
