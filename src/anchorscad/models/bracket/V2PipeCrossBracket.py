'''
Created on 26 Aug 2021

@author: gianni
'''


from dataclasses import dataclass

import ParametricSolid.core as core
import ParametricSolid.extrude as extrude
import ParametricSolid.linear as l
from anchorscad.models.screws.CountersunkScrew import CountersunkScrew, FlatSunkScrew
import numpy as np



@core.shape('anchorscad/models/bracket/PipeCrossBracket')
@dataclass
class V2PipeCrossBracket(core.CompositeShape):
    '''
    Pipe cross bracket V2.
    '''
    
    radius1: float=24.6 / 2
    radius2: float=19.31 / 2
    mid_padding: float=0
    base_hang_factor: float=0.5
    clip_hang_factor: float=0.15
    holder_radius_factor:float = 1.5
    cross_angle: float=90
    tie_width: float= 5.5
    tie_height: float=2.5
    tie_wing_size: float=1
    tie_offs: float= 7
    tie_angle: float= -10
    screw1_size_name: str='M2.6'
    screw1_len: float=19.0
    cutter_grade: float = 1.33
    
    fn: int=37
    
    
    NOEXAMPLE_ANCHORS=(
        core.surface_args('bracket', 'top'),
        core.surface_args('bracket', 'base'),
        core.surface_args('side1', 'surface', 0, 0, tangent=False),
        core.surface_args('side2', 'surface', 0, 0, tangent=False),)
    
    
    EXAMPLES_EXTENDED={'small': core.ExampleParams(
                            core.args(radius1=19.31 / 2,
                                      tie_offs=5,
                                      tie_angle=0,
                                      cutter_grade=1.2)), 
                       'large': core.ExampleParams(
                            core.args(radius2=24.6 / 2))}
    
    def __post_init__(self):
        
        holder_rad = (self.holder_radius_factor * self.radius1 * 
                      (1.0 + self.base_hang_factor))
        
        side1_cage = core.Cone(
            h=self.radius1 * (1 + self.clip_hang_factor), 
            r_base=self.radius1 * (1 + self.clip_hang_factor),
            r_top=holder_rad,
            fn=self.fn
            )
        
        side2_cage = core.Cone(
            h=self.radius2 * (1 + self.clip_hang_factor), 
            r_base=self.radius2 * (1 + self.clip_hang_factor),
            r_top=holder_rad,
            fn=self.fn
            )
        
        pad = core.Cone(
            h=self.mid_padding, 
            r_base=holder_rad,
            r_top=holder_rad,
            fn=self.fn
            )
        
        holder = core.Cone(
            h=side1_cage.h + side2_cage.h + self.mid_padding, 
            r_base=holder_rad,
            r_top=holder_rad,
            fn=self.fn
            )
        
        maker = holder.cage('bracket').at('base')
        

        maker.add_at(
            side1_cage.solid('side1').at('base'), 'bracket', 'base')
        

        maker.add_at(
            side2_cage.solid('side2').at('base', pre=l.rotZ(self.cross_angle)),
            'bracket', 'top')
        
        maker.add_at(pad.solid('pad').at('base'), 
                     'side1', 'top',
                      post=l.translate([0, 0, self.mid_padding]))
        
        pipe1_hole = core.Cone(
            h=10 * self.radius1,
            r_base=self.radius1,
            r_top=self.radius1,
            fn=self.fn
            )
        pipe2_hole = core.Cone(
            h=10 * self.radius2,
            r_base=self.radius2,
            r_top=self.radius2,
            fn=self.fn
            )
        
        xlation1 = [0, -self.radius1 * self.clip_hang_factor, 0]
        xlation2 = [0, -self.radius2 * self.clip_hang_factor, 0]
        
        maker.add_at(pipe1_hole.hole('pipe1').at('centre'), 
                     'side1', 'base',
                     post=l.ROTX_90 * l.translate(xlation1))
        maker.add_at(pipe2_hole.hole('pipe2').at('centre'), 
                     'side2', 'base',
                     post=l.ROTX_90 * l.translate(xlation2))
        
        # Tie grooves/slots
        
        tie_mid = self.tie_width / 2.0
        wing_point_x = self.tie_wing_size + tie_mid
        wing_point_y = self.tie_height / 2.0
        tie_height = self.tie_height
        
        tie_path = (extrude.PathBuilder()
            .move([0, 0])
            .line([-tie_mid, 0], 'base_l')
            .line([-wing_point_x, wing_point_y], 'wing_l_lower')
            .line([-tie_mid, tie_height], 'wing_l_upper')
            .line([0, tie_height], 'top_l')
            .line([tie_mid, tie_height], 'top_r')
            .line([wing_point_x, wing_point_y], 'wing_r_upper')
            .line([tie_mid, 0], 'winr_r_lower')
            .line([0, 0], 'base_r')
            .build())
        
        twist_factor = 0.9
        extrude_height = (self.radius1 + self.radius2 + self.mid_padding) * twist_factor
        tie_shape1 = extrude.LinearExtrude(
            tie_path, 
            extrude_height,
            twist=-90, 
            fn=self.fn)
        
        tie_shape2 = extrude.LinearExtrude(
            tie_path, 
            extrude_height,
            twist=90, 
            fn=self.fn)
        
        tie_args = (
            (tie_shape1, 1, 1, 0),
            (tie_shape2, -1, 1, 180),
            (tie_shape1, 1, 1, 180),
            (tie_shape2, -1, 1, 0),
                )
        
        for i in range(4):
            rotAngle = 45 + i * 90
            trans = [- tie_args[i][1] * self.tie_offs, tie_args[i][2] * self.tie_offs, -3]
            maker.add_at(
                tie_args[i][0].hole(('tie', i))
                .at('base_l', 
                    post=l.rotY(rotAngle + tie_args[i][3]) * l.rotZ(self.tie_angle * tie_args[i][1]),
                     pre=l.translate(trans)),
                     'bracket', 'surface', 0, rotAngle)
            
        # Flatten cutter
        cutter_scale = 4
        cutter_size = holder_rad * cutter_scale
        cutter_h = cutter_scale * side1_cage.h * self.cutter_grade
        cutter_cage_shape = core.Box([cutter_size, cutter_size, cutter_h])
                                              
        cutter_path = (extrude.PathBuilder()
                       .move([0, 0])
                       .line([0, cutter_h], 'side')
                       .line([cutter_size, 0], 'hypot')
                       .line([0, 0], 'base')
                       .build())
        cutter_shape = extrude.LinearExtrude(
            cutter_path, h=cutter_size)
        
        cutter_cage = cutter_cage_shape.cage('cutter_cage').at('face_edge', 1, 0)
        cutter_cage.add_at(cutter_shape.hole('cutter')
                           .at('base', 0.5, cutter_shape.h), post=l.ROTX_180)
        
        maker.add_at(cutter_cage.composite('face_cutter')
                     .at('face_edge', 1, 3, post=l.ROTX_90), 
                     'side1', 'surface', 0, 90, tangent=False,
                     post=l.ROTX_180 * l.translate([0, 0, 8]))
        
        # Add lock screw holes
        
        lock_screw = FlatSunkScrew(
            shaft_overall_length=self.radius1 * 2,
            shaft_thru_length=self.radius1 * 0.1,
            size_name=self.screw1_size_name,
            include_thru_shaft=False,
            include_tap_shaft=False,
            head_depth_factor=0.5,
            as_solid=False,
            fn=self.fn)
        
        maker.add_at(lock_screw.composite('screw1').at('top'),
                     'side1', 'surface', side1_cage.h * 0.7, 23, tangent=False)
        maker.add_at(lock_screw.composite('screw2').at('top'),
                     'side2', 'surface', side2_cage.h * 0.7, 23, tangent=False)
              
        
        self.maker = maker
    
if __name__ == "__main__":
    core.anchorscad_main(False)
