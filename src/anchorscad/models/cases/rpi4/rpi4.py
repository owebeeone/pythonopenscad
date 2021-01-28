'''
Created on 25 Jan 2021

@author: gianni
'''

from dataclasses import dataclass

import ParametricSolid.core as core
import ParametricSolid.linear as l
import anchorscad.models.basic.box_side_bevels as bbox
import numpy as np


def tranX(x):
    return l.translate([x, 0, 0])

SIDE_ANCHOR=core.args('face_corner', 4, 0)
FRONT_ANCHOR=core.args('face_corner', 4, 1)
BOX_ANCHOR=core.args('face_edge', 1, 0)
CYL_ANCHOR=core.args('surface', 0, 0)

ETHERNET=(core.Box([16, 21.25, 13.7]), [0, 3.0, 0], BOX_ANCHOR)

USBA=(core.Box([14.9,  17.5, 16.4]), [0, 3.0, 0], BOX_ANCHOR)

MICRO_HDMI=(core.Box([7.1,  8, 3.6]), [0, 1.8, -0.5], BOX_ANCHOR)

USBC=(core.Box([9,  7.5, 3.2]), [0, 1.8, 0], BOX_ANCHOR)

AUDIO=(core.Cylinder(h=15, r=3, fn=20), [0, 2.7, 0], CYL_ANCHOR)

MICRO_SD=(core.Box([12,  11.35, 1.4]), [0, -3, 0], BOX_ANCHOR)

SIDE_ACCESS=(core.args('face_corner', 4, 0), (
    ('usbC', USBC, tranX(3.5 + 7.7)),
    ('hdmi1', MICRO_HDMI, tranX(3.5 + 7.7 + 14.8)),
    ('hdmi2', MICRO_HDMI, tranX(3.5 + 7.7 + 14.8 + 13.5)),
    ('audio', AUDIO, tranX(3.5 + 7.7 + 14.8 + 13.5 + 7 + 7.5)),
    ))

FRONT_ACCESS=(core.args('face_corner', 4, 1), (
    ('usbA2', USBA, tranX(9)),
    ('usbA3', USBA, tranX(27)),
    ('rj45', ETHERNET, tranX(45.75)),
    ))

BOTTOM_ACCESS=(core.args('face_corner', 1, 3), (
    ('micro_sd', MICRO_SD, tranX((34.15 + 22.15) / 2)),
    ))


DELTA=0.02

HOLE_RADIUS=2.7/2
HOLE_POSITIONS=((3.5, 3.5), (3.5, 3.5 + 49), (3.5 + 58, 3.5), (3.5 + 58, 3.5 + 49))

@core.shape('anchorscad/models/cases/rpi4_model')
@dataclass
class RaspberryPi4Case(core.CompositeShape):
    '''
    A Raspberry Pi 4 Model
    '''
    board_size: tuple=(85, 56, 1.5)
    bevel_radius: float=3.0
    fn: int=None
    fa: float=None
    fs: float=None
    
    
    EXAMPLE_SHAPE_ARGS=core.args(fn=20)
    EXAMPLE_ANCHORS=tuple(
        core.surface_args(('mount_hole', i), 'base') for i in range(4)
        )
    
    def __post_init__(self):
        maker = bbox.BoxSideBevels(
            size=self.board_size, bevel_radius=self.bevel_radius, fn=20).solid(
            'board').at('face_centre', 4)
        
        self.maker = maker
        
#         for i in range(6):
#             maker.add_at(core.Text(str(i)).solid(('face_label', i)).at(), 'face_centre', i)  
#             
        params = core.non_defaults_dict(self, include=('fn', 'fa', 'fs'))
        mount_hole = core.Cylinder(
            h=self.board_size[2] + 2 * DELTA, r=HOLE_RADIUS, **params)
        
        for i, t in enumerate(HOLE_POSITIONS):
            maker.add_at(
                mount_hole.hole(('mount_hole', i)).at('base'), 
                'face_corner', 4, 0, post=l.translate(t + (DELTA,)))

        for base_anchor, items in (SIDE_ACCESS, FRONT_ACCESS, BOTTOM_ACCESS):
            for name, model, xform in items:
                maker.add_at(
                    model[0].solid(name).at(args=model[2], post=l.translate(model[1])),
                    args=base_anchor, post=xform * l.ROTY_180
                    )

        

    

# @core.shape('anchorscad/models/cases/rpi4_case')
# @dataclass
# class RaspberryPi4Case(core.CompositeShape):
#     '''
#     A Raspberry Pi 4 Case
#     '''
#     

if __name__ == "__main__":
    core.anchorscad_main(False)

