'''
Created on 25 Jan 2021

@author: gianni
'''

from builtins import staticmethod
from dataclasses import dataclass

import ParametricSolid.core as core
import ParametricSolid.linear as l
import anchorscad.models.basic.box_side_bevels as bbox
import numpy as np


def tranX(x):
    return l.translate([x, 0, 0])

def tranZ(z):
    return l.translate([0, 0, z])

Z_DELTA=tranZ(-0.01)

SIDE_ANCHOR=core.args('face_corner', 4, 0)
FRONT_ANCHOR=core.args('face_corner', 4, 1)
BOX_ANCHOR=core.args('face_edge', 1, 0)
OBOX_ANCHOR=core.args('face_centre', 3)
IBOX_ANCHOR=core.args('face_centre', 4)
CYL_ANCHOR=core.args('surface', 0, -90)
OCYL_ANCHOR=core.args('base')

def box_expander(expansion_size):
    def expander(maker, name, anchor, box):
        expanded_size = l.GVector(expansion_size) + box.size
        new_shape = core.Box(expanded_size)
        maker.add_at(new_shape.solid((name, 'outer')).at(*anchor[0], **anchor[1]),
                     name, *anchor[0], **anchor[1], post=Z_DELTA * l.ROTX_180)
    return expander


def cyl_expander(expansion_r):
    def expander(maker, name, anchor, cyl):
        expanded_r = expansion_r + cyl.r_base
        new_shape = core.Cylinder(cyl.h, expanded_r)
        maker.add_at(new_shape.solid((name, 'outer')).at(*anchor[0], **anchor[1]),
                     name, *anchor[0], **anchor[1], post=Z_DELTA * l.ROTX_180)
    return expander

def no_op(*args):
    pass

ETHERNET=(core.Box([16, 21.25, 13.7]), [0, 3.0, 0], BOX_ANCHOR, OBOX_ANCHOR, box_expander([0.3] * 3))

USBA=(core.Box([14.9,  17.5, 16.4]), [0, 3.0, 0], BOX_ANCHOR,OBOX_ANCHOR, box_expander([0.3] * 3))

MICRO_HDMI=(core.Box([7.1,  8, 3.6]), [0, 1.8, -0.5], BOX_ANCHOR, OBOX_ANCHOR, box_expander([5, 0, 4.5]))

USBC=(core.Box([9,  7.5, 3.3]), [0, 1.8, 0], BOX_ANCHOR, OBOX_ANCHOR, box_expander([5, 0, 4]))

AUDIO=(core.Cylinder(h=15, r=3, fn=20), [0, 2.7, 0], CYL_ANCHOR, OCYL_ANCHOR, cyl_expander(2))

MICRO_SD=(core.Box([12,  11.35, 1.4]), [0, -3, 0], BOX_ANCHOR, OBOX_ANCHOR, box_expander([1, 1, 1]))

CPU_PACKAGE=(core.Box([15,  15, 2.4]), [0, -25, 0], core.args('face_edge', 1, 0, 1), IBOX_ANCHOR, no_op)

HEADER_100=(core.Box([51,  5.1, 8.7]), [0, -1.75, 0], core.args('face_edge', 1, 0, 1), IBOX_ANCHOR, no_op)

SIDE_ACCESS=(core.args('face_corner', 4, 0), (
    ('usbC', USBC, tranX(3.5 + 7.7)),
    ('hdmi1', MICRO_HDMI, tranX(3.5 + 7.7 + 14.8)),
    ('hdmi2', MICRO_HDMI, tranX(3.5 + 7.7 + 14.8 + 13.5)),
    ('audio', AUDIO, tranX(3.5 + 7.7 + 14.8 + 13.5 + 7 + 7.5)),
    ('cpu', CPU_PACKAGE, tranX(22.0)),
    ))

OSIDE_ACCESS=(core.args('face_corner', 4, 2), (
    ('header100', HEADER_100, tranX(27.0)),
    ))

FRONT_ACCESS=(core.args('face_corner', 4, 1), (
    ('usbA2', USBA, tranX(9)),
    ('usbA3', USBA, tranX(27)),
    ('rj45', ETHERNET, tranX(45.75)),
    ))

BOTTOM_ACCESS=(core.args('face_corner', 1, 3), (
    ('micro_sd', MICRO_SD, tranX((34.15 + 22.15) / 2)),
    ))

ALL_ACCESS_ITEMS=(SIDE_ACCESS, FRONT_ACCESS, BOTTOM_ACCESS, OSIDE_ACCESS)

DELTA=0.02

HOLE_RADIUS=2.7/2
HOLE_POSITIONS=((3.5, 3.5), (3.5, 3.5 + 49), (3.5 + 58, 3.5), (3.5 + 58, 3.5 + 49))

@core.shape('anchorscad/models/cases/rpi4_model')
@dataclass
class RaspberryPi4Case(core.CompositeShape):
    '''
    A Raspberry Pi 4 basic model.
    '''
    board_size: tuple=(85, 56, 1.5)
    bevel_radius: float=3.0
    fn: int=None
    fa: float=None
    fs: float=None
    
    def make_access_anchors():
        anchor_specs = []
        for _, items in ALL_ACCESS_ITEMS:
            for name, model, xform in items:
                o_anchor = model[3]
                anchor_specs.append(
                    core.surface_args(name, *o_anchor[0], **o_anchor[1]))
        return tuple(anchor_specs)
    
    EXAMPLE_SHAPE_ARGS=core.args(fn=20)
    EXAMPLE_ANCHORS=tuple(
            core.surface_args(('mount_hole', i), 'base') for i in range(4)
        ) + make_access_anchors()
        
        
    def __post_init__(self):
        maker = bbox.BoxSideBevels(
            size=self.board_size, bevel_radius=self.bevel_radius, fn=20).solid(
            'board').at('face_centre', 4)
        
        self.maker = maker
        
#         for i in range(6):
#             maker.add_at(core.Text(str(i)).solid(('face_label', i)).at(), 'face_centre', i)  
             
        params = core.non_defaults_dict(self, include=('fn', 'fa', 'fs'))
        mount_hole = core.Cylinder(
            h=self.board_size[2] + 2 * DELTA, r=HOLE_RADIUS, **params)
        
        for i, t in enumerate(HOLE_POSITIONS):
            maker.add_at(
                mount_hole.hole(('mount_hole', i)).at('base'), 
                'face_corner', 4, 0, post=l.translate(t + (DELTA,)))

        for base_anchor, items in ALL_ACCESS_ITEMS:
            for name, model, xform in items:
                maker.add_at(
                    model[0].solid(name).colour([0, 1, 0.5]).at(args=model[2], post=l.translate(model[1])),
                    args=base_anchor, post=xform * l.ROTY_180
                    )
                # Add the outer hole.
                model[4](maker, name, model[3], model[0])


    

# @core.shape('anchorscad/models/cases/rpi4_case')
# @dataclass
# class RaspberryPi4Case(core.CompositeShape):
#     '''
#     A Raspberry Pi 4 Case
#     '''
#     

if __name__ == "__main__":
    core.anchorscad_main(False)

