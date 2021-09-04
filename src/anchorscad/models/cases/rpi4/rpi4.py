'''
Created on 25 Jan 2021

@author: gianni
'''

from dataclasses import dataclass

from ParametricSolid.core import Box
import ParametricSolid.core as core
from ParametricSolid.linear import tranX, tranY, tranZ, ROTX_180
import ParametricSolid.linear as l
import anchorscad.models.basic.box_side_bevels as bbox
from anchorscad.models.screws.holes import SelfTapHole
import numpy as np


Z_DELTA=tranZ(-0.01)

SIDE_ANCHOR=core.args('face_corner', 4, 0)
FRONT_ANCHOR=core.args('face_corner', 4, 1)
BOX_ANCHOR=core.args('face_edge', 1, 0)
OBOX_ANCHOR=core.args('face_centre', 3)
IBOX_ANCHOR=core.args('face_centre', 4)
CYL_ANCHOR=core.args('surface', 0, -90)
OCYL_ANCHOR=core.args('base')

def box_expander(expansion_size, post=None):
    def expander(maker, name, anchor, box):
        expanded_size = l.GVector(expansion_size) + box.size
        new_shape = core.Box(expanded_size)
        post_xform = Z_DELTA * l.ROTX_180
        if post:
            post_xform = post *  post_xform
        maker.add_at(new_shape.solid((name, 'outer')).at(*anchor[0], **anchor[1]),
                     name, *anchor[0], **anchor[1], post=post_xform)
    return expander


def cyl_expander(expansion_r, post=None):
    def expander(maker, name, anchor, cyl):
        expanded_r = expansion_r + cyl.r_base
        new_shape = core.Cylinder(cyl.h, expanded_r)
        post_xform = Z_DELTA * l.ROTX_180
        if post:
            post_xform = post *  post_xform
        maker.add_at(new_shape.solid((name, 'outer')).at(*anchor[0], **anchor[1]),
                     name, *anchor[0], **anchor[1], post=post_xform)
    return expander

def no_op(*args):
    pass

ETHERNET=(core.Box([16, 21.25, 13.7]), [0, 3.0, 0], BOX_ANCHOR, OBOX_ANCHOR, box_expander([0.3] * 3))

USBA=(core.Box([14.9,  17.5, 16.4]), [0, 3.0, 0], BOX_ANCHOR,OBOX_ANCHOR, box_expander([0.3] * 3))

MICRO_HDMI=(core.Box([7.1,  8, 3.6]), [0, 1.8, -0.5], BOX_ANCHOR, OBOX_ANCHOR, box_expander([5, 0, 4.5]))

USBC=(core.Box([9,  7.5, 3.3]), [0, 1.8, 0], BOX_ANCHOR, OBOX_ANCHOR, box_expander([5, 0, 4]))

AUDIO=(core.Cylinder(h=15, r=3, fn=20), [0, 2.7, 0], CYL_ANCHOR, OCYL_ANCHOR, cyl_expander(2))

MICRO_SD=(core.Box([12,  11.35, 1.4]), [0, -3, 0], BOX_ANCHOR, OBOX_ANCHOR, 
          box_expander([1, 1, 6], post=l.translate([0, -3, 0])))

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
HOLE_SUPPORT_RADIUS=5.5/2
HOLE_POSITIONS=((3.5, 3.5), (3.5, 3.5 + 49), (3.5 + 58, 3.5), (3.5 + 58, 3.5 + 49))

@core.shape('anchorscad/models/cases/rpi4_model')
@dataclass
class RaspberryPi4Outline(core.CompositeShape):
    '''
    A Raspberry Pi 4 outline model.
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
            core.surface_args(('mount_hole', i), 'top') for i in range(4)
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


    

@core.shape('anchorscad/models/cases/rpi4_case')
@dataclass
class RaspberryPi4Case(core.CompositeShape):
    '''A Raspberry Pi 4 Case.'''
    outline_model: core.Shape=RaspberryPi4Outline()
    inner_size_delta: tuple=(3, 2, 22)
    inner_offset: tuple=(-1.5, 1, 3)
    wall_thickness: float=2
    inner_bevel_radius: float=outline_model.bevel_radius + (-inner_offset[0] - inner_offset[1]) / 2
    screw_clearannce: float=0.2
    board_screw_min_len: float=6
    show_outline: bool=False
    show_cut_box: bool=False
    make_case_top: bool=False
    
    split_box_delta: float=40
    fn: int=None
    fa: float=None
    fs: float=None
    
    EXAMPLE_ANCHORS=(core.surface_args('shell', 'face_centre', 1),)
    
    def __post_init__(self):
        inner_size = l.GVector(self.inner_size_delta) + l.GVector(self.outline_model.board_size)
        outer_size = (inner_size + (self.wall_thickness * 2,) * 3).A[0:3]
        bevel_radius = self.inner_bevel_radius + self.wall_thickness
        params = core.non_defaults_dict(self, include=('fn', 'fa', 'fs'))
        maker = bbox.BoxShell(
            size=outer_size, 
            bevel_radius=bevel_radius, 
            shell_size=self.wall_thickness, 
            **params).solid('shell').at('face_centre', 4)
        
        
        maker.add_at(self.outline_model.hole('outline').at('face_corner', 5, 0),
                     'inner', 'face_corner', 5, 0, pre=l.translate(self.inner_offset))

        split_box_cage = core.Box(outer_size).cage('split_box_cage').at('centre')
        split_box_size = outer_size + self.split_box_delta
        split_box = core.Box(split_box_size).solid('split_box').at('centre')
        split_box_cage.add(split_box)
        
        cut_point = (l.ROTX_90 * maker.at('outline', 'audio', 'base')).get_translation()
        cut_ref = maker.at('inner', 'face_centre', 4).get_translation()
        cut_xlation = cut_point - cut_ref
        
        cut_xform = l.IDENTITY if self.make_case_top else l.ROTX_180 
            
        cut_box_mode = core.ModeShapeFrame.HOLE if self.show_cut_box else core.ModeShapeFrame.HOLE
        
        maker.add_at(
            split_box_cage
                .named_shape('split_box', cut_box_mode)
                .transparent(self.show_cut_box)
                .at('split_box', 'face_centre', 4), 
            'face_centre', 4, post=tranZ(-cut_xlation.y) * cut_xform)
        
        
        bottom_loc = maker.at('shell', 'face_centre', 1).get_translation()
        screw_hole_loc = maker.at('outline', ('mount_hole', 0), 'top').get_translation()
        screw_hole_top_loc = maker.at('outline', ('mount_hole', 0), 'base').get_translation()
        max_allowable_screw_hole_height = screw_hole_loc.z - bottom_loc.z - self.screw_clearannce
        max_allowable_screw_size = screw_hole_top_loc.z - bottom_loc.z - self.screw_clearannce
        
        assert max_allowable_screw_size >= self.board_screw_min_len, (
            f'Board mounting screw hole height {max_allowable_screw_size} is smaller than the '
            f'mnimum size {self.board_screw_min_len}.')
        
        board_screw_hole = SelfTapHole(
            thru_len=1, 
            tap_len=max_allowable_screw_hole_height -1,
            outer_dia=HOLE_SUPPORT_RADIUS * 2,
            dia=2.6)
        
        for i in range(4):
            maker.add_at(board_screw_hole.composite(('support', i)).at('start', post=ROTX_180),
                         'outline', ('mount_hole', i), 'top')
        
     
        top_maker = maker.solid('main').at('centre')
        self.maker = top_maker
        
        if self.show_outline:
            top_maker.add_at(
                self.outline_model
                    .solid('outline2')
                    .transparent(True)
                    .at('centre'),
                'main', 'outline', 'centre')

if __name__ == "__main__":
    core.anchorscad_main(False)

