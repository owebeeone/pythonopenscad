'''
Created on 13 Nov 2021

Tools for building outlines and access holes.

@author: Gianni Mariani
'''

from dataclasses import dataclass
from ParametricSolid.linear import tranX, tranY, tranZ, ROTX_180, \
                                   translate, GVector
import ParametricSolid.core as core
import numpy as np
import anchorscad.models.basic.connector.hdmi.hdmi_outline as hdmi


Z_DELTA=tranZ(-0.01)

def box_expander(expansion_size=None, actual_size=None, post=None):
    '''
    '''
    def expander(maker, name, anchor, box):
        if actual_size:
            expanded_size = GVector(actual_size)
        else:
            expanded_size = GVector(expansion_size) + box.size
        new_shape = core.Box(expanded_size)
        post_xform = Z_DELTA * ROTX_180
        if post:
            post_xform = post *  post_xform
        maker.add_at(new_shape.solid((name, 'outer')).at(*anchor[0], **anchor[1]),
                     name, *anchor[0], **anchor[1], post=post_xform)
    return expander


def cyl_expander(expansion_r, post=None):
    def expander(maker, name, anchor, cyl):
        expanded_r = expansion_r + cyl.r_base
        params = core.non_defaults_dict(cyl, include=('fn', 'fa', 'fs'))
        new_shape = core.Cylinder(h=cyl.h, r=expanded_r, **params)
        post_xform = Z_DELTA * ROTX_180
        if post:
            post_xform = post *  post_xform
        maker.add_at(new_shape.solid((name, 'outer')).at(*anchor[0], **anchor[1]),
                     name, *anchor[0], **anchor[1], post=post_xform)
    return expander

def no_op(*args):
    pass

@dataclass
class ShapeFactory:
    clazz: type
    shape_args: tuple
    offset: tuple
    anchor1: tuple
    anchor2: tuple
    expander: tuple
    
    def create(self, extra_params: dict):
        params = (dict(((k, v) 
                        for k, v in extra_params.items() 
                        if hasattr(self.clazz, k))))
        
        params.update(self.shape_args[1])
        return self.clazz(*self.shape_args[0], **params)
    
    
SIDE_ANCHOR=core.args('face_corner', 4, 0)
FRONT_ANCHOR=core.args('face_corner', 4, 1)
BOX_ANCHOR=core.args('face_edge', 1, 0)
OBOX_ANCHOR=core.args('face_centre', 3)
IBOX_ANCHOR=core.args('face_centre', 4)
CYL_ANCHOR=core.args('surface', 0, -90)
OCYL_ANCHOR=core.args('base')
        
ETHERNET = ShapeFactory(
    core.Box, core.args([16, 21.25, 13.7]), 
    [0, 3.0, 0], 
    BOX_ANCHOR, 
    OBOX_ANCHOR, 
    box_expander([0.3] * 3))

USBA=ShapeFactory(
    core.Box, core.args([14.9,  17.5, 16.4]), 
    [0, 3.0, 0], 
    BOX_ANCHOR,
    OBOX_ANCHOR, 
    box_expander([0.3] * 3))

MICRO_HDMI=ShapeFactory(
    core.Box, core.args([7.1,  8, 3.6]), 
    [0, 1.8, -0.5], 
    BOX_ANCHOR, 
    OBOX_ANCHOR, 
    box_expander([5, 0, 4.5]))

HDMI_A=ShapeFactory(
    hdmi.HdmiOutline, core.args(), 
    [0, 1.8, -0.5], 
    BOX_ANCHOR, 
    OBOX_ANCHOR, 
    box_expander(actual_size=[21, 0, 10.6]))

USBC=ShapeFactory(
    core.Box, core.args([9,  7.5, 3.3]), 
    [0, 1.8, -(4.14 - 2.83 - 1.44)], 
    BOX_ANCHOR, 
    OBOX_ANCHOR, 
    box_expander([5, 0, 4]))

USBCMICRO=ShapeFactory(
    core.Box, core.args([7.6, 5.6, 2.9]), 
    [0, 1.02, 0], 
    BOX_ANCHOR, 
    OBOX_ANCHOR, 
    box_expander([5, 0, 4]))

AUDIO=ShapeFactory(
    core.Cylinder, core.args(h=15, r=3), 
    [0, 2.7, 0], 
    CYL_ANCHOR, 
    OCYL_ANCHOR, 
    cyl_expander(2))

MICRO_SD=ShapeFactory(
    core.Box, core.args([12,  11.35, 1.4]), 
    [0, -3, 0], 
    BOX_ANCHOR, 
    OBOX_ANCHOR, 
    box_expander([1, 1, 6], post=translate([0, -3, 0])))

CPU_PACKAGE=ShapeFactory(
    core.Box, 
    core.args([15,  15, 2.4]),
    [0, 0, 0], 
    core.args('face_edge', 1, 0, 1), 
    IBOX_ANCHOR, 
    no_op)

HEADER_100=ShapeFactory(
    core.Box, 
    core.args([51,  5.1, 8.7]), [0, -1.75, 0], 
    core.args('face_edge', 1, 0, 1), 
    IBOX_ANCHOR, 
    no_op)

