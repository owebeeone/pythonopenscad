'''
Created on 14 Feb 2021

@author: gianni
'''

from dataclasses import dataclass

from ParametricSolid.core import shape, CompositeShape, non_defaults_dict, Cylinder, args, \
    surface_args, anchorscad_main, Cone, create_from, anchor
from ParametricSolid.extrude import PathBuilder
from ParametricSolid.linear import GVector, rotZSinCos, rotZ, translate
from anchorscad.models.basic.pipe import Pipe
from anchorscad.models.screws.dims import HoleDimensions, holeMetricDims
import numpy as np


PRINTER_HOLE_DIALATION=0.21

@dataclass(frozen=True)
class ArcLineBevelPoints:
    p_line: GVector
    p_arc: GVector
    p_unit_arc: GVector
    

def calc_arc_line_bevel(ra, rb, lo, degrees_orientation, radians_orientation):
    sina = (rb + lo) / (ra + rb)
    cosa = np.sqrt(1 - sina * sina)
    rot_a = rotZSinCos(sina, cosa)
    rot_oa = rotZSinCos(cosa, sina)
    
    p_unit_arc = rot_a * GVector([1, 0])
    p_arc = ra * p_unit_arc

    c_bevel = (ra + rb) * p_unit_arc
    p_line = GVector([c_bevel[0], lo, 0])
    
    
    rot_orientation = rotZ(degrees_orientation, radians_orientation)
    return ArcLineBevelPoints(
        rot_orientation * p_line,
        rot_orientation * p_arc, 
        rot_orientation * p_unit_arc)
    

@shape('anchorscad/models/screws/holes/self_tap_hole')
@dataclass
class SelfTapHole(CompositeShape):
    h: float=50
    i_dia: float=24.9 + PRINTER_HOLE_DIALATION
    o_dia: float=i_dia + 10
    bevel_r: float=6
    t_len: float=35
    t_width: float=19.9 + 10
    fn: int=None
    fa: float=None
    fs: float=None
    
    
    EXAMPLE_SHAPE_ARGS=args()
    EXAMPLE_ANCHORS=(surface_args('start'), surface_args('bottom'))
    
    def __post_init__(self):
        pass