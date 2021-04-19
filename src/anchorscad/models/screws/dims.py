'''
Created on 25 Jan 2021

@author: gianni
'''

from dataclasses import dataclass

from ParametricSolid.core import shape, CompositeShape
from anchorscad.models.basic.pipe import Pipe

@dataclass(frozen=True)
class HoleDimensions(object):
    '''Contains dimensions screw holes.
    '''
    thru_dia: float
    tap_dia: float

@dataclass(frozen=True)
class ShaftDimensions(object):
    '''Contains dimensions for a screw type.
    '''

    head_dia: float
    head_height: float


M_HOLE = {
    2 : HoleDimensions(thru_dia=2.22, tap_dia=2.09),
    2.5 : HoleDimensions(thru_dia=2.70, tap_dia=2.65),
    2.6 : HoleDimensions(thru_dia=2.80, tap_dia=2.67),
    3 : HoleDimensions(thru_dia=3.25, tap_dia=3.05),
    }

def holeMetricDims(m_size):
    return M_HOLE[m_size]