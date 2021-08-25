'''
Created on 25 Jan 2021

@author: gianni
'''

from dataclasses import dataclass

@dataclass(frozen=True)
class ShaftDimensions(object):
    '''Contains diameter dimensions for a screw type.
    '''

    actual: float
    thru_d: float
    tapping_d: float
    
SHAFT_MAP = {
    'M2.6' : ShaftDimensions(2.6, 2.8, 2.61)
    }


@dataclass(frozen=True)
class HeadDimensions(object):
    '''Contains dimensions for a screw type.
    '''

    head_top_d: float
    head_bot_d: float
    head_protrusion_height: float
    head_mid_depth: float
    head_countersink_depth: float
    
    def overall_screw_head_height(self):
        return self.head_countersink_depth + self.head_mid_depth; 


