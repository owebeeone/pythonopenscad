'''
Created on 25 Jan 2021

@author: gianni
'''

from dataclasses import dataclass

@dataclass(frozen=True)
class Dimensions(object):
    '''Contains dimensions for a screw type.
    '''

    thru_dia: float
    tapping_dia: float
    head_dia: float
    head_height: float


