'''
Created on 30 Sep 2021

@author: gianni
'''

from dataclasses import dataclass
import ParametricSolid.core as core
import ParametricSolid.linear as l


@core.shape('anchorscad.models.quilting.machne.SpoolHolder')
@dataclass
class SpoolHolder(core.CompositeShape):
    '''
    <description>
    '''
    h: float=11.3
    shaft_r: float=6.0 / 2
    shrink_r: float=0.21 / 2
    washer_h: float=3
    washer_r: float=23.0
    stem_top_rd: float=1.5
    stem_base_rd: float=2.5
    fn: int=64
    
    EXAMPLE_SHAPE_ARGS=core.args()
    NOEXAMPLE_ANCHORS=(core.surface_args('base'),)
    
    def __post_init__(self):
        maker = core.Cylinder(h=self.h, r=self.shaft_r, fn=self.fn).hole(
            'shaft').colour([1, 1, 0, 0.5]).at('base')
        
        washer_shape = core.Cylinder(
            h=self.washer_h, r=self.washer_r, fn=self.fn).solid(
            'washer').colour([0, 1, 0, 0.5]).at('base')
            
        maker.add_at(washer_shape, 'base')
        
        stem_shape = core.Cone(
            h=self.h - self.washer_h,
            r_top=self.stem_top_rd + self.shaft_r, 
            r_base=self.stem_base_rd + self.shaft_r, 
            fn=self.fn).solid(
            'stem').colour([0, 0.5, 1, 0.5]).at('base')
            
        maker.add_at(stem_shape, 'washer', 'base', rh=1)
        
        self.maker = maker

    @core.anchor('An example anchor specifier.')
    def side(self, *args, **kwds):
        return self.maker.at('face_edge', *args, **kwds)

if __name__ == '__main__':
    core.anchorscad_main(False)
