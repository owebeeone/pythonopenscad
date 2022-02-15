'''
Created on 12 Feb 2022

@author: gianni
'''

import anchorscad as ad
EPSILON=1.0e-3

@ad.shape('anchorscad.models.basic.SquarePipe')
@ad.datatree
class SquarePipe(ad.CompositeShape):
    '''
    Pipe consisting of an outer box with an inner box removed.
    '''
    size: tuple
    wall_size: float=5
    
    EXAMPLE_SHAPE_ARGS=ad.args((70, 50, 30))
    EXAMPLE_ANCHORS=(ad.surface_args('face_centre', 5),
                     ad.surface_args('inner', 'face_centre', 2),)
    
    def __post_init__(self):
        maker = ad.Box(self.size).solid('outer').at('centre')
        inner_size = (self.size[0] - 2 * self.wall_size,
                      self.size[1] - 2 * self.wall_size,
                      self.size[2] + EPSILON)
        maker.add_at(ad.Box(inner_size).hole('hole').at('centre'),
                     'centre')
        
        self.maker = maker

    @ad.anchor('Inner hole anchor.')
    def inner(self, *args, **kwds):
        return self.maker.at('hole', *args, **kwds) * ad.ROTX_180


@ad.shape('anchorscad.models.basic.SquarePipeTee')
@ad.datatree
class SquarePipeTee(ad.CompositeShape):
    '''
    A tee of SquarePipe.
    '''
    size: tuple
    wall_size: float=5

    EXAMPLE_SHAPE_ARGS=ad.args((50, 50, 35))
    
    def __post_init__(self):
        
        shape = SquarePipe(self.size, self.wall_size)

        maker = shape.solid('LHS').at('face_centre', 1)
        
        maker.add_at(shape.solid('RHS').at('face_centre', 1),
                     'face_centre', 1, post=ad.ROTX_180)
        
        maker.add_at(shape.solid('stem').at('face_centre', 1),
                     'face_centre', 1, post=ad.ROTX_90)
        
        self.maker = maker

if __name__ == '__main__':
    ad.anchorscad_main(False)
