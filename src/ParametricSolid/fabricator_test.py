'''
Created on 24 Nov 2021

@author: gianni
'''


import ParametricSolid.linear as l
import ParametricSolid.core as core
from ParametricSolid.datatree import datatree, Node
from ParametricSolid.fabricator import Fabricator, Recipe, \
    StlRecipeArgs, ImageRecipeArgs


@datatree
@core.fabricator
class TestBoxFabricator(Fabricator):
    '''
    Example fabricator to generate stl and png image.
    '''
    file_basename: str=None  # Use the class name if unset.
    stl_args: StlRecipeArgs=StlRecipeArgs(True)
    image_args: ImageRecipeArgs=ImageRecipeArgs(imgsize=(1280, 1024))

    def __post_init__(self):
        self.recipies = (
            Recipe(
                shape=core.Box((10, 20, 30)),
                anchor=core.surface_args(
                            'face_centre', 0, post=l.tranX(40)),
                place_at=core.surface_args(
                            'plate', 100, 100)
                ),
            Recipe(
                shape=core.Box((10, 20, 30)),
                anchor=core.surface_args(
                            'face_centre', 3, post=l.tranX(40)),
                place_at=core.surface_args(
                            'plate', 100, -100)
                ),
            )

    
if __name__ == "__main__":
    core.anchorscad_main(False)
