'''
Created on 11 Jan 2022

@author: gianni
'''

from dataclasses import dataclass

@dataclass(frozen=True)
class StlRecipeArgs:
    '''Parameters for generating STL files.'''
    enable: bool=True


@dataclass(frozen=True)
class ImageRecipeArgs:
    '''Parameters for generating image files.'''
    enable: bool=True
    imgsize: tuple=(1280, 1024)


@dataclass(frozen=True)
class GraphRecipeArgs:
    '''Parameters for generating image files.'''
    enable: bool=True
    enable_svg: bool=True


@dataclass(frozen=True)
class Recipe:
    '''The shape to 'fabricate' '''
    shape: object=None
    anchor: object=None
    place_at: object=None


@dataclass
class Fabricator:
    '''
    Data class of collective fabricator parameters.
    '''
    recipies: tuple=()
    stl_args: StlRecipeArgs=StlRecipeArgs(True)
    image_args: ImageRecipeArgs=ImageRecipeArgs(imgsize=(1280, 1024))
    graph_args: GraphRecipeArgs=GraphRecipeArgs()
    file_basename: str=None # Use the class name if unset.
