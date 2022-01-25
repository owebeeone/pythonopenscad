'''
Created on 22 Jan 2022

@author: gianni
'''

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import List
from debugpy.common.json import default

@dataclass_json
@dataclass
class RunnerExampleResults(object):
    '''
    Status type for an example of a shape.
    '''
    example_name: str
    error_str: str=None # None indicates no error.
    output_file_name: str=None
    output_file_size: int=None
    error_file_name: str=None
    error_file_size: int=None
    scad_file: str=None
    stl_file: str=None
    png_file: str=None
    graph_file: str=None
    graph_svg_dot_file: str=None
    graph_svg_file: str=None
    shape_pickle_file: str=None
    stl_file: str=None


@dataclass_json
@dataclass
class RunnerShapeResults(object):
    '''
    Status type for a shape class run.
    '''
    class_name: str
    examples_with_error_output: int=0
    example_results: List[RunnerExampleResults]=field(default_factory=list)


@dataclass_json
@dataclass
class RunnerModuleStatus(object):
    '''
    Status type for a module run.
    '''
    module_name: str
    shape_results: List[RunnerShapeResults]
    examples_with_error_output: int

 
@dataclass_json
@dataclass
class RunnerxStatus(object):
    '''
    Status type for a module run.
    '''
    module_name: str
    shape_results: List[RunnerShapeResults]
    examples_with_error_output: int
   

@dataclass_json
@dataclass
class RunnerStatus2(RunnerModuleStatus):
    other_thing: str=None


example = RunnerModuleStatus(
    'mod_name',
    (RunnerShapeResults('shape1', 
                        (RunnerExampleResults('ex_name11', 'sf', 'gf', 'pf', 'stl'),
                         RunnerExampleResults('ex_name12', 'sf', 'gf', 'pf', 'stl'))),
    RunnerShapeResults('shape2', 
                        (RunnerExampleResults('ex_name21', 'sf', 'gf', 'pf', 'stl'),
                         RunnerExampleResults('ex_name22', 'sf', 'gf', 'pf', 'stl'))),
    ),
    examples_with_error_output=0
    )

def main():
    s = example.to_json(indent=4)
    js = RunnerStatus2.from_json(s)

    print(s)
    print(js)
    print(js.to_json(indent=4))

if __name__ == "__main__":
    main()
