'''
Created on 6 Jan 2022

@author: gianni
'''

import numpy as np
import ParametricSolid.core as core
from ParametricSolid.datatree import datatree, Node
import ParametricSolid.extrude as extrude
import ParametricSolid.linear as l


@core.shape('anchorscad.models.components.buttons.ButtonCap')
@datatree
class ButtonCap(core.CompositeShape):
    '''
    <description>
    '''
    r: float=17.4
    h: float=5.7
    shaft_diameter: float=3.09
    shaft_taper: tuple=(0.3, 0.8)
    shaft_height: float=4.5
    edge_height: float=4.5
    rim_radius: float=0.75
    bottom_flange: tuple=(0.3, 0.4)
    bc_cage_shape: Node=core.ShapeNode(core.Cylinder, 'r', 'h')
    cageof_node: Node=Node(core.cageof, prefix='bc_cage_')
    spline1_meta_data: object=core.ModelAttributes().with_fn(15)
    spline2_meta_data: object=core.ModelAttributes().with_fn(5)
    extruder: Node=core.ShapeNode(extrude.RotateExtrude, {})
    
    EXAMPLE_SHAPE_ARGS=core.args(fn=128, bc_cage_as_cage=False)
    EXAMPLE_ANCHORS=((core.surface_args('cage', 'base'),
                      core.surface_args('cage', 'top'),))
    
    def __post_init__(self):
        start_point = [self.shaft_diameter / 2.0 + self.shaft_taper[0], 0]
        end_taper_point = [self.shaft_diameter / 2.0, self.shaft_taper[1]]
    
        shaft_height = self.shaft_height
        end_shaft_point1 = [self.shaft_diameter / 2.0, shaft_height]
        end_shaft_point2 = [0, shaft_height]
    
        top_point = [0, self.h]
    
        top_spline_points = [
            [self.r / 2, self.h],
            [self.r / 2, self.h],
            [self.r - self.rim_radius / 2, self.edge_height]]
    
        cp1 = np.array(top_spline_points[1])
        ep1 = np.array(top_spline_points[2])
        dir_vec = ep1 - cp1
        direction = dir_vec / np.linalg.norm(dir_vec)
        
        top_spline_to_rim_spline_tangent = [
            direction * self.rim_radius / 2 + ep1,
            [self.r, self.edge_height - self.rim_radius / 2],
            [self.r, self.edge_height - self.rim_radius]]
        
        rim_bottom_edge = [self.r, self.bottom_flange[1]]
        rim_bottom = [self.r - self.bottom_flange[0], 0]
        
        path = (extrude.PathBuilder()
            .move(start_point)
            .line(end_taper_point, 'shaft_taper')
            .line(end_shaft_point1, 'shaft_side')
            .line(end_shaft_point2, 'shaft_top')
            .line(top_point, 'centre_line')
            .spline(top_spline_points,
                          name='top_part1',
                          metadata=self.spline1_meta_data)
            .spline(top_spline_to_rim_spline_tangent,
                          name='top_part2',
                          metadata=self.spline2_meta_data)
            .line(rim_bottom_edge, 'outer')
            .line(rim_bottom, 'outer_to_base')
            .line(start_point, 'base')).build()
            
        shape = self.extruder(path)
        maker = shape.solid('button_cap').at('centre_line', 1.0, post=l.ROTX_270)
        maker.add_at(self.cageof_node().at('top'), 
                     'centre_line', 1.0, post=l.ROTX_270)
        
        self.maker = maker

if __name__ == '__main__':
    core.anchorscad_main(False)
