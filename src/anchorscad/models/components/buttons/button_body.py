'''
Created on 9 Jan 2022

@author: gianni
'''


import ParametricSolid.core as core
import ParametricSolid.extrude as e
from ParametricSolid.datatree import datatree, Node
import ParametricSolid.linear as l

import anchorscad.models.components.switches.tactile_tl1105 as tactile_switches
import anchorscad.models.components.buttons.button_cap as button_cap

EPSILON=1.0e-3

@core.shape('anchorscad.models.components.buttons.button_body')
@datatree
class ButtonBody(core.CompositeShape):
    '''
    Substrate of a button housing. This can make a fat switch from a
    small tactile switch but retaining the operating ease of the tactile switch.
    
    This is designed as a press fit onto the shaft of a 10-11mm tactile switch.
    The button cap and housing will limit travel and hence the operating forces
    are limited resulting in limited forces on the tactile switch regardless of
    the applied force on the button cap.
    '''
    bottom_plate_height: float=3
    top_plate_height: float=3
    inner_rim_height: float=3
    inner_rim_r: float=15 / 2
    outer_rim_height: float=4
    outer_rim_r: float=17.7 / 2
    outer_r: float=22.5 / 2
    inside_rim_top_r: float=1
    ouside_rim_top_r: float=2
    path: e.Path=None
    extents: tuple=None
    
    metadata: object=core.EMPTY_ATTRS.with_fn(8)
    cage_shape_node: Node=core.ShapeNode(core.Cylinder, {})
    degrees: float=360
    extrude_node: Node=core.ShapeNode(e.RotateExtrude, 'degrees')
    cage_node: Node=core.CageOfNode()
    plate_cage_node: Node=core.CageOfNode(prefix='plate_cage_')
    rim_cage_node: Node=core.CageOfNode(prefix='rim_cage_')
    fn: int=64
    
    EXAMPLE_SHAPE_ARGS=core.args(as_cage=True,
                                plate_cage_as_cage=False, 
                                rim_cage_as_cage=False,
                                degrees=270)
    EXAMPLE_ANCHORS=(core.surface_args('base', scale_anchor=0.5),
                     core.surface_args('plate', scale_anchor=0.5),)
    
    def __post_init__(self):
        
        outside_radius = self.outer_r
        height = (self.top_plate_height
            + self.inner_rim_height
            + self.outer_rim_height)
        
        start_point = [0, -self.bottom_plate_height]
        rim_outer = [self.outer_rim_r,
                     self.top_plate_height + self.inner_rim_height]

        rim_spline1 = [
            [rim_outer[0], height - self.inside_rim_top_r / 3.0],
            [rim_outer[0] + self.inside_rim_top_r / 3.0, height],
            [rim_outer[0] + self.inside_rim_top_r, height]]
        
        rim_spline2 = [
            [rim_outer[0] + self.inside_rim_top_r + self.ouside_rim_top_r / 2.0, 
             height],
            [outside_radius, height - self.ouside_rim_top_r / 2.0],
            [outside_radius, height - self.ouside_rim_top_r]]
        
        path = (e.PathBuilder()
                .move(start_point)
                .line([0, self.top_plate_height], 'plate_centre')
                .line([self.inner_rim_r, self.top_plate_height], 'plate_outer')
                .line([self.inner_rim_r,
                     self.top_plate_height + self.inner_rim_height], 'rim_inner')
                .line(rim_outer, 'rim_outer')
                .spline(rim_spline1, metadata=self.metadata, name='rim1')
                .spline(rim_spline2, metadata=self.metadata, name='rim2')
                .line([outside_radius, - self.bottom_plate_height], 'bottom_outer')
                .line(start_point, 'axis')
                .build())
        self.path = path
        self.extents = path.extents()
        extents = path.extents()
        
        cage_shape = self.cage_shape_node(h=extents[1][1] - extents[0][1], 
                                   r=extents[1][0] - extents[0][0])
        maker = self.cage_node(cage_shape).at('base', post=l.ROTX_180)
        
        cage_plate_shape = self.cage_shape_node(h=self.top_plate_height,
                                                r=self.inner_rim_r)
        maker.add_at(
            self.plate_cage_node(cage_plate_shape, cage_name='plate_cage')
            .at('base'), 'base', h=self.bottom_plate_height)
        
        rim_plate_shape = self.cage_shape_node(h=self.top_plate_height,
                                                r=self.outer_rim_r)
        maker.add_at(
            self.rim_cage_node(rim_plate_shape, cage_name='rim_plate_cage')
            .at('base'), 'plate_cage', 'base', rh=1)

        shape = self.extrude_node(path)
        maker.add_at(shape.solid('body').at('plate_centre'),
                     'base', post=l.ROTY_180 * l.ROTX_90)
        
        self.maker = maker

    @core.anchor('Plate top.')
    def plate(self, *args, **kwds):
        return self.maker.at('plate_cage', 'top', *args, **kwds)


@core.shape('anchorscad.models.components.buttons.button_for_tactile_switch')
@datatree
class ButtonForTactileSwitch(core.CompositeShape):
    '''
    <description>
    '''
    
    leads_as_cages: bool=True
    switch_type: str=None
    tl1105_node: Node=core.ShapeNode(tactile_switches.TactileSwitchTL1105, 
                                     {'leada_node': 'tl1105_leada_node'})
    tl59_node: Node=core.ShapeNode(tactile_switches.TactileSwitchTL59, 
                                     {'leada_node': 'tl59_leada_node'})
    outline_node: Node=core.ShapeNode(tactile_switches.TactileSwitchOutline)
    body_node: Node=core.ShapeNode(ButtonBody, prefix='body_')
    fn: int=64
    
    EXAMPLE_SHAPE_ARGS=core.args(switch_type='TL59')
    EXAMPLE_ANCHORS=tuple()
    
    def __post_init__(self):
        body = self.body_node()
        
        maker = body.solid('body').at()
        
        switch_node = self.select_switch()
        
        switch_shape = self.outline_node(switch_shape=switch_node())
        
        maker.add_at(switch_shape.hole('switch_hole').at('switch_top'),
                     'plate_cage', 'top', post=l.tranZ(EPSILON))
        
        self.maker = maker

    def select_switch(self):
        if self.switch_type is None:
            return self.tl1105_node
        
        if self.switch_type == 'TL59':
            return self.tl59_node
        
        if self.switch_type == 'TL1105':
            return self.tl1105_node
        
        assert False, f'Failed to find switch_type {self.switch_type!r}.'
        

@core.shape('anchorscad.models.components.buttons.button_for_tactile_switch')
@datatree
class ButtonAssemblyTest(core.CompositeShape):
    '''
    <description>
    '''
    
    base_node: Node=core.ShapeNode(ButtonForTactileSwitch)
    cap_node: Node=core.ShapeNode(button_cap.ButtonCap)

    EXAMPLE_SHAPE_ARGS=core.args(switch_type='TL1105',
                                 body_as_cage=True,
                                 body_plate_cage_as_cage=False, 
                                 body_degrees=270, 
                                 ex_degrees=270,
                                 fn=64)
    EXAMPLE_ANCHORS=tuple()
    
    def __post_init__(self):
        
        shape = self.base_node()
        
        maker = shape.solid('base').at(post=l.ROTZ_90)

        switch = shape.select_switch()(leads_as_cages=False)
        
        maker.add_at(switch.solid('switch').colour([1, 0.3, 0.1, 1]).at('switch_base'),
                     'switch_hole', 'switch_base')
        
        cap_shape = self.cap_node()
        
        maker.add_at(cap_shape.solid('cap').colour([1, 0.3, 0.8, 1]).at('base'),
                     'rim_plate_cage', 'base', rh=1, post=l.ROTZ_180)
        
        self.maker = maker


if __name__ == '__main__':
    core.anchorscad_main(False)
