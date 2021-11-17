'''
Created on 16 Nov 2021

@author: gianni
'''

from dataclasses import dataclass

import ParametricSolid.core as core
from ParametricSolid.linear import tranY, tranZ, ROTX_180, ROTX_270, \
    ROTX_90, ROTY_270, ROTY_180, translate, GVector, IDENTITY, \
    plane_line_intersect
import anchorscad.models.basic.box_side_bevels as bbox
from anchorscad.models.basic.TriangularPrism import TriangularPrism
from anchorscad.models.grille.case_vent.basic import RectangularGrilleHoles
from anchorscad.models.fastners.snaps import Snap
from anchorscad.models.vent.fan.fan_vent import FanVent
from anchorscad.models.screws.screw_tab import ScrewTab
import anchorscad.models.cases.outline_tools as ot 
from time import time
from anchorscad.models.cases.rpi4.rpi4_outline import RaspberryPi4Outline

# The time of life.
MODEL_V0=1632924118

DELTA=ot.DELTA


@core.shape('anchorscad/models/cases/rpi_case')
@dataclass
class RaspberryPiCase(core.CompositeShape):
    '''A Generic Raspberry Pi Case.'''
    outline_model: core.Shape=None
    outline_model_class: type=RaspberryPi4Outline
    inner_size_delta: tuple=(3, 2, 22)
    inner_offset: tuple=(-1.5, 1, 3)
    wall_thickness: float=2
    inner_bevel_radius: float=None
    screw_clearannce: float=0.2
    board_screw_min_len: float=6
    board_screw_size: float=2.6
    front_flange_depth: float=20
    vent_hole: tuple= (50, 10)
    show_outline: bool=False
    show_cut_box: bool=False
    make_case_top: bool=False
    rhs_grille_size: float=9
    rhs_grille_y_offs: float=4
    fastener_side: core.Shape=Snap(size=(15, 9.5, 3))
    fastener_rear: core.Shape=Snap(size=(15, 9.5, 4))
    snap_pry_hole_size: tuple=(10, wall_thickness * 0.75, 1.7)
    epsilon: float=0.01
    upper_fan: object=FanVent(grille_as_cutout=True,
                              vent_thickness=wall_thickness + epsilon,
                              screw_hole_extension=wall_thickness-0.5)
    version: object=core.Text(
        text=f'-{int(time())-1632924118:X}', 
        size=5, 
        depth=0.3 if wall_thickness > 0.5 else wall_thickness * 0.5)
    do_versioned_example: bool=False
    split_box_delta: float=40
    fn: int=None
    fa: float=None
    fs: float=None
    
    EXAMPLE_ANCHORS=(core.surface_args('shell', 'face_centre', 1),)
    EXAMPLE_SHAPE_ARGS=core.args(fn=36)
    
    # Some anchor locations for locating flange position and sizes.
    USBA2_A2 = core.surface_args(
        'outline', ('usbA2', 'outer'), 'face_edge', 1, 0, 0)
    USBA3_A1 = core.surface_args(
        'outline', ('usbA3', 'outer'), 'face_edge', 1, 0, 1)
    USBA3_A2 = core.surface_args(
        'outline', ('usbA3', 'outer'), 'face_edge', 1, 0, 0)
    ETH_A1 = core.surface_args(
        'outline', ('rj45', 'outer'), 'face_edge', 1, 0, 1)
    BOUND_LINES = (USBA2_A2, USBA3_A1, USBA3_A2, ETH_A1)
    
    BOX_TOP = core.surface_args('inner', 'face_centre', 4)
    CUT_PLANE = core.surface_args('outline', 'audio', 'base', post=ROTX_270)
    
    
    HEADER_CORNER = core.surface_args(
        'outline', 'header100', 'face_edge', 3, 0, 0.5,
        post=translate([0, -rhs_grille_y_offs, 0]))
    
    BOX_RHS = core.surface_args('shell_centre', 'face_centre', 3)
    BOX_LHS = core.surface_args('shell_centre', 'face_centre', 0)
    
    SNAP_RHS = core.surface_args(
        'shell_centre', 'face_edge', 3, 0, 0.88)
    SNAP_LHS = core.surface_args(
        'shell_centre', 'face_edge', 0, 2, 1 - 0.88)
    SNAP_REAR_LHS = core.surface_args(
        'shell_centre', 'face_edge', 2, 2, 0.19)
    SNAP_REAR_RHS = core.surface_args(
        'shell_centre', 'face_edge', 2, 2, 1 - 0.19)
    SNAP_ANCHOR=core.surface_args('snap', post=translate((0, -1, -0.3)))
    
    FAN_FIXING_PLANE=core.surface_args(
        'shell_centre', 'face_centre', 4)
    
    PRY_RHS = core.surface_args(
        'shell', 'face_edge', 3, 0, 0.7, post=tranZ(epsilon) * ROTY_180)
    PRY_REAR = core.surface_args(
        'shell', 'face_edge', 2, 2, 0.5, post=tranZ(epsilon) * ROTY_180)
    
    FAN_POSITION=core.surface_args(
        'outline', 'cpu', 'face_centre', 1, post=translate([-6, -2, 0]))
    
    TAB_RHS = core.surface_args(
        'shell', 'face_edge', 3, 2, 1 - 0.8)
    TAB_LHS = core.surface_args(
        'shell', 'face_edge', 0, 0, 0.8)
    TAB_REAR_LHS = core.surface_args(
        'shell', 'face_edge', 2, 0, 0.2)
    TAB_REAR_RHS = core.surface_args(
        'shell', 'face_edge', 2, 0, 0.8)
    
    VERS_UPPER = core.surface_args(
        'shell', 'face_edge', 4, 1, 0.15, post=translate([0, 2, epsilon]))
    VERS_LOWER = core.surface_args(
        'shell', 'face_edge', 1, 1, 0.15, post=translate([0, 2, epsilon]))
    
    EXAMPLES_EXTENDED={'bottom': core.ExampleParams(
                            shape_args=core.args(fn=36)),
                       'top': core.ExampleParams(
                            core.args(make_case_top=True, 
                                      fn=36),
                            anchors=())}

    def __post_init__(self):
        params = core.non_defaults_dict(self, include=('fn', 'fa', 'fs'))
        if self.outline_model is None:
            self.outline_model = self.outline_model_class(**params)
        if self.inner_bevel_radius is None:
            self.inner_bevel_radius = self.outline_model.bevel_radius + (-self.inner_offset[0] - self.inner_offset[1]) / 2
        inner_size = GVector(self.inner_size_delta) + GVector(self.outline_model.board_size)
        outer_size = (inner_size + (self.wall_thickness * 2,) * 3).A[0:3]
        bevel_radius = self.inner_bevel_radius + self.wall_thickness
        maker = bbox.BoxShell(
            size=outer_size, 
            bevel_radius=bevel_radius, 
            shell_size=self.wall_thickness, 
            **params).solid('shell').at('face_centre', 4)
        
        
        maker.add_at(self.outline_model.hole('outline').at('face_corner', 5, 0),
                     'inner', 'face_corner', 5, 0, pre=translate(self.inner_offset))

        split_box_cage = core.Box(outer_size).cage('split_box_cage').at('centre')
        split_box_size = outer_size + self.split_box_delta
        split_box = core.Box(split_box_size).solid('split_box').at('centre')
        split_box_cage.add(split_box)
        
        cut_point = (ROTX_90 * maker.at('outline', 'audio', 'base')).get_translation()
        cut_ref = maker.at('inner', 'face_centre', 4).get_translation()
        cut_xlation = cut_point - cut_ref
        
        cut_xform = IDENTITY if self.make_case_top else ROTX_180 
            
        cut_box_mode = core.ModeShapeFrame.HOLE if self.show_cut_box else core.ModeShapeFrame.HOLE
        
        maker.add_at(
            split_box_cage
                .named_shape('split_box', cut_box_mode)
                .transparent(self.show_cut_box)
                .at('split_box', 'face_centre', 4), 
            'face_centre', 4, post=tranZ(-cut_xlation.y) * cut_xform)
        
        # Adds a flange to support the thin columns at the front of the
        # case. Here we project some lines from the edges of the USB
        # and RJ45 connector expanded access holes to the cut line
        # and then to the top of the case. This uses the intersecting
        # points between the top and bottom planes to find the dimensions
        # of the flange.
        support_bound_planes = (self.BOX_TOP, self.CUT_PLANE)
        support_bound_lines = self.BOUND_LINES
        
        top_points = self.find_all_intersect(
            maker, support_bound_planes[0], *support_bound_lines)
        
        bottom_points = self.find_all_intersect(
            maker, support_bound_planes[1], *support_bound_lines)
        
        face_top_locs = []
        for i, m in enumerate(top_points):
            v = m.I * GVector([0, 0, 0,])
            face_top_locs.append(v)
        
        face_bot_locs = []
        for i, m in enumerate(bottom_points):
            v = m.I * GVector([0, 0, 0,])
            face_bot_locs.append(v)
            
        usb_usb_flange = self.make_flange(
            (face_top_locs[1] - face_top_locs[0]).x,
            (face_bot_locs[0] - face_top_locs[0]).z + self.wall_thickness)
        
        
        usb_rj45_flange = self.make_flange(
            (face_top_locs[3] - face_top_locs[2]).x,
            (face_bot_locs[2] - face_top_locs[2]).z + self.wall_thickness)
        
        maker.add_at(usb_usb_flange.solid('usb_usb_flange')
                     .at('prism', 'face3', 1),
                     post=top_points[0] * ROTY_270 * ROTX_90)
        
        maker.add_at(usb_rj45_flange.solid('usb_rj45_flange')
                     .at('prism', 'face3', 1),
                     post=top_points[2] * ROTY_270 * ROTX_90)
        
        # Add air grilles
        
        grille_holes = RectangularGrilleHoles(
            [50, self.wall_thickness + 0.01, self.rhs_grille_size])
        
        maker.add_at(grille_holes.hole('rhs_grille').at('centre', post=ROTX_90),
                     post=plane_line_intersect(
                         core.apply_anchor_args(maker, self.BOX_RHS),
                         core.apply_anchor_args(maker, self.HEADER_CORNER)
                         ))
        
        maker.add_at(grille_holes.hole('lhs_grille').at('centre', post=ROTX_90),
                     post=plane_line_intersect(
                         core.apply_anchor_args(maker, self.BOX_LHS),
                         core.apply_anchor_args(maker, self.HEADER_CORNER)
                         ))
        
        bottom_loc = maker.at('shell', 'face_centre', 1).get_translation()
        screw_hole_loc = maker.at('outline', ('mount_hole', 0), 'top').get_translation()
        screw_hole_top_loc = maker.at('outline', ('mount_hole', 0), 'base').get_translation()
        max_allowable_screw_hole_height = screw_hole_loc.z - bottom_loc.z - self.screw_clearannce
        max_allowable_screw_size = screw_hole_top_loc.z - bottom_loc.z - self.screw_clearannce
        
        assert max_allowable_screw_size >= self.board_screw_min_len, (
            f'Board mounting screw hole height {max_allowable_screw_size} is smaller than the '
            f'mnimum size {self.board_screw_min_len}.')
        
        # Add Fan
        
        fan_fix_plane = core.apply_anchor_args(maker, self.FAN_FIXING_PLANE)
        fan_fix_pos = core.apply_anchor_args(maker, self.FAN_POSITION)
        
        fan_pos = plane_line_intersect(fan_fix_plane, fan_fix_pos)
        
        maker.add_at(self.upper_fan.composite('fan')
                     .at('grille_centre'),
                     post=fan_pos)
        
        # Add screw holes.
        
        params = core.non_defaults_dict(self, include=('fn', 'fa', 'fs'))

        for i, t in enumerate(self.outline_model.HOLE_POSITIONS):
            board_screw_hole = t.spec.screw_hole(
                tap_len=max_allowable_screw_hole_height -1,
                dia=self.board_screw_size, 
                thru_len=1,
                params=params)
            
            maker.add_at(board_screw_hole
                         .composite(('support', i))
                         .at('start', post=ROTX_180),
                         'outline', ('mount_hole', i), 'top')
            
        # Add mounting screw tabs.
        
        tab_anchors = (self.TAB_RHS, 
                        self.TAB_LHS, 
                        self.TAB_REAR_RHS, 
                        self.TAB_REAR_LHS)
        tab_trans = ROTY_180
        tab_shape = ScrewTab()
        for i, a in enumerate(tab_anchors):
            maker.add_at(tab_shape
                    .composite(('tab', i))
                    .at('face_edge', 0, 0),
                    post=core.apply_anchor_args(maker, a) * tab_trans)
        
     
        top_maker = maker.solid('main').at('centre')
        self.maker = top_maker
        
        if self.show_outline:
            top_maker.add_at(
                self.outline_model
                    .solid('outline2')
                    .transparent(True)
                    .at('centre'),
                'main', 'outline', 'centre')
            
        # Add fasteners.
        fastener_mode = (core.ModeShapeFrame.SOLID 
                         if self.make_case_top 
                         else core.ModeShapeFrame.HOLE)
        
        clip_anchors = ((self.fastener_side, self.SNAP_RHS), 
                        (self.fastener_side, self.SNAP_LHS), 
                        (self.fastener_rear, self.SNAP_REAR_RHS), 
                        (self.fastener_rear, self.SNAP_REAR_LHS))
        cut_trans = tranY(-cut_xlation.y)
        for i, fa in enumerate(clip_anchors):
            f, a = fa
            top_maker.add_at(f
                    .named_shape(('clip', i), fastener_mode)
                    .at(*self.SNAP_ANCHOR[1][0], **self.SNAP_ANCHOR[1][1]),
                    post=core.apply_anchor_args(top_maker, a) * -cut_trans)
        
        # Add pry holes
        pry_shape = core.Box(self.snap_pry_hole_size)
        pry_anchors = (self.PRY_RHS, self.PRY_REAR)
        for i, a in enumerate(pry_anchors):
            top_maker.add_at(pry_shape
                    .hole(('pry', i))
                    .at('face_centre', 0),
                    post=core.apply_anchor_args(top_maker, a) * -cut_trans)
            
        # Add version text
        text_anchor, text_name = ((self.VERS_UPPER, 'upper') 
                       if self.make_case_top 
                       else (self.VERS_LOWER, 'lower'))
        top_maker.add_at(self.version
                .hole((('version', text_name), i))
                .at('default', rd=0.4),
                post=core.apply_anchor_args(top_maker, text_anchor))
        
            
    def make_flange(self, width, height):
        return TriangularPrism([
            self.front_flange_depth,
            height,
            width])    
        
    
    def find_all_intersect(self, maker, plane_anchor, *line_anchors):
        return tuple(self.find_intersection(maker, plane_anchor, la) 
                     for la in line_anchors)
    
    def find_intersection(self, maker, plane_anchor, line_anchor):
        plane = core.apply_at_args(
            maker, *plane_anchor[1][0], **plane_anchor[1][1])
        line = core.apply_at_args(
            maker, *line_anchor[1][0], **line_anchor[1][1])
        return plane_line_intersect(plane, line)

    def example_version(self):
        return self.version.text if self.do_versioned_example else None

if __name__ == "__main__":
    core.anchorscad_main(False)
