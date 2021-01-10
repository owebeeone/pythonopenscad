'''
Created on 4 Jan 2021

@author: gianni
'''

import copy
from dataclasses import dataclass

from ParametricSolid import core
from ParametricSolid import linear as l
import pythonopenscad as posc


class EmptyRenderStack(core.BaseException):
    '''Before adding items to renderer a renderer.push must be called.'''
    
class UnpoppedItemsOnRenderStack(core.BaseException):
    '''Before closing the renderer, all pushed frames must be pop()ed..'''
    
class PopCalledTooManyTimes(core.BaseException):
    '''The render stack ran out of elements to pop..'''

HEAD_CONTAINER=1
SOLID_CONTAINER=2
HOLE_CONTAINER=3

class Container():
    def __init__(self, model):
        self.model = model
        self.containers = {}
        
    def _get_or_create_container(self, container_id):
        if container_id in self.containers:
            return self.containers[container_id]
        result = []
        self.containers[container_id] = result
        return result
        
        
    def _get_container(self, container_id):
        return self.containers.get(container_id, None)

    def add_solid(self, *obj):
        container = self._get_or_create_container(SOLID_CONTAINER)
        container.extend(obj)
        
    def add_hole(self, *obj):
        container = self._get_or_create_container(HOLE_CONTAINER)
        container.extend(obj)
        
    def add_head(self, *obj):
        container = self._get_or_create_container(HEAD_CONTAINER)
        container.extend(obj)
        
    def _combine_solids_and_holes(self):
        holes = self._get_container(HOLE_CONTAINER)
        solids = self._get_container(SOLID_CONTAINER)
        
        if holes:
            if not solids:
                solid_obj = self.model.Union()
            elif len(solids) == 1:
                solid_obj = solids[0]
            else:
                solid_obj = self.model.Union()(*solids)
                
            return [self.model.Difference()(solid_obj, *holes)]
        
        # No holes.
        if solids:
            return solids
        return []
    
    def _combine_heads(self, heads):
        
        top_head = None
        last_head = None
        if heads:
            top_head = heads[0]
            last_head = heads[-1]
            for i in range(len(heads)-1):
                heads[i].append(heads[i + 1])
                
        return top_head, last_head
    
    def build_combine(self):
        '''Combines solids and holes if any and returns the representative list of objects.'''
        top_head, last_head = self._combine_heads(heads=self._get_container(HEAD_CONTAINER))

        if not top_head:
            return self._combine_solids_and_holes()
        
        last_head.append(*self._combine_solids_and_holes())
        return [top_head]
    
    def build_composite(self):
        '''Returns a list of solids and holes.'''
        
        holes = self._get_or_create_container(HOLE_CONTAINER)
        solids = self._get_or_create_container(SOLID_CONTAINER)    
        
        head_copies = [None, None]
        
        heads = self._get_container(HEAD_CONTAINER)
        if heads:
            if holes and solids:
                head_copies[0] = self._combine_heads(copy.deepcopy(heads))
                head_copies[0][1].append(*solids)
                head_copies[1] = self._combine_heads(heads)
                head_copies[1][1].append(*holes)
            elif holes:
                head_copies[1] = self._combine_heads(heads)
                head_copies[1][1].append(*holes)
            elif solids:
                head_copies[0] = self._combine_heads(heads)
                head_copies[0][1].append(*solids)
            else:
                return [], []
                
            return [head_copies[0][0]], [head_copies[0][0]]
        else:
            return solids, holes
    
    def get_or_create_first_head(self):
        heads = self._get_or_create_container(HEAD_CONTAINER)
        if not heads:
            head = self.model.Union()
            self.add_head(head)
        return heads[0]
    
    def close(self, mode, parent_container):
        if mode.mode == core.ModeShapeFrame.SOLID.mode:
            solids = self.build_combine()
            parent_container.add_solid(*solids)
        elif mode.mode == core.ModeShapeFrame.HOLE.mode:
            holes = self.build_combine()
            parent_container.add_hole(*holes)
        elif mode.mode == core.ModeShapeFrame.COMPOSITE.mode:
            solids, holes = self.build_composite()
            parent_container.add_solid(*solids)
            parent_container.add_hole(*holes)
        elif mode.mode == core.ModeShapeFrame.CAGE.mode:
            pass # Drop these. They are not part of the model.
            

@dataclass(frozen=True)
class ContextEntry():
    
    container: Container
    mode: core.ModeShapeFrame
    reference_frame: l.GMatrix
    attributes: core.ModelAttributes = None

class Context():

    def __init__(self, renderer):
        self.stack = [] # A stack of ContextEntry
        self.renderer = renderer
        self.model = renderer.model
        
    def push(self, 
             mode: core.ModeShapeFrame, 
             reference_frame: l.GMatrix, 
             attributes: core.ModelAttributes):
        container = Container(model=self.model)
        last_attrs = self.get_last_attributes()
        merged_attrs = last_attrs.merge(attributes)
        diff_attrs = last_attrs.diff(merged_attrs)
        
        entry = ContextEntry(container, mode, reference_frame, merged_attrs)

        self.stack.append(entry)
        
        if reference_frame:
            container.add_head(self.model.Multmatrix(reference_frame.m.A))
            
        if diff_attrs.colour:
            container.add_head(self.model.Color(c=diff_attrs.colour.value))
        
        if diff_attrs.disable:
            head = container.get_or_create_first_head()
            head.add_modifier(self.model.DISABLE)
        if diff_attrs.show_only:
            head = container.get_or_create_first_head()
            head.add_modifier(self.model.SHOW_ONLY)
        if diff_attrs.debug:
            head = container.get_or_create_first_head()
            head.add_modifier(self.model.DEBUG)
        if diff_attrs.transparent:
            head = container.get_or_create_first_head()
            head.add_modifier(self.model.TRANSPARENT)
            
    def pop(self):
        last = self.stack[-1]
        del self.stack[-1]
        if self.stack:
            last.container.close(last.mode, self.stack[-1].container)
            return None
        else:
            objs = last.container.build_combine()
            if not objs:
                return self.model.Union()
            if len(objs) > 1:
                return self.model.Union().append(*objs)
            return objs[0]
            

    def get_last_attributes(self):
        if self.stack:
            attrs = self.stack[-1].attributes
            if not attrs is None:
                return attrs
        return core.EMPTY_ATTRS
        
    
    def get_last_container(self):
        if not self.stack:
            raise EmptyRenderStack('renderer stack is empty.')
        return self.stack[-1].container
        
class Renderer():
    '''Provides renderer machinery for ParametricSolid. Renders to PythonOpenScad models.'''
    model = posc
    
    def __init__(self, initial_frame=None, initial_attrs=None):
        self.context = Context(self)
        self.result = None
        # Push an item on the stack that will collect the final objects.
        self.context.push(core.ModeShapeFrame.SOLID, initial_frame, initial_attrs)
        
    def close(self):
        count = len(self.context.stack)
        if count != 1:
            raise UnpoppedItemsOnRenderStack(
                f'{count - 1} items remain on the render stack.')
        self.result = self.context.pop()
        self.context = Context(self) # Prepare for the next object just in case.
        return self.result

    def push(self, mode, reference_frame, attributes):
        self.context.push(mode, reference_frame, attributes)

    def pop(self):
        self.context.pop()
        # The last item on the stack is for office us only.
        if len(self.context.stack) < 1:
            raise PopCalledTooManyTimes('pop() called more times than push() - stack underrun.')
        
    def get_current_attributes(self):
        return self.context.get_last_attributes()
        
    def add(self, *object):
        self.context.get_last_container().add_solid(*object)


def render(shape, initial_frame=None, initial_attrs=None):
    '''Renders a shape and returns the model root object.'''
    renderer = Renderer(initial_frame, initial_attrs)
    shape.render(renderer)
    return renderer.close()

