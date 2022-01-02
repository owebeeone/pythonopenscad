'''
Created on 31 Dec 2021

@author: gianni
'''
import cgi
from dataclasses import dataclass, field
import re

try:
    import graphviz
    _graphviz_imported = True
except:
    _graphviz_imported = False
    
_num = 0

def _num_gen():
    global _num
    _num += 1
    return str(_num)
    

@dataclass
class Node:
    label: str
    ident: str=field(default_factory=_num_gen)
    
    def __repr__(self):
        return f'{self.get_id()} [label="{self.get_label()}"];'
    
    def get_label(self):
        l = str(self.label)
        return re.escape(l)
    
    def get_id(self):
        label = self.label
        if isinstance(label, tuple):
            label = label[0]
        if isinstance(label, str) and label.isidentifier():
            return f'{label}_{self.ident}'
        return self.ident
        
@dataclass
class Edge:
    start: Node
    end: Node
        
    def __repr__(self):
        return f'{self.start.get_id()} -> {self.end.get_id()};'
    
    
@dataclass
class DirectedGraph:
    '''
    classdocs
    '''

    nodes: list=field(default_factory=lambda:list())
    edges: list=field(default_factory=lambda:list())
    
    def new_node(self, label):
        node = Node(label)
        self.nodes.append(node)
        return node
    
    def add_edge(self, start, end):
        self.edges.append(Edge(start, end))
    
    def get_last_node(self):
        return self.nodes[-1]
        
    def __repr__(self):
        return self.dump('D')
        
    def dump(self, name):
        nodes_str = '\n'.join(f'    {n}' for n in self.nodes)
        edges_str = '\n'.join(f'    {e}' for e in self.edges)
        return '\n'.join((f'digraph {name} {{', 
                          nodes_str,
                          edges_str,
                          '}\n')) 
    
    def write(self, filename, name='D'):
        '''Writes the GraphViz syntax to the given file name.
        Args:
            filename: The filename to create.
        '''
        with open(filename, 'w') as fp:
            fp.write(self.dump(name))  
            
    def write_svg(self, filename, name='D'):  
        '''Writes an SVG and DOT files to the given file name.
        Args:
            filename: The DOT filename to create, with SVG created
            by appending ".svg" to this filename.
        '''
        if not _graphviz_imported:
            raise Exception('Unable to generate SVG file. '
                            'GraphViz must be installed. '
                            'To install, run "pip3 install graphviz" in shell. ')
        dot = graphviz.Digraph(name=name)
        for node in self.nodes:
            label = node.get_label()
            escape_label = cgi.html.escape(label)
            url=f'javascript:alert(&quot;{escape_label}&quot;)'
            dot.node(node.get_id(), label, href=url)
        for edge in self.edges:
            dot.edge(edge.start.get_id(), edge.end.get_id())
        
        dot.render(filename, format='svg')
    
    
    
    
    

