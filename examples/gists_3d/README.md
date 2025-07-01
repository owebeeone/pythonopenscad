# 3D Shapes

These gists demonstrate the creation of 3D shapes using the pythonopenscad library.

## Cube
    
### Python Code:
```python
from pythonopenscad import Cube

MODEL = Cube(size=[10, 15, 5], center=True)
```
    
### OpenSCAD Code:
```js
cube(size=[10.0, 15.0, 5.0], center=true);

```
    
### How to run this example in a viewer:
```bash
python -m pythonopenscad.examples.gists_3d.cube_example --view
```

### Image:
![Cube](cube_example.py.png)
     
## Sphere
    
### Python Code:
```python
from pythonopenscad import Sphere

MODEL = Sphere(r=10, _fn=128)
```
    
### OpenSCAD Code:
```js
sphere(r=10.0, $fn=128);

```
    
### How to run this example in a viewer:
```bash
python -m pythonopenscad.examples.gists_3d.sphere_example --view
```

### Image:
![Sphere](sphere_example.py.png)
     
## Cylinder
    
### Python Code:
```python
from pythonopenscad import Cylinder

MODEL = Cylinder(h=20, r1=5, r2=10, center=True, _fn=128)
```
    
### OpenSCAD Code:
```js
cylinder(h=20.0, r1=5.0, r2=10.0, center=true, $fn=128);

```
    
### How to run this example in a viewer:
```bash
python -m pythonopenscad.examples.gists_3d.cylinder_example --view
```

### Image:
![Cylinder](cylinder_example.py.png)
     
## Polyhedron
    
### Python Code:
```python
from pythonopenscad import Polyhedron

MODEL = Polyhedron(
    points=[[10,10,0],[10,-10,0],[-10,-10,0],[-10,10,0], [0,0,10]],
    faces=[[0,1,4],[1,2,4],[2,3,4],[3,0,4], [3,2,1,0]]
)
```
    
### OpenSCAD Code:
```js
polyhedron(points=[[10.0, 10.0, 0.0], [10.0, -10.0, 0.0], [-10.0, -10.0, 0.0], [-10.0, 10.0, 0.0], [0.0, 0.0, 10.0]], faces=[[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [3, 2, 1, 0]], convexity=10);

```
    
### How to run this example in a viewer:
```bash
python -m pythonopenscad.examples.gists_3d.polyhedron_example --view
```

### Image:
![Polyhedron](polyhedron_example.py.png)
     
