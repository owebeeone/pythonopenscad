# Transformations

These gists demonstrate the use of transformations on shapes.

## Translate
    
### Python Code:
```python
from pythonopenscad import Cube, Translate

MODEL = Translate([10, -10, 5])(Cube(size=5, center=True))
```
    
### OpenSCAD Code:
```js
translate(v=[10.0, -10.0, 5.0]) {
  cube(size=5.0, center=true);
}

```
    
### How to run this example in a viewer:
```bash
python -m pythonopenscad.examples.gists_transforms.translate_example --view
```

### Image:
![Translate](pythonopenscad/examples/gists_transforms/translate_example.png)
     
## Rotate
    
### Python Code:
```python
from pythonopenscad import Cube, Rotate

MODEL = Rotate([45, 45, 0])(Cube(size=[10, 15, 5]))
```
    
### OpenSCAD Code:
```js
rotate(a=[45.0, 45.0, 0.0]) {
  cube(size=[10.0, 15.0, 5.0]);
}

```
    
### How to run this example in a viewer:
```bash
python -m pythonopenscad.examples.gists_transforms.rotate_example --view
```

### Image:
![Rotate](pythonopenscad/examples/gists_transforms/rotate_example.png)
     
## Scale
    
### Python Code:
```python
from pythonopenscad import Scale, Sphere

MODEL = Scale([1.5, 1, 0.5])(Sphere(r=10))
```
    
### OpenSCAD Code:
```js
scale(v=[1.5, 1.0, 0.5]) {
  sphere(r=10.0);
}

```
    
### How to run this example in a viewer:
```bash
python -m pythonopenscad.examples.gists_transforms.scale_example --view
```

### Image:
![Scale](pythonopenscad/examples/gists_transforms/scale_example.png)
     
## Resize
    
### Python Code:
```python
from pythonopenscad import Resize, Sphere

MODEL = Resize(newsize=[30, 10, 5])(Sphere(r=5))
```
    
### OpenSCAD Code:
```js
resize(newsize=[30.0, 10.0, 5.0]) {
  sphere(r=5.0);
}

```
    
### How to run this example in a viewer:
```bash
python -m pythonopenscad.examples.gists_transforms.resize_example --view
```

### Image:
![Resize](pythonopenscad/examples/gists_transforms/resize_example.png)
     
## Mirror
    
### Python Code:
```python
from pythonopenscad import Color, Cube, Mirror, Translate

MODEL = Mirror([1, 1, 0])(Translate([10, 0, 0])(Color('green')(Cube(size=5)))) \
    + Translate([10, 0, 0])(Color('red')(Cube(size=5))).setMetadataName("not mirrored")
```
    
### OpenSCAD Code:
```js
union() {
  mirror(v=[1.0, 1.0, 0.0]) {
    translate(v=[10.0, 0.0, 0.0]) {
      color(c="green") {
        cube(size=5.0);
      }
    }
  }
  // 'not mirrored'
  translate(v=[10.0, 0.0, 0.0]) {
    color(c="red") {
      cube(size=5.0);
    }
  }
}

```
    
### How to run this example in a viewer:
```bash
python -m pythonopenscad.examples.gists_transforms.mirror_example --view
```

### Image:
![Mirror](pythonopenscad/examples/gists_transforms/mirror_example.png)
     
## Color
    
### Python Code:
```python
from pythonopenscad import Color, Sphere

MODEL = Color("green")(Sphere(r=10))
```
    
### OpenSCAD Code:
```js
color(c="green") {
  sphere(r=10.0);
}

```
    
### How to run this example in a viewer:
```bash
python -m pythonopenscad.examples.gists_transforms.color_example --view
```

### Image:
![Color](pythonopenscad/examples/gists_transforms/color_example.png)
     
## Multmatrix
    
### Python Code:
```python
from pythonopenscad import Cube, Multmatrix

MODEL = Multmatrix(m=[
    [1, 0.5, 0, 5],
    [0, 1, 0.5, 10],
    [0.5, 0, 1, 0],
    [0, 0, 0, 1]
])(Cube(size=10))
```
    
### OpenSCAD Code:
```js
multmatrix(m=[[1.0, 0.5, 0.0, 5.0], [0.0, 1.0, 0.5, 10.0], [0.5, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) {
  cube(size=10.0);
}

```
    
### How to run this example in a viewer:
```bash
python -m pythonopenscad.examples.gists_transforms.multmatrix_example --view
```

### Image:
![Multmatrix](pythonopenscad/examples/gists_transforms/multmatrix_example.png)
     
## Projection
    
### Python Code:
```python
from pythonopenscad import Cube, Projection, Sphere, Translate

MODEL = Projection(cut=True)(Cube(7) + Translate([0,0,2.5])(Sphere(r=5)))
```
    
### OpenSCAD Code:
```js
projection(cut=true) {
  union() {
    cube(size=7.0);
    translate(v=[0.0, 0.0, 2.5]) {
      sphere(r=5.0);
    }
  }
}

```
    
### How to run this example in a viewer:
```bash
python -m pythonopenscad.examples.gists_transforms.projection_example --view
```

### Image:
![Projection](pythonopenscad/examples/gists_transforms/projection_example.png)
     
## Linear_Extrude
    
### Python Code:
```python
from pythonopenscad import Linear_Extrude, Square

MODEL = Linear_Extrude(height=5, center=True, scale=0.5, twist=90)(Square(10))
```
    
### OpenSCAD Code:
```js
linear_extrude(height=5.0, center=true, twist=90.0, scale=0.5) {
  square(size=10.0);
}

```
    
### How to run this example in a viewer:
```bash
python -m pythonopenscad.examples.gists_transforms.linear_extrude_example --view
```

### Image:
![Linear_Extrude](pythonopenscad/examples/gists_transforms/linear_extrude_example.png)
     
## Rotate_Extrude
    
### Python Code:
```python
from pythonopenscad import Circle, Rotate_Extrude, Translate

MODEL = Rotate_Extrude(angle=270, _fn=128)(Translate([5,0,0])(Circle(r=2)))
```
    
### OpenSCAD Code:
```js
rotate_extrude(angle=270.0, $fn=128) {
  translate(v=[5.0, 0.0, 0.0]) {
    circle(r=2.0);
  }
}

```
    
### How to run this example in a viewer:
```bash
python -m pythonopenscad.examples.gists_transforms.rotate_extrude_example --view
```

### Image:
![Rotate_Extrude](pythonopenscad/examples/gists_transforms/rotate_extrude_example.png)
     
## Offset
    
### Python Code:
```python
from pythonopenscad import Offset, Square

MODEL = Offset(delta=2)(Square(10)) - Square(10)
```
    
### OpenSCAD Code:
```js
difference() {
  offset(delta=2.0, chamfer=false) {
    square(size=10.0);
  }
  square(size=10.0);
}

```
    
### How to run this example in a viewer:
```bash
python -m pythonopenscad.examples.gists_transforms.offset_example --view
```

### Image:
![Offset](pythonopenscad/examples/gists_transforms/offset_example.png)
     
