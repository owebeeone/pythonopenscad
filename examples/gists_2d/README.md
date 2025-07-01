# 2D Shapes

These gists demonstrate the creation of 2D shapes using the pythonopenscad library.
These shapes are 2D and do not have a depth but when rendered they are extruded to a depth of 1.



## Circle
    
### Python Code:
```python
from pythonopenscad import Circle

MODEL = Circle(d=10, _fn=64)
```
    
### OpenSCAD Code:
```js
circle(d=10.0, $fn=64);

```
    
### How to run this example in a viewer:
```bash
python -m pythonopenscad.examples.gists_2d.circle_example --view
```

### Image:
![Circle](/pythonopenscad/examples/gists_2d/circle_example.png)
     
## Square
    
### Python Code:
```python
from pythonopenscad import Square

MODEL = Square(size=[15, 10], center=True)
```
    
### OpenSCAD Code:
```js
square(size=[15.0, 10.0], center=true);

```
    
### How to run this example in a viewer:
```bash
python -m pythonopenscad.examples.gists_2d.square_example --view
```

### Image:
![Square](pythonopenscad/examples/gists_2d/square_example.png)
     
## Polygon
    
### Python Code:
```python
from pythonopenscad import Polygon

MODEL = Polygon(points=[[0,0], [0,10], [10,10]])
```
    
### OpenSCAD Code:
```js
polygon(points=[[0.0, 0.0], [0.0, 10.0], [10.0, 10.0]]);

```
    
### How to run this example in a viewer:
```bash
python -m pythonopenscad.examples.gists_2d.polygon_example --view
```

### Image:
![Polygon](pythonopenscad/examples/gists_2d/polygon_example.png)
     
## Text
    
### Python Code:
```python
from pythonopenscad import Text

MODEL = Text(text="POSC", size=10, font="Liberation Sans:style=Bold", halign="center")
```
    
### OpenSCAD Code:
```js
text(text="POSC", size=10.0, font="Liberation Sans:style=Bold", halign="center");

```
    
### How to run this example in a viewer:
```bash
python -m pythonopenscad.examples.gists_2d.text_example --view
```

### Image:
![Text](pythonopenscad/examples/gists_2d/text_example.png)
     
