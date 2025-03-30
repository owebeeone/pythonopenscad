r"""
Doc for OpenSCAD text module.

The text module creates text as a 2D geometric object, using fonts installed on the local system or provided as separate font file.

[Note: Requires version 2015.03]

Parameters

text
String. The text to generate.
size
Decimal. The generated text has an ascent (height above the baseline) of approximately this value. Default is 10. Fonts vary and may be a different height, typically slightly smaller. The formula to convert the size value to "points" is pt = size/3.937, so a size argument of 3.05 will give about 12pt text, for instance. Note: if you know a point is 1/72" this may not look right, but point measurements of text are the distance from ascent to descent, not from ascent to baseline as in this case.
font
String. The name of the font that should be used. This is not the name of the font file, but the logical font name (internally handled by the fontconfig library). This can also include a style parameter, see below. A list of installed fonts & styles can be obtained using the font list dialog (Help -> Font List).
halign
String. The horizontal alignment for the text. Possible values are "left", "center" and "right". Default is "left".
valign
String. The vertical alignment for the text. Possible values are "top", "center", "baseline" and "bottom". Default is "baseline".
spacing
Decimal. Factor to increase/decrease the character spacing. The default value of 1 results in the normal spacing for the font, giving a value greater than 1 causes the letters to be spaced further apart.
direction
String. Direction of the text flow. Possible values are "ltr" (left-to-right), "rtl" (right-to-left), "ttb" (top-to-bottom) and "btt" (bottom-to-top). Default is "ltr".
language
String. The language of the text (e.g., "en", "ar", "ch"). Default is "en".
script
String. The script of the text (e.g., "latin", "arabic", "hani"). Default is "latin".
$fn
used for subdividing the curved path segments provided by freetype
Example


Example 1: Result.
text("OpenSCAD");




Notes
To allow specification of particular Unicode characters, you can specify them in a string with the following escape codes;

\x03     - hex char-value (only hex values from 01 to 7f are supported)
\u0123   - Unicode char with 4 hexadecimal digits (note: lowercase \u)
\U012345 - Unicode char with 6 hexadecimal digits (note: uppercase \U)
The null character (NUL) is mapped to the space character (SP).

 assert(version() == [2019, 5, 0]);
 assert(ord(" ") == 32);
 assert(ord("\x00") == 32);
 assert(ord("\u0000") == 32);
 assert(ord("\U000000") == 32);
Example

t="\u20AC10 \u263A"; // 10 euro and a smilie
Using Fonts & Styles
Fonts are specified by their logical font name; in addition a style parameter can be added to select a specific font style like "bold" or "italic", such as:

font="Liberation Sans:style=Bold Italic"
The font list dialog (available under Help > Font List) shows the font name and the font style for each available font. For reference, the dialog also displays the location of the font file. You can drag a font in the font list, into the editor window to use in the text() statement.


OpenSCAD font list dialog
OpenSCAD includes the fonts Liberation Mono, Liberation Sans, and Liberation Serif. Hence, as fonts in general differ by platform type, use of these included fonts is likely to be portable across platforms.

For common/casual text usage, the specification of one of these fonts is recommended for this reason. Liberation Sans is the default font to encourage this.


In addition to the installed fonts ( for windows only fonts installed as admin for all users ), it's possible to add project specific font files. Supported font file formats are TrueType Fonts (*.ttf) and OpenType Fonts (*.otf). The files need to be registered with use<>.

 use <ttf/paratype-serif/PTF55F.ttf>
After the registration, the font is listed in the font list dialog, so in case logical name of a font is unknown, it can be looked up as it was registered.

OpenSCAD uses fontconfig to find and manage fonts, so it's possible to list the system configured fonts on command line using the fontconfig tools in a format similar to the GUI dialog.

$ fc-list -f "%-60{{%{family[0]}%{:style[0]=}}}%{file}\n" | sort

...
Liberation Mono:style=Bold Italic /usr/share/fonts/truetype/liberation2/LiberationMono-BoldItalic.ttf
Liberation Mono:style=Bold        /usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf
Liberation Mono:style=Italic      /usr/share/fonts/truetype/liberation2/LiberationMono-Italic.ttf
Liberation Mono:style=Regular     /usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf
...
Under Windows, fonts are stored in the Windows Registry. To get a file with the font file names, use the command:

reg query "HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts" /s > List_Fonts_Windows.txt


Example


Example 2: Result.
 square(10);
 
 translate([15, 15]) {
   text("OpenSCAD", font = "Liberation Sans");
 }
 
 translate([15, 0]) {
   text("OpenSCAD", font = "Liberation Sans:style=Bold Italic");
 }


Alignment
Vertical alignment
top
The text is aligned so the top of the tallest character in your text is at the given Y coordinate.
center
The text is aligned with the center of the bounding box at the given Y coordinate. This bounding box is based on the actual sizes of the letters, so taller letters and descenders will affect the positioning.
baseline
The text is aligned with the font baseline at the given Y coordinate. This is the default, and is the only option that makes different pieces of text align vertically, as if they were written on lined paper, regardless of character heights and descenders.
bottom
The text is aligned so the bottom of the lowest-reaching character in your text is at the given Y coordinate.

OpenSCAD vertical text alignment
 text = "Align";
 font = "Liberation Sans";
 
 valign = [
   [  0, "top"],
   [ 40, "center"],
   [ 75, "baseline"],
   [110, "bottom"]
 ];
 
 for (a = valign) {
   translate([10, 120 - a[0], 0]) {
     color("red") cube([135, 1, 0.1]);
     color("blue") cube([1, 20, 0.1]);
     linear_extrude(height = 0.5) {
       text(text = str(text,"_",a[1]), font = font, size = 20, valign = a[1]);
     }
   }
 }

The text() module doesn't support multi-line text, but you can make a separate call for each line, using translate() to space them. A spacing of 1.4*size is the minimum required to prevent lines overlapping (if they include descenders). 1.6*size is approximately the default single-spacing in many word processing programs. To get evenly spaced lines, use "baseline" vertical alignment; otherwise, lines may be lower or higher depending on their contents.

Horizontal alignment
left
The text is aligned with the left side of the bounding box at the given X coordinate. This is the default.
center
The text is aligned with the center of the bounding box at the given X coordinate.
right
The text is aligned with the right of the bounding box at the given X coordinate.

OpenSCAD horizontal text alignment
 text = "Align";
 font = "Liberation Sans";
 
 halign = [
   [10, "left"],
   [50, "center"],
   [90, "right"]
 ];
 
 for (a = halign) {
   translate([140, a[0], 0]) {
     color("red") cube([115, 2,0.1]);
     color("blue") cube([2, 20,0.1]);
     linear_extrude(height = 0.5) {
       text(text = str(text,"_",a[1]), font = font, size = 20, halign = a[1]);
     }
   }
 }


3D text

3D text example
Text can be changed from a 2 dimensional object into a 3D object by using the linear_extrude function.

//3d Text Example
linear_extrude(4)
    text("Text");
Metrics
[Note: Requires version Development snapshot]

textmetrics()
The textmetrics() function accepts the same parameters as text(), and returns an object describing how the text would be rendered.

The returned object has these members:

position: the position of the lower-left corner of the generated text.
size: the size of the generated text.
ascent: the amount that the text extends above the baseline.
descent: the amount that the text extends below the baseline.
offset: the lower-left corner of the box containing the text, including inter-glyph spacing before the first glyph.
advance: the "other end" of the text, the point at which additional text should be positioned.
   s = "Hello, World!";
   size = 20;
   font = "Liberation Serif";
   
   tm = textmetrics(s, size=size, font=font);
   echo(tm);
   translate([0,0,1]) text("Hello, World!", size=size, font=font);
   color("black") translate(tm.position) square(tm.size);
yields (reformatted for readability):


Using textmetrics() to draw a box around text
   ECHO: {
       position = [0.7936, -4.2752];
       size = [149.306, 23.552];
       ascent = 19.2768;
       descent = -4.2752;
       offset = [0, 0];
       advance = [153.09, 0];
   }

"""

import numpy as np
from datatrees import datatree, dtfield
import anchorscad_lib.linear as l

EPSILON = 1e-6


def to_gvector(np_array):
    if len(np_array) == 2:
        return l.GVector([np_array[0], np_array[1], 0, 1])
    else:
        return l.GVector(np_array)

def extentsof(p: np.ndarray) -> np.ndarray:
    return np.array((p.min(axis=0), p.max(axis=0)))

@datatree(frozen=True)
class CubicSpline():
    '''Cubic spline evaluator, extents and inflection point finder.'''
    p: object=dtfield(doc='The control points for the spline.')
    dimensions: int=dtfield(
        self_default=lambda s: len(s.p[0]),
        init=True, 
        doc='The number of dimensions in the spline.')
    
    COEFFICIENTS=np.array([
        [-1.,  3, -3,  1 ],
        [  3, -6,  3,  0 ],
        [ -3,  3,  0,  0 ],
        [  1,  0,  0,  0 ]])
    
    #@staticmethod # For some reason this breaks on Raspberry Pi OS.
    def _dcoeffs_builder(dims):
        zero_order_derivative_coeffs=np.array([[1.] * dims, [1] * dims, [1] * dims, [1] * dims])
        derivative_coeffs=np.array([[3.] * dims, [2] * dims, [1] * dims, [0] * dims])
        second_derivative=np.array([[6] * dims, [2] * dims, [0] * dims, [0] * dims])
        return (zero_order_derivative_coeffs, derivative_coeffs, second_derivative)
    
    DERIVATIVE_COEFFS = tuple((
        _dcoeffs_builder(1), 
        _dcoeffs_builder(2), 
        _dcoeffs_builder(3), ))
    
    def _dcoeffs(self, deivative_order):
        return self.DERIVATIVE_COEFFS[self.dimensions - 1][deivative_order]
        
    def __post_init__(self):
        object.__setattr__(self, 'coefs', np.matmul(self.COEFFICIENTS, self.p))
    
    def _make_ta3(self, t):
        t2 = t * t
        t3 = t2 * t
        ta = np.tile([t3, t2, t, 1], (self.dimensions, 1)).T
        return ta
        
    def _make_ta2(self, t):
        t2 = t * t
        ta = np.tile([t2, t, 1, 0], (self.dimensions, 1)).T
        return ta
    
    def evaluate(self, t):
        return np.sum(np.multiply(self.coefs, self._make_ta3(t)), axis=0)
  
    @classmethod
    def find_roots(cls, a, b, c, *, t_range: tuple[float, float]=(0.0, 1.0)):
        '''Find roots of quadratic polynomial that are between t_range.'''
        # a, b, c are quadratic coefficients i.e. at^2 + bt + c
        if a == 0:
            # Degenerate curve is a linear. Only one possible root.
            if b == 0:
                # Degenerate curve is constant so there is no 0 gradient.
                return ()
            t = -c / b
            
            return (t,) if  t >= t_range[0] and t <= t_range[1] else ()
    
        b2_4ac = b * b - 4 * a * c
        if b2_4ac < 0:
            if b2_4ac > -EPSILON: # Could be a rounding error, so treat as 0.
                b2_4ac = 0
            else:
                # Complex roots - no answer.
                return ()
    
        sqrt_b2_4ac = np.sqrt(b2_4ac)
        two_a = 2 * a
    
        values = ((-b + sqrt_b2_4ac) / two_a, (-b - sqrt_b2_4ac) / two_a)
        return tuple(t for t in values if t >= t_range[0] and t <= t_range[1])
    
    # Solve for minima and maxima over t. There are two possible locations 
    # for each axis. The results for t outside of the bounds 0-1 are ignored
    # since the cubic spline is only interpolated in those bounds.
    def curve_maxima_minima_t(self, t_range: tuple[float, float]=(0.0, 1.0)):
        '''Returns a dict with an entry for each dimension containing a list of
        t for each minima or maxima found.'''
        # Splines are defined only for t in the range [0..1] however the curve may
        # go beyond those points. Each axis has a potential of two roots.
        d_coefs = self.coefs * self._dcoeffs(1)
        return dict((i, self.find_roots(*(d_coefs[0:3, i]), t_range=t_range)) 
                    for i in range(self.dimensions))

    def curve_inflexion_t(self, t_range: tuple[float, float]=(0.0, 1.0)):
        '''Returns a dict with an entry for each dimension containing a list of
        t for each inflection point found.'''
        # Splines are defined only for t in the range [0..1] however the curve may
        # go beyond those points. Each axis has a potential of two roots.
        d_coefs = self.coefs * self._dcoeffs(2)
        return dict((i, self.find_roots(0., *(d_coefs[0:2, i]), t_range=t_range))
                    for i in range(self.dimensions))
    
    def derivative(self, t):
        return -np.sum(
            np.multiply(
                np.multiply(self.coefs, self._dcoeffs(1)), self._make_ta2(t)), axis=0)
    
    def normal2d(self, t, dims=[0, 1]):
        '''Returns the normal to the curve at t for the 2 given dimensions.'''
        d = self.derivative(t)
        vr = np.array([d[dims[1]], -d[dims[0]]])
        d = np.sqrt(np.sum(vr**2))
        return vr / d
    
    def extremes(self):
        roots = self.curve_maxima_minima_t()
        t_values = [0.0, 1.0]
        for v in roots.values():
            t_values.extend(v)
        t_values.sort()
        return np.array(tuple(self.evaluate(t) for t in t_values if t >= 0 and t <= 1))
    
    def extents(self):
        extr = self.extremes()
        return extentsof(extr)
    
    def transform(self, m: l.GMatrix) -> 'CubicSpline':
        '''Returns a new spline transformed by the matrix m.'''
        new_p = list((m * to_gvector(p)).A[0:self.dimensions] for p in self.p)
        return CubicSpline(np.array(new_p), self.dimensions)
    
    
    def azimuth_t(self, angle: float | l.Angle=0, t_end: bool=False, 
                t_range: tuple[float, float]=(0.0, 1.0)) -> tuple[float, ...]:
        '''Returns the list of t where the tangent is at the given angle from the beginning of the
        given t_range. The angle is in degrees or Angle.'''
        
        angle = l.angle(angle)
        
        start_slope = self.normal2d(t_range[1 if t_end else 0])
        start_rot: l.GMatrix = l.rotZ(sinr_cosr=(start_slope[1], -start_slope[0]))
        
        qs: CubicSpline = self.transform(l.rotZ(angle.inv()) * start_rot)
        
        roots = qs.curve_maxima_minima_t(t_range)

        return sorted(roots[0])


@datatree(frozen=True)
class QuadraticSpline():
    '''Quadratic spline evaluator, extents and inflection point finder.'''
    p: object=dtfield(doc='The control points for the spline.')
    dimensions: int=dtfield(
        self_default=lambda s: len(s.p[0]),
        init=True, 
        doc='The number of dimensions in the spline.')
    
    COEFFICIENTS=np.array([
        [  1., -2,  1 ],
        [ -2.,  2,  0 ],
        [  1.,  0,  0 ]])
    
    #@staticmethod # For some reason this breaks on Raspberry Pi OS.
    def _dcoeffs_builder(dims):
        zero_order_derivative_coeffs=np.array([[1.] * dims, [1] * dims, [1] * dims])
        derivative_coeffs=np.array([[2] * dims, [1] * dims, [0] * dims])
        second_derivative=np.array([[2] * dims, [0] * dims, [0] * dims])
        return (zero_order_derivative_coeffs, derivative_coeffs, second_derivative)
    
    DERIVATIVE_COEFFS = tuple((
        _dcoeffs_builder(1), 
        _dcoeffs_builder(2), 
        _dcoeffs_builder(3), ))
    
    def _dcoeffs(self, deivative_order):
        return self.DERIVATIVE_COEFFS[self.dimensions - 1][deivative_order]
        
    def __post_init__(self):
        object.__setattr__(self, 'coefs', np.matmul(self.COEFFICIENTS, self.p))
    
    def _qmake_ta2(self, t):
        ta = np.tile([t**2, t, 1], (self.dimensions, 1)).T
        return ta
        
    def _qmake_ta1(self, t):
        ta = np.tile([t, 1, 0], (self.dimensions, 1)).T
        return ta
    
    def evaluate(self, t):
        return np.sum(np.multiply(self.coefs, self._qmake_ta2(t)), axis=0)
    
    @classmethod
    def find_roots(cls, a, b, *, t_range: tuple[float, float]=(0.0, 1.0)):
        '''Find roots of linear equation that are between t_range.'''
        # There either 1 or no roots.
        if a == 0:
            # Degenerate curve is constant so there is no 0 gradient.
            return ()
        
        # Only return the root if it is within the range.
        t = -b / a
        return (t,) if  t >= t_range[0] and t <= t_range[1] else ()

    # Solve for minima and maxima over t. There are two possible locations 
    # for each axis. The results for t outside of the bounds 0-1 are ignored
    # since the cubic spline is only interpolated in those bounds.
    def curve_maxima_minima_t(self, t_range: tuple[float, float]=(0.0, 1.0)):
        '''Returns a dict with an entry for each dimension containing a list of
        t for each minima or maxima found.'''
        # Splines are defined only for t in the range [0..1] however the curve may
        # go beyond those points. Each axis has a potential of two roots.
        d_coefs = self.coefs * self._dcoeffs(1)
        return dict((i, self.find_roots(*(d_coefs[0:2, i]), t_range=t_range)) 
                    for i in range(self.dimensions))

    def curve_inflexion_t(self, t_range: tuple[float, float]=(0.0, 1.0)):
        '''Returns a dict with an entry for each dimension containing a list of
        t for each inflection point found.'''
        
        # Quadradic splines have no inflection points since their second order
        # derivative is constant.
        return dict((i, ()) for i in range(self.dimensions))
    
    def derivative(self, t):
        return -np.sum(
            np.multiply(
                np.multiply(self.coefs, self._dcoeffs(1)), self._qmake_ta1(t)), axis=0)
    
    def normal2d(self, t, dims=[0, 1]):
        '''Returns the normal to the curve at t for the 2 given dimensions.'''
        d = self.derivative(t)
        vr = np.array([d[dims[1]], -d[dims[0]]])
        d = np.sqrt(np.sum(vr**2))
        return vr / d
    
    def extremes(self):
        roots = self.curve_maxima_minima_t()
        t_values = [0.0, 1.0]
        for v in roots.values():
            t_values.extend(v)
        t_values.sort()
        return np.array(tuple(self.evaluate(t) for t in t_values if t >= 0 and t <= 1))
    
    def extents(self):
        extr = self.extremes()
        return extentsof(extr)
    
    def transform(self, m: l.GMatrix) -> 'QuadraticSpline':
        '''Returns a new spline transformed by the matrix m.'''
        new_p = list((m * to_gvector(p)).A[0:self.dimensions] for p in self.p)
        return QuadraticSpline(np.array(new_p), self.dimensions)
    
    def azimuth_t(self, angle: float | l.Angle=0, t_end: bool=False, 
                t_range: tuple[float, float]=(0.0, 1.0)) -> tuple[float, ...]:
        '''Returns the list of t where the tangent is at the given angle from the beginning of the
        given t_range. The angle is in degrees or Angle.'''
        
        angle = l.angle(angle)
        
        start_slope = self.normal2d(t_range[1 if t_end else 0])
        start_rot: l.GMatrix = l.rotZ(sinr_cosr=(-start_slope[1], start_slope[0]))
        
        qs: QuadraticSpline = self.transform(angle.inv().rotZ * start_rot)
        
        roots = qs.curve_maxima_minima_t(t_range)

        return sorted(roots[0])

@datatree
class TextContext:
    text: str=dtfield(default="")
    size: float=dtfield(default=10.0)
    font: str=dtfield(default="Liberation Sans")
    halign: str=dtfield(default="left")
    valign: str=dtfield(default="baseline")
    spacing: float=dtfield(default=1.0)
    direction: str=dtfield(default="ltr")
    language: str=dtfield(default="en")
    script: str=dtfield(default="latin")
    fa: float=dtfield(default=12.0)
    fs: float=dtfield(default=2.0)
    fn: int=dtfield(default=0)
    base_direction: str=dtfield(default="ltr")
    
    def __post_init__(self):
        import fontTools.ttLib
        from fontTools.pens.recordingPen import RecordingPen
        from PIL import ImageFont, ImageDraw, Image
        self._font_tools = fontTools
        self._recording_pen = RecordingPen
        self._image_font = ImageFont
        self._image_draw = ImageDraw
        self._image = Image
        
        # Default to Liberation Sans if not specified
        if not self.font:
            self.font = "Liberation Sans"
            
        # Validate alignment values
        if self.halign not in ("left", "center", "right"):
            raise ValueError(f"Invalid halign value: {self.halign}. Must be 'left', 'center', or 'right'")
        
        if self.valign not in ("top", "center", "baseline", "bottom"):
            raise ValueError(f"Invalid valign value: {self.valign}. Must be 'top', 'center', 'baseline', or 'bottom'")
            
        if self.direction not in ("ltr", "rtl", "ttb", "btt"):
            raise ValueError(f"Invalid direction value: {self.direction}. Must be 'ltr', 'rtl', 'ttb', or 'btt'")
            
        if self.base_direction not in ("ltr", "rtl"):
            raise ValueError(f"Invalid base_direction value: {self.base_direction}. Must be 'ltr' or 'rtl'")
        
        # Load the font
        self._load_font()
        
        # Determine if the font has an inverted Y-axis compared to the standard TTF/OTF coordinate system
        self._detect_font_orientation()
        
        # Also load fallback fonts for characters not supported by the primary font
        self._load_fallback_fonts()
        
        # Process bidirectional text if needed
        if self.direction in ("ltr", "rtl"):
            self._process_bidirectional_text()
            
    def _detect_font_orientation(self):
        """Determine if the font uses an inverted Y-axis coordinate system"""
        self._y_axis_inverted = False
        
        # Special case fonts that are known to have inverted Y-axis
        known_inverted_fonts = [
            "New Gulim", "Gulim", "GulimChe", "Dotum", "DotumChe",  # Korean fonts 
            "MS Mincho", "MS Gothic", "MS PMincho", "MS PGothic",    # Japanese fonts
            "SimSun", "NSimSun", "SimHei", "FangSong", "KaiTi"       # Chinese fonts
        ]
        
        # Check if our font is in the known inverted fonts list
        if any(inverted_font in self.font for inverted_font in known_inverted_fonts):
            self._y_axis_inverted = True
            return
            
        # If we have fontTools font loaded, we can try to detect orientation
        # based on metrics
        if self._font and hasattr(self._font, 'getBestCmap'):
            try:
                # Try a capital 'A' as it's likely in most fonts
                # and has a simple shape that's good for detection
                cmap = self._font.getBestCmap()
                if ord('A') in cmap:
                    # Get glyph set and draw an 'A'
                    glyph_set = self._font.getGlyphSet()
                    pen = self._recording_pen()
                    glyph_name = cmap.get(ord('A'))
                    if glyph_name and glyph_name in glyph_set:
                        glyph_set[glyph_name].draw(pen)
                        
                        # Sample y-coordinates to see where points are
                        y_values = []
                        for command, args in pen.value:
                            if command in ('moveTo', 'lineTo') and args:
                                y_values.append(args[0][1])
                        
                        # If more than 70% of points are below the baseline
                        # font likely has inverted Y-axis
                        if y_values:
                            points_below = sum(1 for y in y_values if y < 0)
                            percentage_below = points_below / len(y_values)
                            if percentage_below > 0.7:
                                self._y_axis_inverted = True
            except Exception as e:
                print(f"Warning: Error detecting font orientation: {e}")
    
    def _load_fallback_fonts(self):
        """Load fallback fonts for characters not supported by the primary font"""
        # List of common fonts that support Arabic and Hebrew
        self._fallback_fonts = {}
        fallback_font_names = ["Arial", "Times New Roman", "DejaVu Sans", "Liberation Sans"]
        
        try:
            for font_name in fallback_font_names:
                if font_name != self.font:  # Don't duplicate the primary font
                    try:
                        fallback_font = self._image_font.truetype(font_name, size=int(self.size * 3.937))
                        self._fallback_fonts[font_name] = fallback_font
                    except:
                        pass
        except Exception as e:
            print(f"Warning: Error loading fallback fonts: {e}")
    
    def _process_bidirectional_text(self):
        """Process bidirectional text according to the Unicode Bidirectional Algorithm."""
        try:
            import bidi.algorithm as bidi
            
            # Get the base direction
            base_dir = "R" if self.base_direction == "rtl" else "L"
            
            # Apply the bidirectional algorithm to get the visual representation
            self._visual_text = bidi.get_display(self.text, base_dir=base_dir)
            
            # Store character mapping (logical to visual)
            self._char_mapping = []
            logical_text = self.text
            visual_text = self._visual_text
            
            # For each character in the logical text, find its position in the visual text
            # This is a simplified mapping and might need improvement for complex cases
            for i, char in enumerate(logical_text):
                try:
                    # Find all occurrences of this character in visual text
                    occurrences = [j for j, c in enumerate(visual_text) if c == char]
                    
                    # Find the occurrence that hasn't been mapped yet
                    for pos in occurrences:
                        if pos not in self._char_mapping:
                            self._char_mapping.append(pos)
                            break
                    else:
                        # If all occurrences have been mapped, just add the last one
                        # This is a fallback and might not be accurate for all cases
                        self._char_mapping.append(occurrences[-1] if occurrences else i)
                except:
                    # Fallback to the same position if there's an error
                    self._char_mapping.append(i)
            
        except ImportError:
            print("Warning: python-bidi library not found. Bidirectional text support is limited.")
            self._visual_text = self.text
            self._char_mapping = list(range(len(self.text)))
    
    def _load_font(self):
        """Load the font specified in self.font"""
        try:
            # Parse the font name and style
            font_name = self.font
            font_style = None
            
            if ":" in self.font:
                parts = self.font.split(":", 1)
                font_name = parts[0]
                if len(parts) > 1 and parts[1].startswith("style="):
                    font_style = parts[1][6:]  # Remove 'style='
            
            # Try to load the font using PIL
            try:
                # First attempt to load as a file path
                if font_style:
                    # Not ideal, but PIL doesn't support styles directly in the same way
                    if "bold" in font_style.lower() and "italic" in font_style.lower():
                        self._pil_font = self._image_font.truetype(font_name, size=int(self.size * 3.937), index=0)
                    elif "bold" in font_style.lower():
                        self._pil_font = self._image_font.truetype(font_name, size=int(self.size * 3.937), index=0)
                    elif "italic" in font_style.lower():
                        self._pil_font = self._image_font.truetype(font_name, size=int(self.size * 3.937), index=0)
                    else:
                        self._pil_font = self._image_font.truetype(font_name, size=int(self.size * 3.937), index=0)
                else:
                    self._pil_font = self._image_font.truetype(font_name, size=int(self.size * 3.937), index=0)
            except OSError:
                # If that fails, try to find a matching font in the available fonts
                available_fonts = get_available_fonts()
                
                # Try exact match first
                matched_font_path = None
                if font_name in available_fonts:
                    # Found the font family
                    matched_style = None
                    if font_style:
                        # Look for exact style match
                        if font_style in available_fonts[font_name]:
                            matched_style = font_style
                        else:
                            # Try case insensitive match
                            for style in available_fonts[font_name]:
                                if style.lower() == font_style.lower():
                                    matched_style = style
                                    break
                    else:
                        # No style specified, prefer Regular
                        if "Regular" in available_fonts[font_name]:
                            matched_style = "Regular"
                        else:
                            # Take the first available style
                            matched_style = available_fonts[font_name][0] if available_fonts[font_name] else None
                    
                    if matched_style:
                        # Find the file path for this font/style
                        import os
                        if os.name == 'nt':
                            # On Windows, get the fonts directory
                            import ctypes.wintypes
                            CSIDL_FONTS = 0x14
                            buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
                            ctypes.windll.shell32.SHGetFolderPathW(0, CSIDL_FONTS, 0, 0, buf)
                            fonts_dir = buf.value
                            
                            # Look for a matching font file
                            for file in os.listdir(fonts_dir):
                                if file.lower().endswith(('.ttf', '.otf')):
                                    try:
                                        font_path = os.path.join(fonts_dir, file)
                                        test_font = self._image_font.truetype(font_path, size=12)
                                        if test_font.getname()[0] == font_name:
                                            # Check if it matches our style
                                            matches_style = False
                                            if matched_style == "Regular":
                                                # For regular, avoid bold/italic in filename
                                                if 'bold' not in file.lower() and 'italic' not in file.lower():
                                                    matches_style = True
                                            elif matched_style == "Bold":
                                                if 'bold' in file.lower() and 'italic' not in file.lower():
                                                    matches_style = True
                                            elif matched_style == "Italic":
                                                if 'italic' in file.lower() and 'bold' not in file.lower():
                                                    matches_style = True
                                            elif matched_style == "Bold Italic":
                                                if 'bold' in file.lower() and 'italic' in file.lower():
                                                    matches_style = True
                                                    
                                            if matches_style:
                                                matched_font_path = font_path
                                                break
                                    except:
                                        continue
                
                # If we found a matching font, use it
                if matched_font_path:
                    self._pil_font = self._image_font.truetype(matched_font_path, size=int(self.size * 3.937))
                else:
                    # Try some fallback fonts
                    fallback_fonts = ["Arial", "Times New Roman", "Verdana", "DejaVu Sans", "Liberation Sans"]
                    for fallback in fallback_fonts:
                        try:
                            self._pil_font = self._image_font.truetype(fallback, size=int(self.size * 3.937))
                            print(f"Could not find font '{font_name}', using '{fallback}' instead")
                            font_name = fallback  # Update font name for fontTools loading
                            break
                        except:
                            continue
                    
                    if not hasattr(self, '_pil_font'):
                        raise ValueError(f"Could not find font '{font_name}' or any fallback fonts")
                
            # Also load with fontTools for outline extraction
            try:
                # First try to load directly (might work if it's a system font)
                try:
                    self._font = self._font_tools.ttLib.TTFont(font_name)
                except:
                    # If that fails and we have a PIL font, try to get its path
                    if hasattr(self, '_pil_font') and hasattr(self._pil_font, 'path'):
                        self._font = self._font_tools.ttLib.TTFont(self._pil_font.path)
                    else:
                        # Try to find the font file in the system
                        import os
                        if os.name == 'nt':
                            import ctypes.wintypes
                            CSIDL_FONTS = 0x14
                            buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
                            ctypes.windll.shell32.SHGetFolderPathW(0, CSIDL_FONTS, 0, 0, buf)
                            fonts_dir = buf.value
                            
                            # Try to find a matching font file
                            font_path = None
                            for file in os.listdir(fonts_dir):
                                if file.lower().startswith(font_name.lower()) and (file.lower().endswith('.ttf') or file.lower().endswith('.otf')):
                                    font_path = os.path.join(fonts_dir, file)
                                    break
                            
                            if font_path:
                                self._font = self._font_tools.ttLib.TTFont(font_path)
                            else:
                                raise ValueError(f"Could not find font file for {font_name}")
                        else:
                            raise ValueError(f"Could not load font {font_name} with fontTools")
            except Exception as e:
                # If fontTools can't load it, we can still use PIL for size calculations,
                # but we won't be able to get outline data
                print(f"Warning: Could not load font {font_name} with fontTools: {e}")
                # Create a dummy _font attribute so other methods don't break
                self._font = None
                    
        except Exception as e:
            raise ValueError(f"Error loading font {self.font}: {str(e)}")
    
    def _calculate_metrics(self):
        """Calculate metrics for the text"""
        # Use PIL to get initial metrics
        img = self._image.new('RGB', (1, 1), color=(255, 255, 255))
        draw = self._image_draw.Draw(img)
        
        # Calculate metrics for the whole text
        text_bbox = draw.textbbox((0, 0), self.text, font=self._pil_font, spacing=int(self.spacing*self.size))
        
        # Get font ascent and descent
        # Some versions of PIL don't expose ascent/descent directly
        ascent = 0
        descent = 0
        
        # Try to get ascent/descent from the font
        if hasattr(self._pil_font, 'ascent') and hasattr(self._pil_font, 'descent'):
            ascent = self._pil_font.ascent / self._pil_font.size * self.size
            descent = -self._pil_font.descent / self._pil_font.size * self.size
        else:
            # Fallback: estimate from the metrics of a capital letter and a descender
            ascent_bbox = draw.textbbox((0, 0), "A", font=self._pil_font)
            descent_bbox = draw.textbbox((0, 0), "g", font=self._pil_font)
            
            # Ascent is roughly the height of a capital letter
            ascent = (ascent_bbox[3] - ascent_bbox[1]) * 0.8
            # Descent is roughly the difference between lowercase g and capital A
            descent = (descent_bbox[3] - ascent_bbox[3]) * 0.8
        
        # Convert to the format expected by OpenSCAD
        metrics = {
            'position': [text_bbox[0], text_bbox[1]],
            'size': [text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]],
            'ascent': ascent,
            'descent': descent,
            'offset': [0, 0],
            'advance': [text_bbox[2] - text_bbox[0], 0]
        }
        
        return metrics
    
    def _get_glyph_outlines(self, char_code):
        """Get the outlines for a specific character"""
        try:
            # --- Font/Glyph Loading & Fallbacks ---
            # (Keep the initial loading, support checks, and fallback logic as before)
            # ... (omitted for brevity, assume this part is working) ...

            # --- Get Glyph and Pen ---
            glyph_set = self._font.getGlyphSet()
            cmap = self._font.getBestCmap()
            glyph_name = cmap.get(ord(char_code))
            if not glyph_name or glyph_name not in glyph_set:
                # (Handle fallback rectangle generation here if not already done)
                print(f"Error: Glyph '{char_code}' not found in font after loading.")
                return [] # Or return rectangle

            pen = self._recording_pen()
            glyph_set[glyph_name].draw(pen)

            # --- Process Pen Commands ---
            polygons = []
            current_contour = [] # Store raw (x, y) tuples

            for command, args in pen.value:
                try: # Wrap each command processing
                    if command == 'moveTo':
                        if current_contour: # Finalize previous
                            polygons.append(np.array(current_contour, dtype=float))
                        # Start new contour
                        if args and len(args[0]) == 2:
                            pt_raw = args[0]
                            if hasattr(self, '_y_axis_inverted') and self._y_axis_inverted:
                                current_contour = [(pt_raw[0], -pt_raw[1])] # Start tuple
                            else:
                                current_contour = [tuple(pt_raw)] # Start tuple
                        else: current_contour = []

                    elif command == 'lineTo':
                        if current_contour and args and len(args[0]) == 2:
                            pt_raw = args[0]
                            if hasattr(self, '_y_axis_inverted') and self._y_axis_inverted:
                                current_contour.append((pt_raw[0], -pt_raw[1])) # Append tuple
                            else:
                                current_contour.append(tuple(pt_raw)) # Append tuple

                    elif command == 'qCurveTo':
                        if not args: continue
                        if not current_contour: continue

                        start_point_tuple = current_contour[-1] # Get start tuple

                        # Ensure args are tuples before proceeding
                        args_tuples = [tuple(p) for p in args if isinstance(p, (list, tuple)) and len(p) == 2]
                        if len(args_tuples) != len(args):
                             print(f"Warning: Invalid data in qCurveTo args: {args}. Skipping.")
                             continue

                        full_pts_tuples_raw = [start_point_tuple] + args_tuples

                        # Apply Y-axis inversion (results are tuples)
                        if hasattr(self, '_y_axis_inverted') and self._y_axis_inverted:
                            full_pts_tuples = [(p[0], -p[1]) for p in full_pts_tuples_raw]
                        else:
                            full_pts_tuples = full_pts_tuples_raw

                        # Check for redundant start point (using tuples)
                        if len(full_pts_tuples) > 1 and np.allclose(full_pts_tuples[0], full_pts_tuples[1]):
                             full_pts_tuples = [full_pts_tuples[0]] + full_pts_tuples[2:]

                        if len(full_pts_tuples) < 3:
                             print(f"Warning: Not enough points ({len(full_pts_tuples)}) for qCurveTo. Joining last points.")
                             final_pt_tuple = full_pts_tuples[-1]
                             if not np.allclose(current_contour[-1], final_pt_tuple):
                                 current_contour.append(final_pt_tuple)
                             continue

                        num_bezier_segments = len(full_pts_tuples) - 2

                        for i in range(num_bezier_segments):
                            # Get points for this segment AS TUPLES first
                            p1_tuple = full_pts_tuples[i+1]

                            if i == 0:
                                p0_tuple = full_pts_tuples[0]
                            else:
                                # Calculate midpoint using arrays, convert back to tuple
                                p0_prev_arr = np.asarray(full_pts_tuples[i], dtype=float)
                                p1_arr_temp = np.asarray(p1_tuple, dtype=float)
                                p0_mid_arr = (p0_prev_arr + p1_arr_temp) / 2.0
                                p0_tuple = tuple(p0_mid_arr)

                            if i == num_bezier_segments - 1:
                                p2_tuple = full_pts_tuples[i+2]
                            else:
                                # Calculate midpoint using arrays, convert back to tuple
                                p1_arr_temp = np.asarray(p1_tuple, dtype=float)
                                p2_next_arr = np.asarray(full_pts_tuples[i+2], dtype=float)
                                p2_mid_arr = (p1_arr_temp + p2_next_arr) / 2.0
                                p2_tuple = tuple(p2_mid_arr)

                            # --- Convert to arrays ONLY for spline creation ---
                            try:
                                p0_arr = np.asarray(p0_tuple, dtype=float)
                                p1_arr = np.asarray(p1_tuple, dtype=float)
                                p2_arr = np.asarray(p2_tuple, dtype=float)

                                # **ULTIMATE CHECK**: Ensure shapes are EXACTLY (2,)
                                if p0_arr.shape != (2,) or p1_arr.shape != (2,) or p2_arr.shape != (2,):
                                     raise ValueError(f"Inconsistent shapes before creating bezier_points: {p0_arr.shape}, {p1_arr.shape}, {p2_arr.shape}")

                                bezier_points = np.array([p0_arr, p1_arr, p2_arr]) # Should be (3, 2)

                            except Exception as conversion_error:
                                print(f"ERROR converting points for spline: {conversion_error}")
                                print(f"  Problematic tuples: p0={p0_tuple}, p1={p1_tuple}, p2={p2_tuple}")
                                continue # Skip this segment

                            # --- Tessellation ---
                            # (Keep tessellation logic as in the previous attempt)
                            if np.allclose(p0_arr, p1_arr) and np.allclose(p1_arr, p2_arr):
                                p2_final_tuple = tuple(p2_arr) # Ensure it's a tuple
                                if not current_contour or not np.allclose(current_contour[-1], p2_final_tuple):
                                     current_contour.append(p2_final_tuple)
                                continue

                            radius = max(np.linalg.norm(p1_arr-p0_arr), np.linalg.norm(p2_arr-p1_arr))
                            radius = max(radius, EPSILON)
                            num_steps = get_fragments_from_fn_fa_fs(radius, self.fn, self.fa, self.fs)
                            num_steps = max(int(num_steps * 1.5), 8)

                            try:
                                spline = QuadraticSpline(bezier_points) # Pass validated (3, 2) array
                                t_values = np.linspace(0, 1, num_steps + 1)[1:]

                                if len(t_values) > 0:
                                    new_points = np.array([spline.evaluate(t) for t in t_values])
                                    if new_points.ndim == 2 and new_points.shape[1] == 2:
                                         current_contour.extend([tuple(pt) for pt in new_points]) # Extend tuples
                                    elif new_points.shape == (2,): # Single point case
                                         current_contour.append(tuple(new_points))
                                # else: (Warning if shape is wrong)

                                # Ensure final point is added if not already last
                                final_bezier_pt_tuple = tuple(p2_arr)
                                if not current_contour or not np.allclose(current_contour[-1], final_bezier_pt_tuple):
                                     current_contour.append(final_bezier_pt_tuple)

                            except Exception as spline_error:
                                print(f"ERROR during spline processing/evaluation: {spline_error}")
                                # Fallback: Add line to end point
                                final_bezier_pt_tuple = tuple(p2_arr)
                                if not current_contour or not np.allclose(current_contour[-1], final_bezier_pt_tuple):
                                    current_contour.append(final_bezier_pt_tuple)
                                continue
                            # --- End Tessellation ---

                    elif command == 'curveTo':
                        # (Apply similar "keep as tuple, convert late" logic here for cubic curves)
                         if not args or len(args) != 3: continue
                         if not current_contour: continue

                         try:
                             start_point_tuple = current_contour[-1]
                             control1_tuple_raw = tuple(args[0])
                             control2_tuple_raw = tuple(args[1])
                             end_point_tuple_raw = tuple(args[2])

                             # Apply Y inversion
                             if hasattr(self, '_y_axis_inverted') and self._y_axis_inverted:
                                 p0_tuple = (start_point_tuple[0], -start_point_tuple[1])
                                 p1_tuple = (control1_tuple_raw[0], -control1_tuple_raw[1])
                                 p2_tuple = (control2_tuple_raw[0], -control2_tuple_raw[1])
                                 p3_tuple = (end_point_tuple_raw[0], -end_point_tuple_raw[1])
                             else:
                                 p0_tuple, p1_tuple, p2_tuple, p3_tuple = start_point_tuple, control1_tuple_raw, control2_tuple_raw, end_point_tuple_raw

                             # Convert just before creating array
                             p0_arr = np.asarray(p0_tuple, dtype=float)
                             p1_arr = np.asarray(p1_tuple, dtype=float)
                             p2_arr = np.asarray(p2_tuple, dtype=float)
                             p3_arr = np.asarray(p3_tuple, dtype=float)

                             if p0_arr.shape != (2,) or p1_arr.shape != (2,) or p2_arr.shape != (2,) or p3_arr.shape != (2,):
                                 raise ValueError("Inconsistent shapes for Cubic points")

                             points = np.array([p0_arr, p1_arr, p2_arr, p3_arr]) # Shape (4, 2)

                             # --- Tessellation --- (similar logic as qCurveTo, use CubicSpline)
                             if np.allclose(p0_arr,p1_arr) and np.allclose(p1_arr,p2_arr) and np.allclose(p2_arr,p3_arr):
                                  p3_final_tuple = tuple(p3_arr)
                                  if not current_contour or not np.allclose(current_contour[-1], p3_final_tuple):
                                       current_contour.append(p3_final_tuple)
                                  continue

                             radius = max(np.linalg.norm(p1_arr-p0_arr), np.linalg.norm(p2_arr-p1_arr), np.linalg.norm(p3_arr-p2_arr))
                             radius = max(radius, EPSILON)
                             num_steps = get_fragments_from_fn_fa_fs(radius, self.fn, self.fa, self.fs)
                             num_steps = max(int(num_steps * 1.5), 12)

                             spline = CubicSpline(points)
                             t_values = np.linspace(0, 1, num_steps + 1)[1:]

                             if len(t_values) > 0:
                                 new_points = np.array([spline.evaluate(t) for t in t_values])
                                 if new_points.ndim == 2 and new_points.shape[1] == 2:
                                      current_contour.extend([tuple(pt) for pt in new_points])
                                 elif new_points.shape == (2,):
                                      current_contour.append(tuple(new_points))

                             # Ensure final point is added
                             final_cubic_pt_tuple = tuple(p3_arr)
                             if not current_contour or not np.allclose(current_contour[-1], final_cubic_pt_tuple):
                                  current_contour.append(final_cubic_pt_tuple)
                             # --- End Tessellation ---

                         except Exception as e:
                             print(f"  ERROR processing curveTo points: {e}")
                             # Recovery attempt...
                             try:
                                 final_pt_tuple = p3_tuple # Use tuple from start of block
                                 if not current_contour or not np.allclose(current_contour[-1], final_pt_tuple):
                                     current_contour.append(final_pt_tuple)
                             except: pass
                             continue

                    elif command == 'closePath':
                        if current_contour:
                            if not np.allclose(current_contour[0], current_contour[-1]):
                                current_contour.append(current_contour[0]) # Append tuple
                            polygons.append(np.array(current_contour, dtype=float)) # Convert final
                            current_contour = []

                except Exception as command_error:
                    print(f"ERROR processing command {command} {args}: {command_error}")
                    # Attempt to prevent cascade failure, maybe just reset contour?
                    # current_contour = [] # This might lose data, use with caution

            # Finalize any open contour
            if current_contour:
                polygons.append(np.array(current_contour, dtype=float))

            return polygons # List of numpy arrays

        except Exception as e:
            print(f"FATAL Error getting glyph outline for '{char_code}': {str(e)}")
            import traceback
            traceback.print_exc() # Print full traceback for fatal errors
            return []
    
    def _apply_alignment(self, polygons, metrics):
        """Apply alignment transformations to the polygons"""
        import copy
        aligned_polygons = copy.deepcopy(polygons)
        
        # Calculate offset based on alignment
        offset_x = 0
        offset_y = 0
        
        # Horizontal alignment
        if self.halign == "center":
            offset_x = -metrics['size'][0] / 2
        elif self.halign == "right":
            offset_x = -metrics['size'][0]
        
        # Vertical alignment
        if self.valign == "top":
            offset_y = metrics['ascent']
        elif self.valign == "center":
            offset_y = (metrics['ascent'] + metrics['descent']) / 2
        elif self.valign == "bottom":
            offset_y = metrics['descent']
        # "baseline" is the default, no adjustment needed
        
        # Apply offset to all points in all polygons
        for i, polygon in enumerate(aligned_polygons):
            aligned_polygons[i] = np.array([
                [p[0] + offset_x, p[1] + offset_y] for p in polygon
            ])
        
        return aligned_polygons
    
    def _determine_winding_order(self, polygon):
        """Determine if a polygon is clockwise (hole) or counterclockwise (solid)"""
        # Shoelace formula to calculate the signed area
        area = 0
        for i in range(len(polygon) - 1):
            area += (polygon[i+1][0] - polygon[i][0]) * (polygon[i+1][1] + polygon[i][1])
        
        # TTF/OTF fonts use the opposite convention from OpenSCAD:
        # - In fonts: outer contours are clockwise (positive area), holes are counterclockwise (negative area)
        # - For OpenSCAD: this needs to be reversed for the correct winding interpretation
        # 
        # Return True for counterclockwise (solid) and False for clockwise (hole)
        return area < 0
    
    def _check_and_fix_orientation(self, polygons):
        """Check if the orientation of glyphs is correct, if not flip them"""
        import copy
        corrected_polygons = copy.deepcopy(polygons)
        
        if not corrected_polygons:
            return corrected_polygons
            
        # First, check if we need to flip Y coordinates
        # Different fonts may use different coordinate systems
        # Some fonts like New Gulim might have inverted Y-axis
        
        # Sample points to determine orientation
        sample_y_values = []
        for poly in corrected_polygons[:min(5, len(corrected_polygons))]:
            if len(poly) > 2:  # Need at least 3 points to determine anything meaningful
                y_values = [p[1] for p in poly]
                sample_y_values.extend(y_values)
        
        if not sample_y_values:
            return corrected_polygons
            
        # For regular Latin fonts:
        # - Most points should be above the baseline (y > 0) for uppercase letters
        # - Descenders go below the baseline (y < 0)
        # 
        # For fonts with inverted Y-axis:
        # - Most points would be below the baseline (y < 0) for uppercase letters
        
        # Calculate what percentage of points are below baseline
        points_below_baseline = sum(1 for y in sample_y_values if y < 0)
        percentage_below = points_below_baseline / len(sample_y_values) if sample_y_values else 0
        
        # If more than 60% of points are below baseline, font might have inverted Y-axis
        y_flip_needed = percentage_below > 0.6
        
        if y_flip_needed:
            # Flip Y coordinates on all polygons
            for i, polygon in enumerate(corrected_polygons):
                corrected_polygons[i] = np.array([
                    [p[0], -p[1]] for p in polygon
                ])
        
        # Check winding direction of polygons - this is independent of Y-axis orientation
        areas = []
        for poly in corrected_polygons[:min(5, len(corrected_polygons))]:
            area = 0
            for i in range(len(poly) - 1):
                area += (poly[i+1][0] - poly[i][0]) * (poly[i+1][1] + poly[i][1])
            areas.append(area)
        
        # In TrueType/OpenType, outer contours should have positive area
        # If most have negative area, we might need to reverse contours
        flip_winding = sum(1 for a in areas if a < 0) > len(areas) / 2
        
        if flip_winding:
            # Reverse the winding order of all polygons
            for i, polygon in enumerate(corrected_polygons):
                corrected_polygons[i] = np.array(polygon[::-1])
        
        return corrected_polygons
    
    def _apply_text_direction(self, polygons, metrics):
        """Apply text direction transformations"""
        import copy
        directed_polygons = copy.deepcopy(polygons)
        
        # Handle direction
        if self.direction == "rtl":
            # Right-to-left: mirror horizontally
            for i, polygon in enumerate(directed_polygons):
                directed_polygons[i] = np.array([
                    [-p[0] + metrics['size'][0], p[1]] for p in polygon
                ])
        elif self.direction == "ttb":
            # Top-to-bottom: rotate 90 degrees clockwise
            for i, polygon in enumerate(directed_polygons):
                directed_polygons[i] = np.array([
                    [p[1], -p[0] + metrics['size'][0]] for p in polygon
                ])
        elif self.direction == "btt":
            # Bottom-to-top: rotate 90 degrees counterclockwise
            for i, polygon in enumerate(directed_polygons):
                directed_polygons[i] = np.array([
                    [-p[1] + metrics['size'][1], p[0]] for p in polygon
                ])
        
        return directed_polygons
    
    def get_polygons(self) -> tuple[np.ndarray, list[np.ndarray]]:
        """Returns the polygons for the text.
        
        Returns:
            tuple[np.ndarray, list[np.ndarray]]: A tuple containing the points array 
                and a list of contours (indices into the points array).
        """
        # Get metrics for the text
        metrics = self._calculate_metrics()
        
        # Get all character outlines
        all_polygons = []
        positions = {}  # Store positions for each character
        
        # Create an image to measure character advances
        img = self._image.new('RGB', (1, 1), color=(255, 255, 255))
        draw = self._image_draw.Draw(img)
        
        # Get advance widths using fontTools if available
        use_fonttools_advances = self._font is not None and hasattr(self._font, 'getGlyphSet')
        if use_fonttools_advances:
            try:
                glyph_set = self._font.getGlyphSet()
                hmtx = self._font['hmtx'].metrics
                cmap = self._font.getBestCmap()
            except:
                use_fonttools_advances = False
        
        # Use bidirectional text if available
        display_text = getattr(self, '_visual_text', self.text)
        
        # Process each character
        x_pos = 0
        char_advances = {}  # Store advance width for each character
        
        for i, char in enumerate(display_text):
            # Get character metrics - using fontTools for accurate advance width if available
            advance_width = 0
            
            if use_fonttools_advances:
                try:
                    glyph_name = cmap.get(ord(char))
                    if glyph_name and glyph_name in hmtx:
                        # Get the advance width from the horizontal metrics table
                        advance_width = hmtx[glyph_name][0]
                        # Scale by font size
                        scale_factor = self.size / self._font['head'].unitsPerEm
                        advance_width *= scale_factor
                    else:
                        # Fallback to PIL
                        use_fonttools_advances = False
                except:
                    use_fonttools_advances = False
            
            if not use_fonttools_advances:
                # Use PIL to get advance width
                char_bbox = draw.textbbox((0, 0), char, font=self._pil_font)
                advance_width = (char_bbox[2] - char_bbox[0]) * (self.size / self._pil_font.size)
                
                # If the advance width is too small (indicating unsupported glyph), use a reasonable minimum width
                if advance_width < 0.1:
                    advance_width = self.size * 0.5  # Use half the font size as a reasonable width
            
            # Store the current position for this character
            positions[i] = x_pos
            char_advances[i] = advance_width
            
            # Move to next character position
            x_pos += advance_width * self.spacing
        
        # Get character outlines in visual order
        for i, char in enumerate(display_text):
            # Get character outline
            char_polygons = self._get_glyph_outlines(char)
            
            # Apply transformation to position the character
            for j, polygon in enumerate(char_polygons):
                # Scale by font size - either using fontTools unitsPerEm or PIL font size
                if use_fonttools_advances:
                    scale_factor = self.size / self._font['head'].unitsPerEm
                else:
                    scale_factor = self.size / self._pil_font.size
                
                # Apply scaling and positioning
                char_polygons[j] = np.array([
                    [p[0] * scale_factor + positions[i], p[1] * scale_factor] for p in polygon
                ])
            
            # Add to all polygons
            all_polygons.extend(char_polygons)
        
        # Check and correct orientation if needed
        all_polygons = self._check_and_fix_orientation(all_polygons)
        
        # Apply alignment
        all_polygons = self._apply_alignment(all_polygons, metrics)
        
        # Apply text direction
        all_polygons = self._apply_text_direction(all_polygons, metrics)
        
        # Separate points and contours
        all_points = []
        contours = []
        
        for polygon in all_polygons:
            start_idx = len(all_points)
            all_points.extend(polygon)
            contours.append(np.arange(start_idx, start_idx + len(polygon)))
        
        return np.array(all_points), contours
    
    def get_polygons_at(self, pos: int) -> tuple[np.ndarray, list[np.ndarray]]:
        """Returns the polygons for the character at the given position.
        
        Args:
            pos: The position of the character in the text.
            
        Returns:
            tuple[np.ndarray, list[np.ndarray]]: A tuple containing the points array 
                and a list of contours (indices into the points array).
        """
        if pos < 0 or pos >= len(self.text):
            raise ValueError(f"Position {pos} is out of range for text of length {len(self.text)}")
        
        # Get metrics for the text
        metrics = self._calculate_metrics()
        
        # Create an image to measure character advances
        img = self._image.new('RGB', (1, 1), color=(255, 255, 255))
        draw = self._image_draw.Draw(img)
        
        # Get advance widths using fontTools if available
        use_fonttools_advances = self._font is not None and hasattr(self._font, 'getGlyphSet')
        if use_fonttools_advances:
            try:
                glyph_set = self._font.getGlyphSet()
                hmtx = self._font['hmtx'].metrics
                cmap = self._font.getBestCmap()
            except:
                use_fonttools_advances = False
                
        # Use bidirectional text if available
        if hasattr(self, '_visual_text') and hasattr(self, '_char_mapping'):
            # Map logical position to visual position
            visual_pos = self._char_mapping[pos]
            display_text = self._visual_text
        else:
            visual_pos = pos
            display_text = self.text
        
        # Calculate position of the character
        x_pos = 0
        char = display_text[visual_pos]  # Get the target character
        
        # Calculate position of the character by adding advances of all previous characters
        for i in range(visual_pos):
            c = display_text[i]
            advance_width = 0
            
            if use_fonttools_advances:
                try:
                    glyph_name = cmap.get(ord(c))
                    if glyph_name and glyph_name in hmtx:
                        # Get the advance width from the horizontal metrics table
                        advance_width = hmtx[glyph_name][0]
                        # Scale by font size
                        scale_factor = self.size / self._font['head'].unitsPerEm
                        advance_width *= scale_factor
                    else:
                        # Fallback to PIL
                        char_bbox = draw.textbbox((0, 0), c, font=self._pil_font)
                        advance_width = (char_bbox[2] - char_bbox[0]) * (self.size / self._pil_font.size)
                        
                        # If the advance width is too small (indicating unsupported glyph), use a reasonable minimum width
                        if advance_width < 0.1:
                            advance_width = self.size * 0.5  # Use half the font size as a reasonable width
                except:
                    # Fallback to PIL
                    char_bbox = draw.textbbox((0, 0), c, font=self._pil_font)
                    advance_width = (char_bbox[2] - char_bbox[0]) * (self.size / self._pil_font.size)
                    
                    # If the advance width is too small (indicating unsupported glyph), use a reasonable minimum width
                    if advance_width < 0.1:
                        advance_width = self.size * 0.5  # Use half the font size as a reasonable width
            else:
                # Use PIL to get advance width
                char_bbox = draw.textbbox((0, 0), c, font=self._pil_font)
                advance_width = (char_bbox[2] - char_bbox[0]) * (self.size / self._pil_font.size)
                
                # If the advance width is too small (indicating unsupported glyph), use a reasonable minimum width
                if advance_width < 0.1:
                    advance_width = self.size * 0.5  # Use half the font size as a reasonable width
            
            # Move to next character position
            x_pos += advance_width * self.spacing
        
        # Get character outline for the target character
        char_polygons = self._get_glyph_outlines(char)
        
        # Apply transformation to position the character
        for i, polygon in enumerate(char_polygons):
            # Scale by font size - either using fontTools unitsPerEm or PIL font size
            if use_fonttools_advances:
                scale_factor = self.size / self._font['head'].unitsPerEm
            else:
                scale_factor = self.size / self._pil_font.size
            
            # Apply scaling and positioning
            char_polygons[i] = np.array([
                [p[0] * scale_factor + x_pos, p[1] * scale_factor] for p in polygon
            ])
        
        # Check and correct orientation if needed
        char_polygons = self._check_and_fix_orientation(char_polygons)
        
        # Apply alignment
        char_polygons = self._apply_alignment(char_polygons, metrics)
        
        # Apply text direction
        char_polygons = self._apply_text_direction(char_polygons, metrics)
        
        # Separate points and contours
        all_points = []
        contours = []
        
        for polygon in char_polygons:
            start_idx = len(all_points)
            all_points.extend(polygon)
            contours.append(np.arange(start_idx, start_idx + len(polygon)))
        
        return np.array(all_points), contours

def get_fragments_from_fn_fa_fs(r: float, fn: int | None, fa: float | None, fs: float | None) -> int:
    # From openscad/src/utils/calc.cpp
    # int Calc::get_fragments_from_r(double r, double fn, double fs, double fa)
    # {
    #   // FIXME: It would be better to refuse to create an object. Let's do more strict error handling
    #   // in future versions of OpenSCAD
    #   if (r < GRID_FINE || std::isinf(fn) || std::isnan(fn)) return 3;
    #   if (fn > 0.0) return static_cast<int>(fn >= 3 ? fn : 3);
    #   return static_cast<int>(ceil(fmax(fmin(360.0 / fa, r * 2 * M_PI / fs), 5)));
    # }
    GRID_FINE = 0.00000095367431640625
    if r < GRID_FINE:
        return 3
    if fn is not None and (np.isinf(fn) or np.isnan(fn)):
        return 3
    if fn is not None and fn > 0:
        return int(max(fn, 3))
    if fa is None:
        fa = 12.0  # Default fragment angle
    if fs is None:
        fs = 2.0   # Default fragment size
    return int(max(5, np.ceil(min(360.0 / fa, r * 2 * np.pi / fs))))

def get_available_fonts():
    """Returns a list of available fonts on the system.
    
    Returns:
        dict: A dictionary mapping font family names to lists of available styles.
    """
    from PIL import ImageFont
    import os
    
    available_fonts = {}
    
    # Try to get system fonts directory
    fonts_dir = None
    
    # Windows
    if os.name == 'nt':
        import ctypes.wintypes
        CSIDL_FONTS = 0x14
        try:
            buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
            ctypes.windll.shell32.SHGetFolderPathW(0, CSIDL_FONTS, 0, 0, buf)
            fonts_dir = buf.value
        except Exception as e:
            print(f"Error getting Windows fonts directory: {e}")
    
    # Linux/Unix
    elif os.name == 'posix':
        potential_dirs = [
            '/usr/share/fonts',
            '/usr/local/share/fonts',
            os.path.expanduser('~/.fonts'),
            os.path.expanduser('~/.local/share/fonts')
        ]
        for dir_path in potential_dirs:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                fonts_dir = dir_path
                break
    
    # macOS
    elif os.name == 'darwin':
        potential_dirs = [
            '/Library/Fonts',
            '/System/Library/Fonts',
            os.path.expanduser('~/Library/Fonts')
        ]
        for dir_path in potential_dirs:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                fonts_dir = dir_path
                break
    
    if fonts_dir:
        # Walk through fonts directory and find all font files
        for root, _, files in os.walk(fonts_dir):
            for file in files:
                if file.lower().endswith(('.ttf', '.otf')):
                    try:
                        font_path = os.path.join(root, file)
                        # Try to load the font to get its family name
                        try:
                            # Use size=12 as a default size
                            font = ImageFont.truetype(font_path, size=12)
                            
                            # Try to get font family and style
                            family = font.getname()[0]
                            style = "Regular"  # Default style
                            
                            # Try to detect style from filename
                            file_lower = file.lower()
                            if 'bold' in file_lower and 'italic' in file_lower:
                                style = "Bold Italic"
                            elif 'bold' in file_lower:
                                style = "Bold"
                            elif 'italic' in file_lower or 'oblique' in file_lower:
                                style = "Italic"
                            
                            # Add to available fonts
                            if family not in available_fonts:
                                available_fonts[family] = []
                            
                            if style not in available_fonts[family]:
                                available_fonts[family].append(style)
                            
                        except Exception as e:
                            # Skip this font if it cannot be loaded
                            continue
                    except Exception as e:
                        # Skip this font if there's an error
                        continue
    
    # Check builtin fonts
    try:
        # These fonts are likely to be included with PIL
        default_fonts = ["arial.ttf", "times.ttf", "cour.ttf", "DejaVuSans.ttf", "DejaVuSerif.ttf"]
        for font_name in default_fonts:
            try:
                font = ImageFont.truetype(font_name, size=12)
                family = font.getname()[0]
                
                if family not in available_fonts:
                    available_fonts[family] = ["Regular"]
                elif "Regular" not in available_fonts[family]:
                    available_fonts[family].append("Regular")
            except:
                pass
    except Exception:
        pass
    
    return available_fonts

def get_fonts_list():
    """Returns a list of available fonts in OpenSCAD format (family:style=Style).
    
    Returns:
        list: A list of strings in the format "FamilyName:style=Style"
    """
    fonts_dict = get_available_fonts()
    fonts_list = []
    
    for family, styles in fonts_dict.items():
        for style in styles:
            if style == "Regular":
                fonts_list.append(family)  # Regular style doesn't need to be specified
            else:
                fonts_list.append(f"{family}:style={style}")
    
    return sorted(fonts_list)

def text(text: str, 
            size: float=10, 
            font: str="Liberation Sans", 
            halign: str="left", 
            valign: str="baseline", 
            spacing: float=1.0, 
            direction: str="ltr", 
            language: str="en", 
            script: str="latin", 
            fa: float=12.0, 
            fs: float=2.0, 
            fn: int=0,
            base_direction: str="ltr") -> tuple[np.ndarray, list[np.ndarray]]:
    """Render text as polygons.
    
    Args:
        text: The text to render.
        size: The size of the text.
        font: The font to use.
        halign: Horizontal alignment ('left', 'center', 'right').
        valign: Vertical alignment ('top', 'center', 'baseline', 'bottom').
        spacing: Character spacing factor.
        direction: Text direction ('ltr', 'rtl', 'ttb', 'btt').
        language: Text language.
        script: Text script.
        fa: Fragment angle.
        fs: Fragment size.
        fn: Fragment number.
        base_direction: Base direction for bidirectional text ('ltr' or 'rtl').
        
    Returns:
        tuple[np.ndarray, list[np.ndarray]]: A tuple containing the points array 
            and a list of contours (indices into the points array).
    """
    context = TextContext(
        text=text,
        size=size,
        font=font,
        halign=halign,
        valign=valign,
        spacing=spacing,
        direction=direction,
        language=language,
        script=script,
        fa=fa,
        fs=fs,
        fn=fn,
        base_direction=base_direction
    )
    return context.get_polygons()

    
# Example Usage (optional)
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Display available fonts if requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--list-fonts':
        print("Available Fonts:")
        fonts = get_fonts_list()
        for font in fonts:
            print(f"  {font}")
        sys.exit(0)
    
    # Or display available fonts by family
    if len(sys.argv) > 1 and sys.argv[1] == '--list-fonts-by-family':
        print("Available Font Families:")
        fonts = get_available_fonts()
        for family, styles in sorted(fonts.items()):
            print(f"{family}:")
            for style in styles:
                print(f"  - {style}")
        sys.exit(0)

    # Read font from command line if provided
    font_to_use = "New Gulim"
    font_to_use = "Arial"
    font_to_use = "Times New Roman"
    font_to_use = "Courier New"
    if len(sys.argv) > 1 and sys.argv[1] != '--list-fonts' and sys.argv[1] != '--list-fonts-by-family':
        font_to_use = sys.argv[1]

    print("Generating text...")
    
    # Test bidirectional text
    # Mix of English, Arabic, and Hebrew
    mixed_text = "Hello! مرحباً שלום"
    
    print(f"Testing bidirectional text: {mixed_text}")
    
    try:
        # Regular text first with high quality setting (fn=128 for smooth curves)
        polygons1 = text(
            "Test!", 
            size=25, 
            font=font_to_use, 
            halign="center", 
            valign="center", 
            fn=128)
            
        # Bidirectional text with LTR base direction
        polygons2 = text(
            mixed_text,
            size=25, 
            font=font_to_use, 
            halign="center", 
            valign="center",
            fn=128,
            base_direction="ltr")
            
        # Bidirectional text with RTL base direction
        polygons3 = text(
            mixed_text,
            size=25, 
            font=font_to_use, 
            halign="center", 
            valign="center",
            fn=128,
            base_direction="rtl")

    except Exception as e:
        print(f"Error: {e}")
        print("\nTry running the script with --list-fonts to see available fonts")
        print("Example usage:")
        print("  python text_render.py --list-fonts")
        print("  python text_render.py \"Arial:style=Bold\"")
        sys.exit(1)

    print(f"Generated {len(polygons1[1])} polygons for 'Test!'")
    print(f"Generated {len(polygons2[1])} polygons for bidirectional text (LTR base)")
    print(f"Generated {len(polygons3[1])} polygons for bidirectional text (RTL base)")

    # Create subplot
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # Function to determine polygon winding order
    def is_clockwise(points):
        """Determine if polygon has clockwise winding order using shoelace formula."""
        # Return True for clockwise (positive area), False for counterclockwise (negative area)
        if len(points) < 3:
            return False
        area = 0
        for i in range(len(points) - 1):
            area += (points[i+1][0] - points[i][0]) * (points[i+1][1] + points[i][1])
        return area > 0
    
    # Plot regular text
    points, contours = polygons1
    for contour in contours:
        contour_points = points[contour]
        # Determine winding and choose styling accordingly
        clockwise = is_clockwise(contour_points)
        fill_color = 'blue'  # Keep original color
        edge_color = 'red' if clockwise else 'black'  # Red edge for inverted
        hatch_pattern = '///' if clockwise else None  # Hatching for inverted
        
        patch = plt.Polygon(contour_points, closed=True, facecolor=fill_color, 
                            edgecolor=edge_color, alpha=0.6, hatch=hatch_pattern)
        axes[0].add_patch(patch)
    
    axes[0].set_aspect('equal', adjustable='box')
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    axes[0].set_xlim(min_x - 5, max_x + 5)
    axes[0].set_ylim(min_y - 5, max_y + 5)
    axes[0].set_title(f"Regular Text: 'Test!'")
    axes[0].grid(True)
    
    # Plot bidirectional text with LTR base
    points, contours = polygons2
    for contour in contours:
        contour_points = points[contour]
        # Determine winding and choose styling accordingly
        clockwise = is_clockwise(contour_points)
        fill_color = 'green'  # Keep original color
        edge_color = 'red' if clockwise else 'black'  # Red edge for inverted
        hatch_pattern = '///' if clockwise else None  # Hatching for inverted
        
        patch = plt.Polygon(contour_points, closed=True, facecolor=fill_color, 
                            edgecolor=edge_color, alpha=0.6, hatch=hatch_pattern)
        axes[1].add_patch(patch)
    
    axes[1].set_aspect('equal', adjustable='box')
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    axes[1].set_xlim(min_x - 5, max_x + 5)
    axes[1].set_ylim(min_y - 5, max_y + 5)
    axes[1].set_title(f"Bidirectional Text (LTR base): '{mixed_text}'")
    axes[1].grid(True)
    
    # Plot bidirectional text with RTL base
    points, contours = polygons3
    for contour in contours:
        contour_points = points[contour]
        # Determine winding and choose styling accordingly
        clockwise = is_clockwise(contour_points)
        fill_color = 'red'  # Keep original color
        edge_color = 'red' if clockwise else 'black'  # Red edge for inverted
        hatch_pattern = '///' if clockwise else None  # Hatching for inverted
        
        patch = plt.Polygon(contour_points, closed=True, facecolor=fill_color, 
                            edgecolor=edge_color, alpha=0.6, hatch=hatch_pattern)
        axes[2].add_patch(patch)
    
    axes[2].set_aspect('equal', adjustable='box')
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    axes[2].set_xlim(min_x - 5, max_x + 5)
    axes[2].set_ylim(min_y - 5, max_y + 5)
    axes[2].set_title(f"Bidirectional Text (RTL base): '{mixed_text}'")
    axes[2].grid(True)
    
    # Add a legend explaining the hatching
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', edgecolor='black', alpha=0.6, label='Normal Winding'),
        Patch(facecolor='blue', edgecolor='red', hatch='///', alpha=0.6, label='Inverted Winding')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)  # Make room for the legend
    
    # Save figure to file for viewing
    plt.savefig("text_render_bidi_test.png", dpi=150)
    
    print("Figure saved to 'text_render_bidi_test.png'")
    
    try:
        plt.show()
    except:
        print("Could not display plot - but image was saved to file")