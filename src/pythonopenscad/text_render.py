import numpy as np
from datatrees import datatree, dtfield
import anchorscad_lib.linear as l
import logging
import sys
import os
import io
from functools import lru_cache
from pythonopenscad.text_utils import CubicSpline, QuadraticSpline, extentsof

"""
Text Rendering Module
--------------------

This module implements a text rendering system using a combination of two specialized libraries:

1. fontTools/PIL (FreeType-based):
   - Used for font discovery, loading, and accessing font properties
   - Provides access to glyph outlines and vector data
   - Handles font file parsing and metrics

2. HarfBuzz:
   - Used for complex text shaping across different writing systems
   - Manages bidirectional text (mixed RTL and LTR scripts)
   - Handles proper glyph substitution, positioning, and OpenType features
   - Applies language-specific rendering rules

The workflow involves:
1. Loading the font file using fontTools/PIL
2. Creating a HarfBuzz font object from the font data
3. Using HarfBuzz for text shaping to get properly positioned glyphs
4. Extracting glyph outlines via fontTools
5. Converting the outlines to polygon data for rendering

This dual-library approach is standard in modern text rendering pipelines, where
each library handles the tasks it specializes in.

- Claude (2024)
"""

try:
    import uharfbuzz as hb
except ImportError:
    print("ERROR: uharfbuzz library not found. Text shaping will be unavailable.", file=sys.stderr)
    print("Please install it: pip install uharfbuzz fonttools[ufo]", file=sys.stderr)
    hb = None

try:
    import fontTools.ttLib
    from fontTools.pens.recordingPen import RecordingPen
    from PIL import ImageFont
except ImportError:
    print(
        "ERROR: fontTools or PIL library not found. Font handling will be unavailable.",
        file=sys.stderr,
    )
    print("Please install it: pip install fonttools[ufo]", file=sys.stderr)
    fontTools = None
    RecordingPen = None
    ImageFont = None


logging.basicConfig(level=logging.WARNING)  # Or logging.WARNING
log = logging.getLogger(__name__)

EPSILON = 1e-6


class FontCacheEntry:
    """
    Handles loading and caching of font data for both fontTools and HarfBuzz.
    Acts as a container for the loaded font objects.
    """

    def __init__(self, font_name, size=10.0):
        self.font_name = font_name
        self.size = size
        self.pil_font = None  # PIL ImageFont object
        self.ft_font = None  # fontTools TTFont object
        self.hb_font = None  # HarfBuzz Font object
        self.glyph_set = None  # fontTools GlyphSet
        self.y_axis_inverted = False
        self.scale_factor = 1.0
        self.fallback_fonts = {}

        self._load_font()
        self._setup_harfbuzz()
        self._detect_font_orientation()
        self._load_fallback_fonts()

    def _load_font(self):
        """Load the font using PIL and fontTools"""
        if not ImageFont or not fontTools:
            raise RuntimeError("Font loading dependencies (PIL/fontTools) not available.")

        font_path_found = None
        font_name, font_style = self.font_name, None

        if ":" in self.font_name:
            parts = self.font_name.split(":", 1)
            font_name = parts[0]
            if len(parts) > 1 and parts[1].startswith("style="):
                font_style = parts[1][6:]

        # Try loading with PIL (using path or name)
        try:
            # Attempt direct path/name load with PIL
            self.pil_font = ImageFont.truetype(self.font_name, size=int(self.size * 3.937))
            # If successful, try to get path for fontTools
            if hasattr(self.pil_font, "path"):
                font_path_found = self.pil_font.path
        except OSError:
            # If direct load fails, try finding font file path
            log.debug(f"Direct PIL load failed for '{self.font_name}', searching system...")
            font_path_found = self._find_font_file(font_name, font_style)
            if font_path_found:
                try:
                    self.pil_font = ImageFont.truetype(font_path_found, size=int(self.size * 3.937))
                except OSError as e:
                    log.warning(f"PIL failed to load found path '{font_path_found}': {e}")
                    font_path_found = None  # Reset if PIL fails on found path
            else:
                log.warning(
                    f"Could not find font file for '{font_name}' with style '{font_style}'."
                )

        # If PIL loading failed entirely, try fallbacks
        if self.pil_font is None:
            log.warning(
                f"Could not load primary font '{self.font_name}' with PIL. Trying fallbacks..."
            )
            fallback_fonts = [
                "Arial",
                "Times New Roman",
                "Verdana",
                "DejaVu Sans",
                "Liberation Sans",
            ]
            for fallback in fallback_fonts:
                try:
                    self.pil_font = ImageFont.truetype(fallback, size=int(self.size * 3.937))
                    log.debug(f"Using fallback PIL font '{fallback}'.")
                    # Try to get path for fontTools fallback
                    if hasattr(self.pil_font, "path"):
                        font_path_found = self.pil_font.path
                    else:
                        font_path_found = self._find_font_file(fallback, None)
                    break  # Stop on first successful fallback
                except OSError:
                    continue  # Try next fallback
            if self.pil_font is None:
                raise ValueError(
                    f"Could not load font '{self.font_name}' or any fallback fonts with PIL."
                )

        # Now load with fontTools, using found path if available
        if font_path_found:
            try:
                self.ft_font = fontTools.ttLib.TTFont(font_path_found)
            except fontTools.ttLib.TTLibError as e:
                log.error(f"fontTools failed to load path '{font_path_found}': {e}")
                self.ft_font = None  # Ensure None on failure
        else:
            # If no path, try loading by name (might work for system fonts sometimes)
            try:
                # Use the potentially updated font_name if PIL used a fallback
                effective_font_name = self.pil_font.getname()[0] if self.pil_font else font_name
                self.ft_font = fontTools.ttLib.TTFont(effective_font_name)
            except fontTools.ttLib.TTLibError as e:
                log.error(f"fontTools failed to load font by name '{effective_font_name}': {e}")
                self.ft_font = None

        if self.ft_font is None:
            raise ValueError(f"fontTools could not load font '{self.font_name}'")

        # Get units per EM and calculate scale factor
        try:
            units_per_em = self.ft_font["head"].unitsPerEm
            if units_per_em <= 0:
                units_per_em = 1000  # Fallback default
        except Exception as e:
            units_per_em = 1000  # Fallback
            log.warning(f"Could not read unitsPerEm from font: {e}. Assuming {units_per_em}")

        self.scale_factor = self.size / units_per_em

        # Get glyph set for drawing outlines
        try:
            self.glyph_set = self.ft_font.getGlyphSet()
        except Exception as e:
            log.error(f"Failed to get glyphSet from font: {e}", exc_info=True)
            self.glyph_set = None
            raise ValueError(f"Failed to get glyphSet from font: {e}") from e

    def _setup_harfbuzz(self):
        """Create a HarfBuzz font object from the fontTools font data"""
        if not hb:
            raise ImportError("HarfBuzz (uharfbuzz) is required for text shaping")

        try:
            # Create an in-memory binary buffer
            mem_file = io.BytesIO()

            # Save the font to the in-memory buffer
            self.ft_font.save(mem_file)

            # Get the font data from the buffer
            mem_file.seek(0)
            face_data = mem_file.read()

            # Create HarfBuzz face and font objects
            hb_face = hb.Face(face_data)
            self.hb_font = hb.Font(hb_face)

            # Set scale based on unitsPerEm
            units_per_em = hb_face.upem
            if units_per_em <= 0:
                units_per_em = 1000

            self.hb_font.scale = (units_per_em, units_per_em)

        except Exception as e:
            log.error(f"Failed to create HarfBuzz font object: {e}", exc_info=True)
            self.hb_font = None  # Ensure it's None on failure
            raise ValueError(f"Failed to create HarfBuzz font object: {e}") from e

    def _detect_font_orientation(self):
        """Detect if the font has inverted Y axis"""
        self.y_axis_inverted = False
        known_inverted_fonts = [
            "New Gulim",
            "Gulim",
            "GulimChe",
            "Dotum",
            "DotumChe",
            "MS Mincho",
            "MS Gothic",
            "MS PMincho",
            "MS PGothic",
            "SimSun",
            "NSimSun",
            "SimHei",
            "FangSong",
            "KaiTi",
        ]

        if (
            self.ft_font
            and hasattr(self.ft_font, "sfntVersion")
            and self.ft_font.sfntVersion == "ttcf"
        ):
            pass
        elif any(inverted_font in self.font_name for inverted_font in known_inverted_fonts):
            self.y_axis_inverted = True
            return

        if self.ft_font and hasattr(self.ft_font, "getBestCmap"):
            try:
                cmap = self.ft_font.getBestCmap()
                glyph_set = self.ft_font.getGlyphSet()
                if ord("A") in cmap:
                    pen = RecordingPen()
                    glyph_name = cmap.get(ord("A"))
                    if glyph_name and glyph_name in glyph_set:
                        glyph_set[glyph_name].draw(pen)
                        y_values = [
                            args[0][1]
                            for cmd, args in pen.value
                            if cmd in ("moveTo", "lineTo") and args
                        ]
                        if y_values:
                            if sum(1 for y in y_values if y < 0) / len(y_values) > 0.7:
                                self.y_axis_inverted = True
                                log.info(f"Detected inverted Y axis for font '{self.font_name}'")
            except Exception as e:
                log.warning(f"Warning: Error detecting font orientation: {e}")

    def _load_fallback_fonts(self):
        """Load fallback fonts for character support"""
        self.fallback_fonts = {}
        if not ImageFont:
            return  # Skip if PIL failed

        fallback_font_names = ["Arial", "Times New Roman", "DejaVu Sans", "Liberation Sans"]
        try:
            for font_name in fallback_font_names:
                if font_name != self.font_name:
                    try:
                        # Use a nominal size for loading, actual size doesn't matter for support check
                        fallback_font = ImageFont.truetype(font_name, size=10)
                        self.fallback_fonts[font_name] = fallback_font
                    except OSError:
                        pass  # Ignore if fallback not found
        except Exception as e:
            log.warning(f"Warning: Error loading fallback fonts: {e}")

    def _find_font_file(self, family, style):
        """Find font file path based on family and style name"""
        import platform

        font_dirs = []
        system = platform.system()
        if system == "Windows":
            windir = os.environ.get("WINDIR", "C:\\Windows")
            font_dirs.append(os.path.join(windir, "Fonts"))
        elif system == "Darwin":  # macOS
            font_dirs.extend([
                "/Library/Fonts",
                "/System/Library/Fonts",
                os.path.expanduser("~/Library/Fonts"),
            ])
        else:  # Linux/Unix
            font_dirs.extend([
                "/usr/share/fonts",
                "/usr/local/share/fonts",
                os.path.expanduser("~/.fonts"),
                os.path.expanduser("~/.local/share/fonts"),
            ])

        family_lower = family.lower()
        style_lower = style.lower() if style else "regular"

        found_path = None
        best_match_score = -1

        for d in font_dirs:
            if not os.path.isdir(d):
                continue
            try:
                for root, _, files in os.walk(d):
                    for fname in files:
                        if fname.lower().endswith((".ttf", ".otf")):
                            fpath = os.path.join(root, fname)
                            try:
                                # Use fontTools to check name/style if possible
                                temp_font = fontTools.ttLib.TTFont(fpath, lazy=True)
                                # Check Name table for Family and Style
                                f_family, f_style = None, None
                                for rec in temp_font["name"].names:
                                    if rec.nameID == 1:
                                        f_family = rec.toUnicode()  # Family
                                    if rec.nameID == 2:
                                        f_style = rec.toUnicode()  # Style
                                temp_font.close()  # Close lazy loaded font

                                if f_family and f_family.lower() == family_lower:
                                    current_style = (f_style or "Regular").lower()
                                    score = 0
                                    if current_style == style_lower:
                                        score = 10  # Exact match
                                    elif "regular" in style_lower and "regular" in current_style:
                                        score = 5
                                    elif "bold" in style_lower and "bold" in current_style:
                                        score = 5
                                    elif "italic" in style_lower and "italic" in current_style:
                                        score = 5
                                    # Basic style matching
                                    elif (
                                        style_lower == "regular"
                                        and "bold" not in current_style
                                        and "italic" not in current_style
                                    ):
                                        score = 3
                                    elif style is None and current_style == "regular":
                                        score = 8  # Prefer regular if no style requested

                                    if score > best_match_score:
                                        best_match_score = score
                                        found_path = fpath
                                        if score == 10:
                                            return found_path  # Found exact match

                            except Exception:
                                continue  # Ignore fonts that fail to load/parse
            except OSError:
                continue  # Ignore directories we can't access

        if found_path:
            log.debug(
                f"Found font file '{found_path}' for family '{family}' style '{style}' (match score {best_match_score})"
            )
        return found_path


@lru_cache(maxsize=10)
def get_font(font_name, size=10.0) -> FontCacheEntry:
    """
    Factory function that creates or returns a cached FontCache object.
    Uses LRU caching to keep the most recently used fonts in memory.

    Args:
        font_name: Name of the font or path to font file
        size: Font size in units

    Returns:
        FontCache object with loaded fonts
    """
    return FontCacheEntry(font_name, size)


@datatree
class TextContext:
    text: str = dtfield(default="")
    size: float = dtfield(default=10.0)
    font: str = dtfield(default="Liberation Sans")
    halign: str = dtfield(default="left")
    valign: str = dtfield(default="baseline")
    spacing: float = dtfield(default=1.0)
    direction: str = dtfield(
        default="ltr"
    )  # Note: Used for final transform, hb uses base_direction
    language: str = dtfield(default="en")
    script: str = dtfield(default="latin")  # Used as hint for HarfBuzz
    fa: float = dtfield(default=12.0)
    fs: float = dtfield(default=2.0)
    fn: int = dtfield(default=0)
    base_direction: str = dtfield(default="ltr")  # Used for HarfBuzz direction
    quality: float = dtfield(default=1.0)  # Controls curve tessellation quality/density

    # --- Internal fields ---
    _font: object = dtfield(init=False, repr=False, default=None)  # fontTools TTFont object
    _hb_font: object = dtfield(init=False, repr=False, default=None)  # uharfbuzz Font object
    _glyph_set: object = dtfield(init=False, repr=False, default=None)  # fontTools glyph set
    _scale_factor: float = dtfield(init=False, repr=False, default=1.0)
    _pil_font: object = dtfield(
        init=False, repr=False, default=None
    )  # Keep PIL for fallback/metrics if needed
    _y_axis_inverted: bool = dtfield(init=False, repr=False, default=False)
    _fallback_fonts: dict = dtfield(init=False, repr=False, default_factory=dict)
    _font_cache: object = dtfield(init=False, repr=False, default=None)  # FontCache object

    def __post_init__(self):
        # --- Import heavy libraries once ---
        if fontTools is None:
            raise ImportError("Missing required libraries (Pillow or fontTools)")

        if hb is None:
            raise ImportError(
                "uharfbuzz library is required for text shaping but could not be imported."
            )

        # --- Keep validations ---
        if not self.font:
            self.font = "Liberation Sans"
        if self.halign not in ("left", "center", "right"):
            raise ValueError(f"Invalid halign: {self.halign}")
        if self.valign not in ("top", "center", "baseline", "bottom"):
            raise ValueError(f"Invalid valign: {self.valign}")
        if self.direction not in ("ltr", "rtl", "ttb", "btt"):
            raise ValueError(f"Invalid direction: {self.direction}")
        if self.base_direction not in ("ltr", "rtl"):
            raise ValueError(f"Invalid base_direction: {self.base_direction}")
        # --- End validations ---

        # --- Load Font ---
        try:
            # Use the cached font loader
            self._font_cache = get_font(self.font, self.size)

            # Get references to the loaded font objects
            self._font = self._font_cache.ft_font
            self._hb_font = self._font_cache.hb_font
            self._glyph_set = self._font_cache.glyph_set
            self._pil_font = self._font_cache.pil_font
            self._scale_factor = self._font_cache.scale_factor
            self._y_axis_inverted = self._font_cache.y_axis_inverted
            self._fallback_fonts = self._font_cache.fallback_fonts
        except Exception as e:
            log.error(f"Error loading font: {e}", exc_info=True)
            raise ValueError(f"Failed to load font: {e}") from e

    def _get_glyph_outlines_by_name(self, glyph_name):
        """Gets raw outlines in font units for a specific glyph name."""
        # Initialize quality_factor and fn_scaled at the beginning of the function
        quality_factor = 1.0  # Default quality
        if hasattr(self, "quality") and self.quality > 0:
            quality_factor = self.quality

        fn_scaled = 0
        if hasattr(self, "fn") and self.fn is not None and self.fn > 0:
            fn_scaled = self.fn * quality_factor

        if not self._glyph_set or glyph_name not in self._glyph_set:
            log.warning(f"Glyph name '{glyph_name}' not found in glyph set.")
            # Return a fallback square shape in font units
            units_per_em = self._font["head"].unitsPerEm if self._font else 1000
            fallback_size = units_per_em * 0.6  # approx size in font units
            return [
                np.array(
                    [
                        [0, 0],
                        [fallback_size, 0],
                        [fallback_size, fallback_size],
                        [0, fallback_size],
                        [0, 0],
                    ],
                    dtype=float,
                )
            ]

        pen = RecordingPen()
        try:
            glyph = self._glyph_set[glyph_name]
            glyph.draw(pen)
        except Exception as e:
            log.error(f"Error drawing glyph '{glyph_name}': {e}", exc_info=True)
            return []  # Return empty on error

        raw_polygons = []
        current_contour = []  # Store tuples

        for command, args in pen.value:
            try:
                if command == "moveTo":
                    if current_contour:
                        raw_polygons.append(np.array(current_contour, dtype=float))
                    if args and len(args[0]) == 2:
                        current_contour = [tuple(args[0])]  # Start fresh with tuple
                    else:
                        current_contour = []

                elif command == "lineTo":
                    if current_contour and args and len(args[0]) == 2:
                        current_contour.append(tuple(args[0]))  # Append tuple

                elif command == "qCurveTo":
                    if not args:
                        continue
                    if not current_contour:
                        continue
                    start_point_tuple = current_contour[-1]
                    args_tuples = [
                        tuple(p) for p in args if isinstance(p, (list, tuple)) and len(p) == 2
                    ]
                    if len(args_tuples) != len(args):
                        continue
                    full_pts_tuples = [start_point_tuple] + args_tuples
                    if len(full_pts_tuples) > 1 and np.allclose(
                        full_pts_tuples[0], full_pts_tuples[1]
                    ):
                        full_pts_tuples = [full_pts_tuples[0]] + full_pts_tuples[2:]
                    if len(full_pts_tuples) < 3:
                        if not np.allclose(current_contour[-1], full_pts_tuples[-1]):
                            current_contour.append(full_pts_tuples[-1])
                        continue
                    num_bezier_segments = len(full_pts_tuples) - 2
                    for i in range(num_bezier_segments):
                        p1_tuple = full_pts_tuples[i + 1]
                        if i == 0:
                            p0_tuple = full_pts_tuples[0]
                        else:
                            p0_tuple = tuple(
                                (np.asarray(full_pts_tuples[i]) + np.asarray(p1_tuple)) / 2.0
                            )
                        if i == num_bezier_segments - 1:
                            p2_tuple = full_pts_tuples[i + 2]
                        else:
                            p2_tuple = tuple(
                                (np.asarray(p1_tuple) + np.asarray(full_pts_tuples[i + 2])) / 2.0
                            )

                        p0_arr, p1_arr, p2_arr = (
                            np.asarray(p0_tuple),
                            np.asarray(p1_tuple),
                            np.asarray(p2_tuple),
                        )
                        if p0_arr.shape != (2,) or p1_arr.shape != (2,) or p2_arr.shape != (2,):
                            continue
                        bezier_points = np.array([p0_arr, p1_arr, p2_arr])
                        if np.allclose(p0_arr, p1_arr) and np.allclose(p1_arr, p2_arr):
                            if not np.allclose(current_contour[-1], p2_tuple):
                                current_contour.append(p2_tuple)
                            continue
                        radius = max(
                            np.linalg.norm(p1_arr - p0_arr),
                            np.linalg.norm(p2_arr - p1_arr),
                            EPSILON,
                        )

                        # Apply quality parameter for tessellation density
                        quality_factor = 1.0  # Default quality factor
                        if hasattr(self, "quality") and self.quality > 0:
                            quality_factor = self.quality

                        # Determine number of segments
                        if fn_scaled > 0:
                            num_steps = max(int(fn_scaled), 6)  # Minimum 6 for quadratic
                        else:
                            # Get steps using get_fragments_from_fn_fa_fs with quality factor
                            num_steps = get_fragments_from_fn_fa_fs(
                                radius,
                                self._scale_factor,
                                self.fn,
                                self.fa,
                                self.fs,
                                quality_factor,
                            )
                            num_steps = max(
                                num_steps, 6
                            )  # Ensure minimum 6 steps for quadratic curves

                        spline = QuadraticSpline(bezier_points)
                        t_values = np.linspace(0, 1, num_steps + 1)[1:]
                        new_points = spline.evaluate(t_values)
                        if new_points.ndim == 2 and new_points.shape[1] == 2:
                            current_contour.extend([tuple(pt) for pt in new_points])
                        elif new_points.shape == (2,):
                            current_contour.append(tuple(new_points))
                        final_pt_tuple = tuple(p2_arr)  # Ensure endpoint tuple is added
                        if not current_contour or not np.allclose(
                            current_contour[-1], final_pt_tuple
                        ):
                            current_contour.append(final_pt_tuple)

                elif command == "curveTo":
                    if not args or len(args) != 3:
                        continue
                    if not current_contour:
                        continue
                    p0_tuple = current_contour[-1]
                    p1_tuple, p2_tuple, p3_tuple = tuple(args[0]), tuple(args[1]), tuple(args[2])
                    p0_arr, p1_arr, p2_arr, p3_arr = (
                        np.asarray(p0_tuple),
                        np.asarray(p1_tuple),
                        np.asarray(p2_tuple),
                        np.asarray(p3_tuple),
                    )
                    if (
                        p0_arr.shape != (2,)
                        or p1_arr.shape != (2,)
                        or p2_arr.shape != (2,)
                        or p3_arr.shape != (2,)
                    ):
                        continue
                    points = np.array([p0_arr, p1_arr, p2_arr, p3_arr])
                    if (
                        np.allclose(p0_arr, p1_arr)
                        and np.allclose(p1_arr, p2_arr)
                        and np.allclose(p2_arr, p3_arr)
                    ):
                        if not np.allclose(current_contour[-1], p3_tuple):
                            current_contour.append(p3_tuple)
                        continue
                    radius = max(
                        np.linalg.norm(p1_arr - p0_arr),
                        np.linalg.norm(p2_arr - p1_arr),
                        np.linalg.norm(p3_arr - p2_arr),
                        EPSILON,
                    )

                    # Apply quality parameter for tessellation density
                    quality_factor = 1.0  # Default quality factor
                    if hasattr(self, "quality") and self.quality > 0:
                        quality_factor = self.quality

                    # Calculate segments with quality scaling
                    if hasattr(self, "fn") and self.fn > 0:
                        num_steps = max(
                            int(self.fn * quality_factor), 8
                        )  # Scale by quality, minimum 8 for cubic
                    else:
                        # Get steps using get_fragments_from_fn_fa_fs with quality factor
                        num_steps = get_fragments_from_fn_fa_fs(
                            radius, self._scale_factor, self.fn, self.fa, self.fs, quality_factor
                        )
                        num_steps = max(num_steps, 8)  # Ensure minimum 8 steps for cubic curves

                    spline = CubicSpline(points)
                    t_values = np.linspace(0, 1, num_steps + 1)[1:]
                    new_points = np.array([spline.evaluate(t) for t in t_values])
                    if new_points.ndim == 2 and new_points.shape[1] == 2:
                        current_contour.extend([tuple(pt) for pt in new_points])
                    elif new_points.shape == (2,):
                        current_contour.append(tuple(new_points))
                    final_pt_tuple = tuple(p3_arr)  # Ensure endpoint tuple is added
                    if not current_contour or not np.allclose(current_contour[-1], final_pt_tuple):
                        current_contour.append(final_pt_tuple)

                elif command == "closePath":
                    if current_contour:
                        if not np.allclose(current_contour[0], current_contour[-1]):
                            current_contour.append(current_contour[0])
                        raw_polygons.append(np.array(current_contour, dtype=float))
                        current_contour = []
            except Exception as e:
                log.error(f"Error processing pen command {command} {args}: {e}", exc_info=True)
                # Reset contour on error to prevent cascade?
                current_contour = []

        # Finalize any open contour
        if current_contour:
            raw_polygons.append(np.array(current_contour, dtype=float))

        # Apply Y-axis inversion at the very end if detected
        if self._y_axis_inverted:
            for i in range(len(raw_polygons)):
                # Ensure it's a numpy array before slicing
                if isinstance(raw_polygons[i], np.ndarray):
                    raw_polygons[i][:, 1] *= -1  # Flip Y in place
                else:  # Should not happen if logic above is correct
                    log.warning(f"Contour {i} was not a numpy array during Y-flip.")

        return raw_polygons  # List of raw numpy arrays in font units

    def _apply_alignment(self, polygons, min_coord, max_coord):
        """Apply alignment transformations based on calculated bounding box."""
        if not polygons:
            return polygons
        min_coord = np.asarray(min_coord)
        max_coord = np.asarray(max_coord)

        bbox_width = max_coord[0] - min_coord[0]
        bbox_height = max_coord[1] - min_coord[1]
        dx, dy = 0.0, 0.0

        if self.halign == "center":
            dx = -(min_coord[0] + bbox_width / 2.0)
        elif self.halign == "right":
            dx = -max_coord[0]
        else:
            dx = -min_coord[0]  # left

        if self.valign == "top":
            dy = -max_coord[1]
        elif self.valign == "center":
            dy = -(min_coord[1] + bbox_height / 2.0)
        elif self.valign == "bottom":
            dy = -min_coord[1]
        # baseline: dy = 0

        if abs(dx) > EPSILON or abs(dy) > EPSILON:
            translation = np.array([dx, dy])
            # Must create new list if modifying
            aligned_polygons = [poly + translation for poly in polygons]
            return aligned_polygons
        else:
            return polygons  # Return original list if no change

    def _apply_text_direction(self, polygons, min_coord, max_coord):
        """Apply text direction transformations based on actual bounds."""
        # TODO: Implement text direction transformations.
        return polygons

    def get_polygons(self) -> tuple[np.ndarray, list[np.ndarray]]:
        """Generates shaped text polygons using HarfBuzz."""
        if not self.text or self._hb_font is None or self._glyph_set is None:
            log.warning("Text is empty or HarfBuzz/GlyphSet not initialized.")
            return np.empty((0, 2)), []

        # 1. Setup HarfBuzz Buffer
        buf = hb.Buffer()
        buf.add_str(self.text)

        # Map common script names to HB codes
        script_map = {
            "latin": "Latn",
            "arabic": "Arab",
            "hebrew": "Hebr",
            "cyrillic": "Cyrl",
            "greek": "Grek",
        }
        hb_script = script_map.get(
            self.script.lower(), self.script.upper()[:4].ljust(4)
        )  # Use mapping or format

        buf.direction = "rtl" if self.base_direction == "rtl" else "ltr"
        buf.script = hb_script
        buf.language = self.language
        buf.cluster_level = hb.BufferClusterLevel.MONOTONE_CHARACTERS  # Helps map glyphs to chars

        log.debug(
            f"Shaping text: '{self.text}' | Direction: {buf.direction} | Script: {buf.script} | Lang: {buf.language}"
        )

        # 2. Shape the text
        try:
            features = {"kern": True, "liga": True}  # Enable common features
            hb.shape(self._hb_font, buf, features)
        except Exception as e:
            log.error(f"HarfBuzz shaping failed: {e}", exc_info=True)
            return np.empty((0, 2)), []

        glyph_infos = buf.glyph_infos
        glyph_positions = buf.glyph_positions

        if not glyph_infos:
            log.warning("HarfBuzz returned no glyphs.")
            return np.empty((0, 2)), []

        # 3. Process Shaped Glyphs
        all_polygons_scaled_positioned = []
        pen_pos = np.array([0.0, 0.0])  # Start at (0, 0) in final scaled units

        for i, (info, pos) in enumerate(zip(glyph_infos, glyph_positions)):
            gid = info.codepoint
            cluster = info.cluster

            # Get advances and offsets in FONT units
            dx, dy = pos.x_advance, pos.y_advance
            xoff, yoff = pos.x_offset, pos.y_offset

            try:
                glyph_name = self._font.getGlyphName(gid)
            except Exception:
                glyph_name = ".notdef"

            log.debug(
                f"Glyph {i}: GID={gid}, Name='{glyph_name}', Cluster={cluster}, Adv=({dx},{dy}), Off=({xoff},{yoff})"
            )

            # Get raw outlines (in font units, Y potentially flipped)
            raw_contours = self._get_glyph_outlines_by_name(glyph_name)

            if not raw_contours:
                log.info(f"No outline for glyph '{glyph_name}' (GID {gid}). Advancing pen.")
            else:
                # Calculate final drawing position for this glyph in scaled units
                draw_pos = pen_pos + np.array([
                    xoff * self._scale_factor,
                    yoff * self._scale_factor,
                ])

                # Scale and translate contours
                for raw_contour in raw_contours:
                    if raw_contour.ndim == 2 and raw_contour.shape[1] == 2:
                        scaled_contour = raw_contour * self._scale_factor
                        final_contour = scaled_contour + draw_pos
                        all_polygons_scaled_positioned.append(final_contour)
                    else:
                        log.warning(
                            f"Skipping invalid raw contour shape {raw_contour.shape} for glyph '{glyph_name}'"
                        )

            # Update pen position using scaled advances and user spacing
            # Apply spacing factor ONLY to the advance in the primary text direction
            advance_scale = self.spacing if self.direction in ("ltr", "rtl") else 1.0
            pen_pos += np.array([
                dx * self._scale_factor * advance_scale,
                dy * self._scale_factor * advance_scale,
            ])  # Assuming y advance scaling needed too? Usually 0 for horizontal text.

        if not all_polygons_scaled_positioned:
            log.warning("No polygons generated after processing glyphs.")
            return np.empty((0, 2)), []

        # 6. Calculate final bounds and apply Alignment & Direction
        try:
            # Need to handle possibility of empty list if all glyphs failed
            if not all_polygons_scaled_positioned:
                raise ValueError("No polygons to stack")
            combined_points = np.vstack(all_polygons_scaled_positioned)
            min_coord, max_coord = extentsof(combined_points)
        except ValueError:  # Handle empty combined_points
            log.warning("Could not calculate bounds from generated polygons.")
            min_coord, max_coord = (
                np.array([0.0, 0.0]),
                np.array([0.0, 0.0]),
            )  # Default bounds at origin

        # Apply alignment based on calculated bounds
        aligned_polygons = self._apply_alignment(
            all_polygons_scaled_positioned, min_coord, max_coord
        )

        # Apply direction transform (e.g., RTL mirroring) based on calculated bounds
        directed_polygons = self._apply_text_direction(aligned_polygons, min_coord, max_coord)

        # 7. Final Output Format Conversion
        final_all_points = []
        final_contours = []
        point_offset = 0
        for poly in directed_polygons:
            if poly.ndim == 2 and poly.shape[1] == 2 and len(poly) > 0:
                if len(poly) < 3:
                    # This is a degenerate polygon, probably a ligature. Let's ignore it.
                    log.info(f"Skipping polygon with less than 3 points: {poly}")
                    continue
                num_points = len(poly)
                final_all_points.append(poly)
                contour = np.arange(point_offset, point_offset + num_points)
                final_contours.append(contour[::-1])
                point_offset += num_points
            else:
                log.warning(
                    f"Skipping polygon with invalid shape {getattr(poly, 'shape', 'N/A')} during final conversion."
                )

        if not final_all_points:
            log.warning("No valid polygons remaining after final conversion.")
            return np.empty((0, 2)), []

        log.debug(f"Successfully generated {len(final_contours)} contours.")
        return np.vstack(final_all_points), final_contours

    def get_polygons_at(self, pos: int) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Returns polygons for the character at logical index `pos`.
        NOTE: Less efficient. Relies on full shaping first. May not work perfectly for complex ligatures spanning multiple clusters.
        """
        if pos < 0 or pos >= len(self.text):
            raise ValueError(f"Position {pos} out of range for text '{self.text}'")

        # Perform full shaping first
        if self._hb_font is None:
            return np.empty((0, 2)), []
        buf = hb.Buffer()
        buf.add_str(self.text)
        script_map = {"latin": "Latn", "arabic": "Arab", "hebrew": "Hebr"}
        hb_script = script_map.get(self.script.lower(), self.script.upper()[:4].ljust(4))
        buf.direction = "rtl" if self.base_direction == "rtl" else "ltr"
        buf.script = hb_script
        buf.language = self.language
        buf.cluster_level = hb.BufferClusterLevel.MONOTONE_CHARACTERS
        try:
            hb.shape(self._hb_font, buf, {"kern": True, "liga": True})
        except Exception as e:
            return np.empty((0, 2)), []
        glyph_infos = buf.glyph_infos
        glyph_positions = buf.glyph_positions
        if not glyph_infos:
            return np.empty((0, 2)), []

        # --- Find glyph indices matching the cluster ---
        target_glyph_indices = [i for i, info in enumerate(glyph_infos) if info.cluster == pos]
        if not target_glyph_indices:
            return np.empty((0, 2)), []

        # --- Generate ALL polygons first (like get_polygons) ---
        all_polygons_scaled_positioned = []
        pen_pos = np.array([0.0, 0.0])
        glyph_to_polygons_map = {}  # Map glyph index to list of polygon indices

        for i, (info, glyph_pos) in enumerate(zip(glyph_infos, glyph_positions)):
            gid = info.codepoint
            dx, dy = glyph_pos.x_advance, glyph_pos.y_advance
            xoff, yoff = glyph_pos.x_offset, glyph_pos.y_offset
            try:
                glyph_name = self._font.getGlyphName(gid)
            except:
                glyph_name = ".notdef"

            raw_contours = self._get_glyph_outlines_by_name(glyph_name)
            current_glyph_poly_indices = []

            if raw_contours:
                draw_pos = pen_pos + np.array([
                    xoff * self._scale_factor,
                    yoff * self._scale_factor,
                ])
                for raw_contour in raw_contours:
                    if raw_contour.ndim == 2 and raw_contour.shape[1] == 2:
                        final_contour = (raw_contour * self._scale_factor) + draw_pos
                        # Store index of the polygon added
                        current_glyph_poly_indices.append(len(all_polygons_scaled_positioned))
                        all_polygons_scaled_positioned.append(final_contour)

            glyph_to_polygons_map[i] = (
                current_glyph_poly_indices  # Store polygon indices for this glyph index
            )

            # Update pen position
            advance_scale = self.spacing if self.direction in ("ltr", "rtl") else 1.0
            pen_pos += np.array([
                dx * self._scale_factor * advance_scale,
                dy * self._scale_factor * advance_scale,
            ])

        target_polygons = []
        for glyph_idx in target_glyph_indices:
            poly_indices = glyph_to_polygons_map.get(glyph_idx, [])
            for poly_idx in poly_indices:
                if 0 <= poly_idx < len(all_polygons_scaled_positioned):
                    target_polygons.append(all_polygons_scaled_positioned[poly_idx])

        if not target_polygons:
            return np.empty((0, 2)), []

        try:
            combined_points = np.vstack(target_polygons)
            min_coord, max_coord = extentsof(combined_points)
        except ValueError:
            return np.empty((0, 2)), []

        # Apply alignment relative to this character's bounds
        aligned_polygons = self._apply_alignment(target_polygons, min_coord, max_coord)
        # Skip direction transform for single char?
        directed_polygons = aligned_polygons

        # --- Final Output Format ---
        # (Same conversion as get_polygons)
        final_all_points = []
        final_contours = []
        point_offset = 0
        for poly in directed_polygons:
            if poly.ndim == 2 and poly.shape[1] == 2 and len(poly) > 0:
                num_points = len(poly)
                final_all_points.append(poly)
                final_contours.append(np.arange(point_offset, point_offset + num_points))
                point_offset += num_points
        if not final_all_points:
            return np.empty((0, 2)), []
        return np.vstack(final_all_points), final_contours


def render_text(
    text: str,
    size: float = 10,
    font: str = "Liberation Sans",
    halign: str = "left",
    valign: str = "baseline",
    spacing: float = 1.0,
    direction: str = "ltr",
    language: str = "en",
    script: str = "latin",
    fa: float = 12.0,
    fs: float = 2.0,
    fn: int = 0,
    base_direction: str = "ltr",
    quality: float = 1.0,
) -> tuple[np.ndarray, list[np.ndarray]]:
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
        base_direction=base_direction,
        quality=quality,
    )
    return context.get_polygons()


def get_available_fonts():
    import platform

    available_fonts = {}
    font_dirs = []
    system = platform.system()
    # Windows
    if system == "Windows":
        windir = os.environ.get("WINDIR", "C:\\Windows")
        font_dirs.append(os.path.join(windir, "Fonts"))
    # macOS
    elif system == "Darwin":
        font_dirs.extend([
            "/Library/Fonts",
            "/System/Library/Fonts",
            os.path.expanduser("~/Library/Fonts"),
        ])
    # Linux/Unix
    else:
        font_dirs.extend([
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            os.path.expanduser("~/.fonts"),
            os.path.expanduser("~/.local/share/fonts"),
        ])

    font_dirs = [d for d in font_dirs if d and os.path.isdir(d)]  # Filter valid dirs

    for d in font_dirs:
        try:
            for root, _, files in os.walk(d):
                for file in files:
                    if file.lower().endswith((".ttf", ".otf", ".ttc")):  # Include TTC
                        try:
                            font_path = os.path.join(root, file)
                            # Use size=10 as default for probing
                            font = ImageFont.truetype(font_path, size=10)
                            family, style = font.getname()
                            if family not in available_fonts:
                                available_fonts[family] = []
                            if style not in available_fonts[family]:
                                available_fonts[family].append(style)
                        except Exception:
                            continue  # Ignore fonts PIL can't read
        except OSError:
            continue  # Ignore dirs we can't access

    # Add known PIL builtins if they weren't found via path
    try:
        default_fonts = ["arial.ttf", "times.ttf", "cour.ttf"]  # Basic ones
        for font_name in default_fonts:
            try:
                font = ImageFont.truetype(font_name, size=10)
                family, style = font.getname()
                if family not in available_fonts:
                    available_fonts[family] = []
                if style not in available_fonts[family]:
                    available_fonts[family].append(style)
            except OSError:
                pass
    except Exception:
        pass
    return available_fonts


def get_fonts_list():
    fonts_dict = get_available_fonts()
    fonts_list = []
    for family, styles in fonts_dict.items():
        for style in styles:
            # Prefer 'Regular' as default style name if applicable
            style_name = (
                "Regular"
                if "regular" in style.lower()
                and "bold" not in style.lower()
                and "italic" not in style.lower()
                else style
            )
            if style_name == "Regular":
                fonts_list.append(family)
            else:
                # Attempt to format style nicely
                style_formatted = (
                    style.replace("Italic", " Italic").replace("Bold", " Bold").strip()
                )
                fonts_list.append(f"{family}:style={style_formatted}")
    # Remove duplicates that might arise from style name normalization
    return sorted(list(set(fonts_list)))


def get_fragments_from_fn_fa_fs(
    r: float, scale: float, fn: int | None, fa: float | None, fs: float | None, quality: float = 1.0
) -> int:
    # NOTE: This function is designed for circles/arcs in OpenSCAD.
    # Its direct application to Bezier curves is non-standard.
    # The TextContext currently uses a simpler logic based primarily on $fn.
    # This function remains for potential future use or compatibility checks.
    GRID_FINE = 0.00000095367431640625  # From OpenSCAD source
    # OpenSCAD uses doubles, ensure calculations are float
    r = r * scale
    fa = float(fa) if fa is not None else 30.0  # Default $fa from OpenSCAD
    fs = float(fs) if fs is not None else 20.0  # Default $fs from OpenSCAD
    quality = float(quality) if quality is not None else 1.0  # Default quality

    if r < GRID_FINE:
        # print("Warning: Radius too small, using 3 fragments.")
        return 2
    # Handle $fn (number) - overrides angular/size fragmentation
    if fn is not None:
        fn = float(fn)  # Ensure float for checks
        if np.isinf(fn) or np.isnan(fn):
            # print("Warning: $fn is inf or nan, using 3 fragments.")
            return 2
        # $fn = 0 means use $fa/$fs
        if fn > 0:
            # Apply quality factor to fn
            result = int(max(fn * quality, 2.0))
            # print(f"Using $fn with quality: {result}")
            return result

    # Calculate fragments based on $fa (angle) and $fs (size)
    # Ensure fs is not zero to avoid division error
    fs = max(fs, GRID_FINE)
    # Calculate number of fragments from angular resolution ($fa)
    num_angle = 360.0 / fa if fa > 0 else 360.0  # Avoid division by zero, large number if fa=0
    # Calculate number of fragments from segment size ($fs)
    num_size = r / fs

    # Choose the larger number of fragments from angle/size constraints
    # Ensure it's at least 5 (OpenSCAD minimum for arc-based fragmentation)
    base_fragments = np.ceil(max(min(num_angle, num_size), 4.0))
    # Apply quality factor
    fragments = np.ceil(base_fragments * quality)
    # print(f"Using $fa/$fs with quality: num_angle={num_angle}, num_size={num_size}, result={int(fragments)}")
    if fragments < 2:
        return 2
    return int(fragments)
