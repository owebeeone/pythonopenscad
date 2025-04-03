# pythonopenscad/tests/test_text_render.py
import unittest
import numpy as np
import os

# Adjust the import path based on your project structure
# This assumes tests are run from the project root or PYTHONPATH is set
try:
    from pythonopenscad import text_render
except ImportError:
    # If running directly from tests directory, adjust path
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from pythonopenscad import text_render

class TestTextRender(unittest.TestCase):

    def test_simple_text_rendering(self):
        """
        Tests basic text rendering with default parameters.
        Checks if the function runs and returns data in the expected format.
        """
        test_string = "A"
        try:
            points, contours = text_render.render_text(
                text=test_string,
                size=10,
                font="Liberation Sans", # Use a known fallback font
                halign="left",
                valign="baseline",
                spacing=1.0,
                direction="ltr",
                language="en",
                script="latin",
                fa=None, # Use defaults
                fs=None,
                fn=None
            )

            # Basic structural checks
            self.assertIsInstance(points, np.ndarray, "Points should be a NumPy array")
            if points.size > 0: # Only check dimensions if points were generated
                 self.assertEqual(points.ndim, 2, "Points array should be 2-dimensional")
                 self.assertEqual(points.shape[1], 2, "Points should have 2 columns (x, y)")

            self.assertIsInstance(contours, list, "Contours should be a list")
            if contours: # Only check elements if contours exist
                self.assertTrue(all(isinstance(c, np.ndarray) for c in contours),
                                "Each contour should be a NumPy array")
                self.assertTrue(all(c.dtype == np.int64 or c.dtype == np.int32 for c in contours), # Check for integer indices based on system/numpy defaults
                                "Contour arrays should contain integer indices")

            # Very basic check: Did we get *something* back for a simple letter?
            # The exact number is highly dependent on font and fragmentation.
            self.assertGreater(points.shape[0], 3, "Expected more than 3 points for 'A'")
            self.assertGreaterEqual(len(contours), 1, "Expected at least one contour for 'A'")

        except RuntimeError as e:
            # Allow test to pass if font loading fails predictably (e.g., in CI)
            # but fail on other unexpected RuntimeErrors.
            if "Could not load font" in str(e):
                 self.skipTest(f"Skipping test: Default font not found ({e})")
            else:
                 self.fail(f"Text rendering failed with unexpected error: {e}")
        except ImportError as e:
             if 'freetype' in str(e):
                 self.skipTest(f"Skipping test: freetype-py not installed ({e})")
             else:
                 self.fail(f"Text rendering failed with unexpected import error: {e}")


    # TODO: Add more tests:
    # - Test different alignments (halign, valign) by checking bounding boxes
    # - Test 'spacing' by comparing output width
    # - Test 'size' by comparing output height/bounds
    # - Test explicit '$fn' values (e.g., low vs high) by checking point counts
    # - Test font not found error handling more specifically
    # - Test more complex strings ("Hello", strings requiring kerning/ligatures if HarfBuzz is added)
    # - Test different directions (rtl, ttb, btt) when implemented
    # - Test get_polygons_at when implemented

if __name__ == '__main__':
    unittest.main() 