
import sys
import logging
from pythonopenscad.text_render import render_text, get_fonts_list, get_available_fonts, extentsof
log = logging.getLogger(__name__)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    if len(sys.argv) > 1 and sys.argv[1] == "--list-fonts":
        print("Available Fonts:")
        fonts = get_fonts_list()
        for font in fonts:
            print(f"  {font}")
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--list-fonts-by-family":
        print("Available Font Families:")
        fonts = get_available_fonts()
        for family, styles in sorted(fonts.items()):
            print(f"{family}: {', '.join(sorted(styles))}")
        sys.exit(0)

    font_to_use = "Arial"  # Default if no arg
    font_to_use = "Times New Roman"
    font_to_use = "Tahoma"
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        font_to_use = sys.argv[1]
    log.info(f"Using font: '{font_to_use}'")

    print("Generating text...")
    mixed_text = "Hello! مرحباً שלום"  # Mix English, Arabic, Hebrew
    print(f"Testing text: {mixed_text}")

    try:
        # Regular text first
        log.info("Generating LTR example...")
        points1, contours1 = render_text(
            "Test!", size=25, font=font_to_use, halign="center", valign="center", fn=None
        )

        # Bidirectional text with LTR base direction
        log.info("Generating Bidi LTR example...")
        points2, contours2 = render_text(
            mixed_text,
            size=25,
            font=font_to_use,
            halign="center",
            valign="center",
            fn=None,
            base_direction="ltr",
            script="latin",
        )  # Script hint might need adjustment

        # Bidirectional text with RTL base direction
        log.info("Generating Bidi RTL example...")
        points3, contours3 = render_text(
            mixed_text,
            size=25,
            font=font_to_use,
            halign="center",
            valign="center",
            fn=None,
            base_direction="rtl",
            script="arabic",
        )  # Script hint might need adjustment

    except Exception as e:
        log.error(f"Error during text generation: {e}", exc_info=True)
        print("\nTry running with --list-fonts to see available fonts.")
        sys.exit(1)

    print(f"Generated {len(contours1)} polygons, {len(points1)} points for 'Test!'")
    print(f"Generated {len(contours2)} polygons, {len(points2)} points for bidi text (LTR base)")
    print(f"Generated {len(contours3)} polygons, {len(points3)} points for bidi text (RTL base)")

    # --- Plotting ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=False)  # Don't share X

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
    
    def plot_polygons(ax, title, points, contours, color):
        if points is None or points.shape[0] == 0 or not contours:
            ax.text(0.5, 0.5, "No polygons generated", ha="center", va="center")
            ax.set_title(f"{title} (No polygons)")
            return
        for contour_indices in contours:
            # Ensure indices are valid
            valid_indices = contour_indices[contour_indices < points.shape[0]]
            if len(valid_indices) > 1:
                contour_points = points[valid_indices]
                clockwise = is_clockwise(contour_points)
                fill_color = 'pink' if clockwise else color
                patch = plt.Polygon(
                    contour_points, closed=True, facecolor=fill_color, edgecolor="black", alpha=0.6
                )
                ax.add_patch(patch)
            else:
                log.warning(f"Skipping contour with invalid indices: {contour_indices}")

        ax.set_aspect("equal", adjustable="box")
        min_coord, max_coord = extentsof(points)
        ax.set_xlim(min_coord[0] - 5, max_coord[0] + 5)
        ax.set_ylim(min_coord[1] - 5, max_coord[1] + 5)
        ax.set_title(title)
        ax.grid(True)

    plot_polygons(axes[0], f"Regular Text: 'Test!'", points1, contours1, "blue")
    plot_polygons(
        axes[1], f"Bidirectional Text (LTR base): '{mixed_text}'", points2, contours2, "green"
    )
    plot_polygons(
        axes[2], f"Bidirectional Text (RTL base): '{mixed_text}'", points3, contours3, "red"
    )

    plt.tight_layout()
    plt.savefig("text_render_bidi_test_hb.png", dpi=150)
    print("Figure saved to 'text_render_bidi_test_hb.png'")
    try:
        plt.show()
    except Exception:
        print("Could not display plot - but image was saved to file")
