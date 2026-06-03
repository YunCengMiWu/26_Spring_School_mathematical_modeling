import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def _make_cmap():
    # Professional, print-friendly blue→cyan→green→amber gradient
    return LinearSegmentedColormap.from_list(
        "geo_blue_grad",
        [
            (0.00, "#0B1F5C"),  # deep navy
            (0.35, "#1EA7FD"),  # sky blue
            (0.62, "#00C389"),  # teal/green
            (1.00, "#F6C453"),  # amber
        ],
    )


CMAP = _make_cmap()


def _rgba_to_hex(rgba):
    r, g, b, a = rgba
    return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))


def pick_color(t: float) -> str:
    """Pick a color from the shared gradient (t in [0,1])."""
    t = float(max(0.0, min(1.0, t)))
    return _rgba_to_hex(CMAP(t))


def apply_common_style():
    # Fonts (fallbacks for Windows)
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False

    # Clean, paper-friendly look
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "#B0B0B0"
    plt.rcParams["axes.linewidth"] = 0.9

    # Grid: subtle
    plt.rcParams["grid.color"] = "#E6E6E6"
    plt.rcParams["grid.linewidth"] = 0.8
    plt.rcParams["grid.alpha"] = 1.0
    plt.rcParams["grid.linestyle"] = "-"

    # Lines
    plt.rcParams["lines.linewidth"] = 2.0

    # Ensure SVG/PNG anti-aliasing behaves well
    matplotlib.rcParams["savefig.facecolor"] = "white"
