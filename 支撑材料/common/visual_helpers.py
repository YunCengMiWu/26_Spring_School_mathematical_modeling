import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def save_svg_heatmap(data, x_coords, y_coords, outpath):
    """Save a simple SVG heatmap for inclusion in LaTeX."""
    import svgwrite
    dwg = svgwrite.Drawing(outpath, size=(800, 400))
    dwg.add(dwg.rect(insert=(0,0), size=('100%','100%'), fill='white'))
    # simple grid
    nx = len(x_coords)
    ny = len(y_coords)
    cell_w = 700 / nx
    cell_h = 320 / ny
    # normalize
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    for i in range(nx):
        for j in range(ny):
            v = data[i, j]
            if not np.isfinite(v):
                color = '#f0f0f0'
            else:
                frac = (v - vmin) / (vmax - vmin + 1e-9)
                # use viridis-like gradient
                cmap = cm.get_cmap('viridis')
                rgba = cmap(frac)
                color = svgwrite.utils.rgb(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
            x = 50 + i * cell_w
            y = 50 + (ny-1-j) * cell_h
            dwg.add(dwg.rect(insert=(x,y), size=(cell_w,cell_h), fill=color, stroke='none'))
    dwg.save()
