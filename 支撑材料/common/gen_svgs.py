"""Generate SVG slices and 3D renders for selected hours using q4_heatmap utilities."""
from q4_heatmap import load_data, build_spatial_temporal_pressure, save_svg_slice, plot_heatmap_3d
import os

def main():
    spots, slots, users = load_data()
    radius = users['max_walk'].median()
    grid_x, grid_y, times, demand_grid, supply_grid, reachable_grid, pressure_grid = \
        build_spatial_temporal_pressure(spots, slots, users, grid_size=10, radius=radius)

    # select indices for 12 and 13
    idx_12 = min(range(len(times)), key=lambda i: abs(times[i]-12))
    idx_13 = min(range(len(times)), key=lambda i: abs(times[i]-13))

    outdir = 'svg_out'
    os.makedirs(outdir, exist_ok=True)

    save_svg_slice(pressure_grid[:,:,idx_12].copy(), grid_x, grid_y, os.path.join(outdir, 'heatmap_12h.svg'), 12)
    save_svg_slice(pressure_grid[:,:,idx_13].copy(), grid_x, grid_y, os.path.join(outdir, 'heatmap_13h.svg'), 13)

    plot_heatmap_3d(grid_x, grid_y, times, pressure_grid, t_idx=idx_12, outpath=os.path.join(outdir, 'heatmap_12h_3d.png'))
    plot_heatmap_3d(grid_x, grid_y, times, pressure_grid, t_idx=idx_13, outpath=os.path.join(outdir, 'heatmap_13h_3d.png'))

    print('Done: SVGs and 3D renders saved to', outdir)

if __name__ == '__main__':
    main()
