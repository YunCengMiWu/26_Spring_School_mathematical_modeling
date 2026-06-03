"""空间-时间压力热力图分析

将空间划分为网格，对每个网格和时刻计算供需压力比，
直观展示哪些区域、哪些时段停车压力最大。

输出: q4_heatmap.png (多子图热力图)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import sys

from utils import load_data, compute_distance_matrix
from plot_theme import CMAP, apply_common_style, pick_color

apply_common_style()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def build_spatial_temporal_pressure(spots, slots, users, grid_size=10, radius=15.0):
    """
    构建空间-时间压力矩阵。

    参数:
        spots: 车位数据框
        slots: 时段数据框
        users: 用户数据框
        grid_size: 每维网格数（grid_size x grid_size）
        radius: 可达半径（米），用于计算"可达供给"

    返回:
        (grid_x, grid_y, times, demand_grid, supply_grid, reachable_grid, pressure_grid)
        demand_grid[gx, gy, t_idx]: 网格(gx, gy)在时刻t的需求密度
        supply_grid[gx, gy, t_idx]: 网格(gx, gy)在时刻t的原始供给
        reachable_grid[gx, gy, t_idx]: 网格(gx, gy)在时刻t的可达供给
        pressure_grid[gx, gy, t_idx]: 需求/可达供给压力比
    """
    # 空间范围
    x_min, x_max = 0, 100
    y_min, y_max = 0, 100
    cell_w = (x_max - x_min) / grid_size  # 格子宽度
    cell_h = (y_max - y_min) / grid_size

    # 网格中心坐标
    grid_x = np.linspace(x_min + cell_w / 2, x_max - cell_w / 2, grid_size)
    grid_y = np.linspace(y_min + cell_h / 2, y_max - cell_h / 2, grid_size)

    # 时间采样点（半开区间，每小时一个点）
    times = np.arange(8, 22, 1)  # [8, 9, ..., 21]
    n_times = len(times)
    n_cells = grid_size * grid_size

    # 初始化矩阵
    demand_grid = np.zeros((grid_size, grid_size, n_times))
    supply_grid = np.zeros((grid_size, grid_size, n_times))
    reachable_grid = np.zeros((grid_size, grid_size, n_times))
    pressure_grid = np.full((grid_size, grid_size, n_times), np.nan)

    # ---- 用户需求映射 ----
    # 每个用户的目的地坐标映射到网格
    user_positions = users[["dest_x", "dest_y"]].values  # (500, 2)
    user_arrivals = users["arrival"].values
    user_departures = users["departure"].values
    user_max_walk = users["max_walk"].values

    # 为每个用户分配网格
    for j in range(len(users)):
        ux, uy = user_positions[j]
        gx = int(np.clip((ux - x_min) / cell_w, 0, grid_size - 1))
        gy = int(np.clip((uy - y_min) / cell_h, 0, grid_size - 1))

        for t_idx, t in enumerate(times):
            if user_arrivals[j] <= t < user_departures[j]:
                demand_grid[gx, gy, t_idx] += 1

    # ---- 车位供给映射 ----
    spot_positions = spots[["x", "y"]].values  # (300, 2)

    # 构建 slot lookup: spot_id -> [(start, end), ...]
    slot_map = {}
    for _, row in slots.iterrows():
        sid = row["spot_id"]
        if sid not in slot_map:
            slot_map[sid] = []
        slot_map[sid].append((row["start"], row["end"]))

    # 计算每个车位的网格位置，以及每个网格到所有车位的距离
    spot_grid_indices = []
    for sid_idx, sid in enumerate(spots["spot_id"]):
        sx, sy = spot_positions[sid_idx]
        gx = int(np.clip((sx - x_min) / cell_w, 0, grid_size - 1))
        gy = int(np.clip((sy - y_min) / cell_h, 0, grid_size - 1))
        spot_grid_indices.append((gx, gy, sx, sy))

    # 计算所有网格中心到所有车位的距离矩阵 (n_cells x n_spots)
    cell_centers = np.array([[gx, gy] for gy in grid_y for gx in grid_x])  # (100, 2)
    cell_centers_swapped = cell_centers[:, [1, 0]]  # 修正: 需要 [x, y]
    # 重新构建正确的网格中心
    cell_positions = []
    for gy in grid_y:
        for gx in grid_x:
            cell_positions.append([gx, gy])
    cell_positions = np.array(cell_positions)  # (100, 2) = [x, y]

    # 计算距离矩阵: 每个网格中心到每个车位
    spot_coords = spot_positions  # (300, 2)
    from scipy.spatial.distance import cdist
    dist_cell_to_spot = cdist(cell_positions, spot_coords, metric="euclidean")  # (100, 300)

    # 对于每个网格、每个时间、每个车位，检查是否可达且可用
    for gx_idx in range(grid_size):
        for gy_idx in range(grid_size):
            cell_idx = gx_idx + gy_idx * grid_size  # flatten: column-major (gx varies fastest)

            for t_idx, t in enumerate(times):
                # 原始供给：车位在该时刻可用
                raw_count = 0
                reachable_count = 0

                for sid_idx, sid in enumerate(spots["spot_id"]):
                    # 检查时刻 t 是否在车位可用时段内
                    if sid not in slot_map:
                        continue
                    is_available = False
                    for s, e in slot_map[sid]:
                        if s <= t < e:
                            is_available = True
                            break
                    if not is_available:
                        continue

                    raw_count += 1

                    # 检查该车位是否从网格中心可达
                    d = dist_cell_to_spot[cell_idx, sid_idx]
                    # 使用该网格内用户的最大步行距离中位数（若无用户则用全局中位数）
                    if d <= radius:
                        reachable_count += 1

                supply_grid[gx_idx, gy_idx, t_idx] = raw_count
                reachable_grid[gx_idx, gy_idx, t_idx] = reachable_count

                # 压力比
                d_val = demand_grid[gx_idx, gy_idx, t_idx]
                s_val = reachable_count
                if d_val > 0 and s_val > 0:
                    pressure_grid[gx_idx, gy_idx, t_idx] = d_val / s_val
                elif d_val > 0 and s_val == 0:
                    pressure_grid[gx_idx, gy_idx, t_idx] = np.inf

    return grid_x, grid_y, times, demand_grid, supply_grid, reachable_grid, pressure_grid


def plot_heatmap(spots, slots, users, grid_x, grid_y, times,
                  demand_grid, supply_grid, reachable_grid, pressure_grid):
    """生成空间-时间压力热力图"""
    grid_size = len(grid_x)
    n_times = len(times)

    # 选择关键时间点展示
    key_hours = [8, 10, 12, 14, 16, 18, 20]
    key_indices = [np.argmin(np.abs(times - h)) for h in key_hours]

    n_plots = len(key_hours)
    # 2行布局：第1行上午，第2行下午，7张图用2x4（空一个）
    fig, axes = plt.subplots(2, 4, figsize=(22, 12))
    axes_flat = axes.flatten()

    # 颜色映射：使用感知均匀的渐变色板，便于打印和色盲友好
    norm = plt.Normalize(vmin=0.1, vmax=4.0)
    cmap = CMAP
    cmap.set_bad('#f0f0f0')  # NaN/无数据格为浅灰

    for idx, (ax, t_idx) in enumerate(zip(axes_flat[:n_plots], key_indices)):
        hour = times[t_idx]
        pressure_slice = pressure_grid[:, :, t_idx].copy()

        # 标记无供给但有需求的单元格为 NaN（灰色显示）
        no_supply_mask = ~np.isfinite(pressure_slice)
        pressure_slice[no_supply_mask] = np.nan
        # 无需求的单元格也显示灰色
        zero_demand_mask = demand_grid[:, :, t_idx] == 0
        pressure_slice[zero_demand_mask] = np.nan

        # 绘制压力热力图，降低透明度让颜色更清晰/接近实心
        im = ax.imshow(pressure_slice.T, origin='lower',
                       extent=[grid_x[0] - (grid_x[1]-grid_x[0])/2, grid_x[-1] + (grid_x[1]-grid_x[0])/2,
                               grid_y[0] - (grid_y[1]-grid_y[0])/2, grid_y[-1] + (grid_y[1]-grid_y[0])/2],
                       cmap=cmap, aspect='equal', norm=norm, alpha=0.95)

        # 叠加散点
        ax.scatter(spots["x"], spots["y"], c=pick_color(0.25), s=12, alpha=0.45,
                   label='车位', marker='s', edgecolors='white', linewidth=0.3)
        ax.scatter(users["dest_x"], users["dest_y"], c=pick_color(0.75), s=4,
                   alpha=0.12, label='用户目的地')

        # 在无供给但有需求的单元格上标注 ✗
        no_supply_with_demand = no_supply_mask & ~zero_demand_mask
        if no_supply_with_demand.any():
            n_xy = np.where(no_supply_with_demand)
            for gx_i, gy_i in zip(n_xy[0], n_xy[1]):
                cx = grid_x[gx_i]
                cy = grid_y[gy_i]
                d_val = demand_grid[gx_i, gy_i, t_idx]
                ax.text(cx, cy, f'X\n{d_val:.0f}人', ha='center', va='center',
                        fontsize=6, fontweight='bold', color=pick_color(0.85),
                        bbox=dict(boxstyle='round,pad=0.1',
                                  facecolor='white', alpha=0.7))

        # 在供需正常的单元格标注 d/s
        normal_cells = ~zero_demand_mask & ~no_supply_mask
        if normal_cells.any():
            n_xy = np.where(normal_cells)
            for gx_i, gy_i in zip(n_xy[0], n_xy[1]):
                cx = grid_x[gx_i]
                cy = grid_y[gy_i]
                d_val = demand_grid[gx_i, gy_i, t_idx]
                s_val = reachable_grid[gx_i, gy_i, t_idx]
                p_val = pressure_grid[gx_i, gy_i, t_idx]
                ax.text(cx, cy, f'{d_val:.0f}/{s_val:.0f}\n{p_val:.1f}x',
                        ha='center', va='center',
                        fontsize=6, fontweight='bold',
                        color='black' if p_val < 2 else 'darkred', alpha=0.8,
                        bbox=dict(boxstyle='round,pad=0.15',
                                  facecolor='white', alpha=0.55))

        ax.set_title(f'{hour:.0f}:00', fontsize=13, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_yticks(np.arange(0, 101, 20))
        ax.grid(True, alpha=0.2, linestyle='--', color='#B0B0B0')

    # 隐藏多余的子图
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # 添加图例（取最后一个子图位置）
    handles = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=pick_color(0.25),
                   markersize=8, label='车位'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=pick_color(0.75),
                   markersize=5, label='用户目的地'),
        plt.Line2D([0], [0], marker='s', color='w',
                     markerfacecolor='#D9D9D9', markersize=8, label='无车位/无需求'),
    ]
    axes_flat[6].legend(handles=handles, loc='center', fontsize=9,
                        framealpha=0.8, title='图例', title_fontsize=10)
    # 把第7个子图用作文本说明
    axes_flat[7].axis('off')

    # 共用颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('压力比 (需求/可达供给)', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    plt.suptitle('共享停车位空间-时间压力热力图', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig("q4_heatmap.png", dpi=200, bbox_inches='tight')
    print("空间-时间热力图已保存: q4_heatmap.png")

    return fig


def plot_demand_supply_summary(grid_x, grid_y, times,
                                demand_grid, supply_grid, reachable_grid, pressure_grid):
    """生成供需汇总图：时间-空间聚合"""
    grid_size = len(grid_x)
    n_times = len(times)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 1. 总需求 vs 总供给（时间维度）
    ax = axes[0, 0]
    total_demand = demand_grid.sum(axis=(0, 1))
    total_supply = supply_grid.sum(axis=(0, 1))
    total_reachable = reachable_grid.sum(axis=(0, 1))

    ax.plot(times, total_demand, '-o', color=pick_color(0.75), label='总需求（活跃用户）', linewidth=2, markersize=6)
    ax.plot(times, total_reachable, '-s', color=pick_color(0.55), label='总可达供给（车位）', linewidth=2, markersize=6)
    ax.plot(times, total_supply, '--^', color=pick_color(0.25), label='总供给（全部车位）', linewidth=2, markersize=6, alpha=0.5)
    ax.set_xlabel('时间 (h)', fontsize=12)
    ax.set_ylabel('数量', fontsize=12)
    ax.set_title('时空聚合：需求 vs 供给', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(times[::2])

    # 2. 累计压力分布（空间维度）
    ax = axes[0, 1]
    peak_pressure = np.nanmax(np.where(np.isfinite(pressure_grid), pressure_grid, 0), axis=2)
    im = ax.imshow(peak_pressure.T, origin='lower',
                   extent=[grid_x[0] - 5, grid_x[-1] + 5,
                            grid_y[0] - 5, grid_y[-1] + 5],
                   cmap=CMAP, aspect='equal')
    ax.set_xlabel('X 坐标', fontsize=12)
    ax.set_ylabel('Y 坐标', fontsize=12)
    ax.set_title('各网格峰值压力比', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='峰值压力比')

    # 3. 总需求空间分布
    ax = axes[1, 0]
    total_demand_spatial = demand_grid.sum(axis=2).T
    im2 = ax.imshow(total_demand_spatial, origin='lower',
                     extent=[grid_x[0] - 5, grid_x[-1] + 5,
                             grid_y[0] - 5, grid_y[-1] + 5],
                     cmap=CMAP, aspect='equal')
    ax.set_xlabel('X 坐标', fontsize=12)
    ax.set_ylabel('Y 坐标', fontsize=12)
    ax.set_title('累计需求密度', fontsize=13, fontweight='bold')
    plt.colorbar(im2, ax=ax, label='累计活跃用户数')

    # 4. 压力比箱线图
    ax = axes[1, 1]
    pressure_valid = []
    time_labels = []
    for t_idx, t in enumerate(times):
        p_slice = pressure_grid[:, :, t_idx].flatten()
        p_valid = p_slice[np.isfinite(p_slice) & (p_slice > 0)]
        if len(p_valid) > 0:
            pressure_valid.append(p_valid)
            time_labels.append(f'{t:.0f}h')

    bp = ax.boxplot(pressure_valid, labels=time_labels, patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor(pick_color(0.25))
        patch.set_alpha(0.7)

    ax.axhline(y=1.0, color=pick_color(0.05), linestyle='--', alpha=0.7, label='供需平衡线')
    ax.set_xlabel('时间', fontsize=12)
    ax.set_ylabel('网格压力比分布', fontsize=12)
    ax.set_title('各时刻网格压力分布', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("q4_heatmap_summary.png", dpi=300, bbox_inches='tight')
    print("供需汇总图已保存: q4_heatmap_summary.png")

def save_svg_slice(pressure_slice, grid_x, grid_y, outpath, hour_label):
    """Save a single 2D压力切片为SVG，以便论文嵌入。"""
    fig, ax = plt.subplots(figsize=(6,5))
    norm = plt.Normalize(vmin=0.1, vmax=4.0)
    im = ax.imshow(pressure_slice.T, origin='lower',
                    extent=[grid_x[0] - 5, grid_x[-1] + 5,
                            grid_y[0] - 5, grid_y[-1] + 5],
                    cmap=CMAP, aspect='equal', norm=norm)
    ax.set_title(f'Pressure Slice @ {hour_label}h', fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(outpath, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved SVG slice: {outpath}")


def plot_heatmap_3d(grid_x, grid_y, times, pressure_grid, t_idx=None, outpath="q4_heatmap_3d.png"):
    """将单时刻的压力热力图渲染为三维表面图并保存"""
    # 选择时刻
    if t_idx is None:
        # 默认选择压力峰值时刻
        valid_means = np.nanmean(np.where(np.isfinite(pressure_grid), pressure_grid, np.nan), axis=(0,1))
        t_idx = int(np.nanargmax(valid_means))

    X, Y = np.meshgrid(grid_x, grid_y)
    Z = pressure_grid[:, :, t_idx].T  # transpose to match X,Y

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Professional pane/grid styling
    try:
        ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]["color"] = (0.78, 0.78, 0.78, 0.35)
            axis._axinfo["grid"]["linewidth"] = 0.8
    except Exception:
        pass

    # mask invalid
    mask = ~np.isfinite(Z)
    Z_masked = np.ma.array(Z, mask=mask)

    # build full-resolution grid for smooth surface
    try:
        xi = np.linspace(grid_x.min(), grid_x.max(), 200)
        yi = np.linspace(grid_y.min(), grid_y.max(), 200)
        XI, YI = np.meshgrid(xi, yi)
        # use simple nearest-neighbor interpolation from coarse grid to fine grid
        from scipy.interpolate import griddata
        points = np.array([(x, y) for x in grid_x for y in grid_y])
        values = Z_masked.data.flatten().T
        # mask out invalid
        valid = np.isfinite(values)
        if valid.sum() >= 4:
            ZI = griddata(points[valid], values[valid], (XI, YI), method='cubic')
        else:
            ZI = griddata(points, values, (XI, YI), method='nearest')
    except Exception:
        # fallback: simple upsample by nearest
        XI, YI = np.meshgrid(grid_x, grid_y)
        ZI = Z_masked

    # apply a light source shading to give a 3D terrain feel
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(ZI, cmap=CMAP, vert_exag=1.0, blend_mode='soft')

    # plot surface with shaded facecolors
    surf = ax.plot_surface(XI, YI, ZI, rstride=1, cstride=1, facecolors=rgb,
                           linewidth=0, antialiased=True, shade=False)

    # add contour projection at base for readability
    zmin = np.nanmin(ZI)
    offset = zmin - (np.nanmax(ZI) - zmin) * 0.25
    try:
        cs = ax.contourf(XI, YI, ZI, zdir='z', offset=offset, cmap=CMAP, levels=12, alpha=0.7)
    except Exception:
        pass

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Pressure Ratio')
    hour = times[t_idx]
    ax.set_title(f'Spatio-temporal Pressure 3D Surface ({hour}:00)', fontsize=14)

    # style adjustments
    ax.set_box_aspect((np.ptp(grid_x), np.ptp(grid_y), max(1e-6, np.nanmax(ZI)-offset)))
    ax.view_init(elev=30, azim=-60)

    # colorbar as separate mappable
    mappable = cm.ScalarMappable(cmap=CMAP)
    mappable.set_array(ZI)
    fig.colorbar(mappable, shrink=0.6, aspect=12, pad=0.08, label='Pressure Ratio')

    plt.tight_layout()
    # save high-res PNG and SVG for vector/bitmap use
    png_out = outpath
    svg_out = outpath.replace('.png', '.svg')
    plt.savefig(png_out, dpi=300, bbox_inches='tight')
    try:
        plt.savefig(svg_out, format='svg', bbox_inches='tight')
    except Exception:
        pass
    print(f"3D pressure surface saved: {png_out} (and {svg_out} if supported)")


def plot_pressure_bar3d(grid_x, grid_y, times, pressure_grid, t_idx=None, outpath="q4_heatmap_bar3d.png"):
    """3D 柱状图：保留备用（当前主流程不生成）。"""
    # Intentionally left as a utility for experimentation.
    # (Not used in main flow per user request.)
    return


def print_insights(grid_x, grid_y, times, demand_grid, supply_grid, reachable_grid, pressure_grid):
    """打印关键洞察"""
    grid_size = len(grid_x)
    n_times = len(times)

    print("\n" + "=" * 70)
    print("空间-时间压力分析核心发现")
    print("=" * 70)

    # 最紧张的网格-时刻组合
    max_pressure = 0
    max_info = None
    for gx in range(grid_size):
        for gy in range(grid_size):
            for t_idx in range(n_times):
                p = pressure_grid[gx, gy, t_idx]
                if np.isfinite(p) and p > max_pressure:
                    max_pressure = p
                    max_info = (gx, gy, t_idx)

    if max_info:
        gx, gy, t_idx = max_info
        cx, cy = grid_x[gx], grid_y[gy]
        hour = times[t_idx]
        d = demand_grid[gx, gy, t_idx]
        s = reachable_grid[gx, gy, t_idx]
        print(f"\n▶ 最紧张网格: 中心 ({cx:.1f}, {cy:.1f})")
        print(f"  ⏰ 时刻: {hour:.0f}:00")
        print(f"  📊 需求: {d:.0f} 用户 | 可达供给: {s:.0f} 车位 | 压力比: {max_pressure:.2f}")

    # 整体统计
    total_demand_time = demand_grid.sum(axis=(0, 1))
    total_reachable_time = reachable_grid.sum(axis=(0, 1))
    total_supply_time = supply_grid.sum(axis=(0, 1))

    peak_demand_hour = times[np.argmax(total_demand_time)]
    valid_pressure = np.where(total_reachable_time > 0,
                              total_demand_time / total_reachable_time, 0)
    peak_pressure_hour = times[np.argmax(valid_pressure)]

    print(f"\n▶ 需求峰值时刻: {peak_demand_hour:.0f}:00 (最大活跃用户)")
    print(f"▶ 压力峰值时刻: {peak_pressure_hour:.0f}:00 (压力比最高)")

    # 供需失衡网格计数
    print(f"\n▶ 各时刻供需失衡网格（压力明细）:")
    header = f"  {'时刻':<8} {'有需求网格':<12} {'可达供给>0':<12} {'高压(>2)':<12} {'无供给':<12}"
    print(header)
    print(f"  {'-' * 56}")
    for t_idx, t in enumerate(times):
        p_slice = pressure_grid[:, :, t_idx]
        d_slice = demand_grid[:, :, t_idx]

        cells_with_demand = np.sum(d_slice > 0)
        cells_with_supply = np.sum(reachable_grid[:, :, t_idx] > 0)

        finite_p = np.isfinite(p_slice)
        high_pressure = np.sum(finite_p & (p_slice > 2))
        no_supply = np.sum((d_slice > 0) & (reachable_grid[:, :, t_idx] == 0))

        print(f"  {t:.0f}:00    {cells_with_demand:<12} {cells_with_supply:<12} {high_pressure:<12} {no_supply:<12}")

    # 空间聚集分析
    print(f"\n▶ 需求最聚集区域（全天累计，前5）:")
    total_demand_spatial = demand_grid.sum(axis=2)
    flat_idx = np.argsort(-total_demand_spatial.flatten())[:5]
    for idx in flat_idx:
        gx = idx % grid_size
        gy = idx // grid_size
        cx, cy = grid_x[gx], grid_y[gy]
        d_val = total_demand_spatial[gx, gy]
        print(f"  中心 ({cx:5.1f}, {cy:5.1f}) → 累计需求 {d_val:.0f} 用户")

    print()


def main():
    print("=" * 70)
    print("空间-时间压力热力图分析")
    print("=" * 70)

    spots, slots, users = load_data()

    # 计算可达半径：使用用户 max_walk 的中位数
    radius = users["max_walk"].median()
    print(f"\n可达半径: {radius:.2f}m（用户最大步行距离中位数）")

    # 构建压力矩阵
    grid_size = 10
    print(f"网格: {grid_size}x{grid_size}")
    print(f"时间点: 8-22h（每小时）")

    grid_x, grid_y, times, demand_grid, supply_grid, reachable_grid, pressure_grid = \
        build_spatial_temporal_pressure(spots, slots, users,
                                        grid_size=grid_size, radius=radius)

    # 打印洞察
    print_insights(grid_x, grid_y, times, demand_grid, supply_grid,
                   reachable_grid, pressure_grid)

    # 生成热力图
    plot_heatmap(spots, slots, users, grid_x, grid_y, times,
                  demand_grid, supply_grid, reachable_grid, pressure_grid)

    # plot_pressure_bar3d(grid_x, grid_y, times, pressure_grid, outpath="q4_heatmap_bar3d.png")

    # 生成汇总图
    plot_demand_supply_summary(grid_x, grid_y, times,
                                demand_grid, supply_grid, reachable_grid, pressure_grid)

    print("分析完成! 输出文件:")
    print("  - q4_heatmap.png (关键时刻空间压力热力图)")
    print("  - q4_heatmap_summary.png (供需汇总分析)")


if __name__ == "__main__":
    main()
