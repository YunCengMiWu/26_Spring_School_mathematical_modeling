"""
压力测试结果可视化
==================
读取 online_stress_test.py 生成的 CSV，输出两张图：
  stress_test_A_epr_curve.png  — EPR vs 到达集中度
  stress_test_B_epr_curve.png  — EPR vs 步行阈值系数

用法：
  python plot_stress.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
    'figure.dpi': 150,
})


def plot_A():
    path = os.path.join(OUT_DIR, 'stress_test_A_concentration.csv')
    if not os.path.exists(path):
        print(f'[SKIP] 找不到 {path}，请先运行 online_stress_test.py')
        return

    df = pd.read_csv(path)
    df = df.sort_values('kappa_hours')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # 左图：服务人数
    ax = axes[0]
    ax.plot(df['kappa_hours'], df['online_served'], 'o-', label='Greedy-Nearest (online)', color='steelblue')
    ax.plot(df['kappa_hours'], df['offline_served'], 's--', label='Offline MILP (Q2)', color='darkorange')
    ax.set_xlabel('Arrival Concentration κ (hours)', fontsize=11)
    ax.set_ylabel('Users Served', fontsize=11)
    ax.set_title('(A) Served Users vs. Arrival Concentration', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()   # 从宽松到集中（左→右 kappa 减小）

    # 右图：EPR
    ax = axes[1]
    ax.plot(df['kappa_hours'], df['EPR'], 'D-', color='crimson')
    ax.axhline(0.632, color='gray', linestyle=':', linewidth=1.2, label='KVV lower bound (0.632)')
    ax.set_xlabel('Arrival Concentration κ (hours)', fontsize=11)
    ax.set_ylabel('EPR (online / offline)', fontsize=11)
    ax.set_title('(A) EPR vs. Arrival Concentration', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'stress_test_A_epr_curve.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'已保存：{out}')


def plot_B():
    path = os.path.join(OUT_DIR, 'stress_test_B_walk_threshold.csv')
    if not os.path.exists(path):
        print(f'[SKIP] 找不到 {path}，请先运行 online_stress_test.py')
        return

    df = pd.read_csv(path)
    df = df.sort_values('walk_scale')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # 左图：服务人数
    ax = axes[0]
    ax.plot(df['walk_scale'], df['online_served'], 'o-', label='Greedy-Nearest', color='steelblue')
    ax.plot(df['walk_scale'], df['offline_served'], 's--', label='Offline MILP', color='darkorange')
    ax.set_xlabel('Walk Threshold Scale', fontsize=11)
    ax.set_ylabel('Users Served', fontsize=11)
    ax.set_title('(B) Served Users vs. Walk Scale', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 中图：平均步行距离
    ax = axes[1]
    ax.plot(df['walk_scale'], df['online_avg_dist_m'], 'o-', label='Greedy-Nearest', color='steelblue')
    ax.plot(df['walk_scale'], df['offline_avg_dist_m'], 's--', label='Offline MILP', color='darkorange')
    ax.set_xlabel('Walk Threshold Scale', fontsize=11)
    ax.set_ylabel('Avg Walk Distance (m)', fontsize=11)
    ax.set_title('(B) Avg Distance vs. Walk Scale', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 右图：EPR
    ax = axes[2]
    ax.plot(df['walk_scale'], df['EPR'], 'D-', color='crimson')
    ax.axhline(0.632, color='gray', linestyle=':', linewidth=1.2, label='KVV lower bound')
    ax.set_xlabel('Walk Threshold Scale', fontsize=11)
    ax.set_ylabel('EPR (online / offline)', fontsize=11)
    ax.set_title('(B) EPR vs. Walk Scale', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'stress_test_B_epr_curve.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'已保存：{out}')


if __name__ == '__main__':
    plot_A()
    plot_B()
    print('绘图完成。')
