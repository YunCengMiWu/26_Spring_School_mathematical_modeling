"""
在线算法压力测试
================
测试A：到达集中度扫描 — 将用户到达时间向午间压缩，观察 EPR 变化
测试B：步行阈值扰动 — w_j 乘以系数，观察服务人数/平均距离/EPR 变化

输出：
  stress_test_A_concentration.csv
  stress_test_B_walk_threshold.csv

用法：
  python online_stress_test.py

依赖：pandas, numpy, scipy, pulp
随机种子：42
"""

import sys
import os
import copy
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# ── 路径设置：优先使用 deepseek/utils.py 的数据加载逻辑 ──────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'deepseek'))

try:
    from utils import load_data, compute_distance_matrix
except ImportError:
    def load_data():
        """Fallback: 直接搜索 CSV"""
        from glob import glob
        for base in [REPO_ROOT, 'C:\\']:
            for p in glob(os.path.join(base, '**', 'spots.csv'), recursive=True):
                d = os.path.dirname(p)
                if os.path.exists(os.path.join(d, 'slots.csv')) and os.path.exists(os.path.join(d, 'users.csv')):
                    def _read(path):
                        for enc in ['gbk', 'utf-8', 'utf-8-sig']:
                            try:
                                return pd.read_csv(path, encoding=enc)
                            except Exception:
                                continue
                        return pd.read_csv(path)
                    spots = _read(os.path.join(d, 'spots.csv'))
                    slots = _read(os.path.join(d, 'slots.csv'))
                    users = _read(os.path.join(d, 'users.csv'))
                    spots.columns = ['spot_id', 'x', 'y']
                    slots.columns = ['spot_id', 'start', 'end']
                    users.columns = ['user_id', 'dest_x', 'dest_y', 'max_walk', 'arrival', 'departure']
                    return spots, slots, users
        raise FileNotFoundError('找不到数据文件')

    def compute_distance_matrix(spots, users):
        return cdist(users[['dest_x', 'dest_y']].values, spots[['x', 'y']].values)

SEED = 42
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── 核心算法 ──────────────────────────────────────────────────────────────────

def build_time_eligible(users_df, slots_df, dist_matrix, walk_scale=1.0):
    """构建时空可行边集 E^T，返回 bool 矩阵 (n_users, n_spots)"""
    n_users, n_spots = dist_matrix.shape
    max_walk = users_df['max_walk'].values * walk_scale
    dist_ok = dist_matrix <= max_walk[:, np.newaxis]          # (n_users, n_spots)

    time_ok = np.zeros((n_users, n_spots), dtype=bool)
    slot_groups = slots_df.groupby('spot_id')

    for j in range(n_users):
        a, d = users_df.iloc[j]['arrival'], users_df.iloc[j]['departure']
        for idx_i in np.where(dist_ok[j])[0]:
            spot_id = idx_i + 1
            if spot_id not in slot_groups.groups:
                continue
            for _, row in slot_groups.get_group(spot_id).iterrows():
                if row['start'] <= a and d <= row['end']:
                    time_ok[j, idx_i] = True
                    break
    return time_ok


def greedy_nearest(users_df, time_eligible, dist_matrix):
    """
    Greedy-Nearest 在线算法
    用户按 arrival 升序处理，为每位用户选最近的时间兼容且当前空闲的车位。
    返回 (served_count, avg_distance)
    """
    order = users_df['arrival'].argsort().values          # 按到达时间排序
    n_spots = dist_matrix.shape[1]
    # 每个车位的已分配区间列表 [(a, d), ...]
    spot_occupied = [[] for _ in range(n_spots)]

    served = 0
    total_dist = 0.0

    for j in order:
        a, d = users_df.iloc[j]['arrival'], users_df.iloc[j]['departure']
        best_spot = -1
        best_dist = np.inf

        for i in np.where(time_eligible[j])[0]:
            # 检查车位 i 在 [a, d) 是否空闲
            conflict = any(
                not (d <= oa or od <= a)
                for (oa, od) in spot_occupied[i]
            )
            if conflict:
                continue
            if dist_matrix[j, i] < best_dist:
                best_dist = dist_matrix[j, i]
                best_spot = i

        if best_spot >= 0:
            spot_occupied[best_spot].append((a, d))
            served += 1
            total_dist += best_dist

    avg_dist = total_dist / served if served > 0 else 0.0
    return served, avg_dist


def offline_milp(users_df, time_eligible, dist_matrix):
    """
    离线 MILP（Q2 模型）：两阶段字典序优化
    阶段1：最大化服务人数；阶段2：最小化总步行距离
    返回 (served_count, avg_distance)
    """
    try:
        import pulp
    except ImportError:
        print('[WARN] pulp 未安装，离线 MILP 跳过，返回 None')
        return None, None

    n_users, n_spots = time_eligible.shape
    # 构建冲突对
    conflict_pairs = {}   # spot_idx -> list of (j1, j2)
    for i in range(n_spots):
        cands = np.where(time_eligible[:, i])[0]
        pairs = []
        for a_idx in range(len(cands)):
            for b_idx in range(a_idx + 1, len(cands)):
                j1, j2 = cands[a_idx], cands[b_idx]
                a1, d1 = users_df.iloc[j1]['arrival'], users_df.iloc[j1]['departure']
                a2, d2 = users_df.iloc[j2]['arrival'], users_df.iloc[j2]['departure']
                if a1 < d2 and a2 < d1:   # 重叠
                    pairs.append((j1, j2))
        if pairs:
            conflict_pairs[i] = pairs

    edges = [(j, i) for j in range(n_users) for i in range(n_spots) if time_eligible[j, i]]
    if not edges:
        return 0, 0.0

    # 阶段1
    prob1 = pulp.LpProblem('Q2_phase1', pulp.LpMaximize)
    x = {(j, i): pulp.LpVariable(f'x_{j}_{i}', cat='Binary') for (j, i) in edges}
    prob1 += pulp.lpSum(x[e] for e in edges)
    for j in range(n_users):
        es = [(j, i) for (jj, i) in edges if jj == j]
        if es:
            prob1 += pulp.lpSum(x[e] for e in es) <= 1
    for i in range(n_spots):
        es = [(j, i) for (j, ii) in edges if ii == i]
        if es:
            prob1 += pulp.lpSum(x[e] for e in es) <= 1
    for i, pairs in conflict_pairs.items():
        for (j1, j2) in pairs:
            if (j1, i) in x and (j2, i) in x:
                prob1 += x[(j1, i)] + x[(j2, i)] <= 1

    prob1.solve(pulp.PULP_CBC_CMD(msg=0))
    N_star = int(round(pulp.value(prob1.objective)))

    # 阶段2
    prob2 = pulp.LpProblem('Q2_phase2', pulp.LpMinimize)
    x2 = {(j, i): pulp.LpVariable(f'x2_{j}_{i}', cat='Binary') for (j, i) in edges}
    prob2 += pulp.lpSum(dist_matrix[j, i] * x2[(j, i)] for (j, i) in edges)
    for j in range(n_users):
        es = [(j, i) for (jj, i) in edges if jj == j]
        if es:
            prob2 += pulp.lpSum(x2[e] for e in es) <= 1
    for i in range(n_spots):
        es = [(j, i) for (j, ii) in edges if ii == i]
        if es:
            prob2 += pulp.lpSum(x2[e] for e in es) <= 1
    for i, pairs in conflict_pairs.items():
        for (j1, j2) in pairs:
            if (j1, i) in x2 and (j2, i) in x2:
                prob2 += x2[(j1, i)] + x2[(j2, i)] <= 1
    prob2 += pulp.lpSum(x2[e] for e in edges) == N_star

    prob2.solve(pulp.PULP_CBC_CMD(msg=0))

    served = N_star
    total_dist = sum(
        dist_matrix[j, i] * pulp.value(x2[(j, i)])
        for (j, i) in edges
        if pulp.value(x2[(j, i)]) is not None and pulp.value(x2[(j, i)]) > 0.5
    )
    avg_dist = total_dist / served if served > 0 else 0.0
    return served, avg_dist


# ── 测试A：到达集中度扫描 ─────────────────────────────────────────────────────

def compress_arrivals(users_df, kappa, center=12.5, rng=None):
    """
    将用户到达时间向 center（小时）压缩，集中度参数 kappa（小时）。
    压缩后到达时间 = center + (arrival - center) * kappa / original_half_range
    保持停车时长不变（departure = new_arrival + duration）。
    kappa 越小越集中；kappa = original_half_range 时不变。
    """
    users_new = users_df.copy()
    arrivals = users_df['arrival'].values
    durations = (users_df['departure'] - users_df['arrival']).values

    original_half_range = max(abs(arrivals - center).max(), 1e-6)
    scale = kappa / original_half_range
    new_arrivals = center + (arrivals - center) * scale
    # 确保 arrival >= 8.0, departure <= 22.0
    new_arrivals = np.clip(new_arrivals, 8.0, 22.0 - durations)
    new_departures = new_arrivals + durations
    new_departures = np.clip(new_departures, new_arrivals + 0.1, 22.0)

    users_new['arrival'] = new_arrivals
    users_new['departure'] = new_departures
    return users_new


def run_test_A(spots, slots, users, dist_matrix):
    """测试A：集中度扫描"""
    print('=== 测试A：到达集中度扫描 ===')
    arrivals = users['arrival'].values
    center = 12.5
    original_half_range = abs(arrivals - center).max()

    # kappa 从 original_half_range（原始）到 1h（高度集中）
    kappas = [original_half_range, original_half_range * 0.75,
              original_half_range * 0.5, 3.0, 2.0, 1.0]
    kappas = sorted(set(round(k, 2) for k in kappas), reverse=True)

    records = []
    for kappa in kappas:
        users_mod = compress_arrivals(users, kappa, center=center)
        te = build_time_eligible(users_mod, slots, dist_matrix)
        online_served, online_dist = greedy_nearest(users_mod, te, dist_matrix)
        offline_served, offline_dist = offline_milp(users_mod, te, dist_matrix)
        epr = online_served / offline_served if offline_served and offline_served > 0 else None
        rec = {
            'kappa_hours': round(kappa, 2),
            'online_served': online_served,
            'online_avg_dist_m': round(online_dist, 3),
            'offline_served': offline_served,
            'offline_avg_dist_m': round(offline_dist, 3) if offline_dist is not None else None,
            'EPR': round(epr, 4) if epr is not None else None,
        }
        records.append(rec)
        print(f'  kappa={kappa:.2f}h | online={online_served} | offline={offline_served} | EPR={epr}')

    df = pd.DataFrame(records)
    out_path = os.path.join(OUT_DIR, 'stress_test_A_concentration.csv')
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'  → 已保存：{out_path}\n')
    return df


# ── 测试B：步行阈值扰动 ───────────────────────────────────────────────────────

def run_test_B(spots, slots, users, dist_matrix):
    """测试B：步行阈值扰动"""
    print('=== 测试B：步行阈值扰动 ===')
    scales = [0.8, 0.9, 1.0, 1.1, 1.2]
    records = []

    for scale in scales:
        te = build_time_eligible(users, slots, dist_matrix, walk_scale=scale)
        online_served, online_dist = greedy_nearest(users, te, dist_matrix)
        offline_served, offline_dist = offline_milp(users, te, dist_matrix)
        epr = online_served / offline_served if offline_served and offline_served > 0 else None
        rec = {
            'walk_scale': scale,
            'online_served': online_served,
            'online_avg_dist_m': round(online_dist, 3),
            'offline_served': offline_served,
            'offline_avg_dist_m': round(offline_dist, 3) if offline_dist is not None else None,
            'EPR': round(epr, 4) if epr is not None else None,
        }
        records.append(rec)
        print(f'  scale={scale:.1f} | online={online_served} | offline={offline_served} | EPR={epr}')

    df = pd.DataFrame(records)
    out_path = os.path.join(OUT_DIR, 'stress_test_B_walk_threshold.csv')
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'  → 已保存：{out_path}\n')
    return df


# ── 主入口 ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(SEED)
    print('加载数据...')
    spots, slots, users = load_data()
    dist_matrix = compute_distance_matrix(spots, users)
    print(f'  spots={len(spots)}, slots={len(slots)}, users={len(users)}\n')

    df_A = run_test_A(spots, slots, users, dist_matrix)
    df_B = run_test_B(spots, slots, users, dist_matrix)

    print('全部测试完成。')
    print(f'  测试A结果：stress_test_A_concentration.csv')
    print(f'  测试B结果：stress_test_B_walk_threshold.csv')
