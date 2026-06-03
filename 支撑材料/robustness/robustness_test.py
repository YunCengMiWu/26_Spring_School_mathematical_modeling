"""
假设鲁棒性检验
==============
扰动A：到达/离开时间噪声 — epsilon ~ Uniform[-10, 10] min，重复20次
扰动B：取消率 — 随机删除 p% 用户请求，p in {5, 10, 15, 20}

输出：
  robustness_A_time_noise.csv
  robustness_B_cancellation.csv

用法：
  python robustness_test.py

随机种子：42
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'deepseek'))

try:
    from utils import load_data, compute_distance_matrix
except ImportError:
    def load_data():
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


# ── 核心算法（与 online_stress_test.py 保持一致）────────────────────────────

def build_time_eligible(users_df, slots_df, dist_matrix):
    n_users, n_spots = dist_matrix.shape
    max_walk = users_df['max_walk'].values
    dist_ok = dist_matrix <= max_walk[:, np.newaxis]
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


def offline_milp(users_df, time_eligible, dist_matrix):
    try:
        import pulp
    except ImportError:
        return None, None

    n_users, n_spots = time_eligible.shape
    conflict_pairs = {}
    for i in range(n_spots):
        cands = np.where(time_eligible[:, i])[0]
        pairs = []
        for a_idx in range(len(cands)):
            for b_idx in range(a_idx + 1, len(cands)):
                j1, j2 = cands[a_idx], cands[b_idx]
                a1, d1 = users_df.iloc[j1]['arrival'], users_df.iloc[j1]['departure']
                a2, d2 = users_df.iloc[j2]['arrival'], users_df.iloc[j2]['departure']
                if a1 < d2 and a2 < d1:
                    pairs.append((j1, j2))
        if pairs:
            conflict_pairs[i] = pairs

    edges = [(j, i) for j in range(n_users) for i in range(n_spots) if time_eligible[j, i]]
    if not edges:
        return 0, 0.0

    prob1 = pulp.LpProblem('rob_phase1', pulp.LpMaximize)
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

    prob2 = pulp.LpProblem('rob_phase2', pulp.LpMinimize)
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

    total_dist = sum(
        dist_matrix[j, i] * pulp.value(x2[(j, i)])
        for (j, i) in edges
        if pulp.value(x2[(j, i)]) is not None and pulp.value(x2[(j, i)]) > 0.5
    )
    avg_dist = total_dist / N_star if N_star > 0 else 0.0
    return N_star, avg_dist


# ── 扰动A：时间噪声 ───────────────────────────────────────────────────────────

def run_perturbation_A(spots, slots, users, dist_matrix, n_trials=20, noise_min_h=-10/60, noise_max_h=10/60):
    """
    到达/离开时间加均匀噪声，重复 n_trials 次，记录均值±标准差
    noise 单位：小时（10 min = 10/60 h）
    """
    print('=== 扰动A：时间噪声 ===')
    rng = np.random.default_rng(SEED)

    baseline_te = build_time_eligible(users, slots, dist_matrix)
    baseline_served, baseline_dist = offline_milp(users, baseline_te, dist_matrix)
    print(f'  基准（无扰动）：served={baseline_served}, avg_dist={baseline_dist:.3f}m')

    served_list, dist_list = [], []
    for trial in range(n_trials):
        users_mod = users.copy()
        eps_a = rng.uniform(noise_min_h, noise_max_h, size=len(users))
        eps_d = rng.uniform(noise_min_h, noise_max_h, size=len(users))
        durations = (users['departure'] - users['arrival']).values
        new_a = np.clip(users['arrival'].values + eps_a, 8.0, 22.0)
        new_d = np.clip(new_a + durations + eps_d, new_a + 0.05, 22.0)
        users_mod['arrival'] = new_a
        users_mod['departure'] = new_d

        te = build_time_eligible(users_mod, slots, dist_matrix)
        served, avg_dist = offline_milp(users_mod, te, dist_matrix)
        if served is not None:
            served_list.append(served)
            dist_list.append(avg_dist)
        print(f'  trial {trial+1:2d}: served={served}, avg_dist={avg_dist:.3f}m')

    records = [{
        'noise_range_min': noise_min_h * 60,
        'noise_range_max': noise_max_h * 60,
        'n_trials': n_trials,
        'baseline_served': baseline_served,
        'baseline_avg_dist_m': round(baseline_dist, 3) if baseline_dist else None,
        'mean_served': round(np.mean(served_list), 2),
        'std_served': round(np.std(served_list), 2),
        'mean_avg_dist_m': round(np.mean(dist_list), 3),
        'std_avg_dist_m': round(np.std(dist_list), 3),
        'served_change_pct': round((np.mean(served_list) - baseline_served) / baseline_served * 100, 2) if baseline_served else None,
    }]
    df = pd.DataFrame(records)
    out_path = os.path.join(OUT_DIR, 'robustness_A_time_noise.csv')
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'  → 已保存：{out_path}\n')
    return df


# ── 扰动B：取消率 ─────────────────────────────────────────────────────────────

def run_perturbation_B(spots, slots, users, dist_matrix, cancel_rates=None, n_trials=10):
    """
    随机删除 p% 用户请求，重复 n_trials 次，记录均值±标准差
    """
    if cancel_rates is None:
        cancel_rates = [0.05, 0.10, 0.15, 0.20]
    print('=== 扰动B：取消率 ===')
    rng = np.random.default_rng(SEED)

    baseline_te = build_time_eligible(users, slots, dist_matrix)
    baseline_served, baseline_dist = offline_milp(users, baseline_te, dist_matrix)
    print(f'  基准（无取消）：served={baseline_served}')

    records = []
    for p in cancel_rates:
        served_list, dist_list = [], []
        for trial in range(n_trials):
            n_cancel = int(len(users) * p)
            cancel_idx = rng.choice(len(users), size=n_cancel, replace=False)
            keep_mask = np.ones(len(users), dtype=bool)
            keep_mask[cancel_idx] = False
            users_mod = users[keep_mask].reset_index(drop=True)
            dist_mod = dist_matrix[keep_mask]

            te = build_time_eligible(users_mod, slots, dist_mod)
            served, avg_dist = offline_milp(users_mod, te, dist_mod)
            if served is not None:
                served_list.append(served)
                dist_list.append(avg_dist)

        rec = {
            'cancel_rate_pct': int(p * 100),
            'n_trials': n_trials,
            'baseline_served': baseline_served,
            'mean_served': round(np.mean(served_list), 2),
            'std_served': round(np.std(served_list), 2),
            'mean_avg_dist_m': round(np.mean(dist_list), 3),
            'std_avg_dist_m': round(np.std(dist_list), 3),
            'served_change_pct': round((np.mean(served_list) - baseline_served) / baseline_served * 100, 2),
        }
        records.append(rec)
        print(f'  p={int(p*100)}%: mean_served={rec["mean_served"]}±{rec["std_served"]}, change={rec["served_change_pct"]}%')

    df = pd.DataFrame(records)
    out_path = os.path.join(OUT_DIR, 'robustness_B_cancellation.csv')
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

    df_A = run_perturbation_A(spots, slots, users, dist_matrix, n_trials=20)
    df_B = run_perturbation_B(spots, slots, users, dist_matrix, n_trials=10)

    print('全部鲁棒性检验完成。')
