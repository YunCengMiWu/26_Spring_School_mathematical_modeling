"""问题2：时间约束分配

考虑车位的可用时段和用户的时间窗口。
每个车位可服务多个用户，只要他们的时间窗口不重叠。
用户只能使用完全覆盖其[到达,离开]时段的车位时段。

模型：MILP
- x_ij: 用户j分配给车位i
- 每个用户最多一个车位
- 每个车位同一时间只能服务一个用户 → 冲突用户对的互斥约束
- 两级目标：最大化服务用户数，再最小化总步行距离
"""

import pulp
import numpy as np
from utils import load_data, compute_distance_matrix, build_eligibility, build_time_eligibility


def build_spot_time_conflicts(users, slots, eligible):
    """
    对每个车位i，构建内部用户时间冲突列表。
    返回: conflict_map[i] = [(j1, j2), ...] 表示在车位i上j1和j2不能同时被分配
    """
    n_users = len(users)
    n_spots = eligible.shape[1]
    slot_groups = slots.groupby("spot_id")

    conflict_map = [[] for _ in range(n_spots)]
    intervals = users[["arrival", "departure"]].values

    for i in range(n_spots):
        spot_id = i + 1
        if spot_id not in slot_groups.groups:
            continue
        time_windows = list(zip(
            slot_groups.get_group(spot_id)["start"],
            slot_groups.get_group(spot_id)["end"]
        ))

        # 找到对该车位兼容的用户列表
        candidate_users = []
        for j in range(n_users):
            if not eligible[j, i]:
                continue
            a, d = intervals[j, 0], intervals[j, 1]
            for s, e in time_windows:
                if s <= a and d <= e:
                    candidate_users.append(j)
                    break

        # 检查两两时间冲突
        for idx1 in range(len(candidate_users)):
            for idx2 in range(idx1 + 1, len(candidate_users)):
                j1 = candidate_users[idx1]
                j2 = candidate_users[idx2]
                a1, d1 = intervals[j1, 0], intervals[j1, 1]
                a2, d2 = intervals[j2, 0], intervals[j2, 1]
                if a1 < d2 and a2 < d1:
                    conflict_map[i].append((j1, j2))

    return conflict_map


def solve_q2(verbose=True):
    """求解问题2，返回分配结果"""
    spots, slots, users = load_data()
    dist_matrix = compute_distance_matrix(spots, users)
    eligible = build_eligibility(dist_matrix, users)
    time_eligible = build_time_eligibility(users, slots, eligible)

    n_users, n_spots = time_eligible.shape
    user_ids = users["user_id"].tolist()
    spot_ids = spots["spot_id"].tolist()

    # 构建可用边列表（距离+时间都兼容）
    edges = []
    for j in range(n_users):
        for i in range(n_spots):
            if time_eligible[j, i]:
                edges.append((j, i))

    if not edges:
        return {"assigned": [], "unassigned": user_ids, "n_assigned": 0,
                "avg_distance": 0, "utilization": 0, "status": "no_eligible"}

    if verbose:
        print("=" * 60)
        print("问题2：时间约束分配")
        print("=" * 60)
        print(f"距离+时间兼容边数: {len(edges)}")

    conflict_map = build_spot_time_conflicts(users, slots, time_eligible)
    total_conflicts = sum(len(c) for c in conflict_map)
    if verbose:
        print(f"车位内部时间冲突对数: {total_conflicts}")

    # ============ 阶段1：最大化匹配用户数 ============
    if verbose:
        print("\n阶段1：求解最大基数匹配（带时间约束）...")

    prob1 = pulp.LpProblem("Q2_MaxCardinality", pulp.LpMaximize)
    x_vars = {}
    for (j, i) in edges:
        x_vars[(j, i)] = pulp.LpVariable(f"x_{j}_{i}", cat="Binary")

    prob1 += pulp.lpSum([x_vars[e] for e in edges])

    for j in range(n_users):
        incident = [(j, i) for (uj, i) in edges if uj == j]
        if incident:
            prob1 += pulp.lpSum([x_vars[e] for e in incident]) <= 1

    for i in range(n_spots):
        for (j1, j2) in conflict_map[i]:
            if (j1, i) in x_vars and (j2, i) in x_vars:
                prob1 += x_vars[(j1, i)] + x_vars[(j2, i)] <= 1

    prob1.solve(pulp.PULP_CBC_CMD(msg=verbose))

    if prob1.status != pulp.LpStatusOptimal:
        if verbose:
            print(f"阶段1求解失败: {pulp.LpStatus[prob1.status]}")
        return {"assigned": [], "unassigned": user_ids, "n_assigned": 0,
                "avg_distance": 0, "utilization": 0, "status": "phase1_failed"}

    max_assigned = int(pulp.value(prob1.objective))
    if verbose:
        print(f"最大匹配用户数: {max_assigned}")

    # ============ 阶段2：固定匹配数，最小化总步行距离 ============
    if verbose:
        print("\n阶段2：最小化总步行距离...")

    prob2 = pulp.LpProblem("Q2_MinDistance", pulp.LpMinimize)
    x_vars2 = {}
    for (j, i) in edges:
        x_vars2[(j, i)] = pulp.LpVariable(f"x2_{j}_{i}", cat="Binary")

    prob2 += pulp.lpSum([dist_matrix[j, i] * x_vars2[(j, i)] for (j, i) in edges])

    for j in range(n_users):
        incident = [(j, i) for (uj, i) in edges if uj == j]
        if incident:
            prob2 += pulp.lpSum([x_vars2[e] for e in incident]) <= 1

    for i in range(n_spots):
        for (j1, j2) in conflict_map[i]:
            if (j1, i) in x_vars2 and (j2, i) in x_vars2:
                prob2 += x_vars2[(j1, i)] + x_vars2[(j2, i)] <= 1

    prob2 += pulp.lpSum([x_vars2[e] for e in edges]) == max_assigned

    prob2.solve(pulp.PULP_CBC_CMD(msg=verbose))

    if prob2.status != pulp.LpStatusOptimal:
        if verbose:
            print(f"阶段2求解失败: {pulp.LpStatus[prob2.status]}")
        return {"assigned": [], "unassigned": user_ids, "n_assigned": 0,
                "avg_distance": 0, "utilization": 0, "status": "phase2_failed"}

    # ============ 提取结果 ============
    assignment = {}
    for (j, i) in edges:
        if pulp.value(x_vars2[(j, i)]) > 0.5:
            if i not in assignment:
                assignment[i] = []
            assignment[i].append(j)

    # 统计所有被分配的用户
    assigned_set = set()
    assignment_detail = []
    for spot_idx, user_idxs in assignment.items():
        for user_idx in user_idxs:
            assigned_set.add(user_idx)
            assignment_detail.append({
                "spot_id": spot_ids[spot_idx],
                "user_id": user_ids[user_idx],
                "distance": float(dist_matrix[user_idx, spot_idx])
            })

    n_assigned = len(assigned_set)
    total_distance = sum(
        dist_matrix[user_idx, spot_idx]
        for spot_idx, user_idxs in assignment.items()
        for user_idx in user_idxs
    )
    avg_distance = total_distance / n_assigned if n_assigned > 0 else 0
    utilization = n_assigned / n_spots

    assigned_users = sorted([user_ids[u] for u in assigned_set])
    unassigned_users = sorted(set(user_ids) - set(assigned_users))

    result = {
        "assigned": assigned_users,
        "unassigned": unassigned_users,
        "n_assigned": n_assigned,
        "total_distance": total_distance,
        "avg_distance": avg_distance,
        "utilization": utilization,
        "assignment_detail": assignment_detail,
        "status": "optimal"
    }

    if verbose:
        print(f"\n========== 问题2 结果 ==========")
        print(f"被分配用户数: {n_assigned}")
        print(f"总步行距离: {total_distance:.2f} 米")
        print(f"平均步行距离: {avg_distance:.2f} 米")
        print(f"车位利用率: {utilization:.2%} ({n_assigned}/{n_spots})")
        print(f"未分配用户数: {len(unassigned_users)}")

    return result


if __name__ == "__main__":
    result = solve_q2(verbose=True)
