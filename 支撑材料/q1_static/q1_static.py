"""问题1：静态分配（无时间约束）

两阶段优化：
阶段1：最大化匹配用户数（最大基数匹配）
阶段2：在最大匹配数下最小化总步行距离

模型：二分图分配，每个车位最多一个用户，每个用户最多一个车位
"""

import pulp
import numpy as np
from utils import load_data, compute_distance_matrix, build_eligibility


def solve_q1(verbose=True):
    """求解问题1，返回分配结果"""
    spots, slots, users = load_data()
    dist_matrix = compute_distance_matrix(spots, users)
    eligible = build_eligibility(dist_matrix, users)

    n_users, n_spots = eligible.shape
    user_ids = users["user_id"].tolist()
    spot_ids = spots["spot_id"].tolist()

    # 构建可用边列表 (user_idx, spot_idx)
    edges = []
    for j in range(n_users):
        for i in range(n_spots):
            if eligible[j, i]:
                edges.append((j, i))

    if not edges:
        return {"assigned": [], "unassigned": user_ids, "n_assigned": 0,
                "avg_distance": 0, "utilization": 0, "status": "no_eligible"}

    # ============ 阶段1：最大化匹配用户数 ============
    if verbose:
        print("=" * 60)
        print("问题1：静态分配（无时间约束）")
        print("=" * 60)
        print(f"可用边数: {len(edges)} / {n_users * n_spots}")
        print("\n阶段1：求解最大基数匹配...")

    prob1 = pulp.LpProblem("Q1_MaxCardinality", pulp.LpMaximize)
    x_vars = {}
    for (j, i) in edges:
        x_vars[(j, i)] = pulp.LpVariable(f"x_{j}_{i}", cat="Binary")

    # 目标：最大化匹配用户数
    prob1 += pulp.lpSum([x_vars[e] for e in edges])

    # 约束：每个用户最多一个车位
    for j in range(n_users):
        incident = [(j, i) for (uj, i) in edges if uj == j]
        if incident:
            prob1 += pulp.lpSum([x_vars[e] for e in incident]) <= 1

    # 约束：每个车位最多一个用户
    for i in range(n_spots):
        incident = [(j, i) for (j, si) in edges if si == i]
        if incident:
            prob1 += pulp.lpSum([x_vars[e] for e in incident]) <= 1

    # 求解
    prob1.solve(pulp.PULP_CBC_CMD(msg=verbose))

    if prob1.status != pulp.LpStatusOptimal:
        if verbose:
            print(f"阶段1求解失败，状态: {pulp.LpStatus[prob1.status]}")
        return {"assigned": [], "unassigned": user_ids, "n_assigned": 0,
                "avg_distance": 0, "utilization": 0, "status": "phase1_failed"}

    # 获取最大匹配数
    max_assigned = int(pulp.value(prob1.objective))
    if verbose:
        print(f"最大匹配用户数: {max_assigned}")

    # ============ 阶段2：固定匹配数，最小化总步行距离 ============
    if verbose:
        print("\n阶段2：最小化总步行距离...")

    prob2 = pulp.LpProblem("Q1_MinDistance", pulp.LpMinimize)
    x_vars2 = {}
    for (j, i) in edges:
        x_vars2[(j, i)] = pulp.LpVariable(f"x2_{j}_{i}", cat="Binary")

    # 目标：最小化总步行距离
    prob2 += pulp.lpSum([dist_matrix[j, i] * x_vars2[(j, i)] for (j, i) in edges])

    # 约束1：每个用户最多一个车位
    for j in range(n_users):
        incident = [(j, i) for (uj, i) in edges if uj == j]
        if incident:
            prob2 += pulp.lpSum([x_vars2[e] for e in incident]) <= 1

    # 约束2：每个车位最多一个用户
    for i in range(n_spots):
        incident = [(j, i) for (j, si) in edges if si == i]
        if incident:
            prob2 += pulp.lpSum([x_vars2[e] for e in incident]) <= 1

    # 约束3：总匹配数 = 最大匹配数
    prob2 += pulp.lpSum([x_vars2[e] for e in edges]) == max_assigned

    # 求解
    prob2.solve(pulp.PULP_CBC_CMD(msg=verbose))

    if prob2.status != pulp.LpStatusOptimal:
        if verbose:
            print(f"阶段2求解失败，状态: {pulp.LpStatus[prob2.status]}")
        return {"assigned": [], "unassigned": user_ids, "n_assigned": 0,
                "avg_distance": 0, "utilization": 0, "status": "phase2_failed"}

    # ============ 提取结果 ============
    assignment = {}  # spot_idx -> user_idx
    for (j, i) in edges:
        if pulp.value(x_vars2[(j, i)]) > 0.5:
            assignment[i] = j

    assigned_users = sorted([user_ids[assignment[i]] for i in assignment])
    unassigned_users = sorted(set(user_ids) - set(assigned_users))
    n_assigned = len(assigned_users)
    total_distance = sum(dist_matrix[assignment[i], i] for i in assignment)
    avg_distance = total_distance / n_assigned if n_assigned > 0 else 0
    utilization = n_assigned / n_spots

    # 构建详细分配表
    assignment_detail = []
    for spot_idx, user_idx in assignment.items():
        assignment_detail.append({
            "spot_id": spot_ids[spot_idx],
            "user_id": user_ids[user_idx],
            "distance": dist_matrix[user_idx, spot_idx]
        })

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
        print(f"\n========== 问题1 结果 ==========")
        print(f"被分配用户数: {n_assigned}")
        print(f"总步行距离: {total_distance:.2f} 米")
        print(f"平均步行距离: {avg_distance:.2f} 米")
        print(f"车位利用率: {utilization:.2%} ({n_assigned}/{n_spots})")
        print(f"未分配用户数: {len(unassigned_users)}")

    return result


if __name__ == "__main__":
    result = solve_q1(verbose=True)
