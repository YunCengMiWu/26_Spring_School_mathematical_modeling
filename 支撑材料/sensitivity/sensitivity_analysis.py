"""敏感度分析

分析以下参数对结果的影响：
1. 步行距离阈值（max_walk）- 用户可接受的最大步行距离
2. 用户停车时长 - 用户停车时间长度的影响
3. 车位时段数量 - 车位开放更多时段的影响
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import time

from utils import load_data, compute_distance_matrix, build_eligibility, build_time_eligibility
from q2_time import build_spot_time_conflicts
import pulp

from plot_theme import apply_common_style, pick_color

apply_common_style()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def solve_q1_with_data(spots, slots, users, dist_matrix, eligible, verbose=False):
    """使用给定数据求解 Q1（避免重新加载）"""
    n_users, n_spots = eligible.shape

    edges = []
    for j in range(n_users):
        for i in range(n_spots):
            if eligible[j, i]:
                edges.append((j, i))

    if not edges:
        return {"n_assigned": 0, "avg_distance": 0, "utilization": 0, "total_distance": 0}

    # 阶段1: 最大基数匹配
    prob1 = pulp.LpProblem("Q1_S", pulp.LpMaximize)
    x_vars = {(j, i): pulp.LpVariable(f"x_{j}_{i}", cat="Binary") for (j, i) in edges}
    prob1 += pulp.lpSum(x_vars.values())

    for j in range(n_users):
        incident = [x_vars[(j, i)] for (uj, i) in edges if uj == j]
        if incident:
            prob1 += pulp.lpSum(incident) <= 1

    for i in range(n_spots):
        incident = [x_vars[(j, i)] for (j, si) in edges if si == i]
        if incident:
            prob1 += pulp.lpSum(incident) <= 1

    prob1.solve(pulp.PULP_CBC_CMD(msg=False))
    if prob1.status != pulp.LpStatusOptimal:
        return {"n_assigned": 0, "avg_distance": 0, "utilization": 0, "total_distance": 0}

    max_assigned = int(pulp.value(prob1.objective))

    # 阶段2: 最小化距离
    prob2 = pulp.LpProblem("Q1_D", pulp.LpMinimize)
    x2 = {(j, i): pulp.LpVariable(f"y_{j}_{i}", cat="Binary") for (j, i) in edges}
    prob2 += pulp.lpSum([dist_matrix[j, i] * x2[(j, i)] for (j, i) in edges])

    for j in range(n_users):
        incident = [x2[(j, i)] for (uj, i) in edges if uj == j]
        if incident:
            prob2 += pulp.lpSum(incident) <= 1

    for i in range(n_spots):
        incident = [x2[(j, i)] for (j, si) in edges if si == i]
        if incident:
            prob2 += pulp.lpSum(incident) <= 1

    prob2 += pulp.lpSum(x2.values()) == max_assigned

    prob2.solve(pulp.PULP_CBC_CMD(msg=False))
    if prob2.status != pulp.LpStatusOptimal:
        return {"n_assigned": 0, "avg_distance": 0, "utilization": 0, "total_distance": 0}

    total = sum(dist_matrix[j, i] * pulp.value(x2[(j, i)]) for (j, i) in edges)
    avg = total / max_assigned if max_assigned > 0 else 0

    return {
        "n_assigned": max_assigned,
        "total_distance": total,
        "avg_distance": avg,
        "utilization": max_assigned / n_spots,
    }


def solve_q2_with_data(spots, slots, users, dist_matrix, eligible, time_eligible, verbose=False):
    """使用给定数据求解 Q2"""
    n_users, n_spots = time_eligible.shape

    edges = [(j, i) for j in range(n_users) for i in range(n_spots) if time_eligible[j, i]]

    if not edges:
        return {"n_assigned": 0, "avg_distance": 0, "utilization": 0, "total_distance": 0}

    conflict_map = build_spot_time_conflicts(users, slots, time_eligible)

    # 阶段1
    prob1 = pulp.LpProblem("Q2_S", pulp.LpMaximize)
    x = {(j, i): pulp.LpVariable(f"x_{j}_{i}", cat="Binary") for (j, i) in edges}
    prob1 += pulp.lpSum(x.values())

    for j in range(n_users):
        inc = [x[(j, i)] for (uj, i) in edges if uj == j]
        if inc:
            prob1 += pulp.lpSum(inc) <= 1

    for i in range(n_spots):
        for (j1, j2) in conflict_map[i]:
            if (j1, i) in x and (j2, i) in x:
                prob1 += x[(j1, i)] + x[(j2, i)] <= 1

    prob1.solve(pulp.PULP_CBC_CMD(msg=False))
    if prob1.status != pulp.LpStatusOptimal:
        return {"n_assigned": 0, "avg_distance": 0, "utilization": 0, "total_distance": 0}

    max_assigned = int(pulp.value(prob1.objective))

    # 阶段2
    prob2 = pulp.LpProblem("Q2_D", pulp.LpMinimize)
    x2 = {(j, i): pulp.LpVariable(f"y_{j}_{i}", cat="Binary") for (j, i) in edges}
    prob2 += pulp.lpSum([dist_matrix[j, i] * x2[(j, i)] for (j, i) in edges])

    for j in range(n_users):
        inc = [x2[(j, i)] for (uj, i) in edges if uj == j]
        if inc:
            prob2 += pulp.lpSum(inc) <= 1

    for i in range(n_spots):
        for (j1, j2) in conflict_map[i]:
            if (j1, i) in x2 and (j2, i) in x2:
                prob2 += x2[(j1, i)] + x2[(j2, i)] <= 1

    prob2 += pulp.lpSum(x2.values()) == max_assigned
    prob2.solve(pulp.PULP_CBC_CMD(msg=False))
    if prob2.status != pulp.LpStatusOptimal:
        return {"n_assigned": 0, "avg_distance": 0, "utilization": 0, "total_distance": 0}

    total = sum(dist_matrix[j, i] * pulp.value(x2[(j, i)]) for (j, i) in edges)
    avg = total / max_assigned if max_assigned > 0 else 0

    return {
        "n_assigned": max_assigned,
        "total_distance": total,
        "avg_distance": avg,
        "utilization": max_assigned / n_spots,
    }


def sensitivity_walk_distance():
    """敏感度分析1：步行距离阈值"""
    print("\n" + "=" * 70)
    print("敏感度分析 1：步行距离阈值 (max_walk) 倍数变化")
    print("=" * 70)

    spots, slots, users = load_data()
    dist_matrix = compute_distance_matrix(spots, users)
    original_max_walk = users["max_walk"].copy()

    multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    results = []

    for mult in multipliers:
        users["max_walk"] = original_max_walk * mult
        eligible = build_eligibility(dist_matrix, users)

        print(f"\n>>> 步行距离倍数 = {mult:.2f}, 平均阈值 = {users['max_walk'].mean():.2f}m")
        print(f"    可行边数: {eligible.sum()}")

        # Q1
        t0 = time.time()
        r1 = solve_q1_with_data(spots, slots, users, dist_matrix, eligible)
        t1 = time.time() - t0

        # Q2 (需要重新计算 time_eligible)
        time_eligible = build_time_eligibility(users, slots, eligible)
        t0 = time.time()
        r2 = solve_q2_with_data(spots, slots, users, dist_matrix, eligible, time_eligible)
        t2 = time.time() - t0

        print(f"    Q1: 分配={r1['n_assigned']}, 平均距离={r1['avg_distance']:.2f}m, 耗时={t1:.2f}s")
        print(f"    Q2: 分配={r2['n_assigned']}, 平均距离={r2['avg_distance']:.2f}m, 耗时={t2:.2f}s")

        results.append({
            "multiplier": mult,
            "q1_assigned": r1["n_assigned"],
            "q1_avg_dist": r1["avg_distance"],
            "q1_util": r1["utilization"] * 100,
            "q2_assigned": r2["n_assigned"],
            "q2_avg_dist": r2["avg_distance"],
            "q2_util": r2["utilization"] * 100,
            "q1_time": t1,
            "q2_time": t2,
        })

    users["max_walk"] = original_max_walk

    # 打印汇总
    print("\n" + "=" * 70)
    print("步行距离敏感度结果汇总")
    print("=" * 70)
    print(f"{'倍数':<8} {'Q1分配':<8} {'Q1距离':<10} {'Q2分配':<8} {'Q2距离':<10} {'Q2/Q1比':<10}")
    print("-" * 70)
    for r in results:
        ratio = r["q2_assigned"] / r["q1_assigned"] if r["q1_assigned"] > 0 else 0
        print(f"{r['multiplier']:<8.2f} {r['q1_assigned']:<8} {r['q1_avg_dist']:<10.2f} "
              f"{r['q2_assigned']:<8} {r['q2_avg_dist']:<10.2f} {ratio:<10.2%}")

    return pd.DataFrame(results)


def sensitivity_user_duration():
    """敏感度分析2：用户停车时长变化"""
    print("\n" + "=" * 70)
    print("敏感度分析 2：用户停车时长变化")
    print("=" * 70)

    spots, slots, users = load_data()
    dist_matrix = compute_distance_matrix(spots, users)
    eligible = build_eligibility(dist_matrix, users)

    original_arrival = users["arrival"].copy()
    original_departure = users["departure"].copy()

    duration_factors = [0.5, 0.75, 1.0, 1.25, 1.5]
    results = []

    for factor in duration_factors:
        # 调整离开时间（保持开始时间不变）
        duration = original_departure - original_arrival
        users["departure"] = original_arrival + duration * factor

        time_eligible = build_time_eligibility(users, slots, eligible)
        n_compatible = time_eligible.sum()

        print(f"\n>>> 时长倍数 = {factor:.2f}, 平均时长 = {(users['departure']-users['arrival']).mean():.2f}h")
        print(f"    时间兼容边数: {n_compatible}")

        t0 = time.time()
        r2 = solve_q2_with_data(spots, slots, users, dist_matrix, eligible, time_eligible)
        t2 = time.time() - t0

        print(f"    Q2: 分配={r2['n_assigned']}, 平均距离={r2['avg_distance']:.2f}m, 耗时={t2:.2f}s")

        results.append({
            "duration_factor": factor,
            "avg_duration": (users["departure"] - users["arrival"]).mean(),
            "q2_assigned": r2["n_assigned"],
            "q2_avg_dist": r2["avg_distance"],
            "q2_util": r2["utilization"] * 100,
            "compatible_edges": int(n_compatible),
            "time": t2,
        })

    users["arrival"] = original_arrival
    users["departure"] = original_departure

    # 打印汇总
    print("\n" + "=" * 70)
    print("用户停车时长敏感度结果")
    print("=" * 70)
    print(f"{'时长倍数':<10} {'平均时长(h)':<12} {'Q2分配':<8} {'Q2距离':<10} {'兼容边':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['duration_factor']:<10.2f} {r['avg_duration']:<12.2f} "
              f"{r['q2_assigned']:<8} {r['q2_avg_dist']:<10.2f} {r['compatible_edges']:<10}")

    return pd.DataFrame(results)


def sensitivity_slot_extension():
    """敏感度分析3：车位时段扩展"""
    print("\n" + "=" * 70)
    print("敏感度分析 3：车位可用时段扩展（错时共享）")
    print("=" * 70)

    spots, slots, users = load_data()
    dist_matrix = compute_distance_matrix(spots, users)
    eligible = build_eligibility(dist_matrix, users)

    # 原始结果
    print("\n>>> 原始时段:")
    time_eligible_orig = build_time_eligibility(users, slots, eligible)
    print(f"    时间兼容边数: {time_eligible_orig.sum()}")
    t0 = time.time()
    r2_orig = solve_q2_with_data(spots, slots, users, dist_matrix, eligible, time_eligible_orig)
    t_orig = time.time() - t0
    print(f"    Q2: 分配={r2_orig['n_assigned']}, 平均距离={r2_orig['avg_distance']:.2f}m, 耗时={t_orig:.2f}s")

    # 全天开放
    extended_slots_full = pd.DataFrame([{"spot_id": sid, "start": 8.0, "end": 22.0}
                                        for sid in spots["spot_id"]])
    print("\n>>> 全天开放时段 (8-22) - 上限场景:")
    time_eligible_full = build_time_eligibility(users, extended_slots_full, eligible)
    print(f"    时间兼容边数: {time_eligible_full.sum()}")
    t0 = time.time()
    r2_full = solve_q2_with_data(spots, extended_slots_full, users, dist_matrix, eligible, time_eligible_full)
    t_full = time.time() - t0
    print(f"    Q2: 分配={r2_full['n_assigned']}, 平均距离={r2_full['avg_distance']:.2f}m, 耗时={t_full:.2f}s")

    # 单时段补充反向时段
    print("\n>>> 单时段车位补充反向时段 (中等场景):")
    spot_slot_count = slots.groupby("spot_id").size()
    single_slot_spots = spot_slot_count[spot_slot_count == 1].index

    new_rows = []
    for spot_id in single_slot_spots:
        original_slot = slots[slots["spot_id"] == spot_id].iloc[0]
        s, e = original_slot["start"], original_slot["end"]
        if e <= 13:
            new_rows.append({"spot_id": spot_id, "start": 14.0, "end": 18.0})
        elif s >= 13:
            new_rows.append({"spot_id": spot_id, "start": 8.0, "end": 12.0})
        else:
            new_rows.append({"spot_id": spot_id, "start": 18.0, "end": 22.0})

    extended_slots_mid = pd.concat([slots, pd.DataFrame(new_rows)], ignore_index=True)

    print(f"    新增时段数: {len(new_rows)}")
    time_eligible_mid = build_time_eligibility(users, extended_slots_mid, eligible)
    print(f"    时间兼容边数: {time_eligible_mid.sum()}")
    t0 = time.time()
    r2_mid = solve_q2_with_data(spots, extended_slots_mid, users, dist_matrix, eligible, time_eligible_mid)
    t_mid = time.time() - t0
    print(f"    Q2: 分配={r2_mid['n_assigned']}, 平均距离={r2_mid['avg_distance']:.2f}m, 耗时={t_mid:.2f}s")

    # 汇总
    print("\n" + "=" * 70)
    print("时段扩展场景对比")
    print("=" * 70)
    print(f"{'场景':<25} {'分配数':<10} {'平均距离':<12} {'利用率':<10}")
    print("-" * 70)
    print(f"{'原始时段':<25} {r2_orig['n_assigned']:<10} {r2_orig['avg_distance']:<12.2f} {r2_orig['utilization']*100:.2f}%")
    print(f"{'单时段补充反向':<25} {r2_mid['n_assigned']:<10} {r2_mid['avg_distance']:<12.2f} {r2_mid['utilization']*100:.2f}%")
    print(f"{'全天 8-22 开放':<25} {r2_full['n_assigned']:<10} {r2_full['avg_distance']:<12.2f} {r2_full['utilization']*100:.2f}%")

    if r2_orig["n_assigned"] > 0:
        improvement_mid = (r2_mid["n_assigned"] - r2_orig["n_assigned"]) / r2_orig["n_assigned"] * 100
        improvement_full = (r2_full["n_assigned"] - r2_orig["n_assigned"]) / r2_orig["n_assigned"] * 100
        print(f"\n中等扩展提升: {improvement_mid:+.2f}%")
        print(f"全天开放提升: {improvement_full:+.2f}%")

    return {
        "original": r2_orig,
        "mid_extension": r2_mid,
        "full_extension": r2_full,
    }


def plot_results(walk_df, dur_df, slot_results):
    """生成敏感度分析图表"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # 1. 步行距离敏感度 - 分配数
    ax = axes[0, 0]
    ax.plot(walk_df["multiplier"], walk_df["q1_assigned"], 'o-', label='Q1(静态)', linewidth=2, markersize=8, color=pick_color(0.25))
    ax.plot(walk_df["multiplier"], walk_df["q2_assigned"], 's-', label='Q2(时间约束)', linewidth=2, markersize=8, color=pick_color(0.75))
    ax.axvline(x=1.0, color=pick_color(0.05), linestyle='--', alpha=0.5)
    ax.set_xlabel('步行距离阈值倍数', fontsize=12)
    ax.set_ylabel('分配用户数', fontsize=12)
    ax.set_title('步行距离阈值对分配数的影响', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. 步行距离对平均距离的影响
    ax = axes[0, 1]
    ax.plot(walk_df["multiplier"], walk_df["q1_avg_dist"], 'o-', label='Q1', linewidth=2, markersize=8, color=pick_color(0.25))
    ax.plot(walk_df["multiplier"], walk_df["q2_avg_dist"], 's-', label='Q2', linewidth=2, markersize=8, color=pick_color(0.75))
    ax.axvline(x=1.0, color=pick_color(0.05), linestyle='--', alpha=0.5)
    ax.set_xlabel('步行距离阈值倍数', fontsize=12)
    ax.set_ylabel('平均步行距离 (米)', fontsize=12)
    ax.set_title('步行距离阈值对平均距离的影响', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3. 用户停车时长敏感度
    ax = axes[1, 0]
    ax.plot(dur_df["duration_factor"], dur_df["q2_assigned"], 'o-', color=pick_color(0.55), linewidth=2, markersize=8, label='Q2分配数')
    ax.axvline(x=1.0, color=pick_color(0.05), linestyle='--', alpha=0.5)
    ax.set_xlabel('用户停车时长倍数', fontsize=12)
    ax.set_ylabel('Q2 分配用户数', fontsize=12, color=pick_color(0.55))
    ax.tick_params(axis='y', labelcolor=pick_color(0.55))
    ax.set_title('用户停车时长对分配数的影响', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(dur_df["duration_factor"], dur_df["compatible_edges"], 's--', color=pick_color(0.85), alpha=0.7, label='兼容边数')
    ax2.set_ylabel('时间兼容边数', fontsize=12, color=pick_color(0.85))
    ax2.tick_params(axis='y', labelcolor=pick_color(0.85))

    # 4. 时段扩展场景对比
    ax = axes[1, 1]
    scenarios = ['原始时段', '单时段\n补充反向', '全天\n8-22开放']
    counts = [
        slot_results["original"]["n_assigned"],
        slot_results["mid_extension"]["n_assigned"],
        slot_results["full_extension"]["n_assigned"],
    ]
    colors = [pick_color(0.25), pick_color(0.55), pick_color(0.85)]
    bars = ax.bar(scenarios, counts, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('分配用户数', fontsize=12)
    ax.set_title('车位时段扩展场景对比', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f'{count}', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig("sensitivity_analysis.png", dpi=200, bbox_inches='tight')
    print("\n敏感度分析图已保存: sensitivity_analysis.png")


def main():
    print("\n" + "=" * 70)
    print("B 题 - 敏感度分析")
    print("=" * 70)

    t_start = time.time()

    walk_df = sensitivity_walk_distance()
    dur_df = sensitivity_user_duration()
    slot_results = sensitivity_slot_extension()

    plot_results(walk_df, dur_df, slot_results)

    walk_df.to_csv("sensitivity_walk_distance.csv", index=False, encoding="utf-8-sig")
    dur_df.to_csv("sensitivity_user_duration.csv", index=False, encoding="utf-8-sig")

    # ── 假设鲁棒性检验（论文新增：时间噪声 + 取消率扰动）──────────────────────
    print("\n" + "=" * 70)
    print("假设鲁棒性检验（调用 robustness/robustness_test.py）")
    print("=" * 70)
    import os, subprocess, sys
    robustness_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "robustness", "robustness_test.py"
    )
    robustness_script = os.path.normpath(robustness_script)
    if os.path.exists(robustness_script):
        result = subprocess.run(
            [sys.executable, robustness_script],
            capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        print(result.stdout)
        if result.returncode != 0:
            print("[WARN] robustness_test.py 返回非零退出码:")
            print(result.stderr[:500])
    else:
        print(f"[SKIP] 未找到鲁棒性检验脚本：{robustness_script}")
        print("       请先运行 robustness/robustness_test.py")

    print("\n" + "=" * 70)
    print("敏感度分析完成")
    print("=" * 70)
    print(f"\n总耗时: {time.time() - t_start:.2f} 秒")
    print("\n输出文件:")
    print("  - sensitivity_analysis.png              - 综合分析图")
    print("  - sensitivity_walk_distance.csv         - 步行距离敏感度数据")
    print("  - sensitivity_user_duration.csv         - 用户时长敏感度数据")
    print("  - ../robustness/robustness_A_time_noise.csv    - 时间噪声鲁棒性结果")
    print("  - ../robustness/robustness_B_cancellation.csv  - 取消率鲁棒性结果")


if __name__ == "__main__":
    main()
