"""问题4：供需匹配分析与建议

分析内容：
1. 时段供需分析 - 哪些时段车位最紧张
2. 步行距离分布
3. 空间热力图
4. 运营建议
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt

from plot_theme import apply_common_style, pick_color

apply_common_style()
import sys

# 控制台 UTF-8 编码，确保中文正常输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from utils import load_data, compute_distance_matrix, build_eligibility, build_time_eligibility

from q1_static import solve_q1
from q2_time import solve_q2
from q3_online import solve_q3


def analyze_supply_demand(spots, slots, users, verbose=True):
    """分析各时段供需情况"""
    # 时间范围：8-22 小时
    time_range = np.arange(8, 22.5, 0.5)  # 每半小时一个点

    n_slots = len(time_range)
    supply = np.zeros(n_slots)  # 每个时间点可用车位数
    demand = np.zeros(n_slots)  # 每个时间点活跃用户数

    # 计算每个时间点的供给（半开区间 [start, end)）
    for _, row in slots.iterrows():
        s, e = row["start"], row["end"]
        for t_idx, t in enumerate(time_range):
            if s <= t < e:
                supply[t_idx] += 1

    # 计算每个时间点的需求（半开区间 [arrival, departure)）
    for _, row in users.iterrows():
        a, d = row["arrival"], row["departure"]
        for t_idx, t in enumerate(time_range):
            if a <= t < d:
                demand[t_idx] += 1

    supply_demand = pd.DataFrame({
        "time": time_range,
        "supply": supply,
        "demand": demand,
        "gap": supply - demand,
        "pressure": np.where(supply > 0, demand / supply, np.inf)
    })

    if verbose:
        print("\n========== 时段供需分析 ==========")
        print(f"{'时段':<10} {'供给(车位)':<14} {'需求(用户)':<14} {'缺口':<10} {'压力比':<10}")
        print("-" * 58)
        for _, row in supply_demand.iterrows():
            hour = row["time"]
            pressure_str = f"{row['pressure']:.2f}" if np.isfinite(row["pressure"]) else "∞"
            if row["gap"] < 0:
                print(f"{hour:<8.1f} {row['supply']:<14.0f} {row['demand']:<14.0f} {row['gap']:<10.0f} {pressure_str:<10}")
        print("-" * 58)

        # 找出最紧张的时段（过滤掉供给为0的时段，避免inf干扰）
        supply_demand_valid = supply_demand[(supply_demand["demand"] > 0) & (supply_demand["supply"] > 0)].copy()
        tightest = supply_demand_valid.loc[supply_demand_valid["pressure"].idxmax()]
        print(f"\n最紧张时段: {tightest['time']:.1f}h")
        print(f"  需求: {tightest['demand']:.0f} 用户, 供给: {tightest['supply']:.0f} 车位")
        print(f"  压力比: {tightest['pressure']:.2f}")

        loosest = supply_demand_valid.loc[supply_demand_valid["pressure"].idxmin()]
        print(f"\n最宽松时段: {loosest['time']:.1f}h")
        print(f"  需求: {loosest['demand']:.0f} 用户, 供给: {loosest['supply']:.0f} 车位")
        print(f"  压力比: {loosest['pressure']:.2f}")

        # 总供需统计
        avg_pressure = supply_demand_valid["pressure"].mean()
        peak_pressure = supply_demand_valid["pressure"].max()
        print(f"\n平均压力比: {avg_pressure:.2f}")
        print(f"峰值压力比: {peak_pressure:.2f}")

    return supply_demand


def analyze_effective_pressure(spots, slots, users, dist_matrix, w_median=14.9, verbose=True):
    """
    计算三层供需压力指标（对应论文 Q4 宏观有效压力比定义）：
      P(t)     = D(t) / S(t)          — 宏观压力比（下界/乐观估计）
      P_eff(t) = D(t) / S_eff(t)      — 宏观有效压力比（以步行中位数 w_median 为半径）
      rho_bar(t)                       — 需求加权网格压力比（见 q4_heatmap.py）

    参数
    ----
    w_median : float
        步行阈值中位数（米），默认 14.9 m（与论文一致）
    """
    time_range = np.arange(8, 22.5, 0.5)   # 29 个时间点，28 个区间
    n_t = len(time_range)

    supply_all  = np.zeros(n_t)   # S(t)：所有可用车位
    supply_eff  = np.zeros(n_t)   # S_eff(t)：对"典型用户"可达的车位
    demand_arr  = np.zeros(n_t)   # D(t)

    # 预计算：哪些车位对"典型用户"可达（至少有一位用户距离 <= w_median）
    reachable_by_typical = np.any(dist_matrix <= w_median, axis=0)  # shape (n_spots,)

    for t_idx, t in enumerate(time_range):
        for _, row in slots.iterrows():
            s, e = row["start"], row["end"]
            if s <= t < e:
                spot_idx = int(row["spot_id"]) - 1   # 0-indexed
                supply_all[t_idx] += 1
                if 0 <= spot_idx < len(reachable_by_typical) and reachable_by_typical[spot_idx]:
                    supply_eff[t_idx] += 1

        for _, row in users.iterrows():
            a, d = row["arrival"], row["departure"]
            if a <= t < d:
                demand_arr[t_idx] += 1

    # 避免除零
    p_all = np.where(supply_all > 0, demand_arr / supply_all, np.inf)
    p_eff = np.where(supply_eff > 0, demand_arr / supply_eff, np.inf)

    result = {
        "time":       time_range,
        "D":          demand_arr,
        "S_all":      supply_all,
        "S_eff":      supply_eff,
        "P_all":      p_all,
        "P_eff":      p_eff,
    }

    if verbose:
        print("\n========== 三层供需压力指标 ==========")
        print(f"{'时刻':>6}  {'D(t)':>6}  {'S(t)':>6}  {'S_eff':>6}  {'P(t)':>7}  {'P_eff':>7}")
        print("-" * 52)
        for i, t in enumerate(time_range):
            if demand_arr[i] > 0 or supply_all[i] > 0:
                p_str     = f"{p_all[i]:.2f}"  if np.isfinite(p_all[i])  else "∞"
                p_eff_str = f"{p_eff[i]:.2f}"  if np.isfinite(p_eff[i]) else "∞"
                print(f"{t:>6.1f}  {demand_arr[i]:>6.0f}  {supply_all[i]:>6.0f}  "
                      f"{supply_eff[i]:>6.0f}  {p_str:>7}  {p_eff_str:>7}")
        print("-" * 52)
        valid = np.isfinite(p_all) & (demand_arr > 0)
        if valid.any():
            print(f"P(t)    均值={p_all[valid].mean():.3f}  最大={p_all[valid].max():.3f}  "
                  f"（下界/乐观估计）")
        valid_eff = np.isfinite(p_eff) & (demand_arr > 0)
        if valid_eff.any():
            print(f"P_eff(t)均值={p_eff[valid_eff].mean():.3f}  最大={p_eff[valid_eff].max():.3f}  "
                  f"（有效压力，更接近真实）")

    return result


def analyze_walking_distance(dist_matrix, eligible, users, verbose=True):
    """分析步行距离分布"""
    # 所有可行配对的步行距离
    feasible_distances = []
    for j in range(len(users)):
        for i in range(dist_matrix.shape[1]):
            if eligible[j, i]:
                feasible_distances.append(dist_matrix[j, i])

    feasible_distances = np.array(feasible_distances)
    max_walk_values = users["max_walk"].values

    if verbose:
        print("\n========== 步行距离分析 ==========")
        print(f"可行配对总数: {len(feasible_distances)}")
        print(f"可行步行距离:")
        print(f"  最小: {feasible_distances.min():.2f} 米")
        print(f"  最大: {feasible_distances.max():.2f} 米")
        print(f"  平均: {feasible_distances.mean():.2f} 米")
        print(f"  中位数: {np.median(feasible_distances):.2f} 米")
        print(f"  标准差: {feasible_distances.std():.2f} 米")

        # 用户最大步行距离分布
        print(f"\n用户最大步行距离:")
        print(f"  最小: {max_walk_values.min():.2f} 米")
        print(f"  最大: {max_walk_values.max():.2f} 米")
        print(f"  平均: {max_walk_values.mean():.2f} 米")
        print(f"  中位数: {np.median(max_walk_values):.2f} 米")

        # 距离区间分布
        bins = [0, 5, 10, 15, 20, 25, 30, 50]
        labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
        dist_bins = np.digitize(feasible_distances, bins) - 1
        for b_idx in range(len(bins)-1):
            count = np.sum(dist_bins == b_idx)
            pct = 100 * count / len(feasible_distances)
            print(f"  {labels[b_idx]}米: {count} ({pct:.1f}%)")

    return feasible_distances


def analyze_user_max_walk_distribution(users, verbose=True):
    """分析用户最大步行距离设置分布"""
    max_walk = users["max_walk"].values

    # 每5米一个区间
    bins = np.arange(0, max_walk.max() + 5, 5)
    hist, _ = np.histogram(max_walk, bins=bins)

    if verbose:
        print("\n========== 用户最大步行容忍度分布 ==========")
        for i in range(len(hist)):
            if hist[i] > 0:
                print(f"  [{bins[i]:.0f}-{bins[i+1]:.0f})米: {hist[i]} 用户 ({100*hist[i]/len(max_walk):.1f}%)")

    return bins, hist


def generate_plots(spots, slots, users, q1_result=None, q2_result=None, q3_result=None):
    """生成可视化图表"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 车位和用户空间分布
    ax1 = axes[0, 0]
    ax1.scatter(spots["x"], spots["y"], c=pick_color(0.25), s=20, alpha=0.5, label="车位")
    ax1.scatter(users["dest_x"], users["dest_y"], c=pick_color(0.75), s=10, alpha=0.25, label="用户目的地")
    ax1.set_xlabel("X 坐标 (米)")
    ax1.set_ylabel("Y 坐标 (米)")
    ax1.set_title("车位与用户目的地空间分布")
    ax1.legend()
    ax1.set_aspect("equal")

    # 2. 时段供需
    ax2 = axes[0, 1]
    sd = analyze_supply_demand(spots, slots, users, verbose=False)
    ax2.plot(sd["time"], sd["supply"], "-", color=pick_color(0.25), marker="o", linewidth=2, label="供给（可用车位）")
    ax2.plot(sd["time"], sd["demand"], "-", color=pick_color(0.75), marker="o", linewidth=2, label="需求（活跃用户）")
    ax2.fill_between(sd["time"], sd["supply"], sd["demand"],
                      where=sd["demand"] > sd["supply"],
                      color=pick_color(0.75), alpha=0.18, label="供给缺口")
    ax2.set_xlabel("时间 (小时)")
    ax2.set_ylabel("数量")
    ax2.set_title("时段供需分析")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 步行距离分布
    ax3 = axes[0, 2]
    dist_matrix = compute_distance_matrix(spots, users)
    eligible = build_eligibility(dist_matrix, users)
    feasible_distances = analyze_walking_distance(dist_matrix, eligible, users, verbose=False)
    ax3.hist(feasible_distances, bins=30, alpha=0.75, color=pick_color(0.45), edgecolor="black")
    ax3.axvline(feasible_distances.mean(), color=pick_color(0.75), linestyle="--",
                label=f"均值={feasible_distances.mean():.1f}")
    ax3.set_xlabel("步行距离 (米)")
    ax3.set_ylabel("频数")
    ax3.set_title("可行配对步行距离分布")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 用户最大步行距离
    ax4 = axes[1, 0]
    max_walk = users["max_walk"].values
    ax4.hist(max_walk, bins=20, alpha=0.75, color=pick_color(0.15), edgecolor="black")
    ax4.axvline(max_walk.mean(), color=pick_color(0.75), linestyle="--",
                label=f"均值={max_walk.mean():.1f}")
    ax4.set_xlabel("最大步行距离 (米)")
    ax4.set_ylabel("用户数")
    ax4.set_title("用户最大步行容忍度分布")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. 各问题结果对比
    ax5 = axes[1, 1]
    labels = []
    n_vals = []
    dist_vals = []
    if q1_result and ("n_assigned" in q1_result) and ("avg_distance" in q1_result):
        labels.append("问题1\n无时间约束")
        n_vals.append(q1_result["n_assigned"])
        dist_vals.append(q1_result["avg_distance"])
    if q2_result and ("n_assigned" in q2_result) and ("avg_distance" in q2_result):
        labels.append("问题2\n时间约束")
        n_vals.append(q2_result["n_assigned"])
        dist_vals.append(q2_result["avg_distance"])
    if q3_result and ("n_assigned" in q3_result) and ("avg_distance" in q3_result):
        strategy = q3_result.get("strategy", "")
        if strategy:
            labels.append(f"问题3\n在线{strategy}")
        else:
            labels.append("问题3\n在线调度")
        n_vals.append(q3_result["n_assigned"])
        dist_vals.append(q3_result["avg_distance"])

    if len(labels) == 0:
        print("[WARN] 各问题结果对比为空：检查 q1_result/q2_result/q3_result 是否包含 n_assigned 和 avg_distance")
        print("  q1_result keys:", sorted(list(q1_result.keys())) if q1_result else None)
        print("  q2_result keys:", sorted(list(q2_result.keys())) if q2_result else None)
        print("  q3_result keys:", sorted(list(q3_result.keys())) if q3_result else None)

    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax5.bar(x - width/2, n_vals, width, label="服务用户数", color=pick_color(0.25))
    ax5_twin = ax5.twinx()
    bars2 = ax5_twin.bar(x + width/2, dist_vals, width, label="平均步行距离", color=pick_color(0.65))
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels)
    ax5.set_ylabel("服务用户数")
    ax5_twin.set_ylabel("平均步行距离 (米)")
    ax5.set_title("各问题结果对比")
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax5.grid(True, alpha=0.3)

    # 6. 压力比时间序列
    ax6 = axes[1, 2]
    sd_valid = sd[sd["demand"] > 0].copy()
    ax6.plot(sd_valid["time"], sd_valid["pressure"], "-", color=pick_color(0.85), linewidth=2, marker="o")
    ax6.axhline(1.0, color=pick_color(0.05), linestyle="--", alpha=0.6, label="供需平衡线")
    ax6.fill_between(sd_valid["time"], sd_valid["pressure"], 1,
                      where=sd_valid["pressure"] > 1,
                      color=pick_color(0.75), alpha=0.18, label="供不应求")
    ax6.fill_between(sd_valid["time"], sd_valid["pressure"], 1,
                      where=sd_valid["pressure"] < 1,
                      color=pick_color(0.25), alpha=0.18, label="供过于求")
    ax6.set_xlabel("时间 (小时)")
    ax6.set_ylabel("压力比 (需求/供给)")
    ax6.set_title("时段压力比")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, max(sd_valid["pressure"].max() * 1.2, 2))

    plt.tight_layout()
    plt.savefig("q4_analysis.png", dpi=150)
    print("\n图表已保存: q4_analysis.png")
    plt.close()


def generate_recommendations(spots, slots, users, supply_demand, verbose=True):
    """生成运营建议"""
    sd = supply_demand

    # 平均压力比
    avg_pressure = sd[sd["demand"] > 0]["pressure"].mean()
    peak_time = sd.loc[sd["pressure"].idxmax()]["time"]

    # 统计槽位特征
    slots_per_spot = slots.groupby("spot_id").size()
    multi_slot_count = (slots_per_spot > 1).sum()

    if verbose:
        print("\n" + "=" * 60)
        print("问题4：供需匹配分析与改进建议")
        print("=" * 60)

        print(f"\n【供需匹配分析】")
        print(f"总车位: {len(spots)}")
        print(f"总用户请求: {len(users)}")
        print(f"平均时段压力比: {avg_pressure:.2f}")
        print(f"最紧张时段: {peak_time:.1f}h")
        print(f"拥有多时段的车位数: {multi_slot_count}（可错峰共享）")

        print(f"\n【改进建议】")

        print(f"\n建议1：动态定价策略")
        print(f"  问题：时段供需严重不均，{peak_time:.1f}h 压力比达峰值。")
        print(f"  方案：")
        print(f"    - 高峰时段({peak_time-1:.0f}-{peak_time+1:.0f}h)提高停车费率，")
        print(f"      引导非必需用户错峰出行")
        print(f"    - 低峰时段提供折扣费率，提高车位利用率")
        print(f"    - 对预约用户提供价格锁定，鼓励提前规划")

        print(f"\n建议2：鼓励错时共享")
        print(f"  问题：拥有多时段的车位仅 {multi_slot_count} 个，大量车位仅单一时段可用。")
        print(f"  方案：")
        print(f"    - 鼓励车位主开放更多空闲时段（如夜间、周末）")
        print(f"    - 对开放多时段的车位主提供补贴或税收优惠")
        print(f"    - 推广\"上班族-夜班族\"车位时段互换匹配")

        print(f"\n建议3：优化用户步行体验")
        print(f"  方案：")
        print(f"    - 在用户高密度区域（基于目的地聚类）增加共享车位供给")
        print(f"    - 提供精准步行导航，规划最优步行路径")
        print(f"    - 对步行距离较远的用户提供积分补偿")

        print(f"\n建议4：智能预约与推荐系统")
        print(f"  方案：")
        print(f"    - 基于历史数据预测未来时段的供需情况")
        print(f"    - 用户提交请求时，系统推荐最可能成功的车位")
        print(f"    - 若无法满足精确时间，推荐相近可用时段的车位")

    return {
        "avg_pressure": avg_pressure,
        "peak_time": peak_time,
        "multi_slot_count": multi_slot_count
    }


def solve_q4(q1_result=None, q2_result=None, q3_result=None, verbose=True):
    """运行问题4的全部分析"""
    if q1_result is None:
        q1_result = solve_q1(verbose=False)

    if q2_result is None:
        q2_result = solve_q2(verbose=False)

    if q3_result is None:
        q3_nearest = solve_q3(strategy="nearest", verbose=False)
        q3_balance = solve_q3(strategy="balance", verbose=False)
        q3_result = q3_nearest if q3_nearest.get("n_assigned", 0) >= q3_balance.get("n_assigned", 0) else q3_balance

    spots, slots, users = load_data()

    supply_demand = analyze_supply_demand(spots, slots, users, verbose=verbose)

    dist_matrix = compute_distance_matrix(spots, users)
    eligible = build_eligibility(dist_matrix, users)

    # 三层供需压力指标（论文 Q4 新增：P(t) 下界 / P_eff(t) 有效压力）
    w_median = float(np.median(users["max_walk"].values))
    effective_pressure = analyze_effective_pressure(
        spots, slots, users, dist_matrix, w_median=w_median, verbose=verbose
    )

    analyze_walking_distance(dist_matrix, eligible, users, verbose=verbose)
    analyze_user_max_walk_distribution(users, verbose=verbose)

    generate_plots(spots, slots, users, q1_result, q2_result, q3_result)

    recs = generate_recommendations(spots, slots, users, supply_demand, verbose=verbose)

    return {
        "supply_demand":      supply_demand,
        "effective_pressure": effective_pressure,
        "recommendations":    recs,
    }


if __name__ == "__main__":
    solve_q4(verbose=True)
