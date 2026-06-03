"""问题3：在线调度算法

用户按到达时间顺序出现，平台需立即做出分配决策（不可更改）。
目标是最大化服务用户数，同时最小化步行距离。

算法策略：
1. Greedy-Nearest: 选择离目的地最近的可用车位
2. Greedy-Balance: 选择兼容且当前利用率最低的车位
3. 竞争比分析：与问题2离线最优解比较
"""

import time
import numpy as np
from utils import load_data, compute_distance_matrix, build_eligibility


class OnlineAllocator:
    """在线分配器，管理车位状态和分配决策"""

    def __init__(self, spots, slots, users, dist_matrix, eligible):
        self.spots = spots
        self.slots = slots
        self.users = users
        self.dist_matrix = dist_matrix
        self.eligible = eligible

        self.n_spots = len(spots)
        self.n_users = len(users)

        # spot -> list of (slot_start, slot_end) 每个车位可用时段
        self.spot_slots = {}
        for spot_id, group in slots.groupby("spot_id"):
            self.spot_slots[spot_id] = list(zip(group["start"], group["end"]))

        # spot -> list of (user_start, user_end, user_id) 已分配的占用区间
        self.occupancy = {spot_id: [] for spot_id in spots["spot_id"]}

        # 统计
        self.assigned = []
        self.rejected = []
        self.total_distance = 0.0

    def is_time_compatible(self, spot_id, user_arrival, user_departure):
        """检查用户时间窗口是否在车位的某个可用时段内"""
        if spot_id not in self.spot_slots:
            return False
        for s, e in self.spot_slots[spot_id]:
            if s <= user_arrival and user_departure <= e:
                return True
        return False

    def is_spot_free(self, spot_id, user_arrival, user_departure):
        """检查车位在用户时间段内是否空闲"""
        if spot_id not in self.occupancy:
            return True
        for (occ_start, occ_end, _) in self.occupancy[spot_id]:
            if user_arrival < occ_end and occ_start < user_departure:
                return False
        return True

    def assign_user(self, user_idx, spot_idx):
        """分配用户到车位"""
        spot_id = self.spots.iloc[spot_idx]["spot_id"]
        user_id = self.users.iloc[user_idx]["user_id"]
        arrival = self.users.iloc[user_idx]["arrival"]
        departure = self.users.iloc[user_idx]["departure"]
        distance = self.dist_matrix[user_idx, spot_idx]

        self.occupancy[spot_id].append((arrival, departure, user_id))
        self.assigned.append({
            "user_id": user_id,
            "spot_id": spot_id,
            "distance": distance,
            "arrival": arrival,
            "departure": departure
        })
        self.total_distance += distance
        return True

    def find_candidates(self, user_idx):
        """找到用户的所有候选车位（距离+时间兼容且空闲）"""
        arrival = self.users.iloc[user_idx]["arrival"]
        departure = self.users.iloc[user_idx]["departure"]
        candidates = []

        for i in range(self.n_spots):
            if not self.eligible[user_idx, i]:
                continue
            spot_id = self.spots.iloc[i]["spot_id"]
            if not self.is_time_compatible(spot_id, arrival, departure):
                continue
            if not self.is_spot_free(spot_id, arrival, departure):
                continue
            candidates.append(i)

        return candidates

    def greedy_nearest(self, user_idx):
        """策略1：选择最近的车位"""
        candidates = self.find_candidates(user_idx)
        if not candidates:
            return False

        # 选最近
        best_i = min(candidates, key=lambda i: self.dist_matrix[user_idx, i])
        return self.assign_user(user_idx, best_i)

    def greedy_balance(self, user_idx):
        """策略2：选择利用率最低的车位"""
        candidates = self.find_candidates(user_idx)
        if not candidates:
            return False

        # 选当前占用最少（利用率低）的车位
        def spot_load(i):
            spot_id = self.spots.iloc[i]["spot_id"]
            return len(self.occupancy[spot_id])

        best_i = min(candidates, key=lambda i: (
            spot_load(i),
            self.dist_matrix[user_idx, i]
        ))
        return self.assign_user(user_idx, best_i)


def solve_q3(strategy="nearest", verbose=True):
    """运行在线调度模拟"""
    spots, slots, users = load_data()
    dist_matrix = compute_distance_matrix(spots, users)
    eligible = build_eligibility(dist_matrix, users)

    # 按到达时间排序
    sorted_users = users.sort_values("arrival")
    # 相同时刻按用户ID排序
    sorted_users = sorted_users.sort_values(["arrival", "user_id"])

    allocator = OnlineAllocator(spots, slots, users, dist_matrix, eligible)

    t_start = time.time()

    if verbose:
        print("=" * 60)
        print(f"问题3：在线调度算法（策略: {strategy}）")
        print("=" * 60)

    for idx, (_, user_row) in enumerate(sorted_users.iterrows()):
        user_idx = user_row.name  # 原始索引
        if strategy == "nearest":
            accepted = allocator.greedy_nearest(user_idx)
        elif strategy == "balance":
            accepted = allocator.greedy_balance(user_idx)
        else:
            accepted = allocator.greedy_nearest(user_idx)

        if not accepted:
            allocator.rejected.append(user_idx)

        if verbose and (idx + 1) % 100 == 0:
            print(f"  已处理 {idx + 1}/{len(sorted_users)} 用户...")

    elapsed = time.time() - t_start

    n_assigned = len(allocator.assigned)
    avg_distance = allocator.total_distance / n_assigned if n_assigned > 0 else 0
    utilization = n_assigned / allocator.n_spots

    result = {
        "strategy": strategy,
        "assigned": [a["user_id"] for a in allocator.assigned],
        "n_assigned": n_assigned,
        "total_distance": allocator.total_distance,
        "avg_distance": avg_distance,
        "utilization": utilization,
        "assignment_detail": allocator.assigned,
        "elapsed_seconds": elapsed,
        "status": "ok"
    }

    if verbose:
        print(f"\n========== 问题3 结果 ==========")
        print(f"策略: {strategy}")
        print(f"被分配用户数: {n_assigned}")
        print(f"总步行距离: {allocator.total_distance:.2f} 米")
        print(f"平均步行距离: {avg_distance:.2f} 米")
        print(f"车位利用率: {utilization:.2%} ({n_assigned}/{allocator.n_spots})")
        print(f"拒绝用户数: {len(allocator.rejected)}")
        print(f"运行时间: {elapsed:.4f} 秒")

    return result


def compare_with_offline(online_result, offline_result, verbose=True):
    """比较在线与离线结果，计算竞争比"""
    if offline_result["status"] != "optimal" or online_result["status"] != "ok":
        if verbose:
            print("无法比较：结果不完整")
        return

    online_n = online_result["n_assigned"]
    offline_n = offline_result["n_assigned"]

    # 竞争比（服务用户数）
    ratio_users = online_n / offline_n if offline_n > 0 else 0

    # 平均距离比
    online_avg = online_result["avg_distance"]
    offline_avg = offline_result["avg_distance"]
    ratio_dist = online_avg / offline_avg if offline_avg > 0 else float("inf")

    if verbose:
        print(f"\n========== 竞争比分析 ==========")
        print(f"{'指标':<25} {'在线':<12} {'离线':<12} {'比值':<12}")
        print(f"{'='*61}")
        print(f"{'服务用户数':<25} {online_n:<12} {offline_n:<12} {ratio_users:<12.4f}")
        print(f"{'平均步行距离(米)':<25} {online_avg:<12.2f} {offline_avg:<12.2f} {ratio_dist:<12.4f}")
        print(f"\n竞争比（服务用户数）: {ratio_users:.4f}")
        print(f"  含义：在线算法达到了离线最优的 {ratio_users:.2%}")

    return {
        "ratio_users": ratio_users,
        "ratio_distance": ratio_dist,
        "online_n": online_n,
        "offline_n": offline_n,
        "online_avg_dist": online_avg,
        "offline_avg_dist": offline_avg
    }


if __name__ == "__main__":
    result = solve_q3(strategy="nearest", verbose=True)
