"""数据加载与公共函数"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import os
from glob import glob

# Try to locate the data directory dynamically. The original relative path
# may break depending on the current working directory / encoding. We search
# the workspace for a directory that contains the expected CSV files.
_CANDIDATE_NAMES = [
    "重庆大学2026数学建模春季赛题目\\B题附件",
    "重庆大学2026数学建模春季赛题目/B题附件",
]

def _find_data_dir():
    # First try candidate names relative to repository root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    for name in _CANDIDATE_NAMES:
        p = os.path.join(repo_root, name)
        if os.path.isdir(p):
            return p

    # Fallback: search for spots.csv somewhere under repo_root
    for root, dirs, files in os.walk(repo_root):
        if 'spots.csv' in files and 'slots.csv' in files and 'users.csv' in files:
            return root

    # Last resort: search entire C: drive (fast stop if found)
    # Limit search depth to avoid long scans
    drives = ['C:\\'] if os.name == 'nt' else ['/']
    for base in drives:
        for path in glob(os.path.join(base, '**', 'spots.csv'), recursive=True):
            candidate = os.path.dirname(path)
            if os.path.exists(os.path.join(candidate, 'slots.csv')) and os.path.exists(os.path.join(candidate, 'users.csv')):
                return candidate

    return None

DATA_DIR = _find_data_dir()


def load_data():
    """加载三个 CSV 文件，返回 (spots, slots, users)"""
    if DATA_DIR is None:
        raise FileNotFoundError("数据目录未找到：请确保附件 CSV（spots.csv, slots.csv, users.csv）位于项目中")

    # Try common encodings for CSVs (GBK for Windows/Chinese exports, fallback to utf-8)
    encodings = ["gbk", "utf-8", "utf-8-sig"]
    def _read_csv_with_fallback(path):
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                continue
        # let pandas raise the original error
        return pd.read_csv(path)

    spots = _read_csv_with_fallback(os.path.join(DATA_DIR, 'spots.csv'))
    slots = _read_csv_with_fallback(os.path.join(DATA_DIR, 'slots.csv'))
    users = _read_csv_with_fallback(os.path.join(DATA_DIR, 'users.csv'))

    # 统一列名
    spots.columns = ["spot_id", "x", "y"]
    slots.columns = ["spot_id", "start", "end"]
    users.columns = ["user_id", "dest_x", "dest_y", "max_walk", "arrival", "departure"]

    return spots, slots, users


def compute_distance_matrix(spots, users):
    """计算所有用户到所有车位的欧氏距离矩阵 (500 x 300)"""
    spot_coords = spots[["x", "y"]].values  # (n_spots, 2)
    user_coords = users[["dest_x", "dest_y"]].values  # (n_users, 2)
    dist_matrix = cdist(user_coords, spot_coords, metric="euclidean")
    return dist_matrix  # shape (500, 300)


def build_eligibility(dist_matrix, users):
    """
    构建用户-车位可用性矩阵
    返回: eligible[i,j] = True if 用户j可停在车位i
    """
    max_walk = users["max_walk"].values  # (500,)
    eligible = dist_matrix <= max_walk[:, np.newaxis]  # (500, 300)
    return eligible


def build_time_eligibility(users, slots, eligible):
    """
    检查时间兼容性：用户 [arrival, departure] 需完全包含在车位的某个可用时段中。
    返回: time_eligible[i,j] = eligible[i,j] AND 时间兼容
    """
    n_users = len(users)
    n_spots = spots_max_id = slots["spot_id"].max()
    n_spots_found = eligible.shape[1]

    time_eligible = np.zeros_like(eligible, dtype=bool)

    # 按车位分组可用时段
    slot_groups = slots.groupby("spot_id")
    # slot_groups 的 key 是 spot_id（1-indexed）

    for j in range(n_users):
        a, d = users.iloc[j]["arrival"], users.iloc[j]["departure"]
        # 只需检查 eligible 的 spot
        candidate_spots = np.where(eligible[j])[0]  # 0-indexed
        for idx_i in candidate_spots:
            spot_id = idx_i + 1  # spot_id 是 1-indexed（来自CSV）
            if spot_id not in slot_groups.groups:
                continue
            group = slot_groups.get_group(spot_id)
            for _, row in group.iterrows():
                if row["start"] <= a and d <= row["end"]:
                    time_eligible[j, idx_i] = True
                    break

    return time_eligible


def check_time_conflict(users, user_pairs=None):
    """
    检查用户之间是否有时间冲突（区间重叠）
    返回: conflict[j, k] = True if 用户j和用户k时间区间重叠
    
    使用半开区间 [arrival, departure) 约定：
    重叠判定：[a1, d1) ∩ [a2, d2) ≠ ∅ ⇔ a1 < d2 AND a2 < d1
    
    使用用户0-indexed ID
    """
    n = len(users)
    a = users["arrival"].to_numpy()  # (n,)
    d = users["departure"].to_numpy()  # (n,)
    # 正确向量化：[a1, d1) 和 [a2, d2) 重叠 iff a1 < d2 AND a2 < d1
    conflict = (a[:, None] < d[None, :]) & (a[None, :] < d[:, None])  # (n, n)
    np.fill_diagonal(conflict, False)
    return conflict


def find_spot_available_slots(users, slots):
    """
    对每个车位，构建其可用时段列表
    返回: {spot_id: [(start, end), ...]}
    """
    available = {}
    for spot_id, group in slots.groupby("spot_id"):
        available[spot_id] = list(zip(group["start"], group["end"]))
    return available


def find_candidate_users_for_spot(users, slots, eligible):
    """
    对每个车位 i, 找到兼容的用户列表（距离 + 时间兼容）
    返回: candidates[i] = [j1, j2, ...]
    """
    n_users = len(users)
    n_spots = eligible.shape[1]
    slot_groups = slots.groupby("spot_id")

    candidates = [[] for _ in range(n_spots)]

    for i in range(n_spots):
        spot_id = i + 1
        if spot_id not in slot_groups.groups:
            continue
        time_windows = list(zip(
            slot_groups.get_group(spot_id)["start"],
            slot_groups.get_group(spot_id)["end"]
        ))

        for j in range(n_users):
            if not eligible[j, i]:
                continue
            a, d = users.iloc[j]["arrival"], users.iloc[j]["departure"]
            for s, e in time_windows:
                if s <= a and d <= e:
                    candidates[i].append(j)
                    break

    return candidates
