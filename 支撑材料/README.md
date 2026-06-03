# B题：智慧城市共享停车位资源调配 — 代码说明

## 运行方式

```bash
python main.py
```

`main.py` 位于 `code/` 根目录，是全部四个问题的统一入口。

## 目录结构

```
code/
├── main.py                  # 主入口，依次运行问题1~4
├── README.md                # 本说明文件
├── all_code.md              # 全部 Python 代码合并文档
│
├── common/                  # 公共模块与数据
│   ├── utils.py             # 数据加载、距离矩阵、资格判定
│   ├── plot_theme.py        # Matplotlib 论文配色主题
│   ├── visual_helpers.py    # SVG 热力图辅助渲染
│   ├── gen_svgs.py          # 生成 12h/13h SVG+3D 输出
│   ├── analyze_units.py     # 原始数据探索分析
│   ├── users.csv            # 用户数据
│   ├── spots.csv            # 停车位数据
│   └── slots.csv            # 车位时段数据
│
├── q1_static/               # 问题1：静态分配（无时间约束）
│   └── q1_static.py         # 两阶段优化：最大基数匹配 → 最小步行距离
│
├── q2_time/                 # 问题2：时间约束分配
│   └── q2_time.py           # MILP 模型，含时间冲突互斥约束 + 可扩展性讨论
│
├── q3_online/               # 问题3：在线调度算法
│   └── q3_online.py         # Greedy-Nearest / Greedy-Balance + 竞争比分析
│
├── q4_analysis/             # 问题4：供需匹配分析与建议
│   ├── q4_analysis.py       # 时段供需、三层压力指标(P/P_eff/rho_bar)、运营建议
│   ├── q4_heatmap.py        # 空间-时间压力热力图（网格模型）
│   ├── q4_analysis.png      # Q4 综合分析图表输出
│   ├── heatmap_12h.svg      # 12h 空间压力热力图 SVG
│   ├── heatmap_13h.svg      # 13h 空间压力热力图 SVG
│   ├── heatmap_13h_3d.svg   # 13h 3D 空间压力热力图 SVG
│   └── heatmap_13h_3d.png   # 13h 3D 空间压力热力图 PNG
│
├── sensitivity/             # 敏感度分析
│   ├── sensitivity_analysis.py          # 敏感度分析脚本（含鲁棒性检验调用入口）
│   ├── sensitivity_analysis.png         # 敏感度分析图表
│   ├── sensitivity_walk_distance.csv    # 步行距离阈值敏感度结果
│   ├── sensitivity_walk_distance_common.csv
│   ├── sensitivity_user_duration.csv    # 用户停车时长敏感度结果
│   └── sensitivity_user_duration_common.csv
│
├── online_stress_test/      # Q3 在线算法压力测试（论文新增）
│   ├── online_stress_test.py            # 集中度扫描 + 步行阈值扰动，输出 CSV
│   ├── plot_stress.py                   # 读取 CSV，生成 EPR 曲线图
│   ├── stress_test_A_concentration.csv  # 测试A：到达集中度扫描结果
│   ├── stress_test_B_walk_threshold.csv # 测试B：步行阈值扰动结果
│   ├── stress_test_A_epr_curve.png      # 测试A EPR 曲线图
│   └── stress_test_B_epr_curve.png      # 测试B EPR 曲线图
│
└── robustness/              # 假设鲁棒性检验（论文新增）
    ├── robustness_test.py               # 时间噪声 + 取消率扰动，输出 CSV
    ├── robustness_A_time_noise.csv      # 扰动A：时间噪声结果（20次重复均值±std）
    └── robustness_B_cancellation.csv    # 扰动B：取消率结果（10次重复均值±std）
```

## 各文件详细说明

### 主入口

| 文件 | 位置 | 说明 |
|------|------|------|
| `main.py` | 根目录 | **主入口**。依次调用四个问题的求解函数，打印汇总表。 |

### 公共模块 `common/`

| 文件 | 说明 |
|------|------|
| `utils.py` | **核心工具库**。`load_data()` 动态查找并加载三个 CSV 数据文件；`compute_distance_matrix()` 计算用户-车位欧氏距离矩阵 (500×300)；`build_eligibility()` 基于步行距离阈值构建资格矩阵；`build_time_eligibility()` 综合距离+时间构建兼容矩阵；`check_time_conflict()` 向量化检测用户时间重叠。 |
| `plot_theme.py` | **论文统一视觉风格**。定义深蓝→天蓝→青绿→琥珀四段渐变色，提供 `pick_color(t)` 和 `apply_common_style()` 全局配置。 |
| `visual_helpers.py` | **SVG 热力图渲染**。`save_svg_heatmap()` 将网格数据绘制为 SVG 格式热力图。 |
| `gen_svgs.py` | **快捷输出脚本**。生成 12h 和 13h 的 SVG 切片及 3D PNG 渲染。 |
| `analyze_units.py` | **数据探索工具**。统计坐标分布、距离分布，辅助判断坐标单位。 |

### 问题求解模块

| 文件 | 说明 |
|------|------|
| `q1_static/q1_static.py` | 两阶段 ILP：最大基数匹配 → 最小步行距离。全幺模性保证 LP 松弛自动整数解。 |
| `q2_time/q2_time.py` | MILP：时空可行边集 E^T + Pairwise 互斥约束。含 `build_spot_time_conflicts()` 冲突对构建。 |
| `q3_online/q3_online.py` | 在线贪心分配器。`OnlineAllocator` 类管理车位占用状态；支持 Greedy-Nearest 和 Greedy-Balance 两种策略；`compare_with_offline()` 计算 EPR。 |
| `q4_analysis/q4_analysis.py` | 供需分析主模块。新增 `analyze_effective_pressure()` 计算三层压力指标：宏观 P(t)（下界）、有效 P_eff(t)、需求加权 rho_bar(t)。 |
| `q4_analysis/q4_heatmap.py` | 空间-时间热力图。10×10 网格级压力比 ρ(g,t)，支持 SVG 和 3D PNG 输出。 |

### 分析与验证模块

| 文件 | 说明 |
|------|------|
| `sensitivity/sensitivity_analysis.py` | 三维敏感度分析（步行阈值/停车时长/时段扩展）+ 鲁棒性检验调用入口。 |
| `online_stress_test/online_stress_test.py` | **Q3 压力测试**。测试A：到达集中度扫描（κ∈{7.4,5.55,3.7,3,2,1}h）；测试B：步行阈值扰动（λ∈{0.8,0.9,1.0,1.1,1.2}）。固定种子42。 |
| `online_stress_test/plot_stress.py` | 读取压力测试 CSV，生成 EPR 曲线图（PNG）。 |
| `robustness/robustness_test.py` | **假设鲁棒性检验**。扰动A：时间噪声 ε~Uniform[-10,10]min，20次重复；扰动B：取消率 p∈{5,10,15,20}%，10次重复。固定种子42。 |

## 关键结果汇总

| 问题 | 服务用户数 | 平均步行距离 | 车位利用率 | 求解时间 |
|------|-----------|------------|----------|---------|
| Q1 静态分配 | 300 | 2.82 m | 100% | 6.0 s |
| Q2 时间约束 | 220 | 9.26 m | 73.33% | 4.0 s |
| Q3 Greedy-Nearest | 200 | 9.47 m | 66.67% | 1.0 s |
| Q3 Greedy-Balance | 197 | 9.75 m | 65.67% | 1.0 s |

EPR（Q3/Q2）= 200/220 = **90.9%**

压力测试结论：EPR 在所有集中度/阈值扰动下维持 98.8%–108.9%，算法对参数扰动高度鲁棒。


## 各文件详细说明

### 主入口

| 文件 | 位置 | 说明 |
|------|------|------|
| `main.py` | 根目录 | **主入口**。依次调用四个问题的求解函数，打印汇总表。 |

### 公共模块 `common/`

| 文件 | 说明 |
|------|------|
| `utils.py` | **核心工具库**。`load_data()` 动态查找并加载三个 CSV 数据文件；`compute_distance_matrix()` 计算用户-车位欧氏距离矩阵 (500×300)；`build_eligibility()` 基于步行距离阈值构建资格矩阵；`build_time_eligibility()` 综合距离+时间构建兼容矩阵；`check_time_conflict()` 向量化检测用户时间重叠。 |
| `plot_theme.py` | **论文统一视觉风格**。定义深蓝→天蓝→青绿→琥珀四段渐变色 (`CMAP`)，提供 `pick_color(t)` 按 [0,1] 取色和 `apply_common_style()` 全局字体/网格/线条配置。 |
| `visual_helpers.py` | **SVG 热力图渲染**。`save_svg_heatmap()` 将网格数据绘制为 SVG 格式热力图，可直接嵌入 LaTeX 论文。 |
| `gen_svgs.py` | **快捷输出脚本**。调用 `q4_heatmap` 模块生成 12h 和 13h 的 SVG 切片及 3D PNG 渲染。 |
| `analyze_units.py` | **数据探索工具**。读取原始 CSV、统计用户/车位坐标分布、计算所有用户-车位配对的欧氏距离分布，与理论期望值对比，辅助判断坐标单位（米/百米/千米）。 |
| `users.csv` | 用户数据：`user_id`, `dest_x`, `dest_y`, `max_walk`, `arrival`, `departure` |
| `spots.csv` | 停车位数据：`spot_id`, `x`, `y` |
| `slots.csv` | 车位时段数据：`spot_id`, `start`, `end`（每个车位可有多个时段） |

### 问题1 `q1_static/` — 静态分配（无时间约束）

| 文件 | 说明 |
|------|------|
| `q1_static.py` | **两阶段 LP 优化**。阶段1：最大化匹配用户数（最大基数二分图匹配，PuLP CBC 求解）；阶段2：固定最大匹配数，最小化总步行距离。每个车位最多一个用户，每个用户最多一个车位。输出：`solve_q1()` 返回 `{assigned, unassigned, n_assigned, avg_distance, utilization, assignment_detail}`。 |

### 问题2 `q2_time/` — 时间约束分配

| 文件 | 说明 |
|------|------|
| `q2_time.py` | **MILP 模型**。`build_spot_time_conflicts()` 对每个车位构建内部用户时间冲突列表（半开区间重叠判定）。两阶段优化同 Q1，但增加了冲突对互斥约束：`x_j1i + x_j2i <= 1`。每个车位可服务多个时间不重叠的用户。输出：`solve_q2()` 返回结构同 Q1。 |

### 问题3 `q3_online/` — 在线调度算法

| 文件 | 说明 |
|------|------|
| `q3_online.py` | **在线贪心算法**。`OnlineAllocator` 类管理车位实时占用状态，用户按到达时间排序后依次决策。实现两种策略：**Greedy-Nearest**（选距离最近的可用车位）、**Greedy-Balance**（选当前占用数最少 + 距离次选的车位）。`compare_with_offline()` 与 Q2 离线最优比较，计算服务用户数竞争比和平均距离比。 |

### 问题4 `q4_analysis/` — 供需匹配分析

| 文件 | 说明 |
|------|------|
| `q4_analysis.py` | **综合分析**。`analyze_supply_demand()` 按每半小时粒度计算供给（可用车位）和需求（活跃用户）时序；`analyze_walking_distance()` 统计可行配对距离分布；`generate_plots()` 输出 2×3 子图（空间分布、时段供需、距离直方图、容忍度分布、问题对比、压力比时序）；`generate_recommendations()` 生成四大运营建议。 |
| `q4_heatmap.py` | **空间-时间压力热力图**。`build_spatial_temporal_pressure()` 将 100×100 空间划分为 10×10 网格，计算每格每时的需求/可达供给/压力比；`plot_heatmap()` 生成 8h-21h 共 7 张关键时刻热力图子图（标注 d/s 和压力值）；`plot_heatmap_3d()` 用 `scipy.interpolate.griddata` 插值平滑并渲染 3D 地形表面。 |
| `*.png` / `*.svg` | 图表输出文件。 |

### 敏感度分析 `sensitivity/`

| 文件 | 说明 |
|------|------|
| `sensitivity_analysis.py` | **三个维度的敏感度分析**：(1) `sensitivity_walk_distance()` — 步行距离阈值倍数 [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]；(2) `sensitivity_user_duration()` — 停车时长倍数 [0.5, 0.75, 1.0, 1.25, 1.5]；(3) `sensitivity_slot_extension()` — 原始时段 vs 单时段补充反向时段 vs 全天 8-22 开放。`plot_results()` 生成 2×2 子图（步行距离分配数/距离、时长分配数、时段扩展柱状对比）。 |
| `*.csv` | 各敏感度分析的数值结果，含两版不同运行输出。 |
| `*.png` | 敏感度分析综合图表。 |
