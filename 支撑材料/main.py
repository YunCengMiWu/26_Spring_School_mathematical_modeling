"""
B题：智慧城市共享停车位资源调配
====================================
完整建模与求解入口

运行方式: python main.py
"""

import time
import sys

# 控制台 UTF-8 编码，确保中文正常输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from q1_static import solve_q1
from q2_time import solve_q2
from q3_online import solve_q3, compare_with_offline
from q4_analysis import solve_q4


def print_header(text):
    """打印带格式的标题"""
    width = 68
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}")


def main():
    total_start = time.time()

    print_header("B题：智慧城市共享停车位资源调配")
    print("  模型假设与说明:")
    print("  - 步行距离为欧氏距离（车位到用户目的地）")
    print("  - 时间兼容性：用户[到达,离开]完全包含在车位可用时段内")
    print("  - 优化目标：两级优先级——最大服务用户数 > 最小化总步行距离")
    print(f"\n  系统时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ============ 问题1：静态分配（无时间约束） ============
    print_header("问题1：静态分配（无时间约束）")
    t1 = time.time()
    q1_result = solve_q1(verbose=True)
    t1_elapsed = time.time() - t1
    print(f"  求解耗时: {t1_elapsed:.2f} 秒")

    # ============ 问题2：时间约束分配 ============
    print_header("问题2：时间约束分配")
    t2 = time.time()
    q2_result = solve_q2(verbose=True)
    t2_elapsed = time.time() - t2
    print(f"  求解耗时: {t2_elapsed:.2f} 秒")

    # ============ 问题3：在线调度算法 ============
    print_header("问题3：在线调度算法")
    t3 = time.time()

    # 策略1：最近适配
    print("\n【策略1：Greedy-Nearest】")
    q3_nearest = solve_q3(strategy="nearest", verbose=True)

    # 策略2：平衡适配
    print("\n【策略2：Greedy-Balance】")
    q3_balance = solve_q3(strategy="balance", verbose=True)

    t3_elapsed = time.time() - t3

    # 竞争比分析
    print_header("在线与离线结果对比分析")
    print("\n【Greedy-Nearest vs 离线最优】")
    compare_with_offline(q3_nearest, q2_result, verbose=True)
    print("\n【Greedy-Balance vs 离线最优】")
    compare_with_offline(q3_balance, q2_result, verbose=True)

    # 选择最优的在线策略作为 Q3 代表
    if q3_nearest["n_assigned"] >= q3_balance["n_assigned"]:
        q3_best = q3_nearest
    else:
        q3_best = q3_balance

    print(f"\n  在线调度总耗时: {t3_elapsed:.2f} 秒")

    # ============ 问题4：分析与建议 ============
    print_header("问题4：供需匹配分析与改进建议")
    t4 = time.time()
    q4_result = solve_q4(
        q1_result=q1_result,
        q2_result=q2_result,
        q3_result=q3_best,
        verbose=True
    )
    t4_elapsed = time.time() - t4
    print(f"\n  分析耗时: {t4_elapsed:.2f} 秒")

    # ============ 最终汇总 ============
    total_elapsed = time.time() - total_start

    print_header("最终结果汇总")
    print(f"{'=' * 68}")
    print(f"{'问题':<20} {'服务用户':<12} {'平均距离(米)':<14} {'利用率':<10} {'耗时(秒)':<10}")
    print(f"{'-' * 68}")

    if q1_result["status"] == "optimal":
        print(f"{'Q1(无时间约束)':<20} {q1_result['n_assigned']:<12} "
              f"{q1_result['avg_distance']:<14.2f} {q1_result['utilization']:<10.2%} {t1_elapsed:<10.2f}")
    if q2_result["status"] == "optimal":
        print(f"{'Q2(时间约束)':<20} {q2_result['n_assigned']:<12} "
              f"{q2_result['avg_distance']:<14.2f} {q2_result['utilization']:<10.2%} {t2_elapsed:<10.2f}")
    if q3_best["status"] == "ok":
        print(f"{'Q3(在线' + q3_best['strategy'] + ')':<20} {q3_best['n_assigned']:<12} "
              f"{q3_best['avg_distance']:<14.2f} {q3_best['utilization']:<10.2%} {t3_elapsed:<10.2f}")

    print(f"{'-' * 68}")
    print(f"{'总耗时':<61} {total_elapsed:<10.2f}")
    print(f"{'=' * 68}")

    print_header("模型总结与关键发现")
    print("""
1. 无时间约束下（Q1），平台可最大化匹配约 ___ 用户
2. 引入时间约束后（Q2），匹配用户数下降至 ___
3. 在线调度（Q3）达到离线最优的 ___% 左右
4. 最紧张时段为 ___ 时左右，压力比达 ___
5. 建议：动态定价 + 错时共享激励 + 步行优化

（具体数值根据运行结果填入）
""")

    print(f"\n总运行时间: {total_elapsed:.2f} 秒")
    print("程序运行完毕。")


if __name__ == "__main__":
    main()
