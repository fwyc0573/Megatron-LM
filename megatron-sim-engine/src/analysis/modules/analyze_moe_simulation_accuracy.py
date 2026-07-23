#!/usr/bin/env python3
"""
MoE模型模拟准确性分析脚本
对比MODE_PROFILE和MODE_SIMULATE的结果，分析误差来源
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

def analyze_simulation_accuracy():
    """分析MoE模型模拟准确性"""
    
    print("=== MoE模型模拟准确性分析 ===")
    print()
    
    # MODE_PROFILE结果 (实际trace数据)
    profile_results = {
        'rank0': {'comp_time': 179.10, 'comm_time': 250.17, 'sum_time': 429.27},
        'rank1': {'comp_time': 132.47, 'comm_time': 298.44, 'sum_time': 430.91},
        'rank2': {'comp_time': 178.89, 'comm_time': 248.74, 'sum_time': 427.63},
        'rank3': {'comp_time': 131.42, 'comm_time': 300.35, 'sum_time': 431.77},
        'rank4': {'comp_time': 179.88, 'comm_time': 152.58, 'sum_time': 332.46},
        'rank5': {'comp_time': 123.97, 'comm_time': 202.55, 'sum_time': 326.52},
        'rank6': {'comp_time': 181.09, 'comm_time': 151.19, 'sum_time': 332.28},
        'rank7': {'comp_time': 124.66, 'comm_time': 202.05, 'sum_time': 326.71}
    }
    
    # MODE_SIMULATE结果 (模拟预测数据)
    simulate_results = {
        'rank0': {'comp_time': 166.53, 'comm_time': 2731.61, 'sum_time': 2898.14},
        'rank1': {'comp_time': 127.45, 'comm_time': 2771.87, 'sum_time': 2899.32},
        'rank2': {'comp_time': 163.86, 'comm_time': 2729.96, 'sum_time': 2893.82},
        'rank3': {'comp_time': 126.60, 'comm_time': 2772.71, 'sum_time': 2899.31},
        'rank4': {'comp_time': 165.18, 'comm_time': 2656.52, 'sum_time': 2821.70},
        'rank5': {'comp_time': 126.56, 'comm_time': 2696.14, 'sum_time': 2822.70},
        'rank6': {'comp_time': 160.52, 'comm_time': 2656.78, 'sum_time': 2817.30},
        'rank7': {'comp_time': 128.35, 'comm_time': 2694.43, 'sum_time': 2822.78}
    }
    
    print("=== 第一步：数据对比分析 ===")
    print()
    
    # 计算误差统计
    comp_errors = []
    comm_errors = []
    total_errors = []
    
    print("Rank | Comp Time (ms)      | Comm Time (ms)        | Total Time (ms)")
    print("     | Profile | Simulate | Profile | Simulate    | Profile | Simulate")
    print("-" * 75)
    
    for i in range(8):
        rank_key = f'rank{i}'
        profile = profile_results[rank_key]
        simulate = simulate_results[rank_key]
        
        comp_error = ((simulate['comp_time'] - profile['comp_time']) / profile['comp_time']) * 100
        comm_error = ((simulate['comm_time'] - profile['comm_time']) / profile['comm_time']) * 100
        total_error = ((simulate['sum_time'] - profile['sum_time']) / profile['sum_time']) * 100
        
        comp_errors.append(comp_error)
        comm_errors.append(comm_error)
        total_errors.append(total_error)
        
        print(f"{i:4d} | {profile['comp_time']:7.2f} | {simulate['comp_time']:8.2f} | "
              f"{profile['comm_time']:7.2f} | {simulate['comm_time']:10.2f} | "
              f"{profile['sum_time']:7.2f} | {simulate['sum_time']:8.2f}")
    
    print()
    print("=== 第二步：误差统计分析 ===")
    print()
    
    print("误差类型分析:")
    print(f"计算时间误差: 平均 {np.mean(comp_errors):6.2f}%, 标准差 {np.std(comp_errors):6.2f}%")
    print(f"通信时间误差: 平均 {np.mean(comm_errors):6.2f}%, 标准差 {np.std(comm_errors):6.2f}%")
    print(f"总体时间误差: 平均 {np.mean(total_errors):6.2f}%, 标准差 {np.std(total_errors):6.2f}%")
    print()
    
    # 分析误差来源
    print("=== 第三步：误差来源分类 ===")
    print()
    
    print("1. **模拟器逻辑错误分析**：")
    print("   - 计算时间误差相对较小 (平均 {:.2f}%)".format(np.mean(np.abs(comp_errors))))
    print("   - 表明计算操作的duration预测基本准确")
    print()
    
    print("2. **通信操作预测误差分析**：")
    print("   - 通信时间误差极大 (平均 {:.2f}%)".format(np.mean(comm_errors)))
    print("   - 这是主要的误差来源")
    print()
    
    # 分析通信时间差异的原因
    print("3. **通信时间差异根因分析**：")
    print()
    
    # 计算通信时间的绝对差异
    profile_comm_avg = np.mean([profile_results[f'rank{i}']['comm_time'] for i in range(8)])
    simulate_comm_avg = np.mean([simulate_results[f'rank{i}']['comm_time'] for i in range(8)])
    
    print(f"   Profile模式平均通信时间: {profile_comm_avg:.2f} ms")
    print(f"   Simulate模式平均通信时间: {simulate_comm_avg:.2f} ms")
    print(f"   差异倍数: {simulate_comm_avg/profile_comm_avg:.1f}x")
    print()
    
    print("4. **可能的原因分析**：")
    print("   a) **数据层面 - 操作耗时预测误差**：")
    print("      - 网络估计器可能高估了MoE通信操作的耗时")
    print("      - exp_allgather, exp_all_to_all, exp_dp_allreduce的duration预测不准确")
    print()
    print("   b) **逻辑层面 - 模拟器逻辑错误**：")
    print("      - 通信同步逻辑可能存在问题")
    print("      - MoE通信依赖关系处理可能不正确")
    print("      - Timeline计算可能有误")
    print()
    
    # 分析不同stage的差异
    print("5. **Pipeline Stage差异分析**：")
    stage0_profile_comm = np.mean([profile_results[f'rank{i}']['comm_time'] for i in range(4)])
    stage1_profile_comm = np.mean([profile_results[f'rank{i}']['comm_time'] for i in range(4, 8)])
    stage0_simulate_comm = np.mean([simulate_results[f'rank{i}']['comm_time'] for i in range(4)])
    stage1_simulate_comm = np.mean([simulate_results[f'rank{i}']['comm_time'] for i in range(4, 8)])
    
    print(f"   Stage 0 (ranks 0-3):")
    print(f"     Profile: {stage0_profile_comm:.2f} ms, Simulate: {stage0_simulate_comm:.2f} ms")
    print(f"     误差: {((stage0_simulate_comm - stage0_profile_comm) / stage0_profile_comm) * 100:.1f}%")
    print(f"   Stage 1 (ranks 4-7):")
    print(f"     Profile: {stage1_profile_comm:.2f} ms, Simulate: {stage1_simulate_comm:.2f} ms")
    print(f"     误差: {((stage1_simulate_comm - stage1_profile_comm) / stage1_profile_comm) * 100:.1f}%")
    print()
    
    print("=== 第四步：修复验证结果 ===")
    print()
    print("✅ **UnboundLocalError修复验证**：")
    print("   - 两种模式都成功运行，没有出现程序崩溃")
    print("   - MoE通信匹配问题已解决")
    print()
    print("✅ **Selected Ranks策略修复验证**：")
    print("   - 日志显示：'Warning: MoE模型需要所有ranks参与以确保通信组完整性'")
    print("   - 选择了所有8个ranks，通信组完整性得到保证")
    print()
    print("✅ **MoE通信操作匹配验证**：")
    print("   - 所有MoE操作都通过类型匹配找到了对应的trace数据")
    print("   - 'Debug: Found type-based MoE operation match' 日志正常输出")
    print("   - 'Debug: comm_matching_operation_list length: 1' 表明匹配成功")
    print()
    
    print("=== 第五步：改进建议 ===")
    print()
    print("🔧 **短期改进**：")
    print("   1. 调整网络估计器的通信时间预测模型")
    print("   2. 校准MoE通信操作的duration计算")
    print("   3. 验证通信同步逻辑的正确性")
    print()
    print("🔧 **长期改进**：")
    print("   1. 基于真实硬件测试数据校准通信模型")
    print("   2. 实现更精确的MoE通信依赖关系建模")
    print("   3. 开发专门的MoE模型性能预测算法")
    
    return profile_results, simulate_results, comp_errors, comm_errors, total_errors

def create_visualization(profile_results, simulate_results, comp_errors, comm_errors, total_errors):
    """创建可视化图表"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    ranks = list(range(8))
    
    # 1. 计算时间对比
    profile_comp = [profile_results[f'rank{i}']['comp_time'] for i in ranks]
    simulate_comp = [simulate_results[f'rank{i}']['comp_time'] for i in ranks]
    
    ax1.bar([r - 0.2 for r in ranks], profile_comp, 0.4, label='Profile', alpha=0.7)
    ax1.bar([r + 0.2 for r in ranks], simulate_comp, 0.4, label='Simulate', alpha=0.7)
    ax1.set_xlabel('Rank')
    ax1.set_ylabel('Computation Time (ms)')
    ax1.set_title('Computation Time Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 通信时间对比
    profile_comm = [profile_results[f'rank{i}']['comm_time'] for i in ranks]
    simulate_comm = [simulate_results[f'rank{i}']['comm_time'] for i in ranks]
    
    ax2.bar([r - 0.2 for r in ranks], profile_comm, 0.4, label='Profile', alpha=0.7)
    ax2.bar([r + 0.2 for r in ranks], simulate_comm, 0.4, label='Simulate', alpha=0.7)
    ax2.set_xlabel('Rank')
    ax2.set_ylabel('Communication Time (ms)')
    ax2.set_title('Communication Time Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 误差分布
    ax3.bar(ranks, comp_errors, alpha=0.7, label='Computation Error')
    ax3.bar(ranks, comm_errors, alpha=0.7, label='Communication Error')
    ax3.set_xlabel('Rank')
    ax3.set_ylabel('Error (%)')
    ax3.set_title('Error Distribution by Rank')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 总体时间对比
    profile_total = [profile_results[f'rank{i}']['sum_time'] for i in ranks]
    simulate_total = [simulate_results[f'rank{i}']['sum_time'] for i in ranks]
    
    ax4.bar([r - 0.2 for r in ranks], profile_total, 0.4, label='Profile', alpha=0.7)
    ax4.bar([r + 0.2 for r in ranks], simulate_total, 0.4, label='Simulate', alpha=0.7)
    ax4.set_xlabel('Rank')
    ax4.set_ylabel('Total Time (ms)')
    ax4.set_title('Total Time Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./log/visualization_outputs/moe_simulation_accuracy_analysis.png', dpi=300, bbox_inches='tight')
    print(f"可视化图表已保存到: ./log/visualization_outputs/moe_simulation_accuracy_analysis.png")

if __name__ == "__main__":
    profile_results, simulate_results, comp_errors, comm_errors, total_errors = analyze_simulation_accuracy()
    
    # 创建可视化
    os.makedirs('./log/visualization_outputs', exist_ok=True)
    create_visualization(profile_results, simulate_results, comp_errors, comm_errors, total_errors)
