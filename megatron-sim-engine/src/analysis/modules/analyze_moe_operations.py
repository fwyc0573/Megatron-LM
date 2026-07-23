#!/usr/bin/env python3
"""
分析MoE操作的索引分配和一致性
"""

def parse_sub_operations(sub_ops_str):
    """解析sub_operations字符串，提取MoE操作信息"""
    import ast
    
    # 移除外层的方括号
    if sub_ops_str.startswith('[') and sub_ops_str.endswith(']'):
        sub_ops_str = sub_ops_str[1:-1]
    
    # 分割各个操作
    operations = []
    current_op = ""
    bracket_count = 0
    quote_count = 0
    
    for char in sub_ops_str:
        if char == "'" and (len(current_op) == 0 or current_op[-1] != '\\'):
            quote_count = (quote_count + 1) % 2
        elif char == '[' and quote_count == 0:
            bracket_count += 1
        elif char == ']' and quote_count == 0:
            bracket_count -= 1
        elif char == ',' and bracket_count == 0 and quote_count == 0:
            if current_op.strip():
                operations.append(current_op.strip().strip("'"))
            current_op = ""
            continue
        
        current_op += char
    
    # 添加最后一个操作
    if current_op.strip():
        operations.append(current_op.strip().strip("'"))
    
    return operations

def analyze_moe_operations(filepath, data_type):
    """分析文件中的MoE操作"""
    print(f"\n=== 分析 {data_type} 数据: {filepath} ===")
    
    moe_operations = {
        'forward_step': [],
        'backward_step': []
    }
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if 'sub_operations=' in line and ('forward_step' in line or 'backward_step' in line):
                # 提取操作类型
                op_type = 'forward_step' if 'forward_step' in line else 'backward_step'
                
                # 提取sub_operations部分
                start_idx = line.find('sub_operations=') + len('sub_operations=')
                end_idx = line.find(')', start_idx)
                sub_ops_str = line[start_idx:end_idx]
                
                # 解析sub_operations
                try:
                    operations = parse_sub_operations(sub_ops_str)
                    
                    # 过滤MoE操作
                    moe_ops = []
                    for op in operations:
                        if 'group=exp' in op:
                            # 提取关键信息
                            comm_func = None
                            if 'comm_func=allgather' in op:
                                comm_func = 'allgather'
                            elif 'comm_func=all_to_all' in op:
                                comm_func = 'all_to_all'
                            
                            if comm_func:
                                moe_ops.append({
                                    'comm_func': comm_func,
                                    'raw': op[:100] + '...' if len(op) > 100 else op
                                })
                    
                    moe_operations[op_type].extend(moe_ops)
                    
                except Exception as e:
                    print(f"解析错误 (行 {line_num}): {e}")
    
    return moe_operations

def compare_operations(trace_ops, profile_ops):
    """比较trace和profile数据中的操作"""
    print(f"\n=== 操作对比分析 ===")
    
    for op_type in ['forward_step', 'backward_step']:
        print(f"\n{op_type.upper()}:")
        trace_count = len(trace_ops[op_type])
        profile_count = len(profile_ops[op_type])
        
        print(f"  Trace数据: {trace_count} 个MoE操作")
        print(f"  Profile数据: {profile_count} 个MoE操作")
        print(f"  数量匹配: {'✅' if trace_count == profile_count else '❌'}")
        
        # 分析操作类型分布
        trace_types = {}
        profile_types = {}
        
        for op in trace_ops[op_type]:
            comm_func = op['comm_func']
            trace_types[comm_func] = trace_types.get(comm_func, 0) + 1
        
        for op in profile_ops[op_type]:
            comm_func = op['comm_func']
            profile_types[comm_func] = profile_types.get(comm_func, 0) + 1
        
        print(f"  Trace操作类型分布: {trace_types}")
        print(f"  Profile操作类型分布: {profile_types}")
        print(f"  类型分布匹配: {'✅' if trace_types == profile_types else '❌'}")

if __name__ == "__main__":
    # 分析文件路径
    trace_file = "simulation_inputs/megatron_operation_log/h800_moe_mixtral8_1.75b_2pp_1tp_4dp_2ep_4096seq/global_ranks_profile/wd8_tp1_pp2_exp2_expNum8_l8_bs1_rank0_20250809155927.txt"
    profile_file = "simulation_inputs/megatron_operation_log/h800_moe_mixtral8_1.75b_2pp_1tp_4dp_2ep_4096seq/database_profile/wd8_tp1_pp2_exp2_expNum8_numl8_bs1_rank0_20250809161134.txt"
    
    # 分析两个文件
    trace_operations = analyze_moe_operations(trace_file, "Trace")
    profile_operations = analyze_moe_operations(profile_file, "Profile")
    
    # 比较操作
    compare_operations(trace_operations, profile_operations)
    
    print(f"\n=== 结论 ===")
    print("如果数量和类型分布都匹配，则问题在于索引分配机制")
    print("如果不匹配，则存在数据收集不完整的问题")
