import argparse
import numpy as np
import pandas as pd
from scipy.optimize import linprog


def parse_args():
    parser = argparse.ArgumentParser(description='Integer Programming Model')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--objective', type=str, default='min', choices=['min', 'max'], help='Objective function type')
    return parser.parse_args()


def load_data(file_path):
    """加载数据"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def branch_and_bound(c, A_ub, b_ub, A_eq, b_eq, bounds, objective_type):
    """分支定界法求解整数规划问题"""
    # 首先求解松弛问题（线性规划）
    if objective_type == 'max':
        # 最大化问题转换为最小化
        c = -c
    
    # 求解线性规划松弛问题
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not result.success:
        return None, float('inf') if objective_type == 'min' else -float('inf')
    
    # 检查是否所有变量都是整数
    x = result.x
    if all(np.isclose(x_i, round(x_i)) for x_i in x):
        # 找到整数解
        objective_value = result.fun if objective_type == 'min' else -result.fun
        return x, objective_value
    
    # 选择第一个非整数变量进行分支
    branch_var = -1
    for i, x_i in enumerate(x):
        if not np.isclose(x_i, round(x_i)):
            branch_var = i
            break
    
    if branch_var == -1:
        return None, float('inf') if objective_type == 'min' else -float('inf')
    
    # 分支：创建两个子问题
    # 子问题1：x[branch_var] ≤ floor(x[branch_var])
    x_floor = np.floor(x[branch_var])
    A_ub1 = np.vstack([A_ub, np.zeros(len(c))])
    A_ub1[-1, branch_var] = 1
    b_ub1 = np.append(b_ub, x_floor)
    
    # 子问题2：x[branch_var] ≥ ceil(x[branch_var])
    x_ceil = np.ceil(x[branch_var])
    A_ub2 = np.vstack([A_ub, np.zeros(len(c))])
    A_ub2[-1, branch_var] = -1
    b_ub2 = np.append(b_ub, -x_ceil)
    
    # 递归求解子问题
    x1, val1 = branch_and_bound(c, A_ub1, b_ub1, A_eq, b_eq, bounds, objective_type)
    x2, val2 = branch_and_bound(c, A_ub2, b_ub2, A_eq, b_eq, bounds, objective_type)
    
    # 选择最优解
    if objective_type == 'min':
        if val1 < val2:
            return x1, val1
        else:
            return x2, val2
    else:
        if val1 > val2:
            return x1, val1
        else:
            return x2, val2


def optimize_model(data, objective_type):
    """优化模型"""
    # 示例：生产计划问题
    # 决策变量：x1, x2（两种产品的生产数量）
    # 目标：最大化利润或最小化成本
    
    # 系数矩阵
    # 假设数据文件包含利润系数和约束条件
    
    # 目标函数系数
    if 'profit' in data.columns:
        c_profit = data['profit'].values[:2]
    else:
        c_profit = np.array([5, 3])  # 默认利润系数
    
    if 'cost' in data.columns:
        c_cost = data['cost'].values[:2]
    else:
        c_cost = np.array([2, 1])  # 默认成本系数
    
    # 选择目标函数系数
    c = c_cost if objective_type == 'min' else c_profit
    
    # 约束条件
    # 资源约束：原料A和原料B
    A_ub = np.array([
        [2, 1],  # 原料A约束
        [1, 2]   # 原料B约束
    ])
    
    # 资源总量
    if 'resource' in data.columns:
        b_ub = data['resource'].values[:2]
    else:
        b_ub = np.array([10, 8])  # 默认资源总量
    
    # 等式约束（如果有）
    A_eq = None
    b_eq = None
    
    # 变量边界
    bounds = [(0, None), (0, None)]  # 非负整数
    
    # 运行分支定界法
    x_opt, val_opt = branch_and_bound(c, A_ub, b_ub, A_eq, b_eq, bounds, objective_type)
    
    return x_opt, val_opt


def main():
    args = parse_args()
    data = load_data(args.input)
    
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    print("Running Integer Programming Model...")
    print(f"Objective: {'Maximization' if args.objective == 'max' else 'Minimization'}")
    
    # 运行优化
    x_opt, val_opt = optimize_model(data, args.objective)
    
    # 输出结果
    print("\nOptimization Result:")
    if x_opt is not None:
        print(f"Success: True")
        print(f"Optimal solution: {x_opt}")
        print(f"Optimal value: {val_opt}")
        
        # 保存结果
        results = {
            'success': [True],
            'x1': [x_opt[0]],
            'x2': [x_opt[1]],
            'optimal_value': [val_opt]
        }
        
        df_results = pd.DataFrame(results)
        df_results.to_csv(args.output, index=False)
        
        print(f"\nResults saved to: {args.output}")
    else:
        print("Success: False")
        print("No feasible solution found.")
    
    print("\nInteger Programming Model completed!")


if __name__ == "__main__":
    main()
