import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize, NonlinearConstraint


def parse_args():
    parser = argparse.ArgumentParser(description='Nonlinear Programming Model')
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


def nonlinear_objective(x, data, objective_type):
    """非线性目标函数"""
    # 示例：最小化成本函数
    # 假设x是决策变量，data包含相关参数
    cost = 0
    
    # 二次成本函数示例
    cost += 0.5 * x[0]**2 + 0.5 * x[1]**2
    
    # 交叉项
    cost += 0.1 * x[0] * x[1]
    
    # 线性项
    if 'cost_coeff' in data.columns:
        coeffs = data['cost_coeff'].values
        for i, coeff in enumerate(coeffs):
            if i < len(x):
                cost += coeff * x[i]
    
    # 如果是最大化问题，返回负值
    if objective_type == 'max':
        return -cost
    return cost


def nonlinear_constraint1(x):
    """非线性约束1：资源约束"""
    return 10 - (x[0]**2 + x[1]**2)  # x1² + x2² ≤ 10


def nonlinear_constraint2(x):
    """非线性约束2：生产约束"""
    return x[0] + x[1] - 2  # x1 + x2 ≥ 2


def optimize_model(data, objective_type):
    """优化模型"""
    # 初始猜测值
    x0 = np.array([1.0, 1.0])
    
    # 定义约束
    constraints = [
        NonlinearConstraint(nonlinear_constraint1, -np.inf, 0),
        NonlinearConstraint(nonlinear_constraint2, 0, np.inf)
    ]
    
    # 定义边界
    bounds = [
        (0, None),  # x1 ≥ 0
        (0, None)   # x2 ≥ 0
    ]
    
    # 选择求解器
    method = 'trust-constr'
    
    # 求解优化问题
    result = minimize(
        nonlinear_objective,
        x0,
        args=(data, objective_type),
        method=method,
        constraints=constraints,
        bounds=bounds,
        options={'maxiter': 1000, 'disp': True}
    )
    
    return result


def main():
    args = parse_args()
    data = load_data(args.input)
    
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    print("Running Nonlinear Programming Model...")
    print(f"Objective: {'Maximization' if args.objective == 'max' else 'Minimization'}")
    
    # 运行优化
    result = optimize_model(data, args.objective)
    
    # 输出结果
    print("\nOptimization Result:")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Number of iterations: {result.nit}")
    print(f"Optimal solution: {result.x}")
    print(f"Optimal value: {result.fun if args.objective == 'min' else -result.fun}")
    
    # 保存结果
    results = {
        'success': [result.success],
        'message': [result.message],
        'iterations': [result.nit],
        'x1': [result.x[0]],
        'x2': [result.x[1]],
        'optimal_value': [result.fun if args.objective == 'min' else -result.fun]
    }
    
    # 添加约束值
    results['constraint1_value'] = [nonlinear_constraint1(result.x)]
    results['constraint2_value'] = [nonlinear_constraint2(result.x)]
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output, index=False)
    
    print(f"\nResults saved to: {args.output}")
    print("\nNonlinear Programming Model completed successfully!")


if __name__ == "__main__":
    main()
