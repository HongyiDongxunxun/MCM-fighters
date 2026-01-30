import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Objective Optimization Model')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--method', type=str, default='weighted', choices=['weighted', 'epsilon'], help='Multi-objective method')
    return parser.parse_args()


def load_data(file_path):
    """加载数据"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def objective1(x):
    """目标函数1：最小化成本"""
    return 2 * x[0]**2 + 3 * x[1]**2 + x[0] * x[1] + 4 * x[0] + 5 * x[1]


def objective2(x):
    """目标函数2：最小化资源消耗"""
    return x[0]**2 + 2 * x[1]**2 + 2 * x[0] * x[1] + 3 * x[0] + 2 * x[1]


def objective3(x):
    """目标函数3：最大化利润（返回负值用于最小化）"""
    return -(5 * x[0] + 4 * x[1] - 0.5 * x[0]**2 - 0.3 * x[1]**2)


def weighted_sum_objective(x, weights):
    """加权和目标函数"""
    f1 = objective1(x)
    f2 = objective2(x)
    f3 = objective3(x)
    
    # 归一化目标函数值
    f1_norm = f1 / 100  # 假设最大值约为100
    f2_norm = f2 / 50   # 假设最大值约为50
    f3_norm = f3 / 10   # 假设最大值约为10
    
    # 加权和
    total = weights[0] * f1_norm + weights[1] * f2_norm + weights[2] * f3_norm
    return total


def epsilon_constraint_objective(x, epsilon1, epsilon2):
    """ε-约束法目标函数"""
    # 以目标1为主要优化目标，其他目标作为约束
    return objective1(x)


def constraint1(x, epsilon):
    """约束1：目标2 ≤ epsilon"""
    return epsilon - objective2(x)


def constraint2(x, epsilon):
    """约束2：目标3 ≤ epsilon"""
    return epsilon - objective3(x)


def constraint3(x):
    """约束3：资源约束"""
    return 10 - (x[0] + x[1])  # x1 + x2 ≤ 10


def constraint4(x):
    """约束4：生产约束"""
    return x[0] + x[1] - 1  # x1 + x2 ≥ 1


def optimize_weighted_sum(data):
    """使用加权和法优化"""
    # 权重设置
    if 'weights' in data.columns:
        weights = data['weights'].values[:3]
        # 归一化权重
        weights = weights / np.sum(weights)
    else:
        weights = np.array([0.4, 0.3, 0.3])  # 默认权重
    
    print(f"Using weights: {weights}")
    
    # 初始猜测值
    x0 = np.array([1.0, 1.0])
    
    # 约束
    cons = [
        {'type': 'ineq', 'fun': constraint3},
        {'type': 'ineq', 'fun': constraint4}
    ]
    
    # 边界
    bounds = [(0, None), (0, None)]
    
    # 优化
    result = minimize(
        weighted_sum_objective,
        x0,
        args=(weights,),
        method='SLSQP',
        constraints=cons,
        bounds=bounds,
        options={'maxiter': 1000, 'disp': True}
    )
    
    return result


def optimize_epsilon_constraint(data):
    """使用ε-约束法优化"""
    # 设置ε值
    if 'epsilon' in data.columns:
        epsilon1 = data['epsilon'].values[0]
        epsilon2 = data['epsilon'].values[1]
    else:
        epsilon1 = 20  # 默认ε1
        epsilon2 = -5   # 默认ε2
    
    print(f"Using epsilon values: ε1={epsilon1}, ε2={epsilon2}")
    
    # 初始猜测值
    x0 = np.array([1.0, 1.0])
    
    # 约束
    cons = [
        {'type': 'ineq', 'fun': constraint3},
        {'type': 'ineq', 'fun': constraint4},
        {'type': 'ineq', 'fun': lambda x: constraint1(x, epsilon1)},
        {'type': 'ineq', 'fun': lambda x: constraint2(x, epsilon2)}
    ]
    
    # 边界
    bounds = [(0, None), (0, None)]
    
    # 优化
    result = minimize(
        epsilon_constraint_objective,
        x0,
        args=(epsilon1, epsilon2),
        method='SLSQP',
        constraints=cons,
        bounds=bounds,
        options={'maxiter': 1000, 'disp': True}
    )
    
    return result


def generate_pareto_front(data):
    """生成帕累托前沿"""
    pareto_points = []
    pareto_values = []
    
    # 尝试不同的权重组合
    weight_combinations = [
        [0.7, 0.2, 0.1],
        [0.5, 0.3, 0.2],
        [0.3, 0.4, 0.3],
        [0.1, 0.5, 0.4],
        [0.2, 0.2, 0.6]
    ]
    
    for weights in weight_combinations:
        # 初始猜测值
        x0 = np.array([1.0, 1.0])
        
        # 约束
        cons = [
            {'type': 'ineq', 'fun': constraint3},
            {'type': 'ineq', 'fun': constraint4}
        ]
        
        # 边界
        bounds = [(0, None), (0, None)]
        
        # 优化
        result = minimize(
            weighted_sum_objective,
            x0,
            args=(weights,),
            method='SLSQP',
            constraints=cons,
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        if result.success:
            x_opt = result.x
            f1 = objective1(x_opt)
            f2 = objective2(x_opt)
            f3 = -objective3(x_opt)  # 转换回最大化的利润
            
            pareto_points.append(x_opt)
            pareto_values.append([f1, f2, f3])
    
    return pareto_points, pareto_values


def main():
    args = parse_args()
    data = load_data(args.input)
    
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    print("Running Multi-Objective Optimization Model...")
    print(f"Method: {'Weighted Sum' if args.method == 'weighted' else 'Epsilon Constraint'}")
    
    # 运行优化
    if args.method == 'weighted':
        result = optimize_weighted_sum(data)
    else:
        result = optimize_epsilon_constraint(data)
    
    # 输出结果
    print("\nOptimization Result:")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Number of iterations: {result.nit}")
    print(f"Optimal solution: {result.x}")
    print(f"Optimal value: {result.fun}")
    
    # 计算各目标函数值
    x_opt = result.x
    f1 = objective1(x_opt)
    f2 = objective2(x_opt)
    f3 = -objective3(x_opt)  # 转换回最大化的利润
    
    print("\nObjective Function Values:")
    print(f"Cost (f1): {f1}")
    print(f"Resource Consumption (f2): {f2}")
    print(f"Profit (f3): {f3}")
    
    # 生成帕累托前沿
    pareto_points, pareto_values = generate_pareto_front(data)
    
    # 保存结果
    results = {
        'success': [result.success],
        'message': [result.message],
        'iterations': [result.nit],
        'x1': [x_opt[0]],
        'x2': [x_opt[1]],
        'cost': [f1],
        'resource_consumption': [f2],
        'profit': [f3]
    }
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output, index=False)
    
    # 保存帕累托前沿
    if pareto_points:
        pareto_data = {
            'x1': [p[0] for p in pareto_points],
            'x2': [p[1] for p in pareto_points],
            'cost': [v[0] for v in pareto_values],
            'resource_consumption': [v[1] for v in pareto_values],
            'profit': [v[2] for v in pareto_values]
        }
        df_pareto = pd.DataFrame(pareto_data)
        pareto_output = args.output.replace('.csv', '_pareto.csv')
        df_pareto.to_csv(pareto_output, index=False)
        print(f"\nPareto front saved to: {pareto_output}")
    
    print(f"\nResults saved to: {args.output}")
    print("\nMulti-Objective Optimization Model completed successfully!")


if __name__ == "__main__":
    main()
