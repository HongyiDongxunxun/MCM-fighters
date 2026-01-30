import argparse
import pandas as pd
import numpy as np
from scipy.optimize import linprog


def parse_args():
    parser = argparse.ArgumentParser(description='Linear Programming model for MCM')
    parser.add_argument('--input', type=str, required=True, help='Input data file path')
    parser.add_argument('--output', type=str, required=True, help='Output result file path')
    parser.add_argument('--objective', type=str, choices=['max', 'min'], default='min', help='Objective function type')
    return parser.parse_args()


def load_data(file_path):
    """加载数据"""
    return pd.read_csv(file_path)


def linear_programming_model(data, objective='min'):
    """线性规划模型"""
    print("执行线性规划模型...")
    
    # 提取目标函数系数
    c = data.iloc[0, :-1].values.astype(float)
    
    # 提取约束条件
    A_ub = data.iloc[1:-1, :-1].values.astype(float)
    b_ub = data.iloc[1:-1, -1].values.astype(float)
    
    # 提取变量 bounds
    bounds = []
    for i in range(len(c)):
        bounds.append((0, None))  # 默认非负约束
    
    # 如果是最大化问题，转换为最小化问题
    if objective == 'max':
        c = -c
    
    # 执行线性规划
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    print("线性规划结果:")
    print(f"目标函数值: {abs(result.fun) if objective == 'max' else result.fun}")
    print(f"决策变量值: {result.x}")
    print(f"状态: {'成功' if result.success else '失败'}")
    if not result.success:
        print(f"消息: {result.message}")
    
    return result


def save_results(result, output_path, objective='min'):
    """保存结果"""
    results = {
        'status': 'success' if result.success else 'failed',
        'objective_value': abs(result.fun) if objective == 'max' else result.fun,
        'variables': list(result.x),
        'message': result.message
    }
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"结果已保存到: {output_path}")


def main():
    args = parse_args()
    data = load_data(args.input)
    result = linear_programming_model(data, args.objective)
    save_results(result, args.output, args.objective)


if __name__ == "__main__":
    main()