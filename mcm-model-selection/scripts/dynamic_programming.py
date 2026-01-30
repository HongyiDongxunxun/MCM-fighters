import argparse
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Dynamic Programming model for MCM')
    parser.add_argument('--input', type=str, required=True, help='Input data file path')
    parser.add_argument('--output', type=str, required=True, help='Output result file path')
    parser.add_argument('--problem_type', type=str, choices=['knapsack', 'investment', 'resource_allocation'], default='knapsack', help='Dynamic programming problem type')
    return parser.parse_args()


def load_data(file_path):
    """加载数据"""
    return pd.read_csv(file_path)


def knapsack_problem(values, weights, capacity):
    """背包问题"""
    n = len(values)
    dp = np.zeros((n + 1, capacity + 1))
    
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]
    
    # 回溯找出选择的物品
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected.append(i-1)
            w -= weights[i-1]
    selected.reverse()
    
    return dp[n][capacity], selected


def investment_problem(returns, budget):
    """投资组合问题"""
    n = len(returns)
    dp = np.zeros(budget + 1)
    
    for i in range(n):
        for w in range(budget, 0, -1):
            for k in range(1, w + 1):
                if dp[w - k] + returns[i] * k > dp[w]:
                    dp[w] = dp[w - k] + returns[i] * k
    
    return dp[budget]


def resource_allocation_problem(resources, projects):
    """资源分配问题"""
    n = len(projects)
    m = resources
    dp = np.zeros((n + 1, m + 1))
    
    for i in range(1, n + 1):
        for r in range(1, m + 1):
            max_value = 0
            for k in range(r + 1):
                value = projects[i-1][k]
                if value > max_value:
                    max_value = value
            dp[i][r] = max(dp[i-1][r], max_value)
    
    return dp[n][m]


def dynamic_programming_model(data, problem_type='knapsack'):
    """动态规划模型"""
    print(f"执行动态规划模型 - {problem_type}问题...")
    
    if problem_type == 'knapsack':
        # 背包问题：values, weights, capacity
        values = data['value'].values
        weights = data['weight'].values
        capacity = int(data['capacity'].iloc[0])
        max_value, selected = knapsack_problem(values, weights, capacity)
        print(f"最大价值: {max_value}")
        print(f"选择的物品: {selected}")
        return {'max_value': max_value, 'selected': selected}
    
    elif problem_type == 'investment':
        # 投资问题：returns, budget
        returns = data['return_rate'].values
        budget = int(data['budget'].iloc[0])
        max_return = investment_problem(returns, budget)
        print(f"最大回报: {max_return}")
        return {'max_return': max_return}
    
    elif problem_type == 'resource_allocation':
        # 资源分配问题：resources, projects
        resources = int(data['resources'].iloc[0])
        projects = []
        for col in data.columns:
            if col != 'resources':
                projects.append(data[col].values)
        max_value = resource_allocation_problem(resources, projects)
        print(f"最大价值: {max_value}")
        return {'max_value': max_value}


def save_results(results, output_path):
    """保存结果"""
    df = pd.DataFrame([results])
    df.to_csv(output_path, index=False)
    print(f"结果已保存到: {output_path}")


def main():
    args = parse_args()
    data = load_data(args.input)
    results = dynamic_programming_model(data, args.problem_type)
    save_results(results, args.output)


if __name__ == "__main__":
    main()