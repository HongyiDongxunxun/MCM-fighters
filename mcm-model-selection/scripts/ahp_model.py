import argparse
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Analytic Hierarchy Process (AHP) model for MCM')
    parser.add_argument('--input', type=str, required=True, help='Input data file path')
    parser.add_argument('--output', type=str, required=True, help='Output result file path')
    return parser.parse_args()


def load_data(file_path):
    """加载数据"""
    return pd.read_csv(file_path)


def calculate_weight(matrix):
    """计算权重"""
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    # 找到最大特征值及其对应的特征向量
    max_eigenvalue_index = np.argmax(eigenvalues)
    max_eigenvalue = eigenvalues[max_eigenvalue_index]
    max_eigenvector = eigenvectors[:, max_eigenvalue_index]
    # 归一化特征向量得到权重
    weight = max_eigenvector / np.sum(max_eigenvector)
    return weight.real, max_eigenvalue.real


def consistency_check(matrix, max_eigenvalue, n):
    """一致性检验"""
    if n == 1:
        return True, 0, 0
    
    # 计算一致性指标CI
    ci = (max_eigenvalue - n) / (n - 1)
    
    # 随机一致性指标RI
    ri_dict = {
        1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12,
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
    }
    ri = ri_dict.get(n, 1.5)
    
    # 计算一致性比率CR
    cr = ci / ri
    
    # 判断一致性
    consistent = cr < 0.1
    
    return consistent, ci, cr


def ahp_model(data):
    """层次分析法模型"""
    print("执行层次分析法(AHP)模型...")
    
    # 提取准则层判断矩阵
    criteria_matrix = data.iloc[:, 1:].values.astype(float)
    n_criteria = criteria_matrix.shape[0]
    
    print(f"准则层判断矩阵大小: {n_criteria}x{n_criteria}")
    
    # 计算准则层权重
    criteria_weights, max_eigenvalue = calculate_weight(criteria_matrix)
    
    # 一致性检验
    consistent, ci, cr = consistency_check(criteria_matrix, max_eigenvalue, n_criteria)
    print(f"准则层一致性检验: {'通过' if consistent else '未通过'}")
    print(f"CI: {ci:.4f}, CR: {cr:.4f}")
    
    # 提取方案层判断矩阵
    # 假设数据格式为：方案名称 + 每个准则下的判断矩阵
    n_alternatives = int((data.shape[0] - n_criteria - 1) / n_criteria)
    alternative_matrices = []
    
    for i in range(n_criteria):
        start_row = n_criteria + 1 + i * n_alternatives
        end_row = start_row + n_alternatives
        matrix = data.iloc[start_row:end_row, 1:1+n_alternatives].values.astype(float)
        alternative_matrices.append(matrix)
    
    # 计算方案层权重
    alternative_weights = []
    for i, matrix in enumerate(alternative_matrices):
        weight, ev = calculate_weight(matrix)
        alternative_weights.append(weight)
        # 一致性检验
        alt_consistent, alt_ci, alt_cr = consistency_check(matrix, ev, n_alternatives)
        print(f"方案层-{i+1}一致性检验: {'通过' if alt_consistent else '未通过'}")
        print(f"CI: {alt_ci:.4f}, CR: {alt_cr:.4f}")
    
    # 计算总权重
    alternative_weights = np.array(alternative_weights)
    total_weights = np.dot(criteria_weights, alternative_weights)
    
    # 排序方案
    rankings = np.argsort(total_weights)[::-1] + 1  # 方案编号从1开始
    
    print("\n结果:")
    print(f"准则层权重: {criteria_weights}")
    print(f"方案层权重矩阵:\n{alternative_weights}")
    print(f"总权重: {total_weights}")
    print(f"方案排名: {rankings}")
    
    return {
        'criteria_weights': criteria_weights.tolist(),
        'alternative_weights': alternative_weights.tolist(),
        'total_weights': total_weights.tolist(),
        'rankings': rankings.tolist(),
        'consistent': consistent,
        'ci': ci,
        'cr': cr
    }


def save_results(results, output_path):
    """保存结果"""
    # 保存权重和排名
    weights_df = pd.DataFrame({
        'total_weight': results['total_weights'],
        'ranking': results['rankings']
    })
    weights_df.to_csv(output_path, index_label='alternative')
    print(f"结果已保存到: {output_path}")


def main():
    args = parse_args()
    data = load_data(args.input)
    results = ahp_model(data)
    save_results(results, args.output)


if __name__ == "__main__":
    main()