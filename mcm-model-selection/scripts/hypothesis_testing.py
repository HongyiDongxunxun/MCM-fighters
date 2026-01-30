import argparse
import pandas as pd
import numpy as np
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser(description='Hypothesis Testing model for MCM')
    parser.add_argument('--input', type=str, required=True, help='Input data file path')
    parser.add_argument('--output', type=str, required=True, help='Output result file path')
    parser.add_argument('--test_type', type=str, choices=['t-test', 'z-test', 'chi-square', 'f-test'], default='t-test', help='Type of hypothesis test')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')
    return parser.parse_args()


def load_data(file_path):
    """加载数据"""
    return pd.read_csv(file_path)


def t_test(data, alpha=0.05):
    """t检验"""
    # 单样本t检验
    if data.shape[1] == 1:
        sample = data.iloc[:, 0].values
        # 假设总体均值为0
        t_stat, p_value = stats.ttest_1samp(sample, 0)
        print(f"单样本t检验: t统计量 = {t_stat:.4f}, p值 = {p_value:.4f}")
        
        # 双侧检验
        reject_null = p_value < alpha
        print(f"显著性水平α = {alpha}, {'拒绝' if reject_null else '不拒绝'}原假设")
        
        return {
            'test_type': 'one-sample t-test',
            't_statistic': t_stat,
            'p_value': p_value,
            'alpha': alpha,
            'reject_null': reject_null
        }
    
    # 两样本t检验
    elif data.shape[1] == 2:
        sample1 = data.iloc[:, 0].values
        sample2 = data.iloc[:, 1].values
        
        # 方差齐性检验
        levene_stat, levene_p = stats.levene(sample1, sample2)
        equal_var = levene_p > alpha
        
        t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=equal_var)
        print(f"两样本t检验: t统计量 = {t_stat:.4f}, p值 = {p_value:.4f}")
        print(f"方差齐性检验: {'通过' if equal_var else '未通过'}")
        
        reject_null = p_value < alpha
        print(f"显著性水平α = {alpha}, {'拒绝' if reject_null else '不拒绝'}原假设")
        
        return {
            'test_type': 'two-sample t-test',
            't_statistic': t_stat,
            'p_value': p_value,
            'alpha': alpha,
            'reject_null': reject_null,
            'equal_variance': equal_var
        }


def z_test(data, alpha=0.05, population_std=None):
    """z检验"""
    # 单样本z检验
    if data.shape[1] == 1:
        sample = data.iloc[:, 0].values
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)
        n = len(sample)
        
        # 如果没有提供总体标准差，使用样本标准差
        if population_std is None:
            population_std = sample_std
        
        # 假设总体均值为0
        z_stat = (sample_mean - 0) / (population_std / np.sqrt(n))
        # 双侧检验
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
        
        print(f"单样本z检验: z统计量 = {z_stat:.4f}, p值 = {p_value:.4f}")
        
        reject_null = p_value < alpha
        print(f"显著性水平α = {alpha}, {'拒绝' if reject_null else '不拒绝'}原假设")
        
        return {
            'test_type': 'one-sample z-test',
            'z_statistic': z_stat,
            'p_value': p_value,
            'alpha': alpha,
            'reject_null': reject_null
        }


def chi_square_test(data, alpha=0.05):
    """卡方检验"""
    # 卡方独立性检验
    if data.shape[1] == 2:
        # 创建列联表
        contingency_table = pd.crosstab(data.iloc[:, 0], data.iloc[:, 1])
        print(f"列联表:\n{contingency_table}")
        
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"卡方检验: 卡方统计量 = {chi2_stat:.4f}, p值 = {p_value:.4f}, 自由度 = {dof}")
        
        reject_null = p_value < alpha
        print(f"显著性水平α = {alpha}, {'拒绝' if reject_null else '不拒绝'}原假设")
        
        return {
            'test_type': 'chi-square independence test',
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'alpha': alpha,
            'reject_null': reject_null
        }


def f_test(data, alpha=0.05):
    """F检验"""
    # 两样本方差齐性检验
    if data.shape[1] == 2:
        sample1 = data.iloc[:, 0].values
        sample2 = data.iloc[:, 1].values
        
        f_stat, p_value = stats.f_oneway(sample1, sample2)
        print(f"F检验: F统计量 = {f_stat:.4f}, p值 = {p_value:.4f}")
        
        reject_null = p_value < alpha
        print(f"显著性水平α = {alpha}, {'拒绝' if reject_null else '不拒绝'}原假设")
        
        return {
            'test_type': 'f-test (one-way ANOVA)',
            'f_statistic': f_stat,
            'p_value': p_value,
            'alpha': alpha,
            'reject_null': reject_null
        }


def hypothesis_testing_model(data, test_type='t-test', alpha=0.05):
    """假设检验模型"""
    print(f"执行假设检验模型 - {test_type}...")
    
    if test_type == 't-test':
        return t_test(data, alpha)
    elif test_type == 'z-test':
        return z_test(data, alpha)
    elif test_type == 'chi-square':
        return chi_square_test(data, alpha)
    elif test_type == 'f-test':
        return f_test(data, alpha)


def save_results(results, output_path):
    """保存结果"""
    df = pd.DataFrame([results])
    df.to_csv(output_path, index=False)
    print(f"结果已保存到: {output_path}")


def main():
    args = parse_args()
    data = load_data(args.input)
    results = hypothesis_testing_model(data, args.test_type, args.alpha)
    save_results(results, args.output)


if __name__ == "__main__":
    main()