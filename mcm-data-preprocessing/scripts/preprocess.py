import argparse
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from scipy.stats import normaltest, levene, shapiro
from statsmodels.tsa.stattools import adfuller
from scipy.spatial.distance import cdist


def parse_args():
    parser = argparse.ArgumentParser(description='MCM data preprocessing')
    parser.add_argument('--input', type=str, required=True, help='Input file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--file_format', type=str, default='csv', choices=['csv', 'excel'], help='Input file format')
    parser.add_argument('--n_components', type=int, default=5, help='Number of PCA components')
    parser.add_argument('--feature_selection', type=bool, default=False, help='Enable feature selection')
    parser.add_argument('--visualization', type=bool, default=False, help='Enable data visualization')
    parser.add_argument('--optimization', type=bool, default=False, help='Enable optimization-specific preprocessing')
    parser.add_argument('--statistics', type=bool, default=False, help='Enable statistics-specific preprocessing')
    parser.add_argument('--normalization', type=str, default='none', choices=['none', 'standard', 'minmax', 'robust'], help='Normalization method')
    parser.add_argument('--encoding', type=str, default='utf-8', help='File encoding')
    parser.add_argument('--sep', type=str, default=None, help='File separator')
    return parser.parse_args()


def load_data(file_path, file_format='csv', encoding='utf-8', sep=None):
    """加载数据"""
    if file_format == 'csv':
        return pd.read_csv(file_path, encoding=encoding, sep=sep)
    elif file_format == 'excel':
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_format}")


def clean_data(df):
    """数据清洗"""
    # 处理缺失值（只删除完全为空的行）
    df = df.dropna(how='all')
    # 不处理异常值，因为对于门票收入数据，异常值可能是有意义的
    return df


def feature_engineering(df):
    """特征工程"""
    # 1. 比赛相关特征
    if 'serve_speed' in df.columns:
        df['serve_efficiency'] = df['first_serve_in'] / df['first_serve_attempts']
    if 'rally_length' in df.columns:
        df['rally_intensity'] = df['rally_length'] * df['shot_velocity']
    
    # 2. 时间特征
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        except:
            pass
    
    # 3. 统计特征
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['score', 'hour', 'day_of_week', 'is_weekend']:
            # 滚动统计特征（如果有时间序列）
            if 'timestamp' in df.columns:
                try:
                    df = df.sort_values('timestamp')
                    df[f'{col}_rolling_mean_3'] = df[col].rolling(window=3).mean()
                    df[f'{col}_rolling_std_3'] = df[col].rolling(window=3).std()
                except:
                    pass
    
    # 4. 交互特征
    if len(numeric_cols) >= 2:
        # 选择前几个重要的数值特征生成交互项
        for i in range(min(3, len(numeric_cols))):
            for j in range(i+1, min(5, len(numeric_cols))):
                col1 = numeric_cols[i]
                col2 = numeric_cols[j]
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
    
    return df


def feature_selection(df, target_col='score'):
    """特征选择"""
    # 选择数值型特征
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 移除目标变量
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    X = df[numeric_cols]
    
    # 1. 方差阈值选择
    var_thresh = VarianceThreshold(threshold=0.01)
    X_var = var_thresh.fit_transform(X)
    selected_by_var = [col for col, selected in zip(numeric_cols, var_thresh.get_support()) if selected]
    
    # 2. 相关性选择
    if target_col in df.columns:
        y = df[target_col]
        corr = X.corrwith(y).abs()
        selected_by_corr = corr[corr > 0.1].index.tolist()
    else:
        selected_by_corr = selected_by_var
    
    # 3. 结合两种方法的结果
    selected_features = list(set(selected_by_var) & set(selected_by_corr))
    
    # 如果没有特征被选择，返回原始特征
    if not selected_features:
        selected_features = numeric_cols
    
    # 保留选择的特征和目标变量
    final_cols = selected_features.copy()
    if target_col in df.columns:
        final_cols.append(target_col)
    
    # 保留非数值型特征
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    final_cols.extend(non_numeric_cols)
    
    return df[final_cols], selected_features

def normalize_data(df, method='none'):
    """数据归一化/标准化"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols]
    
    if method == 'standard':
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)
        print("使用StandardScaler进行数据标准化")
    elif method == 'minmax':
        scaler = MinMaxScaler()
        X_normalized = scaler.fit_transform(X)
        print("使用MinMaxScaler进行数据归一化")
    elif method == 'robust':
        scaler = RobustScaler()
        X_normalized = scaler.fit_transform(X)
        print("使用RobustScaler进行数据标准化")
    else:
        return df
    
    normalized_df = df.copy()
    for i, col in enumerate(numeric_cols):
        normalized_df[f'{col}_normalized'] = X_normalized[:, i]
    
    return normalized_df


def dimensionality_reduction(df, n_components=5):
    """降维处理"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols]
    # 删除包含NaN值的行
    X = X.dropna()
    if len(X) == 0:
        # 如果所有行都包含NaN值，返回原始数据
        return df
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=[f'pca_{i+1}' for i in range(n_components)])
    # 确保索引匹配
    pca_df.index = X.index
    return pd.concat([df, pca_df], axis=1)


def data_visualization(df, output_dir='visualizations'):
    """数据可视化"""
    import os
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 数据分布可视化
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        plt.figure(figsize=(12, 10))
        for i, col in enumerate(numeric_cols[:9]):  # 最多显示9个特征
            plt.subplot(3, 3, i+1)
            sns.histplot(df[col], kde=True)
            plt.title(f'{col} 分布')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'data_distribution.png'))
        plt.close()
    
    # 2. 特征相关性可视化
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('特征相关性矩阵')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_correlation.png'))
        plt.close()
    
    # 3. 目标变量与特征关系可视化（如果有score列）
    if 'score' in df.columns:
        plt.figure(figsize=(12, 10))
        for i, col in enumerate(numeric_cols[:9]):
            if col != 'score':
                plt.subplot(3, 3, i+1)
                sns.scatterplot(x=df[col], y=df['score'])
                plt.title(f'{col} 与 score 的关系')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_target_relationship.png'))
        plt.close()
    
    # 4. 交互可视化（使用plotly）
    if 'score' in df.columns and len(numeric_cols) >= 2:
        fig = px.scatter_matrix(df, dimensions=numeric_cols[:5], color='score', 
                                title='特征散点矩阵')
        fig.write_html(os.path.join(output_dir, 'scatter_matrix.html'))
    
    print(f"数据可视化结果已保存到 {output_dir} 目录")


def optimization_preprocessing(df):
    """适应运筹学方法的预处理"""
    print("执行优化问题专用预处理...")
    
    # 1. 提取约束条件信息
    constraints = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # 2. 识别可能的目标函数和约束条件列
    potential_objectives = []
    potential_constraints = []
    
    for col in numeric_cols:
        # 基于列名识别目标函数和约束条件
        if any(keyword in col.lower() for keyword in ['profit', 'cost', 'revenue', 'objective', 'score']):
            potential_objectives.append(col)
        elif any(keyword in col.lower() for keyword in ['constraint', 'limit', 'capacity', 'requirement']):
            potential_constraints.append(col)
    
    # 3. 计算权重（用于多目标优化）
    weights = {}
    if len(potential_objectives) > 1:
        # 简单权重计算：基于标准差
        for obj in potential_objectives:
            std = df[obj].std()
            weights[obj] = 1 / std if std > 0 else 1
        # 归一化权重
        total_weight = sum(weights.values())
        for obj in weights:
            weights[obj] /= total_weight
    
    # 4. 检查数据范围和可行性
    feasibility = {
        'min_values': df[numeric_cols].min().to_dict(),
        'max_values': df[numeric_cols].max().to_dict(),
        'mean_values': df[numeric_cols].mean().to_dict(),
        'has_negative': any(df[col].min() < 0 for col in numeric_cols)
    }
    
    # 5. 离散化连续变量（如果需要）
    # 这里可以添加离散化逻辑
    
    return {
        'potential_objectives': potential_objectives,
        'potential_constraints': potential_constraints,
        'weights': weights,
        'feasibility': feasibility
    }


def statistics_preprocessing(df):
    """适应传统统计学方法的预处理"""
    print("执行统计问题专用预处理...")
    
    # 1. 分布检验
    distribution_tests = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # 正态分布检验
        stat, p_value = normaltest(df[col].dropna())
        distribution_tests[col] = {
            'normality_p_value': float(p_value),
            'is_normal': bool(p_value > 0.05)
        }
    
    # 2. 方差齐性检验（如果有分组变量）
    homogeneity_tests = {}
    # 检查是否有分组变量
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for cat_col in categorical_cols:
        if df[cat_col].nunique() > 1:
            # 对每个数值列进行方差齐性检验
            for num_col in numeric_cols:
                groups = [group[num_col].values for name, group in df.groupby(cat_col) if len(group) > 1]
                if len(groups) > 1:
                    stat, p_value = levene(*groups)
                    homogeneity_tests[f'{cat_col}_{num_col}'] = {
                        'levene_p_value': float(p_value),
                        'homogeneous_variance': bool(p_value > 0.05)
                    }
    
    # 3. 相关性分析
    correlation_matrix = {}
    corr_matrix = df[numeric_cols].corr()
    for col1 in numeric_cols:
        correlation_matrix[col1] = {}
        for col2 in numeric_cols:
            correlation_matrix[col1][col2] = float(corr_matrix.loc[col1, col2])
    
    # 4. 异常值检测（更详细）
    outliers = {}
    for col in numeric_cols:
        Q1 = float(df[col].quantile(0.25))
        Q3 = float(df[col].quantile(0.75))
        IQR = Q3 - Q1
        lower_bound = float(Q1 - 1.5 * IQR)
        upper_bound = float(Q3 + 1.5 * IQR)
        outlier_count = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
        outlier_percentage = float(outlier_count / len(df) * 100)
        outliers[col] = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count': outlier_count,
            'outlier_percentage': outlier_percentage
        }
    
    return {
        'distribution_tests': distribution_tests,
        'homogeneity_tests': homogeneity_tests,
        'correlation_matrix': correlation_matrix,
        'outliers': outliers
    }


def calculate_indicators(df):
    """计算关键指标"""
    indicators = {}
    # 数据量大小
    data_size = len(df)
    if data_size < 200:
        indicators['data_size'] = '小'
    elif data_size < 500:
        indicators['data_size'] = '中'
    else:
        indicators['data_size'] = '大'
    # 特征维度
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_dim = len(numeric_cols)
    indicators['feature_dim'] = '低' if feature_dim < 10 else '高'
    # 序列平稳性（示例：检查某个时间序列特征）
    if 'timestamp' in df.columns and 'score' in df.columns:
        df_sorted = df.sort_values('timestamp')
        result = adfuller(df_sorted['score'])
        indicators['stationarity'] = '是' if result[1] < 0.05 else '否'
    else:
        indicators['stationarity'] = '否'
    # 动量趋势（示例：检查是否存在连续得分）
    if 'score' in df.columns:
        df['score_change'] = df['score'].diff()
        momentum = (df['score_change'] > 0).sum() > len(df) * 0.6
        indicators['momentum'] = '是' if momentum else '否'
    else:
        indicators['momentum'] = '否'
    # 比赛类型
    if 'gender' in df.columns:
        if 'M' in df['gender'].values:
            indicators['match_type'] = '男单'
        elif 'F' in df['gender'].values:
            indicators['match_type'] = '女单'
        else:
            indicators['match_type'] = '其他'
    else:
        indicators['match_type'] = '其他'
    return indicators


def calculate_optimization_indicators(df):
    """计算运筹学方法需要的指标"""
    print("计算优化问题指标...")
    
    # 执行优化预处理
    opt_info = optimization_preprocessing(df)
    
    indicators = {
        'problem_type': 'optimization',
        'n_objectives': len(opt_info['potential_objectives']),
        'n_constraints': len(opt_info['potential_constraints']),
        'has_weights': len(opt_info['weights']) > 0,
        'data_feasibility': '可行' if not opt_info['feasibility']['has_negative'] else '需检查',
        'variable_types': {
            'continuous': len(df.select_dtypes(include=[np.number]).columns),
            'categorical': len(df.select_dtypes(include=['object', 'category']).columns)
        }
    }
    
    return indicators


def calculate_statistics_indicators(df):
    """计算传统统计学方法需要的指标"""
    print("计算统计问题指标...")
    
    # 执行统计预处理
    stat_info = statistics_preprocessing(df)
    
    # 计算正态分布变量比例
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    normal_count = sum(1 for col in numeric_cols if stat_info['distribution_tests'][col]['is_normal'])
    normal_ratio = normal_count / len(numeric_cols) if len(numeric_cols) > 0 else 0
    
    # 计算异常值比例
    n_outliers = sum(info['outlier_count'] for info in stat_info['outliers'].values())
    outlier_ratio = sum(info['outlier_percentage'] for info in stat_info['outliers'].values()) / len(stat_info['outliers']) if stat_info['outliers'] else 0
    
    # 计算高相关变量对数量
    n_correlated_pairs = sum(1 for i, col1 in enumerate(numeric_cols) for col2 in numeric_cols[i+1:] if abs(df[col1].corr(df[col2])) > 0.7)
    
    # 计算数据质量
    data_quality = '良好' if normal_ratio > 0.5 and outlier_ratio < 5 else '需处理'
    
    indicators = {
        'problem_type': 'statistics',
        'n_normal_variables': normal_count,
        'normal_variable_ratio': normal_ratio,
        'n_outliers': n_outliers,
        'outlier_ratio': outlier_ratio,
        'n_correlated_pairs': n_correlated_pairs,
        'data_quality': data_quality
    }
    
    return indicators


def main():
    args = parse_args()
    
    # 1. 加载数据
    print("加载数据...")
    df = load_data(args.input, args.file_format, args.encoding, args.sep)
    print(f"原始数据形状: {df.shape}")
    
    # 2. 数据清洗
    print("数据清洗...")
    df_cleaned = clean_data(df)
    print(f"清洗后数据形状: {df_cleaned.shape}")
    
    # 3. 特征工程
    print("特征工程...")
    df_features = feature_engineering(df_cleaned)
    print(f"特征工程后数据形状: {df_features.shape}")
    
    # 4. 数据归一化
    if args.normalization != 'none':
        print("数据归一化...")
        df_normalized = normalize_data(df_features, args.normalization)
        print(f"归一化后数据形状: {df_normalized.shape}")
    else:
        df_normalized = df_features
    
    # 5. 特征选择
    if args.feature_selection:
        print("特征选择...")
        df_selected, selected_features = feature_selection(df_normalized)
        print(f"特征选择后数据形状: {df_selected.shape}")
        print(f"选择的特征数: {len(selected_features)}")
    else:
        df_selected = df_normalized
    
    # 6. 降维处理
    print("降维处理...")
    df_reduced = dimensionality_reduction(df_selected, args.n_components)
    print(f"降维后数据形状: {df_reduced.shape}")
    
    # 7. 数据可视化
    if args.visualization:
        print("数据可视化...")
        data_visualization(df_reduced)
    
    # 8. 计算关键指标
    print("计算关键指标...")
    base_indicators = calculate_indicators(df_reduced)
    
    # 9. 优化问题专用预处理和指标
    if args.optimization:
        opt_indicators = calculate_optimization_indicators(df_reduced)
        # 合并指标
        indicators = {**base_indicators, **opt_indicators}
        # 保存优化专用信息
        opt_info = optimization_preprocessing(df_reduced)
        with open('optimization_info.json', 'w', encoding='utf-8') as f:
            json.dump(opt_info, f, ensure_ascii=False, indent=2)
    # 10. 统计问题专用预处理和指标
    elif args.statistics:
        stat_indicators = calculate_statistics_indicators(df_reduced)
        # 合并指标
        indicators = {**base_indicators, **stat_indicators}
        # 保存统计专用信息
        stat_info = statistics_preprocessing(df_reduced)
        with open('statistics_info.json', 'w', encoding='utf-8') as f:
            json.dump(stat_info, f, ensure_ascii=False, indent=2)
    else:
        indicators = base_indicators
    
    # 11. 保存结果
    print("保存结果...")
    df_reduced.to_csv(args.output, index=False)
    # 保存指标到JSON文件
    with open('indicators.json', 'w', encoding='utf-8') as f:
        json.dump(indicators, f, ensure_ascii=False, indent=2)
    
    print("\n预处理完成！")
    print("关键指标：")
    for key, value in indicators.items():
        print(f"{key}: {value}")
    
    print("\n输出文件：")
    print(f"1. 处理后的数据：{args.output}")
    print("2. 关键指标：indicators.json")
    if args.optimization:
        print("3. 优化问题信息：optimization_info.json")
    elif args.statistics:
        print("3. 统计问题信息：statistics_info.json")
    if args.visualization:
        print("4. 可视化结果：visualizations/ 目录")


if __name__ == "__main__":
    main()
