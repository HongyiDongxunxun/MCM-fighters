import argparse
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from scipy.stats import adfuller


def parse_args():
    parser = argparse.ArgumentParser(description='MCM data preprocessing')
    parser.add_argument('--input', type=str, required=True, help='Input file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--file_format', type=str, default='csv', choices=['csv', 'excel'], help='Input file format')
    parser.add_argument('--n_components', type=int, default=5, help='Number of PCA components')
    parser.add_argument('--feature_selection', type=bool, default=False, help='Enable feature selection')
    parser.add_argument('--visualization', type=bool, default=False, help='Enable data visualization')
    return parser.parse_args()


def load_data(file_path, file_format='csv'):
    """加载数据"""
    if file_format == 'csv':
        return pd.read_csv(file_path)
    elif file_format == 'excel':
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_format}")


def clean_data(df):
    """数据清洗"""
    # 处理缺失值
    df = df.dropna()
    # 处理异常值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
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

def dimensionality_reduction(df, n_components=5):
    """降维处理"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=[f'pca_{i+1}' for i in range(n_components)])
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


def main():
    args = parse_args()
    
    # 1. 加载数据
    print("加载数据...")
    df = load_data(args.input, args.file_format)
    print(f"原始数据形状: {df.shape}")
    
    # 2. 数据清洗
    print("数据清洗...")
    df_cleaned = clean_data(df)
    print(f"清洗后数据形状: {df_cleaned.shape}")
    
    # 3. 特征工程
    print("特征工程...")
    df_features = feature_engineering(df_cleaned)
    print(f"特征工程后数据形状: {df_features.shape}")
    
    # 4. 特征选择
    if args.feature_selection:
        print("特征选择...")
        df_features, selected_features = feature_selection(df_features)
        print(f"特征选择后数据形状: {df_features.shape}")
        print(f"选择的特征数: {len(selected_features)}")
    
    # 5. 降维处理
    print("降维处理...")
    df_reduced = dimensionality_reduction(df_features, args.n_components)
    print(f"降维后数据形状: {df_reduced.shape}")
    
    # 6. 数据可视化
    if args.visualization:
        print("数据可视化...")
        data_visualization(df_reduced)
    
    # 7. 计算关键指标
    print("计算关键指标...")
    indicators = calculate_indicators(df_reduced)
    
    # 8. 保存结果
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
    if args.visualization:
        print("3. 可视化结果：visualizations/ 目录")


if __name__ == "__main__":
    main()
