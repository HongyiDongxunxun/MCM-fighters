import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def parse_args():
    parser = argparse.ArgumentParser(description='PCA-TOPSIS model')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    return parser.parse_args()


def topsis(data, weights=None):
    """TOPSIS 方法实现"""
    # 标准化数据
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    # 确定权重
    if weights is None:
        weights = np.ones(data_scaled.shape[1]) / data_scaled.shape[1]
    # 计算加权标准化矩阵
    weighted_data = data_scaled * weights
    # 确定正理想解和负理想解
    ideal_positive = np.max(weighted_data, axis=0)
    ideal_negative = np.min(weighted_data, axis=0)
    # 计算距离
    distance_positive = np.sqrt(np.sum((weighted_data - ideal_positive)**2, axis=1))
    distance_negative = np.sqrt(np.sum((weighted_data - ideal_negative)**2, axis=1))
    # 计算贴近度
    closeness = distance_negative / (distance_positive + distance_negative)
    return closeness


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    # 选择数值型特征
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols]
    # PCA 降维
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X)
    # 应用 TOPSIS
    closeness = topsis(X_pca)
    # 添加排名
    df['closeness'] = closeness
    df['rank'] = df['closeness'].rank(ascending=False)
    # 保存结果
    df.to_csv(args.output, index=False)
    print("PCA-TOPSIS 模型分析完成！")
    print(f"分析结果已保存到 {args.output}")


if __name__ == "__main__":
    main()
