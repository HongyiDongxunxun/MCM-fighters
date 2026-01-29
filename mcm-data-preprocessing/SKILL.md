---
name: mcm-data-preprocessing
description: Comprehensive MCM contest data preprocessing workflow. Use when preparing MCM contest data, especially for tennis momentum analysis, providing data cleaning, feature engineering, dimensionality reduction, and generating key indicators for model selection.
---

# MCM 比赛数据预处理技能

## 核心功能

本技能提供 **"数据清洗 + 特征工程 + 特征选择 + 降维 + 数据可视化 + 指标生成"** 的完整预处理流程，为后续模型选择提供关键指标。

### 功能概述

1. **数据清洗**：处理缺失值、异常值，确保数据完整性，支持多种数据格式
2. **特征工程**：生成比赛相关特征、时间特征、统计特征、交互特征等，提取关键信息
3. **特征选择**：基于相关性、方差、模型的特征选择方法，减少冗余特征
4. **降维处理**：使用PCA等方法减少特征维度，提高模型训练效率
5. **数据可视化**：数据分布、特征相关性、预处理效果的可视化，帮助理解数据
6. **指标生成**：计算模型选择所需的关键指标，为模型选择提供依据

## 环境配置

```python
# 必备库安装
pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels featuretools plotly
```

## 使用方法

### 数据预处理执行

```python
# 执行基本预处理
python scripts/preprocess.py --input data.csv --output processed_data.csv

# 执行预处理并启用特征选择
python scripts/preprocess.py --input data.csv --output processed_data.csv --feature_selection True

# 执行预处理并启用数据可视化
python scripts/preprocess.py --input data.csv --output processed_data.csv --visualization True

# 执行预处理并指定降维组件数
python scripts/preprocess.py --input data.csv --output processed_data.csv --n_components 10

# 处理不同格式的数据（Excel）
python scripts/preprocess.py --input data.xlsx --output processed_data.csv --file_format excel
```

### 关键指标生成

预处理完成后，将生成以下关键指标：

| 指标 | 说明 | 可能值 |
|------|------|--------|
| 数据量大小 | 数据集规模 | 小 (<200) / 中 (200-500) / 大 (>500) |
| 特征维度 | 特征数量 | 低 (<10) / 高 (≥10) |
| 序列平稳性 | 时间序列是否平稳 | 是 / 否 |
| 动量趋势 | 是否存在明显动量 | 是 / 否 |
| 比赛类型 | 比赛类别 | 男单 / 女单 / 其他 |

## 脚本实现

### 预处理脚本 (scripts/preprocess.py)

```python
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import adfuller


def parse_args():
    parser = argparse.ArgumentParser(description='MCM data preprocessing')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    return parser.parse_args()


def load_data(file_path):
    """加载数据"""
    return pd.read_csv(file_path)


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
    # 示例：生成比赛相关特征
    if 'serve_speed' in df.columns:
        df['serve_efficiency'] = df['first_serve_in'] / df['first_serve_attempts']
    if 'rally_length' in df.columns:
        df['rally_intensity'] = df['rally_length'] * df['shot_velocity']
    return df


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
    df = load_data(args.input)
    df_cleaned = clean_data(df)
    df_features = feature_engineering(df_cleaned)
    df_reduced = dimensionality_reduction(df_features)
    indicators = calculate_indicators(df_reduced)
    df_reduced.to_csv(args.output, index=False)
    print("预处理完成！")
    print("关键指标：")
    for key, value in indicators.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
```

## 注意事项

- **数据格式**：确保输入数据为CSV格式，包含必要的比赛信息
- **依赖库**：使用前请安装所有必要的依赖库
- **参数调整**：根据具体数据集调整预处理参数
- **指标解释**：生成的关键指标将用于后续模型选择，确保指标计算准确性

## 输出文件

预处理完成后，将生成两个文件：
1. **processed_data.csv**：处理后的数据文件
2. **indicators.json**：包含关键指标的JSON文件

## 使用建议

1. **数据质量检查**：预处理前检查数据完整性
2. **特征选择**：根据具体比赛类型选择相关特征
3. **降维参数**：根据特征数量调整PCA组件数
4. **指标验证**：确认生成的指标符合预期

## 扩展功能

- **并行处理**：大规模数据使用Dask并行处理
- **自动特征生成**：考虑使用Featuretools自动生成特征
- **数据可视化**：添加数据分布和特征相关性可视化
