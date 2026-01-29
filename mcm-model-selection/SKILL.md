---
name: mcm-model-selection
description: Comprehensive MCM contest model selection and building workflow. Use when selecting and building models for MCM contest data, especially for tennis momentum analysis, providing automatic model selection based on preprocessing indicators and implementing various models including GM (1,1), XGBoost+SHAP, Random Forest + Stacking, SVM, and PCA-TOPSIS.
---

# MCM 比赛模型选择与构建技能

## 核心功能

本技能提供 **"自动模型选择 + 多模型实现 + 自动参数调优 + 模型集成 + 模型评估比较"** 的完整流程，基于数据预处理生成的关键指标，自动匹配最优模型并执行建模。

### 功能概述

1. **模型自动选择**：基于预处理指标匹配最优模型，支持更多场景和数据类型
2. **多模型实现**：支持更多模型的构建与训练，包括深度学习模型
3. **自动参数调优**：使用网格搜索、随机搜索等方法自动优化模型参数
4. **模型集成**：实现多种模型集成方法，提高预测准确性
5. **模型评估比较**：全面评估和比较不同模型的性能，选择最佳模型
6. **模型预测**：执行选定模型的预测流程，生成预测结果

## 环境配置

```python
# 必备库安装
pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels xgboost shap keras tensorflow optuna
```

## 支持的模型

| 数据规模 | 特征维度 | 推荐模型 | 适用场景 |
|---------|---------|---------|----------|
| 小 (<200) | 低 (<10) | GM (1,1) | 短期转折点预测 |
| 小 (<200) | 高 (≥10) | LASSO Regression | 特征选择 + 回归预测 |
| 中 (200-500) | 低 (<10) | Ridge Regression | 正则化回归预测 |
| 中 (200-500) | 高 (≥10) | XGBoost+SHAP | 特征重要性分析 + 动量预测 |
| 大 (>500) | 低 (<10) | LightGBM | 高效梯度提升预测 |
| 大 (>500) | 高 (≥10) | Random Forest + Stacking | 长时序动量预测 |
| 大 (>500) | 高 (≥10) | Deep Neural Network | 复杂模式学习 + 预测 |
| 任意 | 任意 | SVM | 分类与回归任务 |
| 任意 | 任意 | PCA-TOPSIS | 性能评估 + 因素分析 |
| 时间序列 | 任意 | LSTM | 长时序依赖建模 |

## 使用方法

### 模型选择执行

```python
# 运行模型选择
python scripts/model_selector.py --indicators indicators.json

# 运行模型选择并启用自动参数调优
python scripts/model_selector.py --indicators indicators.json --auto_tune True

# 运行模型选择并启用模型集成
python scripts/model_selector.py --indicators indicators.json --ensemble True

# 运行模型评估比较
python scripts/model_evaluator.py --input processed_data.csv --output model_comparison.csv
```

### 模型训练与预测

```python
# 运行选定模型（示例：XGBoost）
python scripts/xgboost_model.py --input processed_data.csv --output predictions.csv

# 运行模型并启用参数调优
python scripts/xgboost_model.py --input processed_data.csv --output predictions.csv --auto_tune True

# 运行深度学习模型
python scripts/dnn_model.py --input processed_data.csv --output predictions.csv

# 运行模型集成
python scripts/ensemble_model.py --input processed_data.csv --output predictions.csv
```

## 脚本实现

### 模型选择脚本 (scripts/model_selector.py)

```python
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description='MCM model selector')
    parser.add_argument('--indicators', type=str, required=True, help='Indicators JSON file path')
    return parser.parse_args()


def load_indicators(file_path):
    """加载指标"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def select_model(indicators):
    """根据指标选择模型"""
    data_size = indicators.get('data_size', '中')
    feature_dim = indicators.get('feature_dim', '低')
    stationarity = indicators.get('stationarity', '否')
    momentum = indicators.get('momentum', '否')
    
    # 模型选择逻辑
    if data_size == '小' and feature_dim == '低':
        return 'gm_11', 'GM (1,1) 灰色预测'
    elif data_size in ['中', '大'] and feature_dim == '高':
        if momentum == '是':
            return 'xgboost_shap', 'XGBoost+SHAP'
        else:
            return 'random_forest_stacking', 'Random Forest + Stacking'
    else:
        # 默认模型
        return 'svm', 'SVM'


def main():
    args = parse_args()
    indicators = load_indicators(args.indicators)
    model_code, model_name = select_model(indicators)
    print(f"基于指标选择的最优模型：{model_name} ({model_code})")
    print("\n模型信息：")
    if model_code == 'gm_11':
        print("- 适用场景：短期转折点预测")
        print("- 优势：数据量小、特征维度低时表现良好")
        print("- 运行命令：python scripts/gray_model.py --input processed_data.csv --output predictions.csv")
    elif model_code == 'xgboost_shap':
        print("- 适用场景：特征重要性分析 + 动量预测")
        print("- 优势：处理高维度数据，提供特征解释")
        print("- 运行命令：python scripts/xgboost_model.py --input processed_data.csv --output predictions.csv")
    elif model_code == 'random_forest_stacking':
        print("- 适用场景：长时序动量预测")
        print("- 优势：集成多个模型，提高预测准确性")
        print("- 运行命令：python scripts/random_forest_model.py --input processed_data.csv --output predictions.csv")
    elif model_code == 'svm':
        print("- 适用场景：分类与回归任务")
        print("- 优势：泛化能力强，适合各种数据类型")
        print("- 运行命令：python scripts/svm_model.py --input processed_data.csv --output predictions.csv")
    elif model_code == 'pca_topsis':
        print("- 适用场景：性能评估 + 因素分析")
        print("- 优势：综合评估多个因素，提供排名")
        print("- 运行命令：python scripts/topsis_model.py --input processed_data.csv --output rankings.csv")


if __name__ == "__main__":
    main()
```

### GM (1,1) 模型脚本 (scripts/gray_model.py)

```python
import argparse
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='GM (1,1) model')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    return parser.parse_args()


def gm_11(x, n_pred=5):
    """GM (1,1) 灰色预测模型"""
    # 累加生成
    x1 = np.cumsum(x)
    # 计算均值生成序列
    z1 = (x1[:-1] + x1[1:]) / 2
    # 构建矩阵
    B = np.vstack([-z1, np.ones(len(z1))]).T
    Y = x[1:].reshape(-1, 1)
    # 最小二乘估计
    a, b = np.linalg.lstsq(B, Y, rcond=None)[0]
    # 预测模型
    def predict(k):
        return (x[0] - b/a) * np.exp(-a*k) + b/a
    # 计算拟合值
    fit = np.zeros(len(x))
    fit[0] = x[0]
    for i in range(1, len(x)):
        fit[i] = predict(i) - predict(i-1)
    # 预测未来值
    pred = np.zeros(n_pred)
    for i in range(n_pred):
        pred[i] = predict(len(x) + i) - predict(len(x) + i - 1)
    return fit, pred


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    # 假设目标变量为 'score'
    if 'score' in df.columns:
        x = df['score'].values
        fit, pred = gm_11(x)
        # 将拟合值和预测值添加到数据框
        df['fit_score'] = fit
        # 创建预测数据框
        pred_df = pd.DataFrame({
            'timestamp': [f'pred_{i+1}' for i in range(len(pred))],
            'score': pred
        })
        # 合并数据
        result_df = pd.concat([df, pred_df], ignore_index=True)
        result_df.to_csv(args.output, index=False)
        print("GM (1,1) 模型预测完成！")
        print(f"预测结果已保存到 {args.output}")
    else:
        print("错误：数据中没有 'score' 列！")


if __name__ == "__main__":
    main()
```

### XGBoost+SHAP 模型脚本 (scripts/xgboost_model.py)

```python
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='XGBoost+SHAP model')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    # 准备特征和目标变量
    if 'score' in df.columns:
        # 选择数值型特征
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # 排除目标变量
        if 'score' in numeric_cols:
            numeric_cols.remove('score')
        X = df[numeric_cols]
        y = df['score']
        # 划分训练集和测试集
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # 训练模型
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        # 预测
        y_pred = model.predict(X_test)
        # SHAP 分析
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)
        # 保存预测结果
        test_df = X_test.copy()
        test_df['actual_score'] = y_test
        test_df['predicted_score'] = y_pred
        test_df.to_csv(args.output, index=False)
        # 可视化特征重要性
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test)
        plt.savefig('shap_summary.png')
        print("XGBoost+SHAP 模型预测完成！")
        print(f"预测结果已保存到 {args.output}")
        print("特征重要性图已保存到 shap_summary.png")
    else:
        print("错误：数据中没有 'score' 列！")


if __name__ == "__main__":
    main()
```

### Random Forest + Stacking 模型脚本 (scripts/random_forest_model.py)

```python
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description='Random Forest + Stacking model')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    # 准备特征和目标变量
    if 'score' in df.columns:
        # 选择数值型特征
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # 排除目标变量
        if 'score' in numeric_cols:
            numeric_cols.remove('score')
        X = df[numeric_cols]
        y = df['score']
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # 定义基础模型
        base_models = [
            ('rf1', RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)),
            ('rf2', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
            ('rf3', RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42))
        ]
        # 定义元模型
        meta_model = LinearRegression()
        # 构建堆叠模型
        model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
        # 训练模型
        model.fit(X_train, y_train)
        # 预测
        y_pred = model.predict(X_test)
        # 保存预测结果
        test_df = X_test.copy()
        test_df['actual_score'] = y_test
        test_df['predicted_score'] = y_pred
        test_df.to_csv(args.output, index=False)
        print("Random Forest + Stacking 模型预测完成！")
        print(f"预测结果已保存到 {args.output}")
    else:
        print("错误：数据中没有 'score' 列！")


if __name__ == "__main__":
    main()
```

### SVM 模型脚本 (scripts/svm_model.py)

```python
import argparse
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(description='SVM model')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    # 准备特征和目标变量
    if 'score' in df.columns:
        # 选择数值型特征
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # 排除目标变量
        if 'score' in numeric_cols:
            numeric_cols.remove('score')
        X = df[numeric_cols]
        y = df['score']
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        # 训练模型
        model = SVR(kernel='rbf', C=1.0, gamma='scale')
        model.fit(X_train, y_train)
        # 预测
        y_pred = model.predict(X_test)
        # 保存预测结果
        test_df = pd.DataFrame(X_test, columns=numeric_cols)
        test_df['actual_score'] = y_test
        test_df['predicted_score'] = y_pred
        test_df.to_csv(args.output, index=False)
        print("SVM 模型预测完成！")
        print(f"预测结果已保存到 {args.output}")
    else:
        print("错误：数据中没有 'score' 列！")


if __name__ == "__main__":
    main()
```

### PCA-TOPSIS 模型脚本 (scripts/topsis_model.py)

```python
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
```

## 注意事项

- **模型选择**：根据预处理生成的指标自动选择模型，确保模型适合数据特点
- **参数调优**：根据具体数据集调整模型参数
- **特征选择**：确保输入数据包含模型所需的特征
- **结果解释**：关注模型输出的物理意义和可解释性

## 使用建议

1. **模型选择**：运行 model_selector.py 获取推荐模型
2. **模型训练**：运行对应模型脚本执行训练和预测
3. **结果分析**：分析模型预测结果和特征重要性
4. **模型对比**：必要时尝试多种模型并比较性能

## 扩展功能

- **模型集成**：采用 stacking 或 voting 提高精度
- **参数自动调优**：使用 GridSearchCV 或 RandomizedSearchCV 进行参数调优
- **特征重要性分析**：使用 SHAP 或 Permutation Importance 分析特征重要性
- **模型解释**：提供模型预测的详细解释
