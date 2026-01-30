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
| 任意 | 任意 | Linear Programming | 资源优化、生产计划、运输问题 |
| 任意 | 任意 | Dynamic Programming | 背包问题、投资组合、资源分配 |
| 任意 | 任意 | Analytic Hierarchy Process (AHP) | 多准则决策、方案排序、权重确定 |
| 任意 | 任意 | Hypothesis Testing | 显著性检验、差异分析、假设验证 |
| 时间序列 | 任意 | Time Series Analysis (ARIMA) | 时间序列预测、趋势分析、季节性分析 |

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

# 运行线性规划模型
python scripts/linear_programming.py --input lp_data.csv --output lp_results.csv --objective min

# 运行动态规划模型
python scripts/dynamic_programming.py --input dp_data.csv --output dp_results.csv --problem_type knapsack

# 运行层次分析法模型
python scripts/ahp_model.py --input ahp_data.csv --output ahp_results.csv

# 运行假设检验模型
python scripts/hypothesis_testing.py --input test_data.csv --output test_results.csv --test_type t-test

# 运行时间序列分析模型
python scripts/time_series_analysis.py --input ts_data.csv --output ts_results.csv --model arima --order 1,1,1
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

### 线性规划模型脚本 (scripts/linear_programming.py)

```python
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
```

### 动态规划模型脚本 (scripts/dynamic_programming.py)

```python
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
```

### 层次分析法模型脚本 (scripts/ahp_model.py)

```python
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
```

### 假设检验模型脚本 (scripts/hypothesis_testing.py)

```python
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
```

### 时间序列分析模型脚本 (scripts/time_series_analysis.py)

```python
import argparse
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Time Series Analysis model for MCM')
    parser.add_argument('--input', type=str, required=True, help='Input data file path')
    parser.add_argument('--output', type=str, required=True, help='Output result file path')
    parser.add_argument('--model', type=str, choices=['arima', 'ma', 'ar'], default='arima', help='Time series model type')
    parser.add_argument('--order', type=str, default='1,1,1', help='ARIMA model order (p,d,q)')
    parser.add_argument('--forecast_steps', type=int, default=5, help='Number of steps to forecast')
    return parser.parse_args()


def load_data(file_path):
    """加载数据"""
    return pd.read_csv(file_path)


def adf_test(series):
    """单位根检验"""
    print("执行单位根检验(ADF)...)
")
    result = adfuller(series, autolag='AIC')
    print(f"ADF统计量: {result[0]:.4f}")
    print(f"p值: {result[1]:.4f}")
    print(f"滞后阶数: {result[2]}")
    print(f"观测值数量: {result[3]}")
    print("临界值:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.4f}")
    print(f"显著性水平α=0.05时, {'拒绝' if result[1] < 0.05 else '不拒绝'}原假设")
    print(f"时间序列{'是' if result[1] < 0.05 else '不是'}平稳的")
    return result


def arima_model(data, order=(1, 1, 1), forecast_steps=5):
    """ARIMA模型"""
    print(f"执行ARIMA模型，阶数: {order}...")
    
    # 提取时间序列数据
    if data.shape[1] == 1:
        series = data.iloc[:, 0].values
    else:
        series = data.iloc[:, 1].values
    
    # 执行单位根检验
    adf_result = adf_test(series)
    
    # 拟合ARIMA模型
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    
    print("\n模型拟合结果:")
    print(model_fit.summary())
    
    # 模型诊断
    residuals = model_fit.resid
    print(f"\n残差均值: {np.mean(residuals):.4f}")
    print(f"残差标准差: {np.std(residuals):.4f}")
    
    # 预测
    forecast = model_fit.forecast(steps=forecast_steps)
    print(f"\n预测未来{forecast_steps}步:")
    print(forecast)
    
    return {
        'model_type': 'ARIMA',
        'order': order,
        'forecast': forecast.tolist(),
        'residuals_mean': np.mean(residuals),
        'residuals_std': np.std(residuals),
        'adf_pvalue': adf_result[1]
    }


def ma_model(data, q=1, forecast_steps=5):
    """MA模型"""
    print(f"执行MA模型，阶数: {q}...")
    return arima_model(data, order=(0, 0, q), forecast_steps=forecast_steps)


def ar_model(data, p=1, forecast_steps=5):
    """AR模型"""
    print(f"执行AR模型，阶数: {p}...")
    return arima_model(data, order=(p, 0, 0), forecast_steps=forecast_steps)


def time_series_analysis_model(data, model_type='arima', order=(1, 1, 1), forecast_steps=5):
    """时间序列分析模型"""
    if model_type == 'arima':
        return arima_model(data, order=order, forecast_steps=forecast_steps)
    elif model_type == 'ma':
        q = order[2]
        return ma_model(data, q=q, forecast_steps=forecast_steps)
    elif model_type == 'ar':
        p = order[0]
        return ar_model(data, p=p, forecast_steps=forecast_steps)


def save_results(results, output_path):
    """保存结果"""
    # 保存预测结果
    forecast_df = pd.DataFrame({
        'forecast': results['forecast']
    })
    forecast_df.to_csv(output_path, index_label='step')
    print(f"\n预测结果已保存到: {output_path}")


def main():
    args = parse_args()
    data = load_data(args.input)
    
    # 解析order参数
    order = tuple(map(int, args.order.split(',')))
    
    results = time_series_analysis_model(data, args.model, order, args.forecast_steps)
    save_results(results, args.output)


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
