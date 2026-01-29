---
name: mcm-model-validation
description: Comprehensive MCM contest model validation workflow. Use when evaluating models for MCM contest data, especially for tennis momentum analysis, providing regression and classification model validation metrics, cross-validation methods, sensitivity analysis, and model performance evaluation.
---

# MCM 比赛模型效果检验技能

## 核心功能

本技能提供 **"模型验证 + 性能评估 + 敏感性分析"** 的完整流程，全面评估模型的准确性、稳定性和可靠性。

### 功能概述

1. **回归模型验证**：计算 MSE、RMSE、MAE、R² 等指标
2. **分类模型验证**：计算准确率、精确率、召回率、F1 分数等指标
3. **交叉验证**：执行 K 折交叉验证，评估模型稳定性
4. **敏感性分析**：分析模型对参数变化的敏感性

## 环境配置

```python
# 必备库安装
pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels
```

## 验证指标

### 回归模型验证指标

| 指标 | 说明 | 代码实现 |
|-----|------|----------|
| MSE | 均方误差 | `mean_squared_error(y_true, y_pred)` |
| RMSE | 均方根误差 | `sqrt(mean_squared_error(y_true, y_pred))` |
| MAE | 平均绝对误差 | `mean_absolute_error(y_true, y_pred)` |
| R² | 决定系数 | `r2_score(y_true, y_pred)` |
| 平均相对误差 | 相对误差均值 | `mean(abs((y_true - y_pred)/y_true))` |
| 残差分析 | 检验模型假设 | `acorr_ljungbox(residuals, lags=20)` |

### 分类模型验证指标

| 指标 | 说明 | 代码实现 |
|-----|------|----------|
| 准确率 | 正确预测比例 | `accuracy_score(y_true, y_pred)` |
| 精确率 | 正例预测正确比例 | `precision_score(y_true, y_pred)` |
| 召回率 | 实际正例被正确预测比例 | `recall_score(y_true, y_pred)` |
| F1分数 | 精确率和召回率的调和平均 | `f1_score(y_true, y_pred)` |
| 混淆矩阵 | 分类结果可视化 | `confusion_matrix(y_true, y_pred)` |
| ROC曲线 | 分类性能评估 | `roc_auc_score(y_true, y_pred_proba)` |

## 使用方法

### 回归模型验证

```python
# 执行回归模型验证
python scripts/validate_regression.py --input predictions.csv
```

### 分类模型验证

```python
# 执行分类模型验证
python scripts/validate_classification.py --input predictions.csv
```

### 交叉验证

```python
# 执行交叉验证
python scripts/cross_validate.py --input processed_data.csv --model xgboost
```

### 敏感性分析

```python
# 执行敏感性分析
python scripts/sensitivity_analysis.py --input processed_data.csv --model xgboost
```

## 脚本实现

### 回归模型验证脚本 (scripts/validate_regression.py)

```python
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import acorr_ljungbox


def parse_args():
    parser = argparse.ArgumentParser(description='Regression model validation')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path with actual and predicted values')
    return parser.parse_args()


def calculate_metrics(y_true, y_pred):
    """计算回归模型指标"""
    metrics = {}
    # MSE
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    # RMSE
    metrics['rmse'] = np.sqrt(metrics['mse'])
    # MAE
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    # R²
    metrics['r2'] = r2_score(y_true, y_pred)
    # 平均相对误差
    metrics['mean_relative_error'] = np.mean(np.abs((y_true - y_pred) / y_true))
    # 残差分析
    residuals = y_true - y_pred
    metrics['residual_mean'] = np.mean(residuals)
    metrics['residual_std'] = np.std(residuals)
    # Ljung-Box 检验
    if len(residuals) > 20:
        lb_result = acorr_ljungbox(residuals, lags=20)
        metrics['lb_statistic'] = lb_result[0][-1]
        metrics['lb_pvalue'] = lb_result[1][-1]
    return metrics


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    # 检查必要的列
    if 'actual_score' in df.columns and 'predicted_score' in df.columns:
        y_true = df['actual_score'].values
        y_pred = df['predicted_score'].values
        # 计算指标
        metrics = calculate_metrics(y_true, y_pred)
        # 打印结果
        print("回归模型验证结果：")
        print("=" * 50)
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        print("=" * 50)
        # 保存结果
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv('regression_metrics.csv', index=False)
        print("验证结果已保存到 regression_metrics.csv")
    else:
        print("错误：数据中没有 'actual_score' 或 'predicted_score' 列！")


if __name__ == "__main__":
    main()
```

### 分类模型验证脚本 (scripts/validate_classification.py)

```python
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description='Classification model validation')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path with actual and predicted values')
    return parser.parse_args()


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """计算分类模型指标"""
    metrics = {}
    # 准确率
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    # 精确率
    metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    # 召回率
    metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    # F1分数
    metrics['f1_score'] = f1_score(y_true, y_pred, average='macro')
    # 混淆矩阵
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    # ROC AUC 分数
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, average='macro', multi_class='ovr')
        except:
            pass
    return metrics


def plot_confusion_matrix(cm, classes):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    # 检查必要的列
    if 'actual_label' in df.columns and 'predicted_label' in df.columns:
        y_true = df['actual_label'].values
        y_pred = df['predicted_label'].values
        # 检查是否有概率列
        y_pred_proba = None
        if 'predicted_prob' in df.columns:
            y_pred_proba = df['predicted_prob'].values
        # 计算指标
        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
        # 打印结果
        print("分类模型验证结果：")
        print("=" * 50)
        for key, value in metrics.items():
            if key != 'confusion_matrix':
                print(f"{key}: {value:.4f}")
        print("\n混淆矩阵：")
        print(metrics['confusion_matrix'])
        print("=" * 50)
        # 绘制混淆矩阵
        classes = np.unique(np.concatenate([y_true, y_pred]))
        plot_confusion_matrix(metrics['confusion_matrix'], classes)
        # 保存结果
        metrics_df = pd.DataFrame([{
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics.get('roc_auc', np.nan)
        }])
        metrics_df.to_csv('classification_metrics.csv', index=False)
        print("验证结果已保存到 classification_metrics.csv")
        print("混淆矩阵已保存到 confusion_matrix.png")
    else:
        print("错误：数据中没有 'actual_label' 或 'predicted_label' 列！")


if __name__ == "__main__":
    main()
```

### 交叉验证脚本 (scripts/cross_validate.py)

```python
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import xgboost as xgb


def parse_args():
    parser = argparse.ArgumentParser(description='Cross validation')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--model', type=str, default='rf', choices=['rf', 'svm', 'lr', 'xgboost'], help='Model type')
    return parser.parse_args()


def get_model(model_type):
    """获取指定类型的模型"""
    if model_type == 'rf':
        return RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'svm':
        return SVR(kernel='rbf', C=1.0, gamma='scale')
    elif model_type == 'lr':
        return LinearRegression()
    elif model_type == 'xgboost':
        return xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    else:
        raise ValueError(f"未知模型类型: {model_type}")


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
        # 获取模型
        model = get_model(args.model)
        # 执行 K 折交叉验证
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
        # 打印结果
        print(f"{args.model.upper()} 模型交叉验证结果：")
        print("=" * 50)
        print(f"交叉验证 R² 得分: {scores.mean():.4f} ± {scores.std():.4f}")
        print(f"各折得分: {[f'{s:.4f}' for s in scores]}")
        print("=" * 50)
        # 保存结果
        cv_df = pd.DataFrame({
            'model': [args.model],
            'mean_r2': [scores.mean()],
            'std_r2': [scores.std()],
            'fold_1': [scores[0]],
            'fold_2': [scores[1]],
            'fold_3': [scores[2]],
            'fold_4': [scores[3]],
            'fold_5': [scores[4]]
        })
        cv_df.to_csv('cross_validation_results.csv', index=False)
        print("交叉验证结果已保存到 cross_validation_results.csv")
    else:
        print("错误：数据中没有 'score' 列！")


if __name__ == "__main__":
    main()
```

### 敏感性分析脚本 (scripts/sensitivity_analysis.py)

```python
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import xgboost as xgb


def parse_args():
    parser = argparse.ArgumentParser(description='Sensitivity analysis')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--model', type=str, default='rf', choices=['rf', 'svm', 'lr', 'xgboost'], help='Model type')
    return parser.parse_args()


def get_model(model_type):
    """获取指定类型的模型"""
    if model_type == 'rf':
        return RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'svm':
        return SVR(kernel='rbf', C=1.0, gamma='scale')
    elif model_type == 'lr':
        return LinearRegression()
    elif model_type == 'xgboost':
        return xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    else:
        raise ValueError(f"未知模型类型: {model_type}")


def sensitivity_analysis(model, X, y, feature_names):
    """执行敏感性分析"""
    # 训练模型
    model.fit(X, y)
    # 计算特征重要性
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        importances = np.ones(X.shape[1]) / X.shape[1]
    # 归一化重要性
    importances = importances / np.sum(importances)
    # 创建重要性字典
    importance_dict = dict(zip(feature_names, importances))
    # 按重要性排序
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_importance


def plot_sensitivity(sorted_importance):
    """绘制敏感性分析结果"""
    features, importances = zip(*sorted_importance)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances, tick_label=features)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance (Sensitivity Analysis)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('sensitivity_analysis.png')


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
        # 获取模型
        model = get_model(args.model)
        # 执行敏感性分析
        sorted_importance = sensitivity_analysis(model, X, y, numeric_cols)
        # 打印结果
        print(f"{args.model.upper()} 模型敏感性分析结果：")
        print("=" * 50)
        print("特征重要性排序：")
        for feature, importance in sorted_importance[:10]:  # 只显示前10个
            print(f"{feature}: {importance:.4f}")
        print("=" * 50)
        # 绘制结果
        plot_sensitivity(sorted_importance[:10])
        # 保存结果
        sensitivity_df = pd.DataFrame(sorted_importance, columns=['feature', 'importance'])
        sensitivity_df.to_csv('sensitivity_analysis.csv', index=False)
        print("敏感性分析结果已保存到 sensitivity_analysis.csv")
        print("特征重要性图已保存到 sensitivity_analysis.png")
    else:
        print("错误：数据中没有 'score' 列！")


if __name__ == "__main__":
    main()
```

## 注意事项

- **数据格式**：确保输入数据包含必要的实际值和预测值列
- **指标选择**：根据模型类型选择合适的验证指标
- **交叉验证**：使用交叉验证评估模型的稳定性和泛化能力
- **敏感性分析**：通过敏感性分析识别关键特征，优化模型

## 使用建议

1. **模型比较**：使用相同的验证指标比较不同模型的性能
2. **参数调优**：根据验证结果调整模型参数
3. **特征选择**：基于敏感性分析结果选择重要特征
4. **结果解释**：关注模型的实际应用价值，而不仅仅是指标数值

## 输出文件

模型验证完成后，将生成以下文件：
1. **regression_metrics.csv** 或 **classification_metrics.csv**：验证指标结果
2. **cross_validation_results.csv**：交叉验证结果
3. **sensitivity_analysis.csv**：敏感性分析结果
4. **confusion_matrix.png** 或 **sensitivity_analysis.png**：可视化结果

## 扩展功能

- **模型集成**：评估集成模型的性能
- **时间序列验证**：使用滚动窗口验证时间序列模型
- **超参数调优**：结合验证指标进行超参数搜索
- **模型解释**：使用 SHAP 或 LIME 解释模型预测
