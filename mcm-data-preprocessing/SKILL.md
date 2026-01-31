---
name: mcm-python-vibe-coding
description: Comprehensive MCM contest data analysis workflow with automatic model selection. Use when analyzing MCM contest data, especially for tennis momentum analysis, providing preprocessing, model selection, and precise modeling with Python code that can be directly adapted to contest datasets.
---

# MCM 比赛 Python Vibe Coding 技能

## 核心工作流程

本技能提供 **"预处理 + 预实验→自动选模型→精准建模"** 的完整流程，无需提前指定模型，只需输入预处理与预实验的关键指标，即可自动匹配最优模型。

### 步骤概览

1. **数据预处理与预实验**：清洗数据、特征工程、降维，生成模型选择指标
2. **模型自动选择**：基于预实验指标匹配最优模型
3. **模型训练与预测**：执行选定模型的建模流程
4. **模型检验与分析**：评估模型性能，进行敏感性分析

## 快速开始

### 环境配置

```python
# 必备库安装
pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels xgboost shap
```

### 数据预处理

运行数据预处理脚本，生成关键指标：

```python
# 执行预处理
python scripts/preprocess.py --input data.csv --output processed_data.csv
```

### 模型选择

输入预处理生成的指标，自动匹配最优模型：

```python
# 运行模型选择
python scripts/model_selector.py
```

## 支持的模型

| 数据规模 | 特征维度 | 推荐模型 | 适用场景 |
|---------|---------|---------|----------|
| 小 (<200) | 低 (<10) | GM (1,1) | 短期转折点预测 |
| 中 (200-500) | 高 (≥10) | XGBoost+SHAP | 特征重要性分析 + 动量预测 |
| 大 (>500) | 高 (≥10) | Random Forest + Stacking | 长时序动量预测 |
| 任意 | 任意 | SVM | 分类与回归任务 |
| 任意 | 任意 | PCA-TOPSIS | 性能评估 + 因素分析 |

## 详细文档

### 1. 数据预处理

**核心功能**：数据清洗、缺失值处理、特征工程、PCA降维

**关键指标**：
- 数据量大小（小/中/大）
- 特征维度（低/高）
- 序列平稳性（是/否）
- 动量趋势（是/否）
- 比赛类型（男单/女单/其他）

详细实现见：[预处理脚本](./scripts/preprocess.py)

### 2. 模型选择逻辑

基于预实验指标的模型匹配逻辑，详见：[模型选择脚本](./scripts/model_selector.py)

### 3. 模型实现

- **GM (1,1) 灰色预测**：[gray_model.py](./scripts/gray_model.py)
- **XGBoost+SHAP**：[xgboost_model.py](./scripts/xgboost_model.py)
- **Random Forest**：[random_forest_model.py](./scripts/random_forest_model.py)
- **SVM**：[svm_model.py](./scripts/svm_model.py)
- **集成学习**：[ensemble_model.py](./scripts/ensemble_model.py)
- **PCA-TOPSIS**：[topsis_model.py](./scripts/topsis_model.py)

### 4. 模型检验与分析

- **模型检验**：[model_validation.py](./scripts/model_validation.py)
- **敏感性分析**：[sensitivity_analysis.py](./scripts/sensitivity_analysis.py)

## 模型效果验证

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

### 交叉验证方法

```python
from sklearn.model_selection import cross_val_score, KFold

# K折交叉验证
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
print(f"交叉验证R²得分: {scores.mean():.4f} ± {scores.std():.4f}")
```

## 使用指南

1. **准备数据**：将赛事数据整理为CSV格式
2. **运行预处理**：生成关键指标
3. **选择模型**：输入指标，获取最优模型
4. **执行建模**：运行对应模型脚本
5. **分析结果**：查看模型输出和可视化结果

## 代码优化建议

- **数据处理**：大规模数据使用Dask并行处理
- **模型训练**：使用GridSearchCV或RandomizedSearchCV进行参数调优
- **特征工程**：考虑使用Featuretools自动生成特征
- **模型集成**：采用stacking或voting提高精度
- **代码复用**：将核心功能封装为类，便于跨项目使用

## 注意事项

- **数据质量**：确保输入数据的完整性和准确性
- **参数调优**：根据具体数据集调整模型参数
- **结果解释**：注重结果的物理意义和可解释性
- **论文撰写**：详细记录模型选择过程和关键参数

## 示例应用

- **网球比赛动量分析**：预测比赛转折点和获胜概率
- **其他体育数据分析**：可通过修改特征工程适配其他体育项目
- **一般时间序列预测**：适用于有明显趋势的时间序列数据
- **分类任务**：如比赛结果预测、球员表现分类等
