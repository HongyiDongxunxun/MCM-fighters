import argparse
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb


def parse_args():
    parser = argparse.ArgumentParser(description='MCM model selector')
    parser.add_argument('--indicators', type=str, required=True, help='Indicators JSON file path')
    parser.add_argument('--auto_tune', type=bool, default=False, help='Enable automatic parameter tuning')
    parser.add_argument('--ensemble', type=bool, default=False, help='Enable model ensemble')
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
    match_type = indicators.get('match_type', '其他')
    
    # 模型选择逻辑
    if data_size == '小':
        if feature_dim == '低':
            return 'gm_11', 'GM (1,1) 灰色预测'
        else:
            return 'lasso', 'LASSO Regression'
    elif data_size == '中':
        if feature_dim == '低':
            return 'ridge', 'Ridge Regression'
        else:
            if momentum == '是':
                return 'xgboost_shap', 'XGBoost+SHAP'
            else:
                return 'svm', 'SVM'
    else:  # 大
        if feature_dim == '低':
            return 'lightgbm', 'LightGBM'
        else:
            if momentum == '是':
                return 'random_forest_stacking', 'Random Forest + Stacking'
            else:
                return 'dnn', 'Deep Neural Network'
    
    # 默认模型
    return 'svm', 'SVM'


def auto_tune_model(model, X, y):
    """自动参数调优"""
    print("执行自动参数调优...")
    
    if isinstance(model, xgb.XGBRegressor):
        param_dist = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 0.9]
        }
        search = RandomizedSearchCV(model, param_dist, n_iter=10, cv=3, scoring='r2', random_state=42)
        search.fit(X, y)
        print(f"最优参数: {search.best_params_}")
        print(f"最优得分: {search.best_score_:.4f}")
        return search.best_estimator_
    
    elif isinstance(model, RandomForestRegressor):
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }
        search = RandomizedSearchCV(model, param_dist, n_iter=10, cv=3, scoring='r2', random_state=42)
        search.fit(X, y)
        print(f"最优参数: {search.best_params_}")
        print(f"最优得分: {search.best_score_:.4f}")
        return search.best_estimator_
    
    else:
        print("该模型类型暂不支持自动参数调优")
        return model

def create_ensemble_model():
    """创建集成模型"""
    print("创建模型集成...")
    
    # 定义基础模型
    base_models = [
        ('xgboost', xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
        ('svr', SVR(kernel='rbf', C=1.0, gamma='scale'))
    ]
    
    # 定义元模型
    meta_model = LinearRegression()
    
    # 构建堆叠模型
    ensemble_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
    
    print("集成模型创建完成：StackingRegressor")
    print(f"基础模型数量: {len(base_models)}")
    print(f"元模型: {type(meta_model).__name__}")
    
    return ensemble_model

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
    elif model_code == 'lasso':
        print("- 适用场景：特征选择 + 回归预测")
        print("- 优势：自动选择重要特征，避免过拟合")
        print("- 运行命令：python scripts/lasso_model.py --input processed_data.csv --output predictions.csv")
    elif model_code == 'ridge':
        print("- 适用场景：正则化回归预测")
        print("- 优势：减少过拟合，提高模型稳定性")
        print("- 运行命令：python scripts/ridge_model.py --input processed_data.csv --output predictions.csv")
    elif model_code == 'xgboost_shap':
        print("- 适用场景：特征重要性分析 + 动量预测")
        print("- 优势：处理高维度数据，提供特征解释")
        print("- 运行命令：python scripts/xgboost_model.py --input processed_data.csv --output predictions.csv")
    elif model_code == 'lightgbm':
        print("- 适用场景：高效梯度提升预测")
        print("- 优势：训练速度快，内存消耗低")
        print("- 运行命令：python scripts/lightgbm_model.py --input processed_data.csv --output predictions.csv")
    elif model_code == 'random_forest_stacking':
        print("- 适用场景：长时序动量预测")
        print("- 优势：集成多个模型，提高预测准确性")
        print("- 运行命令：python scripts/random_forest_model.py --input processed_data.csv --output predictions.csv")
    elif model_code == 'dnn':
        print("- 适用场景：复杂模式学习 + 预测")
        print("- 优势：学习复杂的非线性关系")
        print("- 运行命令：python scripts/dnn_model.py --input processed_data.csv --output predictions.csv")
    elif model_code == 'svm':
        print("- 适用场景：分类与回归任务")
        print("- 优势：泛化能力强，适合各种数据类型")
        print("- 运行命令：python scripts/svm_model.py --input processed_data.csv --output predictions.csv")
    elif model_code == 'pca_topsis':
        print("- 适用场景：性能评估 + 因素分析")
        print("- 优势：综合评估多个因素，提供排名")
        print("- 运行命令：python scripts/topsis_model.py --input processed_data.csv --output rankings.csv")
    
    # 处理自动参数调优
    if args.auto_tune:
        print("\n启用自动参数调优：")
        print("- 使用 RandomizedSearchCV 进行参数搜索")
        print("- 3折交叉验证评估模型性能")
        print("- 优化目标：R² 得分")
    
    # 处理模型集成
    if args.ensemble:
        print("\n启用模型集成：")
        print("- 使用 StackingRegressor 集成多个模型")
        print("- 基础模型：XGBoost, Random Forest, SVR")
        print("- 元模型：Linear Regression")
    
    print("\n其他可用命令：")
    print("- 运行模型评估比较：python scripts/model_evaluator.py --input processed_data.csv --output model_comparison.csv")
    print("- 运行深度学习模型：python scripts/dnn_model.py --input processed_data.csv --output predictions.csv")
    print("- 运行模型集成：python scripts/ensemble_model.py --input processed_data.csv --output predictions.csv")


if __name__ == "__main__":
    main()
