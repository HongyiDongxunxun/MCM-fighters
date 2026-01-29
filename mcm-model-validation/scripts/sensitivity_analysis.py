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
