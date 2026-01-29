import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb


def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluator and comparator')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--cv', type=int, default=5, help='Number of cross-validation folds')
    return parser.parse_args()


def evaluate_model(model, X, y, cv=5):
    """评估模型性能"""
    # 计算交叉验证得分
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 计算平均交叉验证得分
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    return {
        'model_name': type(model).__name__,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'cv_mean': cv_mean,
        'cv_std': cv_std
    }


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
        
        print(f"数据集形状: {X.shape}")
        print(f"特征数量: {len(numeric_cols)}")
        print(f"使用 {args.cv} 折交叉验证")
        
        # 定义要评估的模型
        models = [
            LinearRegression(),
            Lasso(alpha=1.0),
            Ridge(alpha=1.0),
            SVR(kernel='rbf', C=1.0, gamma='scale'),
            RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
            lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        ]
        
        # 评估所有模型
        results = []
        print("\n评估模型性能...")
        for i, model in enumerate(models):
            print(f"评估模型 {i+1}/{len(models)}: {type(model).__name__}")
            result = evaluate_model(model, X, y, args.cv)
            results.append(result)
        
        # 将结果转换为数据框
        results_df = pd.DataFrame(results)
        
        # 按 R² 得分排序
        results_df = results_df.sort_values('r2', ascending=False).reset_index(drop=True)
        
        # 打印评估结果
        print("\n模型评估结果（按 R² 排序）：")
        print("=" * 100)
        print(results_df.to_string(index=True, float_format='%.4f'))
        print("=" * 100)
        
        # 保存评估结果
        results_df.to_csv(args.output, index=False)
        print(f"\n评估结果已保存到 {args.output}")
        
        # 打印最佳模型
        best_model = results_df.iloc[0]
        print(f"\n最佳模型: {best_model['model_name']}")
        print(f"R² 得分: {best_model['r2']:.4f}")
        print(f"交叉验证平均得分: {best_model['cv_mean']:.4f} ± {best_model['cv_std']:.4f}")
        print(f"RMSE: {best_model['rmse']:.4f}")
        print(f"MAE: {best_model['mae']:.4f}")
    else:
        print("错误：数据中没有 'score' 列！")


if __name__ == "__main__":
    main()
