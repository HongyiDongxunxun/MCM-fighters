import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def parse_args():
    parser = argparse.ArgumentParser(description='Ridge Regression model')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--alpha', type=float, default=1.0, help='Ridge regularization parameter')
    parser.add_argument('--auto_tune', type=bool, default=False, help='Enable automatic parameter tuning')
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
        
        if args.auto_tune:
            # 自动参数调优
            from sklearn.model_selection import GridSearchCV
            param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
            grid_search = GridSearchCV(Ridge(random_state=42), param_grid, cv=3, scoring='r2')
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"最优参数: alpha={grid_search.best_params_['alpha']}")
            print(f"最优交叉验证得分: {grid_search.best_score_:.4f}")
        else:
            # 使用指定的 alpha 值
            model = Ridge(alpha=args.alpha, random_state=42)
            model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("Ridge 回归模型评估结果：")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        
        # 特征重要性分析
        feature_importance = pd.DataFrame({
            'feature': numeric_cols,
            'importance': np.abs(model.coef_)
        }).sort_values('importance', ascending=False)
        
        print("\n特征重要性排序：")
        print(feature_importance.head(10))
        
        # 保存预测结果
        test_df = X_test.copy()
        test_df['actual_score'] = y_test
        test_df['predicted_score'] = y_pred
        test_df.to_csv(args.output, index=False)
        
        # 保存特征重要性
        feature_importance.to_csv('ridge_feature_importance.csv', index=False)
        
        print(f"\n预测结果已保存到 {args.output}")
        print("特征重要性已保存到 ridge_feature_importance.csv")
    else:
        print("错误：数据中没有 'score' 列！")


if __name__ == "__main__":
    main()
