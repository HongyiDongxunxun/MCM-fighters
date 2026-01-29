import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def parse_args():
    parser = argparse.ArgumentParser(description='LightGBM model')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of boosting iterations')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Boosting learning rate')
    parser.add_argument('--max_depth', type=int, default=5, help='Maximum tree depth')
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
            from sklearn.model_selection import RandomizedSearchCV
            param_dist = {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
            random_search = RandomizedSearchCV(
                lgb.LGBMRegressor(random_state=42),
                param_distributions=param_dist,
                n_iter=10,
                cv=3,
                scoring='r2',
                random_state=42
            )
            random_search.fit(X_train, y_train)
            model = random_search.best_estimator_
            print(f"最优参数: {random_search.best_params_}")
            print(f"最优交叉验证得分: {random_search.best_score_:.4f}")
        else:
            # 使用指定的参数
            model = lgb.LGBMRegressor(
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                max_depth=args.max_depth,
                random_state=42
            )
            model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("LightGBM 模型评估结果：")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        
        # 特征重要性分析
        feature_importance = pd.DataFrame({
            'feature': numeric_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n特征重要性排序：")
        print(feature_importance.head(10))
        
        # 保存预测结果
        test_df = X_test.copy()
        test_df['actual_score'] = y_test
        test_df['predicted_score'] = y_pred
        test_df.to_csv(args.output, index=False)
        
        # 保存特征重要性
        feature_importance.to_csv('lightgbm_feature_importance.csv', index=False)
        
        print(f"\n预测结果已保存到 {args.output}")
        print("特征重要性已保存到 lightgbm_feature_importance.csv")
    else:
        print("错误：数据中没有 'score' 列！")


if __name__ == "__main__":
    main()
