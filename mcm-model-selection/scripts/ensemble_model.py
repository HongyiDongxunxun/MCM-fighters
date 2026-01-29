import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb


def parse_args():
    parser = argparse.ArgumentParser(description='Model ensemble')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--ensemble_type', type=str, default='stacking', choices=['voting', 'stacking'], help='Ensemble method')
    return parser.parse_args()


def create_voting_ensemble():
    """创建投票集成模型"""
    # 定义基础模型
    base_models = [
        ('xgboost', xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
        ('svr', SVR(kernel='rbf', C=1.0, gamma='scale')),
        ('lightgbm', lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
    ]
    
    # 创建投票集成模型
    ensemble_model = VotingRegressor(estimators=base_models, weights=[1, 1, 1, 1])
    
    return ensemble_model, base_models


def create_stacking_ensemble():
    """创建堆叠集成模型"""
    # 定义基础模型
    base_models = [
        ('xgboost', xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
        ('svr', SVR(kernel='rbf', C=1.0, gamma='scale')),
        ('lightgbm', lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
    ]
    
    # 定义元模型
    meta_model = LinearRegression()
    
    # 创建堆叠集成模型
    ensemble_model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=3)
    
    return ensemble_model, base_models


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
        
        # 创建集成模型
        if args.ensemble_type == 'voting':
            model, base_models = create_voting_ensemble()
            print(f"创建投票集成模型，包含 {len(base_models)} 个基础模型")
        else:  # stacking
            model, base_models = create_stacking_ensemble()
            print(f"创建堆叠集成模型，包含 {len(base_models)} 个基础模型")
        
        # 训练模型
        print("训练集成模型...")
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n{args.ensemble_type} 集成模型评估结果：")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        
        # 评估基础模型
        print("\n基础模型评估结果：")
        base_model_results = []
        for name, base_model in base_models:
            # 训练基础模型
            base_model.fit(X_train, y_train)
            # 预测
            base_y_pred = base_model.predict(X_test)
            # 计算评估指标
            base_mse = mean_squared_error(y_test, base_y_pred)
            base_rmse = np.sqrt(base_mse)
            base_mae = mean_absolute_error(y_test, base_y_pred)
            base_r2 = r2_score(y_test, base_y_pred)
            base_model_results.append((name, base_mse, base_rmse, base_mae, base_r2))
            print(f"{name}:")
            print(f"  MSE: {base_mse:.4f}")
            print(f"  RMSE: {base_rmse:.4f}")
            print(f"  MAE: {base_mae:.4f}")
            print(f"  R²: {base_r2:.4f}")
        
        # 保存预测结果
        test_df = X_test.copy()
        test_df['actual_score'] = y_test
        test_df['predicted_score'] = y_pred
        test_df.to_csv(args.output, index=False)
        
        # 保存基础模型结果
        base_results_df = pd.DataFrame(
            base_model_results,
            columns=['model', 'mse', 'rmse', 'mae', 'r2']
        )
        base_results_df.to_csv('base_model_results.csv', index=False)
        
        print(f"\n预测结果已保存到 {args.output}")
        print("基础模型评估结果已保存到 base_model_results.csv")
    else:
        print("错误：数据中没有 'score' 列！")


if __name__ == "__main__":
    main()
