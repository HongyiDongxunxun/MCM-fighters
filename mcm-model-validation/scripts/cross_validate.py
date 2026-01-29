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
