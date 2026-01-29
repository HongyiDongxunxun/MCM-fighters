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
