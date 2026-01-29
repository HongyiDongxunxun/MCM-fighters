import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='XGBoost+SHAP model')
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
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # 训练模型
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        # 预测
        y_pred = model.predict(X_test)
        # SHAP 分析
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)
        # 保存预测结果
        test_df = X_test.copy()
        test_df['actual_score'] = y_test
        test_df['predicted_score'] = y_pred
        test_df.to_csv(args.output, index=False)
        # 可视化特征重要性
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test)
        plt.savefig('shap_summary.png')
        print("XGBoost+SHAP 模型预测完成！")
        print(f"预测结果已保存到 {args.output}")
        print("特征重要性图已保存到 shap_summary.png")
    else:
        print("错误：数据中没有 'score' 列！")


if __name__ == "__main__":
    main()
