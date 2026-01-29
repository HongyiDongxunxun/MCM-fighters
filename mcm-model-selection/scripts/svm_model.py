import argparse
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(description='SVM model')
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
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        # 训练模型
        model = SVR(kernel='rbf', C=1.0, gamma='scale')
        model.fit(X_train, y_train)
        # 预测
        y_pred = model.predict(X_test)
        # 保存预测结果
        test_df = pd.DataFrame(X_test, columns=numeric_cols)
        test_df['actual_score'] = y_test
        test_df['predicted_score'] = y_pred
        test_df.to_csv(args.output, index=False)
        print("SVM 模型预测完成！")
        print(f"预测结果已保存到 {args.output}")
    else:
        print("错误：数据中没有 'score' 列！")


if __name__ == "__main__":
    main()
