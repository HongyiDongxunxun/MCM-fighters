import argparse
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='GM (1,1) model')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    return parser.parse_args()


def gm_11(x, n_pred=5):
    """GM (1,1) 灰色预测模型"""
    # 累加生成
    x1 = np.cumsum(x)
    # 计算均值生成序列
    z1 = (x1[:-1] + x1[1:]) / 2
    # 构建矩阵
    B = np.vstack([-z1, np.ones(len(z1))]).T
    Y = x[1:].reshape(-1, 1)
    # 最小二乘估计
    a, b = np.linalg.lstsq(B, Y, rcond=None)[0]
    # 预测模型
    def predict(k):
        return (x[0] - b/a) * np.exp(-a*k) + b/a
    # 计算拟合值
    fit = np.zeros(len(x))
    fit[0] = x[0]
    for i in range(1, len(x)):
        fit[i] = predict(i) - predict(i-1)
    # 预测未来值
    pred = np.zeros(n_pred)
    for i in range(n_pred):
        pred[i] = predict(len(x) + i) - predict(len(x) + i - 1)
    return fit, pred


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    # 假设目标变量为 'score'
    if 'score' in df.columns:
        x = df['score'].values
        fit, pred = gm_11(x)
        # 将拟合值和预测值添加到数据框
        df['fit_score'] = fit
        # 创建预测数据框
        pred_df = pd.DataFrame({
            'timestamp': [f'pred_{i+1}' for i in range(len(pred))],
            'score': pred
        })
        # 合并数据
        result_df = pd.concat([df, pred_df], ignore_index=True)
        result_df.to_csv(args.output, index=False)
        print("GM (1,1) 模型预测完成！")
        print(f"预测结果已保存到 {args.output}")
    else:
        print("错误：数据中没有 'score' 列！")


if __name__ == "__main__":
    main()
