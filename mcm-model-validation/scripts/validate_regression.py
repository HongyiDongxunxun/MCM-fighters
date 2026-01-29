import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import acorr_ljungbox


def parse_args():
    parser = argparse.ArgumentParser(description='Regression model validation')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path with actual and predicted values')
    return parser.parse_args()


def calculate_metrics(y_true, y_pred):
    """计算回归模型指标"""
    metrics = {}
    # MSE
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    # RMSE
    metrics['rmse'] = np.sqrt(metrics['mse'])
    # MAE
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    # R²
    metrics['r2'] = r2_score(y_true, y_pred)
    # 平均相对误差
    metrics['mean_relative_error'] = np.mean(np.abs((y_true - y_pred) / y_true))
    # 残差分析
    residuals = y_true - y_pred
    metrics['residual_mean'] = np.mean(residuals)
    metrics['residual_std'] = np.std(residuals)
    # Ljung-Box 检验
    if len(residuals) > 20:
        lb_result = acorr_ljungbox(residuals, lags=20)
        metrics['lb_statistic'] = lb_result[0][-1]
        metrics['lb_pvalue'] = lb_result[1][-1]
    return metrics


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    # 检查必要的列
    if 'actual_score' in df.columns and 'predicted_score' in df.columns:
        y_true = df['actual_score'].values
        y_pred = df['predicted_score'].values
        # 计算指标
        metrics = calculate_metrics(y_true, y_pred)
        # 打印结果
        print("回归模型验证结果：")
        print("=" * 50)
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        print("=" * 50)
        # 保存结果
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv('regression_metrics.csv', index=False)
        print("验证结果已保存到 regression_metrics.csv")
    else:
        print("错误：数据中没有 'actual_score' 或 'predicted_score' 列！")


if __name__ == "__main__":
    main()
