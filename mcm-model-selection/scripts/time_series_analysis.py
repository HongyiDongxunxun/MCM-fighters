import argparse
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Time Series Analysis model for MCM')
    parser.add_argument('--input', type=str, required=True, help='Input data file path')
    parser.add_argument('--output', type=str, required=True, help='Output result file path')
    parser.add_argument('--model', type=str, choices=['arima', 'ma', 'ar'], default='arima', help='Time series model type')
    parser.add_argument('--order', type=str, default='1,1,1', help='ARIMA model order (p,d,q)')
    parser.add_argument('--forecast_steps', type=int, default=5, help='Number of steps to forecast')
    return parser.parse_args()


def load_data(file_path):
    """加载数据"""
    return pd.read_csv(file_path)


def adf_test(series):
    """单位根检验"""
    print("执行单位根检验(ADF)...)\n")
    result = adfuller(series, autolag='AIC')
    print(f"ADF统计量: {result[0]:.4f}")
    print(f"p值: {result[1]:.4f}")
    print(f"滞后阶数: {result[2]}")
    print(f"观测值数量: {result[3]}")
    print("临界值:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.4f}")
    print(f"显著性水平α=0.05时, {'拒绝' if result[1] < 0.05 else '不拒绝'}原假设")
    print(f"时间序列{'是' if result[1] < 0.05 else '不是'}平稳的")
    return result


def arima_model(data, order=(1, 1, 1), forecast_steps=5):
    """ARIMA模型"""
    print(f"执行ARIMA模型，阶数: {order}...")
    
    # 提取时间序列数据
    if data.shape[1] == 1:
        series = data.iloc[:, 0].values
    else:
        series = data.iloc[:, 1].values
    
    # 执行单位根检验
    adf_result = adf_test(series)
    
    # 拟合ARIMA模型
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    
    print("\n模型拟合结果:")
    print(model_fit.summary())
    
    # 模型诊断
    residuals = model_fit.resid
    print(f"\n残差均值: {np.mean(residuals):.4f}")
    print(f"残差标准差: {np.std(residuals):.4f}")
    
    # 预测
    forecast = model_fit.forecast(steps=forecast_steps)
    print(f"\n预测未来{forecast_steps}步:")
    print(forecast)
    
    return {
        'model_type': 'ARIMA',
        'order': order,
        'forecast': forecast.tolist(),
        'residuals_mean': np.mean(residuals),
        'residuals_std': np.std(residuals),
        'adf_pvalue': adf_result[1]
    }


def ma_model(data, q=1, forecast_steps=5):
    """MA模型"""
    print(f"执行MA模型，阶数: {q}...")
    return arima_model(data, order=(0, 0, q), forecast_steps=forecast_steps)


def ar_model(data, p=1, forecast_steps=5):
    """AR模型"""
    print(f"执行AR模型，阶数: {p}...")
    return arima_model(data, order=(p, 0, 0), forecast_steps=forecast_steps)


def time_series_analysis_model(data, model_type='arima', order=(1, 1, 1), forecast_steps=5):
    """时间序列分析模型"""
    if model_type == 'arima':
        return arima_model(data, order=order, forecast_steps=forecast_steps)
    elif model_type == 'ma':
        q = order[2]
        return ma_model(data, q=q, forecast_steps=forecast_steps)
    elif model_type == 'ar':
        p = order[0]
        return ar_model(data, p=p, forecast_steps=forecast_steps)


def save_results(results, output_path):
    """保存结果"""
    # 保存预测结果
    forecast_df = pd.DataFrame({
        'forecast': results['forecast']
    })
    forecast_df.to_csv(output_path, index_label='step')
    print(f"\n预测结果已保存到: {output_path}")


def main():
    args = parse_args()
    data = load_data(args.input)
    
    # 解析order参数
    order = tuple(map(int, args.order.split(',')))
    
    results = time_series_analysis_model(data, args.model, order, args.forecast_steps)
    save_results(results, args.output)


if __name__ == "__main__":
    main()