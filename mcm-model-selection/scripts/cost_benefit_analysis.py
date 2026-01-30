import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Cost-Benefit Analysis Model')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--discount_rate', type=float, default=0.05, help='Discount rate for present value calculations')
    return parser.parse_args()


def load_data(file_path):
    """加载数据"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def calculate_present_value(values, discount_rate):
    """计算现值"""
    present_values = []
    for t, value in enumerate(values):
        if t == 0:
            present_values.append(value)
        else:
            present_values.append(value / (1 + discount_rate)**t)
    return np.array(present_values)


def analyze_cost_benefit(data, discount_rate):
    """分析成本收益"""
    # 从数据中提取成本和收益
    if 'cost' in data.columns and 'benefit' in data.columns:
        # 按时间周期组织的数据
        periods = sorted(data['period'].unique())
        n_periods = len(periods)
        
        costs = []
        benefits = []
        
        for period in periods:
            period_data = data[data['period'] == period]
            period_cost = period_data['cost'].sum()
            period_benefit = period_data['benefit'].sum()
            costs.append(period_cost)
            benefits.append(period_benefit)
    else:
        # 默认数据：体育场馆投资项目
        # 假设5年周期
        periods = [0, 1, 2, 3, 4, 5]
        # 初始投资 + 运营成本
        costs = [-1000, -100, -120, -130, -140, -150]
        # 门票收入 + 赞助收入 + 其他收入
        benefits = [0, 300, 350, 400, 450, 500]
    
    # 计算现值
    present_costs = calculate_present_value(costs, discount_rate)
    present_benefits = calculate_present_value(benefits, discount_rate)
    
    # 计算净现值 (NPV)
    npv = np.sum(present_benefits) + np.sum(present_costs)
    
    # 计算收益成本比 (BCR)
    total_cost = -np.sum(present_costs)
    total_benefit = np.sum(present_benefits)
    bcr = total_benefit / total_cost if total_cost > 0 else 0
    
    # 计算内部收益率 (IRR) - 使用线性插值法近似
    irr = calculate_irr(costs + benefits)
    
    # 计算投资回收期
    payback_period = calculate_payback_period(costs, benefits)
    
    return {
        'periods': periods,
        'costs': costs,
        'benefits': benefits,
        'present_costs': present_costs,
        'present_benefits': present_benefits,
        'npv': npv,
        'bcr': bcr,
        'irr': irr,
        'payback_period': payback_period,
        'total_cost': total_cost,
        'total_benefit': total_benefit
    }


def calculate_irr(cash_flows):
    """计算内部收益率（近似值）"""
    # 使用线性插值法
    # 尝试不同的贴现率
    rates = np.linspace(0, 1, 100)
    npvs = []
    
    for rate in rates:
        pv = 0
        for t, cf in enumerate(cash_flows):
            pv += cf / (1 + rate)**t
        npvs.append(pv)
    
    # 找到NPV符号变化的点
    npvs = np.array(npvs)
    sign_changes = np.where(np.diff(np.sign(npvs)))[0]
    
    if len(sign_changes) > 0:
        # 线性插值
        i = sign_changes[0]
        r1 = rates[i]
        r2 = rates[i+1]
        npv1 = npvs[i]
        npv2 = npvs[i+1]
        
        if npv2 - npv1 != 0:
            irr = r1 - npv1 * (r2 - r1) / (npv2 - npv1)
            return irr
    
    return 0.0


def calculate_payback_period(costs, benefits):
    """计算投资回收期"""
    cumulative_cash_flow = 0
    payback_period = -1
    
    for t in range(len(costs)):
        cash_flow = benefits[t] + costs[t]
        cumulative_cash_flow += cash_flow
        
        if cumulative_cash_flow >= 0 and payback_period == -1:
            payback_period = t
            break
    
    return payback_period


def perform_sensitivity_analysis(data, discount_rate):
    """敏感性分析"""
    # 测试不同贴现率下的NPV
    discount_rates = [0.03, 0.05, 0.07, 0.10]
    npvs = []
    bcrs = []
    
    for rate in discount_rates:
        result = analyze_cost_benefit(data, rate)
        npvs.append(result['npv'])
        bcrs.append(result['bcr'])
    
    return discount_rates, npvs, bcrs


def main():
    args = parse_args()
    data = load_data(args.input)
    
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    print("Running Cost-Benefit Analysis Model...")
    print(f"Discount Rate: {args.discount_rate:.2f}")
    
    # 运行成本收益分析
    cba_result = analyze_cost_benefit(data, args.discount_rate)
    
    print("\nCost-Benefit Analysis Results:")
    print(f"Net Present Value (NPV): {cba_result['npv']:.2f}")
    print(f"Benefit-Cost Ratio (BCR): {cba_result['bcr']:.2f}")
    print(f"Internal Rate of Return (IRR): {cba_result['irr']:.2f}")
    print(f"Payback Period: {cba_result['payback_period']} years")
    print(f"Total Present Cost: {cba_result['total_cost']:.2f}")
    print(f"Total Present Benefit: {cba_result['total_benefit']:.2f}")
    
    # 打印年度数据
    print("\nAnnual Cash Flows:")
    cash_flow_data = {
        'Period': cba_result['periods'],
        'Cost': cba_result['costs'],
        'Benefit': cba_result['benefits'],
        'Net Cash Flow': [b + c for b, c in zip(cba_result['benefits'], cba_result['costs'])],
        'Present Cost': cba_result['present_costs'],
        'Present Benefit': cba_result['present_benefits'],
        'Present Net Cash Flow': cba_result['present_benefits'] + cba_result['present_costs']
    }
    cash_flow_df = pd.DataFrame(cash_flow_data)
    print(cash_flow_df)
    
    # 敏感性分析
    print("\nSensitivity Analysis:")
    discount_rates, npvs, bcrs = perform_sensitivity_analysis(data, args.discount_rate)
    sensitivity_df = pd.DataFrame({
        'Discount Rate': discount_rates,
        'NPV': npvs,
        'BCR': bcrs
    })
    print(sensitivity_df)
    
    # 保存结果
    results = {
        'npv': [cba_result['npv']],
        'bcr': [cba_result['bcr']],
        'irr': [cba_result['irr']],
        'payback_period': [cba_result['payback_period']],
        'total_cost': [cba_result['total_cost']],
        'total_benefit': [cba_result['total_benefit']],
        'discount_rate': [args.discount_rate]
    }
    
    # 添加敏感性分析结果
    for i, rate in enumerate(discount_rates):
        results[f'npv_rate_{rate:.2f}'] = [npvs[i]]
        results[f'bcr_rate_{rate:.2f}'] = [bcrs[i]]
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output, index=False)
    
    # 保存详细的现金流量数据
    cash_flow_output = args.output.replace('.csv', '_cash_flow.csv')
    cash_flow_df.to_csv(cash_flow_output, index=False)
    
    print(f"\nResults saved to: {args.output}")
    print(f"Cash flow details saved to: {cash_flow_output}")
    print("\nCost-Benefit Analysis Model completed successfully!")


if __name__ == "__main__":
    main()
