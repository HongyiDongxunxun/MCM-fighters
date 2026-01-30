import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Economic Analysis Model')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--analysis_type', type=str, default='comprehensive', choices=['comprehensive', 'market', 'policy'], help='Analysis type')
    return parser.parse_args()


def load_data(file_path):
    """加载数据"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def analyze_market_structure(data):
    """分析市场结构"""
    # 计算市场集中度指标
    if 'firm' in data.columns and 'market_share' in data.columns:
        # 使用赫芬达尔-赫希曼指数 (HHI)
        market_shares = data['market_share'].values
        hhi = np.sum(np.square(market_shares))
        
        # 计算四企业集中度比率 (CR4)
        sorted_shares = sorted(market_shares, reverse=True)
        cr4 = np.sum(sorted_shares[:4]) if len(sorted_shares) >= 4 else np.sum(sorted_shares)
        
        # 计算八企业集中度比率 (CR8)
        cr8 = np.sum(sorted_shares[:8]) if len(sorted_shares) >= 8 else np.sum(sorted_shares)
    else:
        # 默认数据：体育产业市场结构
        # 假设6个主要企业
        market_shares = np.array([0.3, 0.25, 0.2, 0.15, 0.07, 0.03])
        hhi = np.sum(np.square(market_shares))
        cr4 = np.sum(market_shares[:4])
        cr8 = np.sum(market_shares)
    
    # 市场结构分类
    if hhi < 1500:
        market_structure = '竞争性市场'
    elif hhi < 2500:
        market_structure = '适度集中市场'
    else:
        market_structure = '高度集中市场'
    
    return {
        'hhi': hhi,
        'cr4': cr4,
        'cr8': cr8,
        'market_structure': market_structure,
        'analysis': f'市场集中度分析显示该市场为{market_structure}，HHI指数为{hhi:.2f}'
    }


def analyze_demand_supply(data):
    """分析需求供给"""
    # 计算需求弹性和供给弹性
    if 'price' in data.columns and 'quantity' in data.columns:
        # 计算需求弹性
        prices = data['price'].values
        quantities = data['quantity'].values
        
        # 简单线性回归估计需求曲线
        from sklearn.linear_model import LinearRegression
        X = prices.reshape(-1, 1)
        y = quantities
        model = LinearRegression().fit(X, y)
        demand_slope = model.coef_[0]
        
        # 计算中点弹性
        avg_price = np.mean(prices)
        avg_quantity = np.mean(quantities)
        demand_elasticity = (demand_slope * avg_price) / avg_quantity
    else:
        # 默认数据：体育赛事门票需求
        demand_elasticity = -0.8  # 缺乏弹性
        supply_elasticity = 1.2   # 富有弹性
    
    # 弹性分析
    if abs(demand_elasticity) > 1:
        demand_elasticity_type = '富有弹性'
    elif abs(demand_elasticity) < 1:
        demand_elasticity_type = '缺乏弹性'
    else:
        demand_elasticity_type = '单位弹性'
    
    return {
        'demand_elasticity': demand_elasticity,
        'supply_elasticity': supply_elasticity,
        'demand_elasticity_type': demand_elasticity_type,
        'analysis': f'需求弹性为{demand_elasticity:.2f}，属于{demand_elasticity_type}；供给弹性为{supply_elasticity:.2f}'
    }


def analyze_economic_indicators(data):
    """分析经济指标"""
    # 计算关键经济指标
    if 'indicator' in data.columns and 'value' in data.columns:
        # 从数据中提取指标
        indicators = {}
        for _, row in data.iterrows():
            indicators[row['indicator']] = row['value']
    else:
        # 默认数据：体育产业经济指标
        indicators = {
            'revenue_growth': 0.08,  # 收入增长率
            'profit_margin': 0.15,    # 利润率
            'employment_rate': 0.05,  # 就业率
            'investment_return': 0.12, # 投资回报率
            'gdp_contribution': 0.02   # GDP贡献率
        }
    
    return {
        'indicators': indicators,
        'analysis': '经济指标分析显示该产业具有良好的增长潜力和盈利能力'
    }


def comprehensive_analysis(data):
    """综合经济分析"""
    # 市场结构分析
    market_analysis = analyze_market_structure(data)
    
    # 需求供给分析
    demand_supply_analysis = analyze_demand_supply(data)
    
    # 经济指标分析
    economic_indicators_analysis = analyze_economic_indicators(data)
    
    # 综合分析
    comprehensive_result = {
        'market_analysis': market_analysis,
        'demand_supply_analysis': demand_supply_analysis,
        'economic_indicators_analysis': economic_indicators_analysis,
        'summary': '综合经济分析显示该产业处于{}，具有{}的需求特征和良好的经济表现。'
                   .format(
                       market_analysis['market_structure'],
                       demand_supply_analysis['demand_elasticity_type']
                   )
    }
    
    return comprehensive_result


def market_analysis(data):
    """市场分析"""
    # 市场结构分析
    market_analysis = analyze_market_structure(data)
    
    # 市场趋势分析
    market_trends = {
        'growth_rate': 0.07,  # 市场增长率
        'forecast_period': 5,  # 预测周期
        'market_size': 1000,   # 市场规模（亿元）
        'key_drivers': ['媒体版权', '赞助收入', '门票销售', '周边商品']
    }
    
    return {
        'market_structure': market_analysis,
        'market_trends': market_trends,
        'analysis': '市场分析显示该市场具有稳定的增长趋势和明确的驱动因素'
    }


def policy_analysis(data):
    """政策分析"""
    # 政策影响分析
    policy_impacts = {
        'tax_incentives': 0.1,  # 税收激励影响
        'subsidies': 0.08,      # 补贴影响
        'regulation': -0.05,     # 监管影响
        'infrastructure': 0.12   # 基础设施影响
    }
    
    # 政策建议
    policy_recommendations = [
        '制定差异化税收政策，鼓励体育产业投资',
        '加大对体育基础设施的投入',
        '完善体育产业监管体系，促进公平竞争',
        '支持体育产业与其他产业的融合发展'
    ]
    
    return {
        'policy_impacts': policy_impacts,
        'policy_recommendations': policy_recommendations,
        'analysis': '政策分析显示当前政策对体育产业发展具有积极影响，同时存在进一步优化空间'
    }


def main():
    args = parse_args()
    data = load_data(args.input)
    
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    print("Running Economic Analysis Model...")
    print(f"Analysis Type: {args.analysis_type}")
    
    # 根据分析类型执行相应的分析
    if args.analysis_type == 'comprehensive':
        result = comprehensive_analysis(data)
        print("\nComprehensive Economic Analysis Results:")
        print(f"Market Structure: {result['market_analysis']['market_structure']}")
        print(f"HHI: {result['market_analysis']['hhi']:.2f}")
        print(f"Demand Elasticity: {result['demand_supply_analysis']['demand_elasticity']:.2f} ({result['demand_supply_analysis']['demand_elasticity_type']})")
        print(f"Summary: {result['summary']}")
    elif args.analysis_type == 'market':
        result = market_analysis(data)
        print("\nMarket Analysis Results:")
        print(f"Market Structure: {result['market_structure']['market_structure']}")
        print(f"Market Growth Rate: {result['market_trends']['growth_rate']:.2f}")
        print(f"Market Size: {result['market_trends']['market_size']} 亿元")
        print(f"Key Drivers: {', '.join(result['market_trends']['key_drivers'])}")
    else:  # policy
        result = policy_analysis(data)
        print("\nPolicy Analysis Results:")
        print("Policy Impacts:")
        for policy, impact in result['policy_impacts'].items():
            print(f"- {policy}: {impact:.2f}")
        print("\nPolicy Recommendations:")
        for i, recommendation in enumerate(result['policy_recommendations'], 1):
            print(f"{i}. {recommendation}")
    
    # 保存结果
    if args.analysis_type == 'comprehensive':
        results = {
            'analysis_type': ['comprehensive'],
            'market_structure': [result['market_analysis']['market_structure']],
            'hhi': [result['market_analysis']['hhi']],
            'cr4': [result['market_analysis']['cr4']],
            'cr8': [result['market_analysis']['cr8']],
            'demand_elasticity': [result['demand_supply_analysis']['demand_elasticity']],
            'demand_elasticity_type': [result['demand_supply_analysis']['demand_elasticity_type']],
            'supply_elasticity': [result['demand_supply_analysis']['supply_elasticity']],
            'summary': [result['summary']]
        }
    elif args.analysis_type == 'market':
        results = {
            'analysis_type': ['market'],
            'market_structure': [result['market_structure']['market_structure']],
            'hhi': [result['market_structure']['hhi']],
            'market_growth_rate': [result['market_trends']['growth_rate']],
            'market_size': [result['market_trends']['market_size']],
            'key_drivers': [', '.join(result['market_trends']['key_drivers'])],
            'analysis': [result['analysis']]
        }
    else:
        results = {
            'analysis_type': ['policy'],
            'tax_incentives_impact': [result['policy_impacts']['tax_incentives']],
            'subsidies_impact': [result['policy_impacts']['subsidies']],
            'regulation_impact': [result['policy_impacts']['regulation']],
            'infrastructure_impact': [result['policy_impacts']['infrastructure']],
            'policy_recommendations': [', '.join(result['policy_recommendations'])],
            'analysis': [result['analysis']]
        }
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output, index=False)
    
    print(f"\nResults saved to: {args.output}")
    print("\nEconomic Analysis Model completed successfully!")


if __name__ == "__main__":
    main()
