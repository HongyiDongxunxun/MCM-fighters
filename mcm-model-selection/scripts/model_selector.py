import argparse
import json
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='MCM model selector')
    parser.add_argument('--indicators', type=str, required=True, help='Indicators JSON file path')
    return parser.parse_args()


def load_indicators(file_path):
    """加载指标"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def select_model(indicators):
    """根据指标选择模型"""
    match_type = indicators.get('match_type', '其他')
    problem_type = indicators.get('problem_type', 'optimization')
    
    # 优先根据问题类型选择模型
    if problem_type == 'optimization':
        # 优化问题 - 选择运筹学方法
        if match_type == 'resource_allocation':
            return 'dynamic_programming', 'Dynamic Programming (Resource Allocation)'
        elif match_type == 'decision_making':
            return 'ahp', 'Analytic Hierarchy Process (AHP)'
        elif match_type == 'nonlinear_optimization':
            return 'nonlinear_programming', 'Nonlinear Programming'
        elif match_type == 'integer_optimization':
            return 'integer_programming', 'Integer Programming'
        elif match_type == 'multi_objective':
            return 'multi_objective_optimization', 'Multi-Objective Optimization'
        else:
            return 'linear_programming', 'Linear Programming'
    elif problem_type == 'hypothesis_testing':
        # 假设检验问题 - 选择传统统计学方法
        return 'hypothesis_testing', 'Hypothesis Testing'
    elif problem_type == 'time_series':
        # 时间序列问题 - 选择时间序列分析方法
        return 'time_series', 'Time Series Analysis (ARIMA)'
    elif problem_type == 'economic_analysis':
        # 经济分析问题 - 选择经济学方法
        if match_type == 'game_theory':
            return 'game_theory', 'Game Theory'
        elif match_type == 'input_output':
            return 'input_output_analysis', 'Input-Output Analysis'
        elif match_type == 'cost_benefit':
            return 'cost_benefit_analysis', 'Cost-Benefit Analysis'
        else:
            return 'economic_analysis', 'Economic Analysis'
    
    # 默认模型
    return 'linear_programming', 'Linear Programming'




def main():
    args = parse_args()
    indicators = load_indicators(args.indicators)
    model_code, model_name = select_model(indicators)
    
    print(f"基于指标选择的最优模型：{model_name} ({model_code})")
    print("\n模型信息：")
    
    # 保留的模型
    if model_code == 'gray_model':
        print("- 适用场景：短期转折点预测")
        print("- 优势：数据量小、特征维度低时表现良好")
        print("- 运行命令：python scripts/gray_model.py --input processed_data.csv --output predictions.csv")
    elif model_code == 'topsis':
        print("- 适用场景：性能评估 + 因素分析")
        print("- 优势：综合评估多个因素，提供排名")
        print("- 运行命令：python scripts/topsis_model.py --input processed_data.csv --output rankings.csv")
    # 运筹学模型
    elif model_code == 'linear_programming':
        print("- 适用场景：资源优化、生产计划、运输问题")
        print("- 优势：求解线性约束下的最优解")
        print("- 运行命令：python scripts/linear_programming.py --input lp_data.csv --output lp_results.csv --objective min")
    elif model_code == 'dynamic_programming':
        print("- 适用场景：背包问题、投资组合、资源分配")
        print("- 优势：求解多阶段决策问题的最优解")
        print("- 运行命令：python scripts/dynamic_programming.py --input dp_data.csv --output dp_results.csv --problem_type knapsack")
    elif model_code == 'ahp':
        print("- 适用场景：多准则决策、方案排序、权重确定")
        print("- 优势：将定性分析与定量分析相结合")
        print("- 运行命令：python scripts/ahp_model.py --input ahp_data.csv --output ahp_results.csv")
    elif model_code == 'nonlinear_programming':
        print("- 适用场景：非线性约束优化、生产调度、投资组合")
        print("- 优势：求解复杂的非线性优化问题")
        print("- 运行命令：python scripts/nonlinear_programming.py --input nlp_data.csv --output nlp_results.csv")
    elif model_code == 'integer_programming':
        print("- 适用场景：整数决策变量优化、排班问题、选址问题")
        print("- 优势：处理离散决策变量的优化问题")
        print("- 运行命令：python scripts/integer_programming.py --input ip_data.csv --output ip_results.csv")
    elif model_code == 'multi_objective_optimization':
        print("- 适用场景：多目标决策、资源分配、权衡分析")
        print("- 优势：处理多个冲突目标的优化问题")
        print("- 运行命令：python scripts/multi_objective_optimization.py --input moo_data.csv --output moo_results.csv")
    # 经济学模型
    elif model_code == 'game_theory':
        print("- 适用场景：策略决策、竞争分析、谈判模型")
        print("- 优势：分析多主体交互决策")
        print("- 运行命令：python scripts/game_theory.py --input game_data.csv --output game_results.csv")
    elif model_code == 'input_output_analysis':
        print("- 适用场景：经济影响分析、产业关联分析")
        print("- 优势：量化经济系统中各部门的相互依赖关系")
        print("- 运行命令：python scripts/input_output_analysis.py --input io_data.csv --output io_results.csv")
    elif model_code == 'cost_benefit_analysis':
        print("- 适用场景：项目评估、投资决策、政策分析")
        print("- 优势：系统化评估成本与收益")
        print("- 运行命令：python scripts/cost_benefit_analysis.py --input cba_data.csv --output cba_results.csv")
    elif model_code == 'economic_analysis':
        print("- 适用场景：综合经济分析、市场预测、政策评估")
        print("- 优势：提供全面的经济视角分析")
        print("- 运行命令：python scripts/economic_analysis.py --input economic_data.csv --output economic_results.csv")
    # 传统统计学模型
    elif model_code == 'hypothesis_testing':
        print("- 适用场景：显著性检验、差异分析、假设验证")
        print("- 优势：提供统计显著性判断")
        print("- 运行命令：python scripts/hypothesis_testing.py --input test_data.csv --output test_results.csv --test_type t-test")
    elif model_code == 'time_series':
        print("- 适用场景：时间序列预测、趋势分析、季节性分析")
        print("- 优势：处理时间依赖数据")
        print("- 运行命令：python scripts/time_series_analysis.py --input ts_data.csv --output ts_results.csv --model arima --order 1,1,1")
    
    print("\n可用命令：")
    print("\n运筹学模型命令：")
    print("- 运行线性规划模型：python scripts/linear_programming.py --input lp_data.csv --output lp_results.csv --objective min")
    print("- 运行动态规划模型：python scripts/dynamic_programming.py --input dp_data.csv --output dp_results.csv --problem_type knapsack")
    print("- 运行层次分析法模型：python scripts/ahp_model.py --input ahp_data.csv --output ahp_results.csv")
    print("- 运行非线性规划模型：python scripts/nonlinear_programming.py --input nlp_data.csv --output nlp_results.csv")
    print("- 运行整数规划模型：python scripts/integer_programming.py --input ip_data.csv --output ip_results.csv")
    print("- 运行多目标优化模型：python scripts/multi_objective_optimization.py --input moo_data.csv --output moo_results.csv")
    print("\n经济学模型命令：")
    print("- 运行博弈论模型：python scripts/game_theory.py --input game_data.csv --output game_results.csv")
    print("- 运行投入产出分析模型：python scripts/input_output_analysis.py --input io_data.csv --output io_results.csv")
    print("- 运行成本收益分析模型：python scripts/cost_benefit_analysis.py --input cba_data.csv --output cba_results.csv")
    print("- 运行综合经济分析模型：python scripts/economic_analysis.py --input economic_data.csv --output economic_results.csv")
    print("\n统计学与其他模型命令：")
    print("- 运行假设检验模型：python scripts/hypothesis_testing.py --input test_data.csv --output test_results.csv --test_type t-test")
    print("- 运行时间序列分析模型：python scripts/time_series_analysis.py --input ts_data.csv --output ts_results.csv --model arima --order 1,1,1")
    print("- 运行灰色预测模型：python scripts/gray_model.py --input processed_data.csv --output predictions.csv")
    print("- 运行TOPSIS模型：python scripts/topsis_model.py --input processed_data.csv --output rankings.csv")


if __name__ == "__main__":
    main()
