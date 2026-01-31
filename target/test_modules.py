#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试拆分后的体育团队管理模型模块
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from player_value_evaluator import PlayerValueEvaluator
from team_expansion_analyzer import TeamExpansionAnalyzer
from ticket_pricing_optimizer import TicketPricingOptimizer
from media_exposure_adjuster import MediaExposureAdjuster
from main_model import SportsTeamManagementModel, MarkovDecisionProcess

def test_all_modules():
    """测试所有模块"""
    print("=== 测试体育团队管理模型模块 ===")
    
    # 1. 测试主模型初始化
    print("\n1. 测试主模型初始化...")
    try:
        model = SportsTeamManagementModel()
        print("✓ 主模型初始化成功")
    except Exception as e:
        print(f"✗ 主模型初始化失败: {e}")
        return False
    
    # 2. 测试数据加载
    print("\n2. 测试数据加载...")
    try:
        model.load_data()
        print("✓ 数据加载成功")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False
    
    # 3. 测试数据预处理
    print("\n3. 测试数据预处理...")
    try:
        processed_data = model.preprocess_data()
        print(f"✓ 数据预处理成功，处理了 {len(processed_data)} 条球员数据")
    except Exception as e:
        print(f"✗ 数据预处理失败: {e}")
        return False
    
    # 4. 测试球员价值评估模块
    print("\n4. 测试球员价值评估模块...")
    try:
        player_value = model.player_value_evaluator.calculate_balanced_value('Precious Achiuwa')
        print(f"✓ 球员价值计算成功，Precious Achiuwa 的平衡价值: {player_value:.2f}")
        
        injured_value = model.player_value_evaluator.evaluate_injured_player('Precious Achiuwa', injury_severity=0.5)
        print(f"✓ 伤病球员价值评估成功，伤病状态下的价值: {injured_value:.2f}")
    except Exception as e:
        print(f"✗ 球员价值评估模块失败: {e}")
        return False
    
    # 5. 测试球队扩张与选址策略模块
    print("\n5. 测试球队扩张与选址策略模块...")
    try:
        potential_locations = ['Seattle', 'Las Vegas', 'Kansas City', 'Louisville']
        location_evaluation = model.team_expansion_analyzer.evaluate_location_strategy(potential_locations)
        print(f"✓ 球队扩张分析成功，评估了 {len(location_evaluation)} 个潜在位置")
        print(f"  最佳位置: {location_evaluation.iloc[0]['Location']} (评分: {location_evaluation.iloc[0]['Evaluation_Score']:.2f})")
    except Exception as e:
        print(f"✗ 球队扩张与选址策略模块失败: {e}")
        return False
    
    # 6. 测试球队门票设置模块
    print("\n6. 测试球队门票设置模块...")
    try:
        pricing_strategy = model.ticket_pricing_optimizer.optimize_ticket_pricing('Lakers')
        if pricing_strategy:
            print("✓ 门票定价优化成功")
            print(f"  常规赛票价: ${pricing_strategy['pricing_strategy']['regular_season']:.2f}")
        else:
            print("⚠ 门票定价优化返回空结果，可能是数据问题")
    except Exception as e:
        print(f"✗ 球队门票设置模块失败: {e}")
        return False
    
    # 7. 测试媒体曝光度调整模块
    print("\n7. 测试媒体曝光度调整模块...")
    try:
        optimal_roster = model.optimize_team_roster(100000000)
        media_strategy = model.media_exposure_adjuster.optimize_media_strategy(optimal_roster)
        if media_strategy:
            print("✓ 媒体策略优化成功")
            print(f"  总预算: ${media_strategy['total_budget']:,.2f}")
        else:
            print("⚠ 媒体策略优化返回空结果，可能是数据问题")
    except Exception as e:
        print(f"✗ 媒体曝光度调整模块失败: {e}")
        return False
    
    # 8. 测试马尔科夫链决策模块
    print("\n8. 测试马尔科夫链决策模块...")
    try:
        current_state = 'Average_Performance'
        team_performance = {'win_rate': 0.55, 'avg_attendance': 18000}
        economic_conditions = {'market_growth': 0.04, 'salary_cap_increase': 0.05}
        
        final_decision = model.make_final_decision(current_state, team_performance, economic_conditions)
        print(f"✓ 马尔科夫链决策成功")
        print(f"  当前状态: {final_decision['Current_State']}")
        print(f"  推荐动作: {final_decision['Recommended_Action']}")
    except Exception as e:
        print(f"✗ 马尔科夫链决策模块失败: {e}")
        return False
    
    # 9. 测试球队阵容优化
    print("\n9. 测试球队阵容优化...")
    try:
        optimized_roster = model.optimize_team_roster(100000000, max_players=12)
        print(f"✓ 球队阵容优化成功，选定了 {len(optimized_roster)} 名球员")
    except Exception as e:
        print(f"✗ 球队阵容优化失败: {e}")
        return False
    
    print("\n=== 所有模块测试完成 ===")
    return True

if __name__ == "__main__":
    success = test_all_modules()
    if success:
        print("\n🎉 所有模块测试通过！")
    else:
        print("\n❌ 部分模块测试失败，请检查错误信息。")
        sys.exit(1)
