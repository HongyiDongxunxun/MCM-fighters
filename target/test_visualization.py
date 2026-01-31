#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试修改后的可视化模块
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualization import SportsTeamVisualizer
from main_model import SportsTeamManagementModel


def test_visualization():
    """测试可视化模块"""
    print("=== 测试可视化模块 ===")
    
    # 1. 初始化可视化器
    print("\n1. 初始化可视化器...")
    try:
        visualizer = SportsTeamVisualizer()
        print("✓ 可视化器初始化成功")
    except Exception as e:
        print(f"✗ 可视化器初始化失败: {e}")
        return False
    
    # 2. 加载数据
    print("\n2. 加载测试数据...")
    try:
        # 初始化主模型以获取测试数据
        model = SportsTeamManagementModel()
        model.load_data()
        player_data = model.preprocess_data()
        print(f"✓ 测试数据加载成功，获取了 {len(player_data)} 条球员数据")
    except Exception as e:
        print(f"✗ 测试数据加载失败: {e}")
        return False
    
    # 3. 测试球员价值分布可视化
    print("\n3. 测试球员价值分布可视化...")
    try:
        value_files = visualizer.visualize_player_value_distribution(player_data)
        print(f"✓ 球员价值分布可视化成功，生成了 {len(value_files)} 个文件")
        for file in value_files[:3]:  # 只显示前3个文件
            print(f"  - {os.path.basename(file)}")
    except Exception as e:
        print(f"✗ 球员价值分布可视化失败: {e}")
        return False
    
    # 4. 测试球队扩张影响可视化
    print("\n4. 测试球队扩张影响可视化...")
    try:
        # 获取扩张数据
        expansion_data = model.team_expansion_analyzer.evaluate_location_strategy(['Seattle', 'Las Vegas', 'Kansas City', 'Louisville'])
        expansion_files = visualizer.visualize_team_expansion_impact(expansion_data)
        print(f"✓ 球队扩张影响可视化成功，生成了 {len(expansion_files)} 个文件")
        for file in expansion_files[:3]:  # 只显示前3个文件
            print(f"  - {os.path.basename(file)}")
    except Exception as e:
        print(f"✗ 球队扩张影响可视化失败: {e}")
        return False
    
    # 5. 测试门票定价可视化
    print("\n5. 测试门票定价可视化...")
    try:
        # 获取定价数据
        pricing_data = model.ticket_pricing_optimizer.optimize_ticket_pricing('Lakers')
        if pricing_data:
            pricing_files = visualizer.visualize_ticket_pricing(pricing_data)
            print(f"✓ 门票定价可视化成功，生成了 {len(pricing_files)} 个文件")
            for file in pricing_files[:3]:  # 只显示前3个文件
                print(f"  - {os.path.basename(file)}")
        else:
            print("⚠ 门票定价数据为空，跳过可视化测试")
    except Exception as e:
        print(f"✗ 门票定价可视化失败: {e}")
        return False
    
    # 6. 测试媒体策略可视化
    print("\n6. 测试媒体策略可视化...")
    try:
        # 直接使用默认的媒体策略数据来测试可视化功能
        media_files = visualizer.visualize_media_strategy(None)
        print(f"✓ 媒体策略可视化成功，生成了 {len(media_files)} 个文件")
        for file in media_files[:3]:  # 只显示前3个文件
            print(f"  - {os.path.basename(file)}")
    except Exception as e:
        print(f"✗ 媒体策略可视化失败: {e}")
        return False
    
    # 7. 测试马尔可夫决策过程可视化
    print("\n7. 测试马尔可夫决策过程可视化...")
    try:
        # 创建一个默认的马尔可夫模型进行测试
        from main_model import MarkovDecisionProcess
        markov_model = MarkovDecisionProcess()
        markov_files = visualizer.visualize_markov_decision(markov_model)
        print(f"✓ 马尔可夫决策过程可视化成功，生成了 {len(markov_files)} 个文件")
        for file in markov_files[:3]:  # 只显示前3个文件
            print(f"  - {os.path.basename(file)}")
    except Exception as e:
        print(f"✗ 马尔可夫决策过程可视化失败: {e}")
        return False
    
    # 8. 测试球队阵容优化可视化
    print("\n8. 测试球队阵容优化可视化...")
    try:
        # 使用player_data创建一个模拟的球队阵容数据
        # 选择前12名球员作为模拟的优化阵容
        optimized_roster = player_data.nlargest(12, 'Value_Index').copy()
        # 确保必要的列存在
        if 'Risk_Score' not in optimized_roster.columns:
            optimized_roster['Risk_Score'] = np.random.uniform(0.1, 0.9, len(optimized_roster))
        if 'Pos' not in optimized_roster.columns:
            optimized_roster['Pos'] = np.random.choice(['PG', 'SG', 'SF', 'PF', 'C'], len(optimized_roster))
        if 'Age' not in optimized_roster.columns:
            optimized_roster['Age'] = np.random.randint(20, 35, len(optimized_roster))
        # 打印优化阵容的信息，以便调试
        print(f"优化阵容信息: {len(optimized_roster)} 名球员")
        print(f"列名: {list(optimized_roster.columns)}")
        roster_files = visualizer.visualize_roster_optimization(optimized_roster)
        print(f"✓ 球队阵容优化可视化成功，生成了 {len(roster_files)} 个文件")
        for file in roster_files[:3]:  # 只显示前3个文件
            print(f"  - {os.path.basename(file)}")
    except Exception as e:
        print(f"✗ 球队阵容优化可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 9. 测试综合仪表盘可视化
    print("\n9. 测试综合仪表盘可视化...")
    try:
        # 准备仪表盘数据
        model_results = {
            'player_data': player_data,
            'expansion_data': expansion_data,
            'pricing_data': pricing_data,
            'roster_data': optimized_roster
        }
        dashboard_files = visualizer.generate_dashboard(model_results)
        print(f"✓ 综合仪表盘可视化成功，生成了 {len(dashboard_files)} 个文件")
        for file in dashboard_files[:3]:  # 只显示前3个文件
            print(f"  - {os.path.basename(file)}")
    except Exception as e:
        print(f"✗ 综合仪表盘可视化失败: {e}")
        return False
    
    print("\n=== 可视化模块测试完成 ===")
    return True


if __name__ == "__main__":
    success = test_visualization()
    if success:
        print("\n🎉 所有可视化测试通过！")
        print("\n生成的可视化文件位于: d:\code\MCM\visualizations 目录")
    else:
        print("\n❌ 部分可视化测试失败，请检查错误信息。")
        sys.exit(1)
