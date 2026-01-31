import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from scipy.optimize import minimize
import os
import warnings
warnings.filterwarnings('ignore')

# 导入各个模块
from player_value_evaluator import PlayerValueEvaluator
from team_expansion_analyzer import TeamExpansionAnalyzer
from ticket_pricing_optimizer import TicketPricingOptimizer
from media_exposure_adjuster import MediaExposureAdjuster
from visualization import SportsTeamVisualizer

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class MarkovDecisionProcess:
    """马尔科夫链决策模型"""
    def __init__(self):
        self.states = ['Poor_Performance', 'Average_Performance', 'Good_Performance']
        self.actions = ['Invest_In_Players', 'Invest_In_Media', 'Balance_Investment']
        self.state_index = {state: i for i, state in enumerate(self.states)}
        self.action_index = {action: i for i, action in enumerate(self.actions)}
        
        # 状态转移矩阵 (动作 -> 状态 -> 下一状态 -> 概率)
        self.transition_matrix = self._initialize_transition_matrix()
        
        # 奖励函数 (状态 -> 动作 -> 奖励)
        self.reward_matrix = self._initialize_reward_matrix()
    
    def _initialize_transition_matrix(self):
        """初始化状态转移矩阵"""
        # 基于历史数据和专家知识初始化转移矩阵
        transition = {
            'Invest_In_Players': {
                'Poor_Performance': {'Poor_Performance': 0.3, 'Average_Performance': 0.5, 'Good_Performance': 0.2},
                'Average_Performance': {'Poor_Performance': 0.1, 'Average_Performance': 0.6, 'Good_Performance': 0.3},
                'Good_Performance': {'Poor_Performance': 0.05, 'Average_Performance': 0.4, 'Good_Performance': 0.55}
            },
            'Invest_In_Media': {
                'Poor_Performance': {'Poor_Performance': 0.4, 'Average_Performance': 0.4, 'Good_Performance': 0.2},
                'Average_Performance': {'Poor_Performance': 0.2, 'Average_Performance': 0.5, 'Good_Performance': 0.3},
                'Good_Performance': {'Poor_Performance': 0.1, 'Average_Performance': 0.3, 'Good_Performance': 0.6}
            },
            'Balance_Investment': {
                'Poor_Performance': {'Poor_Performance': 0.35, 'Average_Performance': 0.45, 'Good_Performance': 0.2},
                'Average_Performance': {'Poor_Performance': 0.15, 'Average_Performance': 0.6, 'Good_Performance': 0.25},
                'Good_Performance': {'Poor_Performance': 0.05, 'Average_Performance': 0.35, 'Good_Performance': 0.6}
            }
        }
        return transition
    
    def _initialize_reward_matrix(self):
        """初始化奖励矩阵"""
        # 基于预期收益和风险初始化奖励矩阵
        reward = {
            'Poor_Performance': {
                'Invest_In_Players': 50,  # 高投资高回报潜力
                'Invest_In_Media': 30,  # 中等投资中等回报
                'Balance_Investment': 40  # 平衡投资
            },
            'Average_Performance': {
                'Invest_In_Players': 40,  # 维持竞争力
                'Invest_In_Media': 45,  # 提升市场价值
                'Balance_Investment': 55  # 平衡发展
            },
            'Good_Performance': {
                'Invest_In_Players': 30,  # 保持阵容
                'Invest_In_Media': 50,  # 最大化市场价值
                'Balance_Investment': 45  # 平衡发展
            }
        }
        return reward
    
    def value_iteration(self, discount_factor=0.9, epsilon=0.01):
        """值迭代算法求解最优策略"""
        # 初始化价值函数
        V = {state: 0 for state in self.states}
        
        while True:
            delta = 0
            # 对每个状态计算新的价值
            for state in self.states:
                max_value = -float('inf')
                
                # 对每个动作计算预期价值
                for action in self.actions:
                    action_value = self.reward_matrix[state][action]
                    
                    # 加上下一状态的预期价值
                    for next_state, probability in self.transition_matrix[action][state].items():
                        action_value += discount_factor * probability * V[next_state]
                    
                    if action_value > max_value:
                        max_value = action_value
                
                # 更新价值函数并计算变化
                delta = max(delta, abs(max_value - V[state]))
                V[state] = max_value
            
            # 检查收敛
            if delta < epsilon:
                break
        
        # 提取最优策略
        optimal_policy = {}
        for state in self.states:
            max_value = -float('inf')
            best_action = None
            
            for action in self.actions:
                action_value = self.reward_matrix[state][action]
                
                for next_state, probability in self.transition_matrix[action][state].items():
                    action_value += discount_factor * probability * V[next_state]
                
                if action_value > max_value:
                    max_value = action_value
                    best_action = action
            
            optimal_policy[state] = best_action
        
        return optimal_policy, V
    
    def make_decision(self, current_state, team_performance=None, economic_conditions=None):
        """基于当前状态和附加信息做出决策"""
        # 使用值迭代算法求解最优策略
        optimal_policy, _ = self.value_iteration()
        
        # 根据当前状态选择最优动作
        recommended_action = optimal_policy.get(current_state, 'Balance_Investment')
        
        # 根据团队表现和经济条件调整决策
        if team_performance:
            win_rate = team_performance.get('win_rate', 0.5)
            if win_rate < 0.4 and recommended_action != 'Invest_In_Players':
                recommended_action = 'Invest_In_Players'
            elif win_rate > 0.6 and recommended_action != 'Invest_In_Media':
                recommended_action = 'Invest_In_Media'
        
        if economic_conditions:
            market_growth = economic_conditions.get('market_growth', 0.03)
            if market_growth > 0.05 and recommended_action != 'Invest_In_Media':
                recommended_action = 'Invest_In_Media'
            elif market_growth < 0.01 and recommended_action != 'Invest_In_Players':
                recommended_action = 'Invest_In_Players'
        
        return recommended_action

class SportsTeamManagementModel:
    """整合所有模块的主模型"""
    def __init__(self):
        self.player_performance_data = None
        self.player_salary_data = None
        self.player_social_data = None
        self.team_market_data = None
        self.ticket_revenue_data = None
        self.team_analysis = None
        
        # 初始化各个模块
        self.player_value_evaluator = None
        self.team_expansion_analyzer = None
        self.ticket_pricing_optimizer = None
        self.media_exposure_adjuster = None
        self.markov_decision = MarkovDecisionProcess()
        self.visualizer = SportsTeamVisualizer()
    
    def load_data(self):
        """加载所有数据"""
        print("加载球员表现数据...")
        self.player_performance_data = pd.read_csv('d:/code/MCM/data_source/player_team_and_performance/2023-2024 NBA Player Stats - Regular.csv', sep=';', encoding='latin1')
        
        print("加载球员薪资数据...")
        self.player_salary_data = pd.read_csv('d:/code/MCM/data_source/player_salaries/NBA Player Stats and Salaries_2000-2025.csv', encoding='latin1')
        
        print("加载球员社交影响力数据...")
        try:
            self.player_social_data = pd.read_csv('d:/code/MCM/data_source/player_social_influence/player_followers.csv', encoding='latin1')
        except:
            print("使用手动方式处理球员社交影响力数据...")
            data = []
            with open('d:/code/MCM/data_source/player_social_influence/player_followers.csv', 'r', encoding='latin1') as f:
                next(f)  # 跳过表头
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) > 2:
                        player_name = ','.join(parts[:-1])
                        fan_count = parts[-1]
                    else:
                        player_name, fan_count = parts
                    data.append({'Player Name': player_name, 'Fan Count': fan_count})
            self.player_social_data = pd.DataFrame(data)
        
        print("加载球队市场数据...")
        self.team_market_data = pd.read_csv('d:/code/MCM/data_source/team_markets/2022 NBA Team Market Size.csv', encoding='latin1')
        self.team_market_data.columns = self.team_market_data.columns.str.replace('ï»¿', '')
        
        print("加载门票收入数据...")
        self.ticket_revenue_data = {}
        for season in ['22-23', '23-24']:
            season_data = {}
            for team_file in os.listdir(f'd:/code/MCM/data_source/tickets_gain/{season}'):
                team_name = team_file.replace('.txt', '')
                if season == '22-23':
                    team_name = team_name.replace('2022-23 ', '')
                elif season == '23-24':
                    team_name = team_name.replace('2023-24 ', '')
                team_name = team_name.strip()
                
                file_path = f'd:/code/MCM/data_source/tickets_gain/{season}/{team_file}'
                try:
                    team_data = pd.read_csv(file_path, encoding='latin1')
                    team_data = team_data.dropna(axis=1, how='all')
                    for col in team_data.columns:
                        if 'attend' in col.lower():
                            team_data[col] = pd.to_numeric(team_data[col], errors='coerce')
                except Exception as e:
                    print(f"读取{team_name}数据失败: {e}")
                    team_data = pd.DataFrame()
                
                season_data[team_name] = team_data
            self.ticket_revenue_data[season] = season_data
    
    def preprocess_data(self):
        """数据预处理"""
        print("预处理球员表现数据...")
        performance_df = self.player_performance_data.copy()
        performance_df = performance_df[performance_df['Tm'] == 'TOT']
        
        # 计算球员效率评分和价值指标
        # 先创建临时评估器计算PER
        temp_evaluator = PlayerValueEvaluator(performance_df, self.player_social_data, self.team_market_data)
        performance_df['PER'] = temp_evaluator.calculate_per(performance_df)
        performance_df['Value_Index'] = temp_evaluator.calculate_value_index(performance_df)
        
        print("预处理球员薪资数据...")
        salary_df = self.player_salary_data.copy()
        salary_df.columns = salary_df.columns.str.replace('ï»¿', '')
        
        salary_2024 = salary_df[salary_df['Year'] == 2024].copy()
        if salary_2024.empty:
            salary_2024 = salary_df[salary_df['Year'] == 2023].copy()
        
        salary_2024 = salary_2024.rename(columns={'Salary': '2023/2024'})
        salary_2024['2023/2024'] = pd.to_numeric(salary_2024['2023/2024'], errors='coerce').fillna(0)
        
        print("预处理球员社交影响力数据...")
        social_df = self.player_social_data.copy()
        if 'Fan Count' in social_df.columns:
            social_df['Fan Count'] = social_df['Fan Count'].astype(str)
            def convert_fan_count(value):
                if value == 'Fan Count':
                    return 0
                try:
                    if 'M' in value:
                        return float(value.replace('M', '')) * 1000000
                    elif 'K' in value:
                        return float(value.replace('K', '')) * 1000
                    else:
                        return float(value)
                except:
                    return 0
            social_df['Fan Count'] = social_df['Fan Count'].apply(convert_fan_count)
        else:
            social_df['Fan Count'] = 0
        
        print("合并数据...")
        merged_df = pd.merge(performance_df, salary_2024[['Player', '2023/2024']], left_on='Player', right_on='Player', how='inner')
        
        if 'Player Name' in social_df.columns:
            merged_df = pd.merge(merged_df, social_df, left_on='Player', right_on='Player Name', how='left')
        elif 'Player' in social_df.columns:
            merged_df = pd.merge(merged_df, social_df, left_on='Player', right_on='Player', how='left')
        
        if 'Fan Count' not in merged_df.columns:
            merged_df['Fan Count'] = 0
        else:
            merged_df['Fan Count'] = merged_df['Fan Count'].fillna(0)
        
        # 剔除薪资为0的球员
        merged_df = merged_df[merged_df['2023/2024'] > 0]
        print(f"数据预处理后球员数量: {len(merged_df)}")
        
        self.team_analysis = merged_df
        
        # 初始化各个模块
        self.player_value_evaluator = PlayerValueEvaluator(merged_df, social_df, self.team_market_data)
        self.team_expansion_analyzer = TeamExpansionAnalyzer(self.team_market_data)
        self.ticket_pricing_optimizer = TicketPricingOptimizer(self.ticket_revenue_data, self.team_market_data)
        self.media_exposure_adjuster = MediaExposureAdjuster(social_df)
        
        # 可视化球员价值分布
        print("生成球员价值分布可视化...")
        self.visualizer.visualize_player_value_distribution(merged_df)
        
        return merged_df
    
    def optimize_team_roster(self, team_budget, max_players=12, min_players=8):
        """优化球队阵容"""
        print(f"优化球队阵容，预算: ${team_budget:,.2f}")
        
        players = self.team_analysis.copy()
        players['Salary_2024'] = players['2023/2024']
        # 剔除薪资为0的球员
        players = players[players['Salary_2024'] > 0]
        print(f"筛选后球员数量: {len(players)}")
        
        # 计算平衡价值
        players['Balanced_Value'] = players.apply(lambda row: 
            self.player_value_evaluator.calculate_balanced_value(row['Player']), axis=1)
        
        # 计算球员的商业价值
        players['Commercial_Value'] = players.apply(lambda row: 
            self.player_value_evaluator.calculate_financial_contribution(row['Player']) / 1000000, axis=1)
        
        # 计算风险评分（基于简化的伤病风险模型）
        players['Risk_Score'] = players.apply(lambda row: 
            self._calculate_player_risk(row['Player']), axis=1)
        
        players['Value_per_Dollar'] = players['Balanced_Value'] / players['Salary_2024']
        players['Composite_Score'] = (players['Balanced_Value'] * 0.4 + 
                                     players['Commercial_Value'] * 0.3 +
                                     players['Value_per_Dollar'] * 1e6 * 0.2 +
                                     (1 - players['Risk_Score']) * 0.1)
        
        top_players = players.nlargest(30, 'Composite_Score')
        
        # 转换为列表格式
        player_list = []
        for _, row in top_players.iterrows():
            player_list.append({
                'Player': row['Player'],
                'Salary': int(row['Salary_2024']),
                'Value': float(row['Composite_Score']),
                'Risk': float(row['Risk_Score']),
                'Commercial': float(row['Commercial_Value'])
            })
        
        n = len(player_list)
        scale_factor = 10000
        max_budget = int(team_budget / scale_factor)
        
        for player in player_list:
            player['Salary'] = max(1, int(player['Salary'] / scale_factor))
        
        # 动态规划优化（考虑多约束）
        dp = [[-float('inf')] * (max_players + 1) for _ in range(max_budget + 1)]
        dp[0][0] = 0
        path = [[[] for _ in range(max_players + 1)] for __ in range(max_budget + 1)]
        
        for i in range(n):
            player = player_list[i]
            salary = player['Salary']
            value = player['Value']
            
            for j in range(max_budget, salary - 1, -1):
                for k in range(max_players, 0, -1):
                    if dp[j - salary][k - 1] != -float('inf') and dp[j - salary][k - 1] + value > dp[j][k]:
                        dp[j][k] = dp[j - salary][k - 1] + value
                        path[j][k] = path[j - salary][k - 1].copy()
                        path[j][k].append(i)
        
        # 找到最优解（满足阵容人数要求）
        max_total_value = -float('inf')
        best_j = 0
        best_k = 0
        
        for j in range(max_budget + 1):
            for k in range(min_players, max_players + 1):
                if dp[j][k] > max_total_value:
                    max_total_value = dp[j][k]
                    best_j = j
                    best_k = k
        
        # 重构选定的球员
        selected_indices = path[best_j][best_k]
        selected_players = []
        total_salary = 0
        total_risk = 0
        total_commercial = 0
        
        for idx in selected_indices:
            player = player_list[idx]
            full_player_info = self.team_analysis[self.team_analysis['Player'] == player['Player']].iloc[0]
            full_player_info['Salary_2024'] = full_player_info['2023/2024']
            selected_players.append(full_player_info)
            total_salary += full_player_info['2023/2024']
            total_risk += player['Risk']
            total_commercial += player['Commercial']
        
        # 如果没有找到解，使用贪心算法
        if not selected_players:
            print("使用贪心算法作为备选方案")
            sorted_players = players.sort_values('Composite_Score', ascending=False)
            
            selected_players = []
            total_salary = 0
            total_risk = 0
            total_commercial = 0
            
            for _, player in sorted_players.iterrows():
                salary = player['2023/2024']
                if total_salary + salary <= team_budget and len(selected_players) < max_players:
                    player['Salary_2024'] = salary
                    selected_players.append(player)
                    total_salary += salary
                    total_risk += player['Risk_Score']
                    total_commercial += player['Commercial_Value']
        
        selected_df = pd.DataFrame(selected_players)
        print(f"选定球员数量: {len(selected_df)}")
        print(f"总薪资: ${total_salary:,.2f}")
        print(f"剩余预算: ${team_budget - total_salary:,.2f}")
        print(f"平均风险评分: {total_risk / len(selected_df):.2f}")
        print(f"总商业价值: ${total_commercial * 1000000:,.2f}")
        
        # 可视化阵容优化结果
        print("生成阵容优化可视化...")
        self.visualizer.visualize_roster_optimization(selected_df)
        
        return selected_df
    
    def _calculate_player_risk(self, player_name):
        """计算球员风险评分"""
        # 简化的伤病风险模型
        # 基于球员表现和假设的伤病风险
        player_data = self.team_analysis[self.team_analysis['Player'] == player_name]
        if player_data.empty:
            return 0.5
        
        per = player_data['PER'].values[0]
        # 假设高PER球员使用率更高，风险也更高
        risk_score = min(1.0, max(0.1, per / 30))
        
        return risk_score
    
    def make_final_decision(self, current_state, team_performance=None, economic_conditions=None, risk_aversion=0.1):
        """使用马尔科夫链进行最终决策"""
        # 使用马尔科夫决策过程做出决策
        recommended_action = self.markov_decision.make_decision(current_state, team_performance, economic_conditions)
        
        # 计算风险调整后的预期结果
        expected_outcomes = self._predict_outcomes(recommended_action, team_performance, economic_conditions)
        
        # 加入风险惩罚项
        risk_adjusted_outcomes = self._adjust_for_risk(expected_outcomes, recommended_action, risk_aversion)
        
        # 生成详细的决策建议
        decision_details = {
            'Current_State': current_state,
            'Recommended_Action': recommended_action,
            'Action_Details': self._get_action_details(recommended_action),
            'Expected_Outcomes': expected_outcomes,
            'Risk_Adjusted_Outcomes': risk_adjusted_outcomes,
            'Risk_Aversion': risk_aversion
        }
        
        return decision_details
    
    def _adjust_for_risk(self, outcomes, action, risk_aversion):
        """根据风险偏好调整预期结果"""
        risk_adjusted = outcomes.copy()
        
        # 不同动作的风险水平
        action_risk = {
            'Invest_In_Players': 0.8,  # 高风险
            'Invest_In_Media': 0.4,  # 中等风险
            'Balance_Investment': 0.6  # 中等风险
        }
        
        risk_level = action_risk.get(action, 0.6)
        
        # 调整预期结果以反映风险
        for key, value in risk_adjusted.items():
            if key in ['Win_Rate_Change', 'Revenue_Growth', 'Fan_Base_Growth']:
                # 风险调整：降低高风险动作的预期收益
                risk_adjusted[key] = value * (1 - risk_aversion * risk_level)
        
        return risk_adjusted
    
    def _get_action_details(self, action):
        """获取动作的详细信息"""
        action_details = {
            'Invest_In_Players': {
                'Description': '增加球员薪资投入，追求竞技表现提升',
                'Key_Initiatives': ['签下顶级自由球员', '通过交易获取明星球员', '优化选秀策略'],
                'Expected_Cost': '高',
                'Expected_Return': '长期竞技表现提升'
            },
            'Invest_In_Media': {
                'Description': '增加媒体和市场营销投入，提升品牌价值',
                'Key_Initiatives': ['社交媒体策略优化', '内容营销升级', '粉丝互动活动'],
                'Expected_Cost': '中等',
                'Expected_Return': '短期商业价值提升'
            },
            'Balance_Investment': {
                'Description': '平衡球员和媒体投入，追求全面发展',
                'Key_Initiatives': ['核心球员续约', '适度媒体投入', '青训体系建设'],
                'Expected_Cost': '中等',
                'Expected_Return': '可持续发展'
            }
        }
        
        return action_details.get(action, {})
    
    def _predict_outcomes(self, action, team_performance, economic_conditions):
        """预测决策结果"""
        base_outcomes = {
            'Invest_In_Players': {
                'Win_Rate_Change': 0.08,
                'Revenue_Growth': 0.04,
                'Fan_Base_Growth': 0.05
            },
            'Invest_In_Media': {
                'Win_Rate_Change': 0.03,
                'Revenue_Growth': 0.07,
                'Fan_Base_Growth': 0.09
            },
            'Balance_Investment': {
                'Win_Rate_Change': 0.05,
                'Revenue_Growth': 0.05,
                'Fan_Base_Growth': 0.06
            }
        }
        
        outcomes = base_outcomes.get(action, base_outcomes['Balance_Investment'])
        
        # 根据当前条件调整预测
        if team_performance:
            win_rate = team_performance.get('win_rate', 0.5)
            if win_rate < 0.4:
                outcomes['Win_Rate_Change'] *= 1.2
            elif win_rate > 0.6:
                outcomes['Win_Rate_Change'] *= 0.8
        
        if economic_conditions:
            market_growth = economic_conditions.get('market_growth', 0.03)
            outcomes['Revenue_Growth'] *= (1 + market_growth)
        
        return outcomes
    
    def generate_visualization_dashboard(self):
        """生成综合可视化仪表盘"""
        print("生成综合可视化仪表盘...")
        
        # 收集模型结果
        model_results = {
            'player_data': self.team_analysis
        }
        
        # 分析球队扩张策略
        try:
            potential_locations = ['Seattle', 'Las Vegas', 'Kansas City', 'Louisville']
            expansion_data = self.team_expansion_analyzer.evaluate_location_strategy(potential_locations)
            model_results['expansion_data'] = expansion_data
            self.visualizer.visualize_team_expansion_impact(expansion_data)
        except Exception as e:
            print(f"球队扩张分析可视化失败: {e}")
        
        # 分析门票定价策略
        try:
            pricing_data = self.ticket_pricing_optimizer.optimize_ticket_pricing('Lakers')
            if pricing_data:
                model_results['pricing_data'] = pricing_data
                self.visualizer.visualize_ticket_pricing(pricing_data)
        except Exception as e:
            print(f"门票定价分析可视化失败: {e}")
        
        # 分析媒体策略
        try:
            optimal_roster = self.optimize_team_roster(100000000)
            media_data = self.media_exposure_adjuster.optimize_media_strategy(optimal_roster)
            if media_data:
                model_results['media_data'] = media_data
                self.visualizer.visualize_media_strategy(media_data)
        except Exception as e:
            print(f"媒体策略分析可视化失败: {e}")
        
        # 分析马尔可夫决策过程
        try:
            self.visualizer.visualize_markov_decision(self.markov_decision)
        except Exception as e:
            print(f"马尔可夫决策可视化失败: {e}")
        
        # 分析阵容优化
        try:
            optimal_roster = self.optimize_team_roster(100000000)
            model_results['roster_data'] = optimal_roster
        except Exception as e:
            print(f"阵容优化可视化失败: {e}")
        
        # 分析决策结果
        try:
            decision_data = self.make_final_decision('Average_Performance')
            model_results['decision_data'] = decision_data
        except Exception as e:
            print(f"决策结果可视化失败: {e}")
        
        # 生成综合仪表盘
        dashboard_path = self.visualizer.generate_dashboard(model_results)
        print(f"综合仪表盘已生成: {dashboard_path}")
        
        return dashboard_path

# 主函数
if __name__ == "__main__":
    print("初始化体育团队管理模型...")
    model = SportsTeamManagementModel()
    
    print("加载数据...")
    model.load_data()
    
    print("预处理数据...")
    model.preprocess_data()
    
    # 1. 评估球员价值（包括伤病球员）
    print("\n1. 评估球员价值...")
    player_value = model.player_value_evaluator.calculate_balanced_value('Precious Achiuwa')
    print(f"Precious Achiuwa 的平衡价值: {player_value:.2f}")
    
    injured_value = model.player_value_evaluator.evaluate_injured_player('Precious Achiuwa', injury_severity=0.5)
    print(f"Precious Achiuwa 伤病状态下的价值: {injured_value:.2f}")
    
    # 2. 分析球队扩张与选址策略
    print("\n2. 分析球队扩张与选址策略...")
    potential_locations = ['Seattle', 'Las Vegas', 'Kansas City', 'Louisville']
    location_evaluation = model.team_expansion_analyzer.evaluate_location_strategy(potential_locations)
    print("潜在扩张位置评估:")
    print(location_evaluation[['Location', 'Evaluation_Score', 'Market_Potential']])
    
    # 3. 球队门票设置
    print("\n3. 球队门票设置...")
    pricing_strategy = model.ticket_pricing_optimizer.optimize_ticket_pricing('Lakers')
    if pricing_strategy:
        print("门票定价策略:")
        for game_type, price in pricing_strategy['pricing_strategy'].items():
            print(f"{game_type}: ${price:.2f}")
    
    # 4. 媒体曝光度调整
    print("\n4. 媒体曝光度调整...")
    optimal_roster = model.optimize_team_roster(100000000)
    media_strategy = model.media_exposure_adjuster.optimize_media_strategy(optimal_roster)
    if media_strategy:
        print("媒体策略推荐:")
        print(f"总预算: ${media_strategy['total_budget']:,.2f}")
        print("平台投资分配:")
        for platform, percentage in media_strategy['platform_strategy'].items():
            print(f"{platform}: {percentage:.2%}")
    
    # 5. 使用马尔科夫链进行最终决策
    print("\n5. 最终决策（基于马尔科夫链）...")
    current_state = 'Average_Performance'
    team_performance = {'win_rate': 0.55, 'avg_attendance': 18000}
    economic_conditions = {'market_growth': 0.04, 'salary_cap_increase': 0.05}
    
    final_decision = model.make_final_decision(current_state, team_performance, economic_conditions)
    print(f"当前状态: {final_decision['Current_State']}")
    print(f"推荐动作: {final_decision['Recommended_Action']}")
    print(f"动作描述: {final_decision['Action_Details']['Description']}")
    print("关键举措:")
    for initiative in final_decision['Action_Details']['Key_Initiatives']:
        print(f"- {initiative}")
    print("预期结果:")
    for outcome, value in final_decision['Expected_Outcomes'].items():
        print(f"- {outcome}: {value:.2f}")
    
    # 生成综合可视化仪表盘
    print("\n生成综合可视化仪表盘...")
    model.generate_visualization_dashboard()
    
    print("\n模型运行完成！")
