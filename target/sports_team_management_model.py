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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class PlayerValueEvaluator:
    """球员价值评估模块（包括伤病球员）"""
    def __init__(self, team_analysis, player_social_data, team_market_data):
        self.team_analysis = team_analysis
        self.player_social_data = player_social_data
        self.team_market_data = team_market_data
    
    def calculate_per(self, df):
        """计算球员效率评分"""
        per = (df['PTS'] + df['TRB'] + df['AST'] + df['STL'] + df['BLK'] -
              (df['FGA'] - df['FG']) - (df['3PA'] - df['3P']) - (df['FTA'] - df['FT']) +
              0.5 * df['ORB'] + 0.5 * df['AST'] + 1.5 * df['STL'] + 0.75 * df['BLK'] -
              0.5 * df['FTA'] - 0.3 * df['PF'])
        return per
    
    def calculate_value_index(self, df):
        """计算球员价值指标"""
        value_index = (df['PER'] * 0.4 +
                      df['PTS'] * 0.2 +
                      df['TRB'] * 0.15 +
                      df['AST'] * 0.15 +
                      df['STL'] * 0.05 +
                      df['BLK'] * 0.05)
        return value_index
    
    def calculate_financial_contribution(self, player_name, team_name=None):
        """计算球员的财务贡献"""
        base_contribution = 1000000  # 基础商业价值
        
        player_data = self.team_analysis[self.team_analysis['Player'] == player_name]
        if player_data.empty:
            return base_contribution
        
        performance_factor = player_data['PER'].values[0] / 20  # 标准化PER
        
        # 人气贡献（使用实际的粉丝数量）
        fan_count = player_data['Fan Count'].values[0] if 'Fan Count' in player_data.columns else 0
        popularity_factor = min(3.0, max(0.5, fan_count / 2000000))
        
        # 市场大小贡献（使用实际的球队市场数据）
        market_factor = 1.0
        if team_name:
            team_market = self.team_market_data[self.team_market_data['Team'] == team_name]
            if not team_market.empty:
                market_size = team_market['Metro Population (millions)'].values[0]
                market_factor = min(3.0, max(0.5, market_size / 3))
        
        # 社交媒体影响力（基于粉丝数量）
        social_media_factor = min(2.0, max(0.8, fan_count / 1000000))
        
        # 商业代言潜力（基于表现和人气的综合）
        endorsement_factor = min(2.0, max(0.8, (performance_factor * 0.6 + popularity_factor * 0.4)))
        
        # 计算总财务贡献
        financial_contribution = (base_contribution * performance_factor *
                                 popularity_factor * market_factor *
                                 social_media_factor * endorsement_factor)
        
        return max(base_contribution, financial_contribution)
    
    def calculate_balanced_value(self, player_name, team_name=None, performance_weight=0.6, financial_weight=0.4):
        """计算平衡的球员价值（表现+财务贡献）"""
        player_data = self.team_analysis[self.team_analysis['Player'] == player_name]
        if player_data.empty:
            return 0
        
        performance_value = player_data['Value_Index'].values[0]
        financial_value = self.calculate_financial_contribution(player_name, team_name) / 1000000  # 标准化
        
        balanced_value = (performance_value * performance_weight +
                         financial_value * financial_weight)
        
        return balanced_value
    
    def evaluate_injured_player(self, player_name, injury_severity=0.5):
        """评估伤病球员的价值"""
        player_data = self.team_analysis[self.team_analysis['Player'] == player_name]
        if player_data.empty:
            return 0
        
        # 计算健康状态下的价值
        healthy_value = self.calculate_balanced_value(player_name)
        
        # 根据伤病严重程度调整价值
        # injury_severity: 0-1，0表示完全健康，1表示完全无法上场
        injured_value = healthy_value * (1 - injury_severity * 0.7)  # 伤病最多影响70%的价值
        
        return injured_value

class TeamExpansionAnalyzer:
    """球队扩张与选址策略模块"""
    def __init__(self, team_market_data):
        self.team_market_data = team_market_data
    
    def analyze_league_expansion(self, new_team_location):
        """分析联盟扩张影响"""
        impact_analysis = []
        for _, team_data in self.team_market_data.iterrows():
            team = team_data['Team']
            market_size = team_data['Metro Population (millions)']
            
            # 计算影响因素
            if new_team_location.lower() in team.lower():
                distance_factor = 0.8  # 同城球队影响最大
            elif any(city in team.lower() for city in ['Los Angeles', 'New York', 'Chicago', 'Boston'] if city.lower() in new_team_location.lower()):
                distance_factor = 0.5  # 同一大城市的球队影响较大
            else:
                distance_factor = 0.2  # 其他球队影响较小
            
            # 计算影响
            impact = market_size * distance_factor
            impact_analysis.append({'Team': team, 'Market_Size': market_size, 'Impact': impact})
        
        impact_df = pd.DataFrame(impact_analysis)
        impact_df = impact_df.sort_values('Impact', ascending=False)
        
        return impact_df
    
    def evaluate_location_strategy(self, potential_locations):
        """评估潜在的扩张选址策略"""
        location_evaluations = []
        
        for location in potential_locations:
            # 分析每个潜在位置的影响
            impact_df = self.analyze_league_expansion(location)
            
            # 计算总影响和平均影响
            total_impact = impact_df['Impact'].sum()
            avg_impact = impact_df['Impact'].mean()
            
            # 计算市场潜力（基于位置的市场大小估算）
            # 这里使用简化的市场潜力计算，实际应用中可以使用更详细的市场数据
            market_potential = self._estimate_market_potential(location)
            
            # 综合评估分数
            evaluation_score = (total_impact * 0.4 +
                               market_potential * 0.4 +
                               (1 - avg_impact / total_impact) * 0.2)  # 影响分布的均衡性
            
            location_evaluations.append({
                'Location': location,
                'Total_Impact': total_impact,
                'Avg_Impact': avg_impact,
                'Market_Potential': market_potential,
                'Evaluation_Score': evaluation_score
            })
        
        evaluation_df = pd.DataFrame(location_evaluations)
        evaluation_df = evaluation_df.sort_values('Evaluation_Score', ascending=False)
        
        return evaluation_df
    
    def _estimate_market_potential(self, location):
        """估算位置的市场潜力"""
        # 基于城市大小和体育市场成熟度估算市场潜力
        major_cities = {
            'New York': 10.0,
            'Los Angeles': 9.5,
            'Chicago': 8.5,
            'Houston': 7.5,
            'Phoenix': 7.0,
            'Philadelphia': 7.0,
            'San Antonio': 6.5,
            'San Diego': 6.5,
            'Dallas': 6.5,
            'San Francisco': 6.0,
            'Seattle': 5.5,
            'Denver': 5.0,
            'Portland': 4.5,
            'Sacramento': 4.0,
            'Las Vegas': 3.5
        }
        
        return major_cities.get(location, 3.0)

class TicketPricingOptimizer:
    """球队门票设置模块"""
    def __init__(self, ticket_revenue_data, team_market_data):
        self.ticket_revenue_data = ticket_revenue_data
        self.team_market_data = team_market_data
    
    def optimize_ticket_pricing(self, team_name):
        """优化门票定价策略"""
        # 获取球队门票数据
        team_data = None
        selected_season = None
        for season, teams in self.ticket_revenue_data.items():
            for available_team, data in teams.items():
                if team_name.lower() in available_team.lower():
                    team_data = data
                    selected_season = season
                    break
            if team_data is not None:
                break
        
        if team_data is None:
            return None
        
        # 找到正确的 attendance 列名
        attendance_col = None
        for col in team_data.columns:
            if 'attend' in col.lower():
                attendance_col = col
                break
        
        if not attendance_col:
            return None
        
        # 计算平均 attendance
        avg_attendance = team_data[attendance_col].mean()
        max_attendance = team_data[attendance_col].max()
        min_attendance = team_data[attendance_col].min()
        
        # 获取球队市场大小
        team_market = None
        for _, data in self.team_market_data.iterrows():
            if team_name.lower() in data['Team'].lower():
                team_market = data
                break
        
        market_factor = 1.0
        if team_market:
            market_size = team_market['Metro Population (millions)']
            market_factor = min(3.0, max(0.5, market_size / 3))
        
        # 基于 attendance 和市场因子计算基础票价
        base_price = 80  # 基础票价
        attendance_factor = min(1.5, max(0.7, avg_attendance / 15000))
        optimal_price = base_price * attendance_factor * market_factor
        
        # 考虑不同类型比赛的票价差异
        pricing_strategy = {
            'regular_season': optimal_price,
            'rivalry_games': optimal_price * 1.3,  # rivalry比赛票价高30%
            'playoff_games': optimal_price * 1.8,  # 季后赛票价高80%
            'premium_seating': optimal_price * 2.5  # 高级座位票价高150%
        }
        
        # 计算预计收入
        estimated_revenue = {
            'regular_season_single_game': pricing_strategy['regular_season'] * avg_attendance,
            'rivalry_game_single_game': pricing_strategy['rivalry_games'] * (avg_attendance * 1.2),
            'playoff_game_single_game': pricing_strategy['playoff_games'] * (avg_attendance * 1.5),
            'regular_season_total': pricing_strategy['regular_season'] * avg_attendance * 41  # 常规赛41主场
        }
        
        return {
            'pricing_strategy': pricing_strategy,
            'estimated_revenue': estimated_revenue,
            'attendance_metrics': {
                'avg_attendance': avg_attendance,
                'max_attendance': max_attendance,
                'min_attendance': min_attendance
            }
        }

class MediaExposureAdjuster:
    """媒体曝光度调整模块"""
    def __init__(self, player_social_data):
        self.player_social_data = player_social_data
    
    def analyze_social_media_influence(self):
        """分析球员社交媒体影响力"""
        if 'Fan Count' not in self.player_social_data.columns:
            return None
        
        # 清理和分析粉丝数据
        social_df = self.player_social_data.copy()
        social_df = social_df[social_df['Fan Count'] > 0]
        
        if social_df.empty:
            return None
        
        # 计算社交媒体影响力指标
        social_df['Social_Influence_Score'] = social_df['Fan Count'] / 1000000  # 转换为百万粉丝单位
        
        # 按影响力排序
        social_df = social_df.sort_values('Social_Influence_Score', ascending=False)
        
        return social_df
    
    def optimize_media_strategy(self, team_roster, budget=1000000):
        """优化媒体曝光策略"""
        # 分析现有球员的社交媒体影响力
        social_influence = self.analyze_social_media_influence()
        if social_influence is None:
            return None
        
        # 评估团队的媒体曝光度
        team_media_analysis = []
        for _, player in team_roster.iterrows():
            player_name = player['Player']
            player_social = social_influence[social_influence['Player Name'] == player_name]
            
            if not player_social.empty:
                social_score = player_social['Social_Influence_Score'].values[0]
            else:
                social_score = 0.1  # 默认值
            
            # 计算媒体投资回报率
            media_roi = self._calculate_media_roi(player_name, social_score, player['PER'])
            
            team_media_analysis.append({
                'Player': player_name,
                'Social_Score': social_score,
                'PER': player['PER'],
                'Media_ROI': media_roi,
                'Recommended_Investment': min(budget * 0.3, budget * media_roi / sum(item['Media_ROI'] for item in team_media_analysis) if team_media_analysis else budget)
            })
        
        # 分配预算
        total_roi = sum(item['Media_ROI'] for item in team_media_analysis)
        for item in team_media_analysis:
            item['Recommended_Investment'] = budget * item['Media_ROI'] / total_roi
        
        # 制定媒体策略
        media_strategy = {
            'total_budget': budget,
            'player_investments': team_media_analysis,
            'platform_strategy': self._recommend_platform_strategy(team_media_analysis),
            'content_strategy': self._recommend_content_strategy(team_media_analysis)
        }
        
        return media_strategy
    
    def _calculate_media_roi(self, player_name, social_score, per):
        """计算媒体投资回报率"""
        # 基于社交媒体影响力和球员表现计算ROI
        base_roi = 1.0
        social_factor = max(0.5, social_score)
        performance_factor = max(0.5, per / 20)
        
        return base_roi * social_factor * performance_factor
    
    def _recommend_platform_strategy(self, team_media_analysis):
        """推荐平台策略"""
        # 基于球员特点推荐不同平台的投资比例
        platform_strategy = {
            'Instagram': 0.4,  # 图片和短视频平台
            'Twitter/X': 0.25,  # 实时更新和互动
            'YouTube': 0.2,  # 长视频内容
            'TikTok': 0.15  # 短视频趋势内容
        }
        
        # 根据团队平均社交媒体影响力调整策略
        avg_social_score = sum(item['Social_Score'] for item in team_media_analysis) / len(team_media_analysis)
        
        if avg_social_score > 5:  # 高影响力团队
            platform_strategy['TikTok'] += 0.05
            platform_strategy['Twitter/X'] -= 0.05
        elif avg_social_score < 2:  # 低影响力团队
            platform_strategy['YouTube'] += 0.05
            platform_strategy['TikTok'] -= 0.05
        
        return platform_strategy
    
    def _recommend_content_strategy(self, team_media_analysis):
        """推荐内容策略"""
        # 基于球员特点推荐内容类型
        content_strategy = {
            'Game_Highlights': 0.3,  # 比赛精彩瞬间
            'Behind_The_Scenes': 0.25,  # 幕后内容
            'Player_Profiles': 0.2,  # 球员个人简介
            'Community_Engagement': 0.15,  # 社区互动
            'Skill_Tutorials': 0.1  # 技能教程
        }
        
        # 根据团队表现调整策略
        avg_per = sum(item['PER'] for item in team_media_analysis) / len(team_media_analysis)
        
        if avg_per > 18:  # 高表现团队
            content_strategy['Game_Highlights'] += 0.05
            content_strategy['Skill_Tutorials'] += 0.05
            content_strategy['Behind_The_Scenes'] -= 0.1
        elif avg_per < 14:  # 低表现团队
            content_strategy['Community_Engagement'] += 0.1
            content_strategy['Player_Profiles'] += 0.05
            content_strategy['Game_Highlights'] -= 0.15
        
        return content_strategy

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
        # 求解最优策略
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
    
    def load_data(self):
        """加载所有数据"""
        print("加载球员表现数据...")
        # 正确读取分号分隔的CSV文件
        self.player_performance_data = pd.read_csv('d:\\code\\MCM\\data_source\\player_team_and_performance\\2023-2024 NBA Player Stats - Regular.csv', sep=';', encoding='latin1')
        
        print("加载球员薪资数据...")
        self.player_salary_data = pd.read_csv('d:\\code\\MCM\\data_source\\player_salaries\\NBA Player Stats and Salaries_2000-2025.csv', encoding='latin1')
        
        print("加载球员社交影响力数据...")
        try:
            self.player_social_data = pd.read_csv('d:\\code\\MCM\\data_source\\player_social_influence\\player_followers.csv', encoding='latin1')
        except:
            print("使用手动方式处理球员社交影响力数据...")
            data = []
            with open('d:\\code\\MCM\\data_source\\player_social_influence\\player_followers.csv', 'r', encoding='latin1') as f:
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
        self.team_market_data = pd.read_csv('d:\\code\\MCM\\data_source\\team_markets\\2022 NBA Team Market Size.csv', encoding='latin1')
        self.team_market_data.columns = self.team_market_data.columns.str.replace('ï»¿', '')
        
        print("加载门票收入数据...")
        self.ticket_revenue_data = {}
        for season in ['22-23', '23-24']:
            season_data = {}
            for team_file in os.listdir(f'd:\\code\\MCM\\data_source\\tickets_gain\\{season}'):
                team_name = team_file.replace('.txt', '')
                if season == '22-23':
                    team_name = team_name.replace('2022-23 ', '')
                elif season == '23-24':
                    team_name = team_name.replace('2023-24 ', '')
                team_name = team_name.strip()
                
                file_path = f'd:\\code\\MCM\\data_source\\tickets_gain\\{season}\\{team_file}'
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
        if not hasattr(self, 'player_value_evaluator') or self.player_value_evaluator is None:
            # 临时创建评估器计算PER和价值指标
            temp_evaluator = PlayerValueEvaluator(performance_df, self.player_social_data, self.team_market_data)
            performance_df['PER'] = temp_evaluator.calculate_per(performance_df)
            performance_df['Value_Index'] = temp_evaluator.calculate_value_index(performance_df)
        else:
            performance_df['PER'] = self.player_value_evaluator.calculate_per(performance_df)
            performance_df['Value_Index'] = self.player_value_evaluator.calculate_value_index(performance_df)
        
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
        
        self.team_analysis = merged_df
        
        # 初始化各个模块
        self.player_value_evaluator = PlayerValueEvaluator(merged_df, social_df, self.team_market_data)
        self.team_expansion_analyzer = TeamExpansionAnalyzer(self.team_market_data)
        self.ticket_pricing_optimizer = TicketPricingOptimizer(self.ticket_revenue_data, self.team_market_data)
        self.media_exposure_adjuster = MediaExposureAdjuster(social_df)
        
        return merged_df
    
    def optimize_team_roster(self, team_budget, max_players=12):
        """优化球队阵容"""
        print(f"优化球队阵容，预算: ${team_budget:,.2f}")
        
        players = self.team_analysis.copy()
        players['Salary_2024'] = players['2023/2024']
        players = players[players['Salary_2024'] > 0]
        
        # 计算平衡价值
        players['Balanced_Value'] = players.apply(lambda row: 
            self.player_value_evaluator.calculate_balanced_value(row['Player']), axis=1)
        
        players['Value_per_Dollar'] = players['Balanced_Value'] / players['Salary_2024']
        players['Composite_Score'] = (players['Balanced_Value'] * 0.6 + 
                                     players['Value_per_Dollar'] * 1e6 * 0.4)
        
        top_players = players.nlargest(30, 'Composite_Score')
        
        # 转换为列表格式
        player_list = []
        for _, row in top_players.iterrows():
            player_list.append({
                'Player': row['Player'],
                'Salary': int(row['Salary_2024']),
                'Value': float(row['Value_Index'])
            })
        
        n = len(player_list)
        scale_factor = 10000
        max_budget = int(team_budget / scale_factor)
        
        for player in player_list:
            player['Salary'] = max(1, int(player['Salary'] / scale_factor))
        
        # 动态规划优化
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
        
        # 找到最优解
        max_total_value = -float('inf')
        best_j = 0
        best_k = 0
        
        for j in range(max_budget + 1):
            for k in range(1, max_players + 1):
                if dp[j][k] > max_total_value:
                    max_total_value = dp[j][k]
                    best_j = j
                    best_k = k
        
        # 重构选定的球员
        selected_indices = path[best_j][best_k]
        selected_players = []
        total_salary = 0
        
        for idx in selected_indices:
            player = player_list[idx]
            full_player_info = self.team_analysis[self.team_analysis['Player'] == player['Player']].iloc[0]
            full_player_info['Salary_2024'] = full_player_info['2023/2024']
            selected_players.append(full_player_info)
            total_salary += full_player_info['2023/2024']
        
        # 如果没有找到解，使用贪心算法
        if not selected_players:
            print("使用贪心算法作为备选方案")
            sorted_players = players.sort_values('Composite_Score', ascending=False)
            
            selected_players = []
            total_salary = 0
            
            for _, player in sorted_players.iterrows():
                salary = player['2023/2024']
                if total_salary + salary <= team_budget and len(selected_players) < max_players:
                    player['Salary_2024'] = salary
                    selected_players.append(player)
                    total_salary += salary
        
        selected_df = pd.DataFrame(selected_players)
        print(f"选定球员数量: {len(selected_df)}")
        print(f"总薪资: ${total_salary:,.2f}")
        print(f"剩余预算: ${team_budget - total_salary:,.2f}")
        
        return selected_df
    
    def make_final_decision(self, current_state, team_performance=None, economic_conditions=None):
        """使用马尔科夫链进行最终决策"""
        # 使用马尔科夫决策过程做出决策
        recommended_action = self.markov_decision.make_decision(current_state, team_performance, economic_conditions)
        
        # 生成详细的决策建议
        decision_details = {
            'Current_State': current_state,
            'Recommended_Action': recommended_action,
            'Action_Details': self._get_action_details(recommended_action),
            'Expected_Outcomes': self._predict_outcomes(recommended_action, team_performance, economic_conditions)
        }
        
        return decision_details
    
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
    
    print("\n模型运行完成！")
