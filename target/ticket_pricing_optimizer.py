import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class TicketPricingOptimizer:
    """球队门票设置模块"""
    def __init__(self, ticket_revenue_data, team_market_data):
        self.ticket_revenue_data = ticket_revenue_data
        self.team_market_data = team_market_data
        self.logistic_model = None
        self.scaler = None
    
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
        
        # 价格歧视策略：一级价格歧视
        pricing_strategy = self._calculate_price_discrimination(optimal_price)
        
        # 计算预计收入
        estimated_revenue = {
            'regular_season_single_game': pricing_strategy['regular_season'] * avg_attendance,
            'rivalry_game_single_game': pricing_strategy['rivalry_games'] * (avg_attendance * 1.2),
            'playoff_game_single_game': pricing_strategy['playoff_games'] * (avg_attendance * 1.5),
            'premium_seating': pricing_strategy['premium_seating'] * (avg_attendance * 0.2),
            'regular_season_total': pricing_strategy['regular_season'] * avg_attendance * 41  # 常规赛41主场
        }
        
        # 预测季票转化率
        season_ticket_conversion = self._predict_season_ticket_conversion(team_name, optimal_price)
        
        return {
            'pricing_strategy': pricing_strategy,
            'estimated_revenue': estimated_revenue,
            'attendance_metrics': {
                'avg_attendance': avg_attendance,
                'max_attendance': max_attendance,
                'min_attendance': min_attendance
            },
            'season_ticket_conversion': season_ticket_conversion,
            'revenue_optimization': self._optimize_revenue(pricing_strategy, avg_attendance)
        }
    
    def _calculate_price_discrimination(self, base_price):
        """计算价格歧视策略"""
        # 一级价格歧视：根据不同因素制定差异化票价
        pricing_strategy = {
            'regular_season': base_price,
            'rivalry_games': base_price * 1.3,  #  rivalry比赛票价高30%
            'playoff_games': base_price * 1.8,  # 季后赛票价高80%
            'premium_seating': base_price * 2.5,  # 高级座位票价高150%
            'weekend_games': base_price * 1.1,  # 周末比赛票价高10%
            'weekday_games': base_price * 0.9,  # 工作日比赛票价低10%
            'star_player_games': base_price * 1.2,  # 有明星球员的比赛票价高20%
            'early_season': base_price * 0.95,  # 赛季初期票价低5%
            'late_season': base_price * 1.05  # 赛季末期票价高5%
        }
        return pricing_strategy
    
    def _predict_season_ticket_conversion(self, team_name, base_price):
        """预测季票转化率"""
        # 简化的逻辑回归模型预测季票转化率
        # 实际应用中应使用历史数据训练模型
        
        # 特征变量
        features = {
            'base_price': base_price,
            'team_performance': 0.6,  # 假设球队胜率
            'market_size': 2.5,  # 假设市场大小
            'fan_loyalty': 0.7  # 假设球迷忠诚度
        }
        
        # 简化的转化率计算
        # 票价越高，转化率越低
        price_factor = max(0.1, 1.0 - (base_price - 50) / 200)
        # 球队表现越好，转化率越高
        performance_factor = features['team_performance'] * 0.3
        # 市场越大，转化率越高
        market_factor = features['market_size'] / 5 * 0.2
        # 球迷忠诚度越高，转化率越高
        loyalty_factor = features['fan_loyalty'] * 0.2
        
        conversion_rate = price_factor + performance_factor + market_factor + loyalty_factor
        conversion_rate = min(0.8, max(0.1, conversion_rate))
        
        return conversion_rate
    
    def _optimize_revenue(self, pricing_strategy, avg_attendance):
        """优化收入"""
        # 目标函数：最大化收入
        # 约束：平衡票价和上座率
        
        # 计算不同票价策略的预期收入
        revenue_scenarios = {
            'current_strategy': sum(self._calculate_scenario_revenue(pricing_strategy, avg_attendance, 1.0)),
            'higher_prices': sum(self._calculate_scenario_revenue(pricing_strategy, avg_attendance, 1.1)),
            'lower_prices': sum(self._calculate_scenario_revenue(pricing_strategy, avg_attendance, 0.9))
        }
        
        # 选择最优策略
        optimal_scenario = max(revenue_scenarios, key=revenue_scenarios.get)
        
        return {
            'optimal_scenario': optimal_scenario,
            'revenue_projections': revenue_scenarios,
            'recommendation': self._get_revenue_recommendation(optimal_scenario)
        }
    
    def _calculate_scenario_revenue(self, pricing_strategy, avg_attendance, price_multiplier):
        """计算特定场景的收入"""
        # 价格变化对上座率的影响
        attendance_multiplier = max(0.7, 1.2 - price_multiplier * 0.3)
        
        revenues = []
        for game_type, price in pricing_strategy.items():
            if game_type == 'regular_season':
                # 常规赛41场
                game_count = 41
                attendance = avg_attendance * attendance_multiplier
                revenue = price * price_multiplier * attendance * game_count
                revenues.append(revenue)
            elif game_type == 'rivalry_games':
                # 假设每个赛季有5场 rivalry 比赛
                game_count = 5
                attendance = avg_attendance * 1.2 * attendance_multiplier
                revenue = price * price_multiplier * attendance * game_count
                revenues.append(revenue)
            elif game_type == 'playoff_games':
                # 假设平均每个赛季有3场季后赛主场比赛
                game_count = 3
                attendance = avg_attendance * 1.5 * attendance_multiplier
                revenue = price * price_multiplier * attendance * game_count
                revenues.append(revenue)
        
        return revenues
    
    def _get_revenue_recommendation(self, optimal_scenario):
        """获取收入优化建议"""
        recommendations = {
            'current_strategy': '保持当前定价策略，平衡票价和上座率',
            'higher_prices': '提高票价以增加收入，但可能影响上座率',
            'lower_prices': '降低票价以提高上座率，通过增加粉丝基础来提升长期收入'
        }
        return recommendations.get(optimal_scenario, '保持当前定价策略')
