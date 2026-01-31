import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

class TeamExpansionAnalyzer:
    """球队扩张与选址策略模块"""
    def __init__(self, team_market_data):
        self.team_market_data = team_market_data
        # 城市坐标数据（简化版）
        self.city_coordinates = self._get_city_coordinates()
    
    def _get_city_coordinates(self):
        """获取城市坐标数据"""
        # 简化的城市坐标数据（经度，纬度）
        return {
            'New York': (-74.0060, 40.7128),
            'Los Angeles': (-118.2437, 34.0522),
            'Chicago': (-87.6298, 41.8781),
            'Houston': (-95.3698, 29.7604),
            'Phoenix': (-112.0740, 33.4484),
            'Philadelphia': (-75.1652, 39.9526),
            'San Antonio': (-98.4936, 29.4241),
            'San Diego': (-117.1611, 32.7157),
            'Dallas': (-96.7970, 32.7767),
            'San Francisco': (-122.4194, 37.7749),
            'Seattle': (-122.3321, 47.6062),
            'Denver': (-104.9903, 39.7392),
            'Portland': (-122.6765, 45.5051),
            'Sacramento': (-121.4944, 38.5816),
            'Las Vegas': (-115.1398, 36.1699),
            'Kansas City': (-94.5786, 39.0997),
            'Louisville': (-85.7585, 38.2527)
        }
    
    def analyze_league_expansion(self, new_team_location):
        """分析联盟扩张影响"""
        impact_analysis = []
        
        for _, team_data in self.team_market_data.iterrows():
            team = team_data['Team']
            market_size = team_data['Metro Population (millions)']
            
            # 空间计量经济学：计算地理距离影响
            distance_factor = self._calculate_distance_factor(team, new_team_location)
            
            # 市场规模影响
            market_factor = self._calculate_market_factor(market_size)
            
            # 竞争强度影响
            competition_factor = self._calculate_competition_factor(team, new_team_location)
            
            # 综合影响
            impact = market_size * distance_factor * market_factor * competition_factor
            
            impact_analysis.append({
                'Team': team, 
                'Market_Size': market_size, 
                'Distance_Factor': distance_factor,
                'Market_Factor': market_factor,
                'Competition_Factor': competition_factor,
                'Impact': impact
            })
        
        impact_df = pd.DataFrame(impact_analysis)
        impact_df = impact_df.sort_values('Impact', ascending=False)
        
        return impact_df
    
    def _calculate_distance_factor(self, existing_team, new_location):
        """计算距离因素"""
        # 提取城市名称
        existing_city = existing_team.split()[-1] if len(existing_team.split()) > 1 else existing_team
        
        # 获取坐标
        existing_coords = self.city_coordinates.get(existing_city, (0, 0))
        new_coords = self.city_coordinates.get(new_location, (0, 0))
        
        # 计算距离（简化为欧几里得距离）
        distance = euclidean(existing_coords, new_coords)
        
        # 距离因子：距离越近，影响越大
        max_distance = 50  # 最大距离（度）
        distance_factor = max(0.1, 1.0 - (distance / max_distance))
        
        return distance_factor
    
    def _calculate_market_factor(self, market_size):
        """计算市场因子"""
        # 市场越大，受到的影响越小（因为市场容量大）
        return max(0.5, 1.0 - (market_size / 20))
    
    def _calculate_competition_factor(self, existing_team, new_location):
        """计算竞争强度因子"""
        # 简化的竞争强度计算
        if new_location.lower() in existing_team.lower():
            return 1.5  # 同城竞争影响最大
        return 1.0
    
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
            market_potential = self._estimate_market_potential(location)
            
            # 系统动力学：模拟扩军后的连锁反应
            systemic_impact = self._simulate_systemic_impact(location, impact_df)
            
            # 综合评估分数
            evaluation_score = (total_impact * 0.3 +
                               market_potential * 0.3 +
                               systemic_impact * 0.2 +
                               (1 - avg_impact / total_impact) * 0.2)  # 影响分布的均衡性
            
            location_evaluations.append({
                'Location': location,
                'Total_Impact': total_impact,
                'Avg_Impact': avg_impact,
                'Market_Potential': market_potential,
                'Systemic_Impact': systemic_impact,
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
            'Las Vegas': 3.5,
            'Kansas City': 4.0,
            'Louisville': 3.5
        }
        
        return major_cities.get(location, 3.0)
    
    def _simulate_systemic_impact(self, new_location, impact_df):
        """模拟扩军后的系统影响"""
        # 系统动力学模型：模拟扩军后的连锁反应
        
        # 1. 赛程调整影响
        schedule_impact = 0.1  # 赛程调整对所有球队的平均影响
        
        # 2. 转播分成变化
        media_impact = -0.05  # 转播收入被摊薄
        
        # 3. 球员流动影响
        player_impact = -0.1  # 扩军选秀对现有球队阵容的削弱
        
        # 4. 新市场带来的整体增长
        new_market_growth = 0.2 * self._estimate_market_potential(new_location) / 10
        
        # 综合系统影响
        systemic_impact = 1.0 + schedule_impact + media_impact + player_impact + new_market_growth
        
        return max(0.5, systemic_impact)
