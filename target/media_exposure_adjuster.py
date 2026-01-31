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
            
            # 检查球员是否有薪资
            if '2023/2024' in player and player['2023/2024'] <= 0:
                continue
            
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
