import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class PlayerValueEvaluator:
    """球员价值评估模块（包括伤病球员）"""
    def __init__(self, team_analysis, player_social_data, team_market_data):
        self.team_analysis = team_analysis
        self.player_social_data = player_social_data
        self.team_market_data = team_market_data
        self.rf_model = None
        self.scaler = None
        self._train_value_model()
    
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
    
    def _train_value_model(self):
        """训练随机森林模型用于球员价值预测"""
        if len(self.team_analysis) < 10:
            return
        
        # 检查必要的列是否存在
        required_columns = ['PER', 'PTS', 'TRB', 'AST', 'STL', 'BLK']
        if not all(col in self.team_analysis.columns for col in required_columns):
            return
        
        # 准备训练数据
        X = self.team_analysis[required_columns].copy()
        
        # 计算目标变量：综合价值得分
        y = []
        for _, row in self.team_analysis.iterrows():
            player_name = row['Player']
            financial_value = self.calculate_financial_contribution(player_name) / 1000000
            performance_value = row['Value_Index']
            total_value = performance_value * 0.6 + financial_value * 0.4
            y.append(total_value)
        
        y = np.array(y)
        
        # 数据标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练随机森林模型
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X_scaled, y)
        
        # 评估模型
        y_pred = self.rf_model.predict(X_scaled)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        print(f"球员价值预测模型训练完成 - MSE: {mse:.2f}, R²: {r2:.2f}")
    
    def calculate_financial_contribution(self, player_name, team_name=None):
        """计算球员的财务贡献"""
        base_contribution = 1000000  # 基础商业价值
        
        player_data = self.team_analysis[self.team_analysis['Player'] == player_name]
        if player_data.empty:
            return base_contribution
        
        # 检查球员是否有薪资
        if '2023/2024' in player_data.columns:
            salary = player_data['2023/2024'].values[0]
            if salary <= 0:
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
    
    def predict_player_value(self, player_stats):
        """使用随机森林模型预测球员价值"""
        if self.rf_model is None:
            return self.calculate_value_index(pd.DataFrame([player_stats]))[0]
        
        # 准备输入数据
        X = pd.DataFrame([player_stats])[['PER', 'PTS', 'TRB', 'AST', 'STL', 'BLK']]
        X_scaled = self.scaler.transform(X)
        
        # 预测价值
        predicted_value = self.rf_model.predict(X_scaled)[0]
        return predicted_value
    
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
    
    def bayesian_injury_impact_analysis(self, player_name, injury_severity=0.5):
        """使用贝叶斯网络分析伤病对球队的连锁影响"""
        # 贝叶斯网络：量化伤病对球队的连锁影响
        
        # 1. 核心球员缺阵概率
        core_player_prob = self._is_core_player(player_name)
        
        # 2. 胜率下降幅度
        win_rate_impact = self._calculate_win_rate_impact(player_name, injury_severity)
        
        # 3. 门票收入减少比例
        ticket_impact = self._calculate_ticket_impact(player_name, injury_severity)
        
        # 4. 季后赛晋级概率变化
        playoff_impact = self._calculate_playoff_impact(win_rate_impact)
        
        # 5. 替补球员替代能力评估
        replacement_analysis = self._evaluate_replacement_players(player_name)
        
        return {
            'core_player_probability': core_player_prob,
            'win_rate_impact': win_rate_impact,
            'ticket_impact': ticket_impact,
            'playoff_impact': playoff_impact,
            'replacement_analysis': replacement_analysis,
            'recommended_actions': self._generate_injury_response_actions(core_player_prob, injury_severity, replacement_analysis)
        }
    
    def _is_core_player(self, player_name):
        """判断是否为核心球员"""
        player_data = self.team_analysis[self.team_analysis['Player'] == player_name]
        if player_data.empty:
            return 0.3  # 默认不是核心球员
        
        # 基于PER和上场时间判断核心球员
        per = player_data['PER'].values[0]
        # 假设核心球员PER > 18
        if per > 18:
            return 0.8
        elif per > 15:
            return 0.6
        else:
            return 0.3
    
    def _calculate_win_rate_impact(self, player_name, injury_severity):
        """计算胜率下降幅度"""
        core_prob = self._is_core_player(player_name)
        # 核心球员受伤影响更大
        base_impact = 0.15 * injury_severity
        core_factor = 1.0 + (core_prob - 0.5) * 1.5
        return base_impact * core_factor
    
    def _calculate_ticket_impact(self, player_name, injury_severity):
        """计算门票收入减少比例"""
        core_prob = self._is_core_player(player_name)
        # 核心球员受伤对门票影响更大
        base_impact = 0.1 * injury_severity
        core_factor = 1.0 + (core_prob - 0.5) * 1.2
        return base_impact * core_factor
    
    def _calculate_playoff_impact(self, win_rate_impact):
        """计算季后赛晋级概率变化"""
        # 胜率下降导致季后赛概率下降
        return -win_rate_impact * 1.2  # 季后赛概率下降幅度更大
    
    def _evaluate_replacement_players(self, injured_player_name):
        """评估替补球员的替代能力"""
        replacement_analysis = []
        
        # 找到受伤球员的位置（简化处理）
        injured_player = self.team_analysis[self.team_analysis['Player'] == injured_player_name]
        if injured_player.empty:
            return replacement_analysis
        
        # 评估所有其他球员作为替代者的能力
        for _, player in self.team_analysis.iterrows():
            if player['Player'] == injured_player_name:
                continue
            
            # 计算替代能力评分
            replacement_score = self._calculate_replacement_score(injured_player_name, player['Player'])
            
            if replacement_score > 0.5:  # 只考虑替代能力大于0.5的球员
                replacement_analysis.append({
                    'Player': player['Player'],
                    'Replacement_Score': replacement_score,
                    'PER': player['PER'],
                    'Value': self.calculate_balanced_value(player['Player'])
                })
        
        # 按替代能力排序
        replacement_analysis.sort(key=lambda x: x['Replacement_Score'], reverse=True)
        
        return replacement_analysis[:3]  # 返回前3个最佳替代者
    
    def _calculate_replacement_score(self, injured_player_name, replacement_player_name):
        """计算替补球员的替代能力评分"""
        injured_data = self.team_analysis[self.team_analysis['Player'] == injured_player_name]
        replacement_data = self.team_analysis[self.team_analysis['Player'] == replacement_player_name]
        
        if injured_data.empty or replacement_data.empty:
            return 0.3
        
        # 基于PER的替代能力
        injured_per = injured_data['PER'].values[0]
        replacement_per = replacement_data['PER'].values[0]
        per_score = min(1.0, replacement_per / injured_per)
        
        # 基于位置相似性的替代能力（简化处理）
        position_score = 0.8  # 假设位置相似
        
        # 综合替代能力评分
        replacement_score = per_score * 0.7 + position_score * 0.3
        
        return replacement_score
    
    def _generate_injury_response_actions(self, core_player_prob, injury_severity, replacement_analysis):
        """生成伤病应对策略"""
        actions = {
            'short_term': [],
            'long_term': []
        }
        
        # 短期应对策略
        if core_player_prob > 0.7 and injury_severity > 0.5:
            if replacement_analysis:
                actions['short_term'].append(f"优先使用替补球员: {replacement_analysis[0]['Player']} (替代评分: {replacement_analysis[0]['Replacement_Score']:.2f})")
            actions['short_term'].append("考虑激活双向合同球员")
            actions['short_term'].append("在交易市场寻找临时替代者")
        else:
            if replacement_analysis:
                actions['short_term'].append(f"使用替补球员: {replacement_analysis[0]['Player']}")
            actions['short_term'].append("调整球队战术体系")
        
        # 长期应对策略
        actions['long_term'].append("调整训练计划降低伤病风险")
        actions['long_term'].append("加强替补阵容深度")
        actions['long_term'].append("考虑购买伤病保险")
        
        return actions
    
    def bilateral_matching(self, team_needs, available_players, budget_constraint):
        """双边匹配模型用于球员招募策略"""
        # 评估每个球员对球队的匹配度
        player_evaluations = []
        
        for _, player in available_players.iterrows():
            player_name = player['Player']
            salary = player.get('2023/2024', 0)
            
            if salary > budget_constraint:
                continue
            
            # 计算球员价值
            player_value = self.calculate_balanced_value(player_name)
            
            # 计算位置匹配度
            position_match = 1.0  # 简化处理，实际应根据球队需求和球员位置计算
            
            # 计算综合匹配得分
            match_score = player_value * position_match * (1 - salary / budget_constraint)
            
            player_evaluations.append({
                'Player': player_name,
                'Salary': salary,
                'Value': player_value,
                'Match_Score': match_score
            })
        
        # 按匹配得分排序
        player_evaluations.sort(key=lambda x: x['Match_Score'], reverse=True)
        
        return player_evaluations
