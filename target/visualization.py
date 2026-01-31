import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SportsTeamVisualizer:
    """体育团队数据可视化模块"""
    
    def __init__(self, output_dir='../visualizations'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def visualize_player_value_distribution(self, player_data, filename='player_value_distribution.png'):
        """可视化球员价值分布"""
        plt.figure(figsize=(12, 8))
        
        # 价值分布直方图
        plt.subplot(2, 2, 1)
        sns.histplot(player_data['Value_Index'], bins=20, kde=True)
        plt.title('球员价值指数分布')
        plt.xlabel('价值指数')
        plt.ylabel('频率')
        
        # 价值与薪资散点图
        plt.subplot(2, 2, 2)
        sns.scatterplot(x='2023/2024', y='Value_Index', data=player_data)
        plt.title('球员价值与薪资关系')
        plt.xlabel('薪资 ($)')
        plt.ylabel('价值指数')
        plt.ticklabel_format(style='plain', axis='x')
        
        # 价值与PER关系
        plt.subplot(2, 2, 3)
        sns.scatterplot(x='PER', y='Value_Index', data=player_data)
        plt.title('球员价值与PER关系')
        plt.xlabel('PER')
        plt.ylabel('价值指数')
        
        # 风险评分分布
        if 'Risk_Score' in player_data.columns:
            plt.subplot(2, 2, 4)
            sns.histplot(player_data['Risk_Score'], bins=10, kde=True)
            plt.title('球员风险评分分布')
            plt.xlabel('风险评分')
            plt.ylabel('频率')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        return os.path.join(self.output_dir, filename)
    
    def visualize_team_expansion_impact(self, location_evaluation, filename='team_expansion_impact.png'):
        """可视化球队扩张影响"""
        plt.figure(figsize=(12, 6))
        
        # 位置评估条形图
        sns.barplot(x='Location', y='Evaluation_Score', data=location_evaluation)
        plt.title('潜在扩张位置评估')
        plt.xlabel('位置')
        plt.ylabel('评估分数')
        plt.xticks(rotation=45)
        
        # 在条形上显示具体数值
        for i, v in enumerate(location_evaluation['Evaluation_Score']):
            plt.text(i, v + 0.1, f'{v:.2f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        return os.path.join(self.output_dir, filename)
    
    def visualize_ticket_pricing(self, pricing_strategy, filename='ticket_pricing_strategy.png'):
        """可视化门票定价策略"""
        plt.figure(figsize=(10, 6))
        
        # 票价策略条形图
        pricing_data = pd.DataFrame.from_dict(pricing_strategy['pricing_strategy'], orient='index', columns=['Price'])
        sns.barplot(x=pricing_data.index, y='Price', data=pricing_data)
        plt.title('门票定价策略')
        plt.xlabel('比赛类型')
        plt.ylabel('票价 ($)')
        plt.xticks(rotation=45)
        
        # 在条形上显示具体数值
        for i, v in enumerate(pricing_data['Price']):
            plt.text(i, v + 2, f'${v:.2f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        return os.path.join(self.output_dir, filename)
    
    def visualize_media_strategy(self, media_strategy, filename='media_strategy.png'):
        """可视化媒体策略"""
        plt.figure(figsize=(12, 8))
        
        # 平台投资分配饼图
        plt.subplot(2, 1, 1)
        platform_data = media_strategy['platform_strategy']
        plt.pie(platform_data.values(), labels=platform_data.keys(), autopct='%1.1f%%')
        plt.title('媒体平台投资分配')
        plt.axis('equal')
        
        # 内容策略分配饼图
        plt.subplot(2, 1, 2)
        content_data = media_strategy['content_strategy']
        plt.pie(content_data.values(), labels=content_data.keys(), autopct='%1.1f%%')
        plt.title('内容类型投资分配')
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        return os.path.join(self.output_dir, filename)
    
    def visualize_markov_decision(self, markov_model, filename='markov_decision_process.png'):
        """可视化马尔可夫决策过程"""
        plt.figure(figsize=(12, 8))
        
        # 状态转移矩阵热图
        states = markov_model.states
        actions = markov_model.actions
        
        for i, action in enumerate(actions):
            plt.subplot(1, len(actions), i+1)
            
            # 构建转移矩阵
            transition_matrix = []
            for state in states:
                row = []
                for next_state in states:
                    row.append(markov_model.transition_matrix[action][state].get(next_state, 0))
                transition_matrix.append(row)
            
            sns.heatmap(transition_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                       xticklabels=states, yticklabels=states)
            plt.title(f'{action} 状态转移')
            plt.xlabel('下一状态')
            plt.ylabel('当前状态')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        return os.path.join(self.output_dir, filename)
    
    def visualize_roster_optimization(self, selected_players, filename='roster_optimization.png'):
        """可视化球队阵容优化结果"""
        plt.figure(figsize=(14, 8))
        
        # 球员价值与薪资对比
        plt.subplot(2, 1, 1)
        players = selected_players['Player']
        values = selected_players['Value_Index']
        salaries = selected_players['2023/2024'] / 1000000  # 转换为百万美元
        
        x = np.arange(len(players))
        width = 0.35
        
        plt.bar(x - width/2, values, width, label='价值指数')
        plt.bar(x + width/2, salaries, width, label='薪资 (百万$)')
        plt.xticks(x, players, rotation=45, ha='right')
        plt.title('优化阵容：球员价值与薪资')
        plt.ylabel('数值')
        plt.legend()
        
        # 球员风险评分
        plt.subplot(2, 1, 2)
        if 'Risk_Score' in selected_players.columns:
            sns.barplot(x='Player', y='Risk_Score', data=selected_players)
            plt.title('球员风险评分')
            plt.xlabel('球员')
            plt.ylabel('风险评分')
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        return os.path.join(self.output_dir, filename)
    
    def visualize_decision_tree(self, model, feature_names, filename='decision_tree.png'):
        """可视化决策树模型"""
        from sklearn.tree import plot_tree
        
        plt.figure(figsize=(20, 10))
        plot_tree(model, feature_names=feature_names, filled=True, rounded=True, fontsize=10)
        plt.title('决策树模型')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        return os.path.join(self.output_dir, filename)
    
    def generate_dashboard(self, model_results, filename='dashboard.png'):
        """生成综合仪表盘"""
        plt.figure(figsize=(16, 12))
        
        # 1. 球员价值分布
        if 'player_data' in model_results:
            plt.subplot(3, 2, 1)
            sns.histplot(model_results['player_data']['Value_Index'], bins=20, kde=True)
            plt.title('球员价值指数分布')
            plt.xlabel('价值指数')
            plt.ylabel('频率')
        
        # 2. 球队扩张评估
        if 'expansion_data' in model_results:
            plt.subplot(3, 2, 2)
            sns.barplot(x='Location', y='Evaluation_Score', data=model_results['expansion_data'])
            plt.title('潜在扩张位置评估')
            plt.xlabel('位置')
            plt.ylabel('评估分数')
            plt.xticks(rotation=45)
        
        # 3. 门票定价策略
        if 'pricing_data' in model_results:
            plt.subplot(3, 2, 3)
            pricing_data = pd.DataFrame.from_dict(
                model_results['pricing_data']['pricing_strategy'], 
                orient='index', 
                columns=['Price']
            )
            sns.barplot(x=pricing_data.index, y='Price', data=pricing_data)
            plt.title('门票定价策略')
            plt.xlabel('比赛类型')
            plt.ylabel('票价 ($)')
            plt.xticks(rotation=45)
        
        # 4. 媒体平台分配
        if 'media_data' in model_results:
            plt.subplot(3, 2, 4)
            platform_data = model_results['media_data']['platform_strategy']
            plt.pie(platform_data.values(), labels=platform_data.keys(), autopct='%1.1f%%')
            plt.title('媒体平台投资分配')
            plt.axis('equal')
        
        # 5. 阵容薪资分布
        if 'roster_data' in model_results:
            plt.subplot(3, 2, 5)
            sns.histplot(model_results['roster_data']['2023/2024'], bins=10, kde=True)
            plt.title('阵容薪资分布')
            plt.xlabel('薪资 ($)')
            plt.ylabel('频率')
            plt.ticklabel_format(style='plain', axis='x')
        
        # 6. 决策结果
        if 'decision_data' in model_results:
            plt.subplot(3, 2, 6)
            decision = model_results['decision_data']
            outcomes = decision['Expected_Outcomes']
            outcome_df = pd.DataFrame.from_dict(outcomes, orient='index', columns=['Value'])
            sns.barplot(x=outcome_df.index, y='Value', data=outcome_df)
            plt.title('预期决策结果')
            plt.xlabel('指标')
            plt.ylabel('数值')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        return os.path.join(self.output_dir, filename)