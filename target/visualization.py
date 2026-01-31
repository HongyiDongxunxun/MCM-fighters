import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Set global style
sns.set_style("whitegrid")  # Use white grid background
# 使用更丰富的颜色方案
sns.set_palette("Set3")  # Use Set3 color scheme for better aesthetics
plt.rcParams['figure.figsize'] = (12, 8)  # Default figure size
plt.rcParams['font.size'] = 12  # Default font size
plt.rcParams['axes.titlesize'] = 16  # Title font size
plt.rcParams['axes.labelsize'] = 14  # Label font size
plt.rcParams['xtick.labelsize'] = 11  # X-axis tick font size
plt.rcParams['ytick.labelsize'] = 11  # Y-axis tick font size
plt.rcParams['legend.fontsize'] = 12  # Legend font size

# Define custom color palettes for different visualization types
COLOR_PALETTES = {
    'value': ['#4C78A8', '#F58518', '#E45756', '#72B7B2', '#54A24B', '#EECA3B', '#B279A2', '#FF9DA6'],
    'expansion': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'],
    'pricing': ['#FF7675', '#74B9FF', '#00B894', '#FDCB6E', '#A29BFE', '#E17055', '#00CEC9', '#6C5CE7'],
    'media': ['#FD79A8', '#E17055', '#00B894', '#FDCB6E', '#6C5CE7', '#0984E3', '#A29BFE', '#FD79A8'],
    'markov': ['#8E44AD', '#3498DB', '#27AE60', '#F39C12', '#E74C3C', '#1ABC9C', '#95A5A6', '#34495E'],
    'roster': ['#E74C3C', '#3498DB', '#2ECC71', '#F1C40F', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E'],
    'dashboard': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
}

class SportsTeamVisualizer:
    """Sports team data visualization module"""
    
    def __init__(self, output_dir='../visualizations'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def visualize_player_value_distribution(self, player_data):
        """Visualize player value distribution with multiple separate files"""
        output_files = []
        
        # 1. Value distribution histogram with kernel density estimate
        output_files.append(self._visualize_value_distribution(player_data))
        
        # 2. Value vs Salary scatter plot with regression line
        output_files.append(self._visualize_value_vs_salary(player_data))
        
        # 3. Value vs PER relationship with hexbin plot
        output_files.append(self._visualize_value_vs_per(player_data))
        
        # 4. Risk score distribution with violin plot and swarm plot
        if 'Risk_Score' in player_data.columns:
            output_files.append(self._visualize_risk_distribution(player_data))
        
        # 5. Top 10 players by value index
        output_files.append(self._visualize_top_players(player_data))
        
        # 6. Value index distribution by position (if position data is available)
        if 'Pos' in player_data.columns:
            output_files.append(self._visualize_value_by_position(player_data))
        else:
            # Fallback to salary distribution if position data is not available
            output_files.append(self._visualize_salary_distribution(player_data))
        
        # 7. Value vs Age scatter plot (if age data is available)
        if 'Age' in player_data.columns:
            output_files.append(self._visualize_value_vs_age(player_data))
        
        # 8. Value distribution by team (if team data is available)
        if 'Team' in player_data.columns:
            output_files.append(self._visualize_value_by_team(player_data))
        
        # 9. Correlation heatmap of player metrics
        output_files.append(self._visualize_player_metrics_correlation(player_data))
        
        return output_files
    
    def _visualize_value_distribution(self, player_data):
        """Visualize value distribution histogram"""
        filename = 'player_value_distribution_histogram.png'
        plt.figure(figsize=(12, 8))
        sns.histplot(player_data['Value_Index'], bins=20, kde=True, color=COLOR_PALETTES['value'][0], edgecolor='black')
        plt.title('Player Value Index Distribution', fontweight='bold')
        plt.xlabel('Value Index')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def visualize_pairplot(self, data, variables, filename='pairplot_visualization.png'):
        """Visualize relationships between multiple variables using pair plot"""
        plt.figure(figsize=(15, 15))
        
        sns.pairplot(data[variables], diag_kind='kde', 
                    plot_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'k'}, 
                    diag_kws={'fill': True, 'alpha': 0.6})
        plt.suptitle('Pair Plot of Key Variables', fontweight='bold', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def visualize_violinplot(self, data, x_col, y_col, filename='violinplot_visualization.png'):
        """Visualize data distribution using violin plot"""
        plt.figure(figsize=(12, 8))
        
        sns.violinplot(x=x_col, y=y_col, data=data, 
                      inner='quartile', palette=COLOR_PALETTES['dashboard'][:len(data[x_col].unique())])
        plt.title(f'{y_col} Distribution by {x_col}', fontweight='bold')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def visualize_bubble_plot(self, data, x_col, y_col, size_col, filename='bubble_plot_visualization.png'):
        """Visualize three variables using bubble plot"""
        plt.figure(figsize=(12, 8))
        
        # Normalize size for better visualization
        normalized_size = (data[size_col] - data[size_col].min()) / (data[size_col].max() - data[size_col].min()) * 500 + 50
        
        scatter = plt.scatter(data[x_col], data[y_col], s=normalized_size, 
                             alpha=0.6, c=normalized_size, cmap='viridis', 
                             edgecolors='black', linewidths=1)
        plt.colorbar(scatter, label=size_col)
        plt.title(f'Bubble Plot: {y_col} vs {x_col} (Size: {size_col})', fontweight='bold')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(axis='both', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def visualize_stacked_bar(self, data, x_col, y_col, hue_col, filename='stacked_bar_visualization.png'):
        """Visualize stacked bar chart for categorical data"""
        plt.figure(figsize=(14, 10))
        
        sns.histplot(data=data, x=x_col, hue=hue_col, multiple='stack', 
                     palette=COLOR_PALETTES['dashboard'][:len(data[hue_col].unique())])
        plt.title(f'Stacked Bar Chart: {y_col} by {x_col} and {hue_col}', fontweight='bold')
        plt.xlabel(x_col)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.legend(title=hue_col)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def visualize_area_chart(self, data, x_col, y_col, hue_col, filename='area_chart_visualization.png'):
        """Visualize area chart for time series data"""
        plt.figure(figsize=(14, 10))
        
        sns.lineplot(data=data, x=x_col, y=y_col, hue=hue_col, 
                     palette=COLOR_PALETTES['dashboard'][:len(data[hue_col].unique())], 
                     linewidth=2, marker='o')
        plt.fill_between(data[x_col], data[y_col], alpha=0.3)
        plt.title(f'Area Chart: {y_col} over {x_col}', fontweight='bold')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='both', alpha=0.3)
        plt.legend(title=hue_col)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def visualize_facet_grid(self, data, col_col, x_col, y_col, filename='facet_grid_visualization.png'):
        """Visualize data using facet grid for multiple categories"""
        g = sns.FacetGrid(data, col=col_col, col_wrap=3, height=4, aspect=1.2)
        g.map(sns.scatterplot, x_col, y_col, alpha=0.6, s=50, edgecolor='k')
        g.set_titles(col_template='{col_name}', fontweight='bold')
        g.set_xlabels(x_col)
        g.set_ylabels(y_col)
        g.tight_layout()
        plt.suptitle(f'Facet Grid: {y_col} vs {x_col} by {col_col}', fontweight='bold', fontsize=14, y=1.02)
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def visualize_density_plot(self, data, x_col, hue_col, filename='density_plot_visualization.png'):
        """Visualize data density using kernel density estimate"""
        plt.figure(figsize=(12, 8))
        
        sns.kdeplot(data=data, x=x_col, hue=hue_col, fill=True, 
                    palette=COLOR_PALETTES['dashboard'][:len(data[hue_col].unique())], 
                    alpha=0.6, linewidth=2)
        plt.title(f'Density Plot: {x_col} by {hue_col}', fontweight='bold')
        plt.xlabel(x_col)
        plt.ylabel('Density')
        plt.grid(axis='y', alpha=0.3)
        plt.legend(title=hue_col)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_value_vs_salary(self, player_data):
        """Visualize value vs salary relationship"""
        filename = 'player_value_vs_salary.png'
        plt.figure(figsize=(12, 8))
        sns.regplot(x='2023/2024', y='Value_Index', data=player_data, 
                    scatter_kws={'alpha':0.6, 'color':COLOR_PALETTES['value'][1]}, 
                    line_kws={'color':COLOR_PALETTES['value'][2]})
        plt.title('Player Value vs Salary Relationship', fontweight='bold')
        plt.xlabel('Salary ($)')
        plt.ylabel('Value Index')
        plt.ticklabel_format(style='plain', axis='x')
        plt.grid(axis='both', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_value_vs_per(self, player_data):
        """Visualize value vs PER relationship"""
        filename = 'player_value_vs_per.png'
        plt.figure(figsize=(12, 8))
        plt.hexbin(x=player_data['PER'], y=player_data['Value_Index'], gridsize=30, cmap='coolwarm', mincnt=1)
        plt.colorbar(label='Density')
        plt.title('Player Value vs PER Relationship', fontweight='bold')
        plt.xlabel('PER')
        plt.ylabel('Value Index')
        plt.grid(axis='both', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_risk_distribution(self, player_data):
        """Visualize risk score distribution"""
        filename = 'player_risk_distribution.png'
        plt.figure(figsize=(12, 8))
        sns.violinplot(y=player_data['Risk_Score'], inner='quartile', color=COLOR_PALETTES['value'][3])
        sns.swarmplot(y=player_data['Risk_Score'], color='black', size=4, alpha=0.6)
        plt.title('Player Risk Score Distribution', fontweight='bold')
        plt.xlabel('')
        plt.ylabel('Risk Score')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_top_players(self, player_data):
        """Visualize top 10 players by value index"""
        filename = 'top_10_players_by_value.png'
        plt.figure(figsize=(12, 8))
        top_10_players = player_data.nlargest(10, 'Value_Index')
        sns.barplot(x='Value_Index', y='Player', data=top_10_players, palette=COLOR_PALETTES['value'][:10])
        plt.title('Top 10 Players by Value Index', fontweight='bold')
        plt.xlabel('Value Index')
        plt.ylabel('Player')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_value_by_position(self, player_data):
        """Visualize value index distribution by position"""
        filename = 'player_value_by_position.png'
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Pos', y='Value_Index', data=player_data, palette=COLOR_PALETTES['value'][:len(player_data['Pos'].unique())])
        sns.swarmplot(x='Pos', y='Value_Index', data=player_data, color='black', size=3, alpha=0.5)
        plt.title('Value Index Distribution by Position', fontweight='bold')
        plt.xlabel('Position')
        plt.ylabel('Value Index')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_salary_distribution(self, player_data):
        """Visualize player salary distribution"""
        filename = 'player_salary_distribution.png'
        plt.figure(figsize=(12, 8))
        sns.histplot(player_data['2023/2024'] / 1000000, bins=15, kde=True, color=COLOR_PALETTES['value'][4], edgecolor='black')
        plt.title('Player Salary Distribution (in Millions $)', fontweight='bold')
        plt.xlabel('Salary (Million $)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_value_vs_age(self, player_data):
        """Visualize value vs age relationship"""
        filename = 'player_value_vs_age.png'
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='Age', y='Value_Index', data=player_data, 
                        size='2023/2024', sizes=(50, 500), alpha=0.7,
                        color=COLOR_PALETTES['value'][5])
        sns.regplot(x='Age', y='Value_Index', data=player_data, scatter=False, color=COLOR_PALETTES['value'][6])
        plt.title('Player Value vs Age Relationship', fontweight='bold')
        plt.xlabel('Age')
        plt.ylabel('Value Index')
        plt.grid(axis='both', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_value_by_team(self, player_data):
        """Visualize value distribution by team"""
        filename = 'player_value_by_team.png'
        plt.figure(figsize=(14, 10))
        team_value = player_data.groupby('Team')['Value_Index'].mean().sort_values(ascending=False)
        team_value_df = pd.DataFrame({'Team': team_value.index, 'Average Value Index': team_value.values})
        sns.barplot(x='Average Value Index', y='Team', data=team_value_df, 
                    palette=COLOR_PALETTES['value'][:len(team_value_df)])
        plt.title('Average Player Value by Team', fontweight='bold')
        plt.xlabel('Average Value Index')
        plt.ylabel('Team')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_player_metrics_correlation(self, player_data):
        """Visualize correlation heatmap of player metrics"""
        filename = 'player_metrics_correlation.png'
        plt.figure(figsize=(14, 12))
        # Select relevant numerical columns for correlation
        metrics_cols = ['Value_Index', '2023/2024', 'PER', 'Age']
        if 'Risk_Score' in player_data.columns:
            metrics_cols.append('Risk_Score')
        if 'PTS' in player_data.columns:
            metrics_cols.append('PTS')
        if 'AST' in player_data.columns:
            metrics_cols.append('AST')
        if 'TRB' in player_data.columns:
            metrics_cols.append('TRB')
        
        # Filter only available columns
        available_cols = [col for col in metrics_cols if col in player_data.columns]
        correlation_data = player_data[available_cols]
        
        # Calculate correlation matrix
        corr_matrix = correlation_data.corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    square=True, linewidths=0.5, cbar_kws={'label': 'Correlation'})
        plt.title('Player Metrics Correlation Heatmap', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def visualize_team_expansion_impact(self, location_evaluation):
        """Visualize team expansion impact with multiple separate files"""
        output_files = []
        
        # Sort data for better visualization
        location_evaluation = location_evaluation.sort_values('Evaluation_Score', ascending=False)
        
        # 1. Horizontal bar chart for location evaluation
        output_files.append(self._visualize_expansion_location_ranking(location_evaluation))
        
        # 2. Radar chart for location attributes (if available)
        if all(col in location_evaluation.columns for col in ['Market_Potential', 'Geographic_Factor', 'Competitive_Intensity']):
            output_files.append(self._visualize_expansion_location_radar(location_evaluation))
        else:
            # Fallback to vertical bar chart
            output_files.append(self._visualize_expansion_location_vertical(location_evaluation))
        
        # 3. Score distribution histogram
        output_files.append(self._visualize_expansion_score_distribution(location_evaluation))
        
        # 4. Location ranking with colored bars
        output_files.append(self._visualize_expansion_location_detailed_ranking(location_evaluation))
        
        # 5. Location attributes comparison (if available)
        if all(col in location_evaluation.columns for col in ['Market_Potential', 'Geographic_Factor', 'Competitive_Intensity']):
            output_files.append(self._visualize_expansion_attributes_comparison(location_evaluation))
        
        # 6. Location score vs market size (if available)
        if 'Market_Size' in location_evaluation.columns:
            output_files.append(self._visualize_expansion_score_vs_market(location_evaluation))
        
        # 7. Location score vs distance to nearest team (if available)
        if 'Distance_to_Nearest_Team' in location_evaluation.columns:
            output_files.append(self._visualize_expansion_score_vs_distance(location_evaluation))
        
        return output_files
    
    def _visualize_expansion_location_ranking(self, location_evaluation):
        """Visualize location evaluation horizontal bar chart"""
        filename = 'team_expansion_location_ranking.png'
        plt.figure(figsize=(12, 8))
        bars = sns.barplot(y='Location', x='Evaluation_Score', data=location_evaluation, palette=COLOR_PALETTES['expansion'][:len(location_evaluation)])
        plt.title('Potential Expansion Location Evaluation', fontweight='bold')
        plt.xlabel('Evaluation Score')
        plt.ylabel('Location')
        plt.grid(axis='x', alpha=0.3)
        
        # Display specific values on bars
        for i, v in enumerate(location_evaluation['Evaluation_Score']):
            plt.text(v + 0.1, i, f'{v:.2f}', va='center', fontweight='bold')
        
        # Add average score line
        mean_score = location_evaluation['Evaluation_Score'].mean()
        plt.axvline(x=mean_score, color='red', linestyle='--', label=f'Average Score: {mean_score:.2f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_expansion_location_radar(self, location_evaluation):
        """Visualize location attributes radar chart"""
        filename = 'team_expansion_location_radar.png'
        plt.figure(figsize=(12, 10), polar=True)
        
        # Normalize data for radar chart
        attributes = ['Market_Potential', 'Geographic_Factor', 'Competitive_Intensity']
        normalized_data = location_evaluation.copy()
        for attr in attributes:
            normalized_data[attr] = (normalized_data[attr] - normalized_data[attr].min()) / (normalized_data[attr].max() - normalized_data[attr].min())
        
        # Plot radar chart for each location
        angles = np.linspace(0, 2 * np.pi, len(attributes), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        for i, (_, row) in enumerate(normalized_data.iterrows()):
            values = row[attributes].tolist()
            values += values[:1]  # Close the loop
            plt.plot(angles, values, 'o-', linewidth=2, label=row['Location'], color=COLOR_PALETTES['expansion'][i % len(COLOR_PALETTES['expansion'])])
            plt.fill(angles, values, alpha=0.25)
        
        plt.title('Location Attributes Radar Chart', fontweight='bold', size=14, pad=20)
        plt.setp(plt.xticks(angles[:-1], attributes), fontsize=12)
        plt.setp(plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0]), fontsize=10)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_expansion_location_vertical(self, location_evaluation):
        """Visualize location evaluation vertical bar chart"""
        filename = 'team_expansion_location_vertical.png'
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Location', y='Evaluation_Score', data=location_evaluation, palette=COLOR_PALETTES['expansion'][:len(location_evaluation)])
        plt.title('Location Evaluation Scores', fontweight='bold')
        plt.xlabel('Location')
        plt.ylabel('Evaluation Score')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_expansion_score_distribution(self, location_evaluation):
        """Visualize evaluation score distribution histogram"""
        filename = 'team_expansion_score_distribution.png'
        plt.figure(figsize=(12, 8))
        sns.histplot(location_evaluation['Evaluation_Score'], bins=len(location_evaluation), kde=True, color=COLOR_PALETTES['expansion'][0], edgecolor='black')
        plt.title('Evaluation Score Distribution', fontweight='bold')
        plt.xlabel('Evaluation Score')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_expansion_location_detailed_ranking(self, location_evaluation):
        """Visualize location detailed ranking with colored bars"""
        filename = 'team_expansion_location_detailed_ranking.png'
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Evaluation_Score', y='Location', data=location_evaluation, palette=sns.color_palette('coolwarm', len(location_evaluation)))
        plt.title('Location Ranking by Evaluation Score', fontweight='bold')
        plt.xlabel('Evaluation Score')
        plt.ylabel('Location')
        plt.grid(axis='x', alpha=0.3)
        
        # Add evaluation metrics if available
        if 'Market_Size' in location_evaluation.columns:
            for i, (_, row) in enumerate(location_evaluation.iterrows()):
                plt.text(row['Evaluation_Score'] + 0.1, i, f'Market: {row.get("Market_Size", "N/A")}', va='center', fontsize=10, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_expansion_attributes_comparison(self, location_evaluation):
        """Visualize location attributes comparison"""
        filename = 'team_expansion_attributes_comparison.png'
        plt.figure(figsize=(14, 10))
        
        # Melt the data for grouped bar chart
        attributes = ['Market_Potential', 'Geographic_Factor', 'Competitive_Intensity']
        melted_data = pd.melt(location_evaluation, id_vars=['Location'], value_vars=attributes, var_name='Attribute', value_name='Score')
        
        sns.barplot(x='Attribute', y='Score', hue='Location', data=melted_data, palette=COLOR_PALETTES['expansion'][:len(location_evaluation)])
        plt.title('Location Attributes Comparison', fontweight='bold')
        plt.xlabel('Attribute')
        plt.ylabel('Score')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.legend(title='Location')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_expansion_score_vs_market(self, location_evaluation):
        """Visualize location score vs market size"""
        filename = 'team_expansion_score_vs_market.png'
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='Market_Size', y='Evaluation_Score', data=location_evaluation, 
                        size='Market_Size', sizes=(100, 500), alpha=0.7,
                        color=COLOR_PALETTES['expansion'][4])
        sns.regplot(x='Market_Size', y='Evaluation_Score', data=location_evaluation, scatter=False, color=COLOR_PALETTES['expansion'][5])
        plt.title('Location Evaluation Score vs Market Size', fontweight='bold')
        plt.xlabel('Market Size')
        plt.ylabel('Evaluation Score')
        plt.grid(axis='both', alpha=0.3)
        
        # Add data labels
        for i, (_, row) in enumerate(location_evaluation.iterrows()):
            plt.text(row['Market_Size'], row['Evaluation_Score'] + 0.1, row['Location'], ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_expansion_score_vs_distance(self, location_evaluation):
        """Visualize location score vs distance to nearest team"""
        filename = 'team_expansion_score_vs_distance.png'
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='Distance_to_Nearest_Team', y='Evaluation_Score', data=location_evaluation, 
                        size='Distance_to_Nearest_Team', sizes=(100, 500), alpha=0.7,
                        color=COLOR_PALETTES['expansion'][6])
        sns.regplot(x='Distance_to_Nearest_Team', y='Evaluation_Score', data=location_evaluation, scatter=False, color=COLOR_PALETTES['expansion'][7])
        plt.title('Location Evaluation Score vs Distance to Nearest Team', fontweight='bold')
        plt.xlabel('Distance to Nearest Team (miles)')
        plt.ylabel('Evaluation Score')
        plt.grid(axis='both', alpha=0.3)
        
        # Add data labels
        for i, (_, row) in enumerate(location_evaluation.iterrows()):
            plt.text(row['Distance_to_Nearest_Team'], row['Evaluation_Score'] + 0.1, row['Location'], ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def visualize_ticket_pricing(self, pricing_strategy):
        """Visualize ticket pricing strategy with multiple separate files"""
        output_files = []
        
        # Ticket pricing strategy bar chart
        pricing_data = pd.DataFrame.from_dict(pricing_strategy['pricing_strategy'], orient='index', columns=['Price'])
        pricing_data = pricing_data.sort_values('Price', ascending=False)
        
        # 1. Main pricing strategy bar chart
        output_files.append(self._visualize_pricing_main_strategy(pricing_data))
        
        # 2. Pricing comparison with previous season (simulated data)
        output_files.append(self._visualize_pricing_season_comparison(pricing_data))
        
        # 3. Price distribution histogram
        output_files.append(self._visualize_pricing_distribution(pricing_data))
        
        # 4. Price vs expected attendance (simulated data)
        output_files.append(self._visualize_pricing_vs_attendance(pricing_data))
        
        # 5. Price vs expected revenue (simulated data)
        output_files.append(self._visualize_pricing_vs_revenue(pricing_data))
        
        # 6. Price sensitivity analysis (simulated data)
        output_files.append(self._visualize_pricing_sensitivity(pricing_data))
        
        # 7. Pricing strategy heatmap (if multiple pricing tiers available)
        output_files.append(self._visualize_pricing_heatmap(pricing_data))
        
        return output_files
    
    def _visualize_pricing_main_strategy(self, pricing_data):
        """Visualize main pricing strategy bar chart"""
        filename = 'ticket_pricing_main_strategy.png'
        plt.figure(figsize=(12, 8))
        bars = sns.barplot(x=pricing_data.index, y='Price', data=pricing_data, palette=COLOR_PALETTES['pricing'][:len(pricing_data)])
        plt.title('Ticket Pricing Strategy', fontweight='bold')
        plt.xlabel('Game Type')
        plt.ylabel('Price ($)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Display specific values on bars
        for i, v in enumerate(pricing_data['Price']):
            plt.text(i, v + 2, f'${v:.2f}', ha='center', fontweight='bold', color='black')
        
        # Add average price reference line
        plt.axhline(y=pricing_data['Price'].mean(), color='green', linestyle='--', label=f'Average Price: ${pricing_data["Price"].mean():.2f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_pricing_season_comparison(self, pricing_data):
        """Visualize pricing comparison with previous season"""
        filename = 'ticket_pricing_season_comparison.png'
        plt.figure(figsize=(12, 8))
        # Create simulated previous season data for comparison
        prev_season_prices = pricing_data['Price'] * np.random.uniform(0.8, 0.95, len(pricing_data))
        comparison_data = pd.DataFrame({
            'Game Type': pricing_data.index,
            'Current Season': pricing_data['Price'],
            'Previous Season': prev_season_prices
        })
        
        comparison_data = comparison_data.melt(id_vars='Game Type', var_name='Season', value_name='Price')
        sns.barplot(x='Game Type', y='Price', hue='Season', data=comparison_data, palette=COLOR_PALETTES['pricing'][:2])
        plt.title('Pricing Comparison: Current vs Previous Season', fontweight='bold')
        plt.xlabel('Game Type')
        plt.ylabel('Price ($)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_pricing_distribution(self, pricing_data):
        """Visualize price distribution histogram"""
        filename = 'ticket_pricing_distribution.png'
        plt.figure(figsize=(12, 8))
        sns.histplot(pricing_data['Price'], bins=len(pricing_data), kde=True, color=COLOR_PALETTES['pricing'][0], edgecolor='black')
        plt.title('Price Distribution', fontweight='bold')
        plt.xlabel('Price ($)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_pricing_vs_attendance(self, pricing_data):
        """Visualize price vs expected attendance"""
        filename = 'ticket_pricing_vs_attendance.png'
        plt.figure(figsize=(12, 8))
        # Create simulated attendance data
        expected_attendance = 15000 + (pricing_data['Price'] * np.random.uniform(10, 50, len(pricing_data)))
        attendance_data = pd.DataFrame({
            'Game Type': pricing_data.index,
            'Price': pricing_data['Price'],
            'Expected Attendance': expected_attendance
        })
        
        sns.scatterplot(x='Price', y='Expected Attendance', data=attendance_data, 
                        s=100, color=COLOR_PALETTES['pricing'][1], alpha=0.7)
        sns.regplot(x='Price', y='Expected Attendance', data=attendance_data, 
                    scatter=False, color=COLOR_PALETTES['pricing'][2])
        plt.title('Price vs Expected Attendance', fontweight='bold')
        plt.xlabel('Price ($)')
        plt.ylabel('Expected Attendance')
        plt.grid(axis='both', alpha=0.3)
        
        # Add data labels
        for i, row in attendance_data.iterrows():
            plt.text(row['Price'], row['Expected Attendance'] + 500, 
                     f'{row["Game Type"]}', ha='center', fontsize=9, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_pricing_vs_revenue(self, pricing_data):
        """Visualize price vs expected revenue"""
        filename = 'ticket_pricing_vs_revenue.png'
        plt.figure(figsize=(12, 8))
        # Create simulated revenue data
        expected_attendance = 15000 + (pricing_data['Price'] * np.random.uniform(10, 50, len(pricing_data)))
        expected_revenue = pricing_data['Price'] * expected_attendance
        revenue_data = pd.DataFrame({
            'Game Type': pricing_data.index,
            'Price': pricing_data['Price'],
            'Expected Revenue': expected_revenue
        })
        
        sns.scatterplot(x='Price', y='Expected Revenue', data=revenue_data, 
                        s=100, color=COLOR_PALETTES['pricing'][3], alpha=0.7)
        sns.regplot(x='Price', y='Expected Revenue', data=revenue_data, 
                    scatter=False, color=COLOR_PALETTES['pricing'][4])
        plt.title('Price vs Expected Revenue', fontweight='bold')
        plt.xlabel('Price ($)')
        plt.ylabel('Expected Revenue ($)')
        plt.grid(axis='both', alpha=0.3)
        
        # Add data labels
        for i, row in revenue_data.iterrows():
            plt.text(row['Price'], row['Expected Revenue'] + 100000, 
                     f'{row["Game Type"]}', ha='center', fontsize=9, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_pricing_sensitivity(self, pricing_data):
        """Visualize price sensitivity analysis"""
        filename = 'ticket_pricing_sensitivity.png'
        plt.figure(figsize=(12, 8))
        
        # Create price sensitivity data
        sensitivity_data = []
        for game_type in pricing_data.index:
            base_price = pricing_data.loc[game_type, 'Price']
            for discount in [-0.2, -0.1, 0, 0.1, 0.2]:
                new_price = base_price * (1 + discount)
                # Simulate attendance change based on price change
                attendance_change = -discount * np.random.uniform(0.5, 1.5)  # Price elasticity
                sensitivity_data.append({
                    'Game Type': game_type,
                    'Price Change': discount * 100,
                    'New Price': new_price,
                    'Attendance Change': attendance_change * 100
                })
        
        sensitivity_df = pd.DataFrame(sensitivity_data)
        sns.lineplot(x='Price Change', y='Attendance Change', hue='Game Type', data=sensitivity_df, 
                     palette=COLOR_PALETTES['pricing'][:len(pricing_data)])
        plt.title('Price Sensitivity Analysis', fontweight='bold')
        plt.xlabel('Price Change (%)')
        plt.ylabel('Attendance Change (%)')
        plt.grid(axis='both', alpha=0.3)
        plt.legend(title='Game Type')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_pricing_heatmap(self, pricing_data):
        """Visualize pricing strategy heatmap"""
        filename = 'ticket_pricing_heatmap.png'
        plt.figure(figsize=(12, 8))
        
        # Create simulated pricing tiers data
        tiers = ['Economy', 'Standard', 'Premium', 'VIP']
        tier_multipliers = [0.7, 1.0, 1.5, 2.5]
        
        heatmap_data = []
        for game_type in pricing_data.index:
            base_price = pricing_data.loc[game_type, 'Price']
            for tier, multiplier in zip(tiers, tier_multipliers):
                heatmap_data.append({
                    'Game Type': game_type,
                    'Tier': tier,
                    'Price': base_price * multiplier
                })
        
        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_pivot = heatmap_df.pivot(index='Game Type', columns='Tier', values='Price')
        
        sns.heatmap(heatmap_pivot, annot=True, fmt='.2f', cmap='coolwarm', 
                    linewidths=0.5, cbar_kws={'label': 'Price ($)'})
        plt.title('Ticket Pricing Tiers Heatmap', fontweight='bold')
        plt.xlabel('Pricing Tier')
        plt.ylabel('Game Type')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def visualize_media_strategy(self, media_strategy):
        """Visualize media strategy with multiple separate files"""
        output_files = []
        
        # Create default platform and content data if media_strategy is not a dictionary
        if not isinstance(media_strategy, dict) or 'platform_strategy' not in media_strategy or 'content_strategy' not in media_strategy:
            platform_data = {
                'Instagram': 0.4,
                'Twitter/X': 0.25,
                'YouTube': 0.2,
                'TikTok': 0.15
            }
            content_data = {
                'Game_Highlights': 0.3,
                'Behind_The_Scenes': 0.25,
                'Player_Profiles': 0.2,
                'Community_Engagement': 0.15,
                'Skill_Tutorials': 0.1
            }
        else:
            # Platform investment distribution pie chart
            platform_data = media_strategy['platform_strategy']
            # Content strategy distribution using horizontal bar chart
            content_data = media_strategy['content_strategy']
        
        output_files.append(self._visualize_media_platform_distribution(platform_data))
        output_files.append(self._visualize_media_content_distribution(content_data))
        output_files.append(self._visualize_media_platform_roi(platform_data))
        output_files.append(self._visualize_media_content_engagement(content_data))
        output_files.append(self._visualize_media_platform_investment_vs_engagement(platform_data))
        output_files.append(self._visualize_media_content_performance_matrix(content_data))
        output_files.append(self._visualize_media_strategy_trends(platform_data, content_data))
        output_files.append(self._visualize_media_mix_optimization(platform_data))
        
        return output_files
    
    def _visualize_media_platform_distribution(self, platform_data):
        """Visualize media platform investment distribution"""
        filename = 'media_platform_distribution.png'
        plt.figure(figsize=(12, 10))
        colors = COLOR_PALETTES['media'][:len(platform_data)]
        wedges, texts, autotexts = plt.pie(
            platform_data.values(), 
            labels=platform_data.keys(), 
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
            textprops={'fontsize': 12}
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        plt.title('Media Platform Investment Distribution', fontweight='bold', fontsize=14)
        plt.axis('equal')
        plt.legend(wedges, platform_data.keys(), loc='best', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_media_content_distribution(self, content_data):
        """Visualize content type investment distribution"""
        filename = 'media_content_distribution.png'
        plt.figure(figsize=(12, 10))
        content_df = pd.DataFrame.from_dict(content_data, orient='index', columns=['Percentage'])
        content_df = content_df.sort_values('Percentage', ascending=True)
        
        sns.barplot(y=content_df.index, x='Percentage', data=content_df, palette=COLOR_PALETTES['media'][:len(content_df)])
        plt.title('Content Type Investment Distribution', fontweight='bold', fontsize=14)
        plt.xlabel('Percentage (%)')
        plt.ylabel('Content Type')
        plt.grid(axis='x', alpha=0.3)
        
        # Display specific values on bars
        for i, v in enumerate(content_df['Percentage']):
            plt.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_media_platform_roi(self, platform_data):
        """Visualize platform ROI comparison"""
        filename = 'media_platform_roi.png'
        plt.figure(figsize=(12, 8))
        # Create simulated ROI data
        platform_roi = {platform: value * np.random.uniform(1.2, 3.0) for platform, value in platform_data.items()}
        roi_df = pd.DataFrame.from_dict(platform_roi, orient='index', columns=['ROI'])
        roi_df = roi_df.sort_values('ROI', ascending=False)
        
        sns.barplot(x=roi_df.index, y='ROI', data=roi_df, palette=COLOR_PALETTES['media'][:len(roi_df)])
        plt.title('Estimated ROI by Media Platform', fontweight='bold')
        plt.xlabel('Platform')
        plt.ylabel('ROI (Return on Investment)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add data labels
        for i, v in enumerate(roi_df['ROI']):
            plt.text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_media_content_engagement(self, content_data):
        """Visualize content engagement metrics"""
        filename = 'media_content_engagement.png'
        plt.figure(figsize=(12, 10))
        # Create simulated engagement data
        content_engagement = {content: percentage * np.random.uniform(0.5, 1.5) for content, percentage in content_data.items()}
        engagement_df = pd.DataFrame.from_dict(content_engagement, orient='index', columns=['Engagement'])
        engagement_df = engagement_df.sort_values('Engagement', ascending=True)
        
        sns.barplot(y=engagement_df.index, x='Engagement', data=engagement_df, palette=COLOR_PALETTES['media'][:len(engagement_df)])
        plt.title('Estimated Content Engagement Metrics', fontweight='bold')
        plt.xlabel('Engagement Score')
        plt.ylabel('Content Type')
        plt.grid(axis='x', alpha=0.3)
        
        # Add data labels
        for i, v in enumerate(engagement_df['Engagement']):
            plt.text(v + 1, i, f'{v:.1f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_media_platform_investment_vs_engagement(self, platform_data):
        """Visualize platform investment vs engagement"""
        filename = 'media_platform_investment_vs_engagement.png'
        plt.figure(figsize=(12, 8))
        # Create simulated engagement data
        platform_engagement = {platform: value * np.random.uniform(1.0, 2.5) for platform, value in platform_data.items()}
        invest_engage_df = pd.DataFrame({
            'Platform': list(platform_data.keys()),
            'Investment': list(platform_data.values()),
            'Engagement': list(platform_engagement.values())
        })
        
        sns.scatterplot(x='Investment', y='Engagement', data=invest_engage_df, 
                        size='Investment', sizes=(100, 500), alpha=0.7,
                        color=COLOR_PALETTES['media'][4])
        sns.regplot(x='Investment', y='Engagement', data=invest_engage_df, scatter=False, color=COLOR_PALETTES['media'][5])
        plt.title('Platform Investment vs Engagement', fontweight='bold')
        plt.xlabel('Investment ($)')
        plt.ylabel('Engagement Score')
        plt.grid(axis='both', alpha=0.3)
        
        # Add data labels
        for i, row in invest_engage_df.iterrows():
            plt.text(row['Investment'], row['Engagement'] + 0.1, row['Platform'], ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_media_content_performance_matrix(self, content_data):
        """Visualize content type performance matrix"""
        filename = 'media_content_performance_matrix.png'
        plt.figure(figsize=(14, 10))
        
        # Create simulated performance metrics
        metrics = ['Engagement', 'Conversion', 'Retention', 'Shareability']
        performance_data = []
        
        for content_type in content_data.keys():
            for metric in metrics:
                # Simulate performance based on investment percentage
                base_value = content_data[content_type]
                performance = base_value * np.random.uniform(0.8, 1.8)
                performance_data.append({
                    'Content Type': content_type,
                    'Metric': metric,
                    'Performance': performance
                })
        
        performance_df = pd.DataFrame(performance_data)
        performance_pivot = performance_df.pivot(index='Content Type', columns='Metric', values='Performance')
        
        sns.heatmap(performance_pivot, annot=True, fmt='.2f', cmap='coolwarm', 
                    linewidths=0.5, cbar_kws={'label': 'Performance Score'})
        plt.title('Content Type Performance Matrix', fontweight='bold')
        plt.xlabel('Metric')
        plt.ylabel('Content Type')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_media_strategy_trends(self, platform_data, content_data):
        """Visualize media strategy trends"""
        filename = 'media_strategy_trends.png'
        plt.figure(figsize=(14, 10))
        
        # Create simulated trend data for 6 months
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        trend_data = []
        
        # Platform trends
        for platform in platform_data.keys():
            base_value = platform_data[platform]
            for month in months:
                # Simulate monthly trend with some variation
                trend = base_value * (1 + np.random.uniform(-0.1, 0.15))
                trend_data.append({
                    'Category': 'Platform',
                    'Name': platform,
                    'Month': month,
                    'Value': trend
                })
        
        # Content trends
        for content_type in content_data.keys():
            base_value = content_data[content_type]
            for month in months:
                # Simulate monthly trend with some variation
                trend = base_value * (1 + np.random.uniform(-0.1, 0.15))
                trend_data.append({
                    'Category': 'Content',
                    'Name': content_type,
                    'Month': month,
                    'Value': trend
                })
        
        trend_df = pd.DataFrame(trend_data)
        
        # Create two subplots for platform and content trends
        plt.subplot(2, 1, 1)
        platform_trends = trend_df[trend_df['Category'] == 'Platform']
        sns.lineplot(x='Month', y='Value', hue='Name', data=platform_trends, 
                     palette=COLOR_PALETTES['media'][:len(platform_data)])
        plt.title('Platform Investment Trends', fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Investment ($)')
        plt.grid(axis='both', alpha=0.3)
        plt.legend(title='Platform')
        
        plt.subplot(2, 1, 2)
        content_trends = trend_df[trend_df['Category'] == 'Content']
        sns.lineplot(x='Month', y='Value', hue='Name', data=content_trends, 
                     palette=COLOR_PALETTES['media'][len(platform_data):len(platform_data)+len(content_data)])
        plt.title('Content Investment Trends', fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Percentage (%)')
        plt.grid(axis='both', alpha=0.3)
        plt.legend(title='Content Type')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_media_mix_optimization(self, platform_data):
        """Visualize media mix optimization analysis"""
        filename = 'media_mix_optimization.png'
        plt.figure(figsize=(12, 8))
        
        # Create simulated optimization data
        platforms = list(platform_data.keys())
        optimization_data = []
        
        for i in range(10):  # Generate 10 different media mixes
            mix = {}
            total_budget = sum(platform_data.values())
            remaining_budget = total_budget
            
            # Randomly allocate budget across platforms
            for j, platform in enumerate(platforms):
                if j == len(platforms) - 1:
                    # Allocate remaining budget to last platform
                    mix[platform] = remaining_budget
                else:
                    allocation = remaining_budget * np.random.uniform(0.1, 0.4)
                    mix[platform] = allocation
                    remaining_budget -= allocation
            
            # Calculate simulated performance for this mix
            performance = 0
            for platform, allocation in mix.items():
                # Performance based on allocation and platform effectiveness
                effectiveness = np.random.uniform(0.8, 1.2)  # Platform effectiveness
                performance += allocation * effectiveness
            
            # Add to optimization data
            for platform, allocation in mix.items():
                optimization_data.append({
                    'Mix': f'Mix {i+1}',
                    'Platform': platform,
                    'Allocation': allocation,
                    'Total Performance': performance
                })
        
        optimization_df = pd.DataFrame(optimization_data)
        
        # Find the mix with highest performance
        best_mix = optimization_df.groupby('Mix')['Total Performance'].max().idxmax()
        best_mix_data = optimization_df[optimization_df['Mix'] == best_mix]
        
        # Plot the best mix allocation
        sns.barplot(x='Platform', y='Allocation', data=best_mix_data, palette=COLOR_PALETTES['media'][:len(platforms)])
        plt.title(f'Optimal Media Mix (Performance: {best_mix_data["Total Performance"].iloc[0]:.2f})', fontweight='bold')
        plt.xlabel('Platform')
        plt.ylabel('Allocation ($)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add data labels
        for i, v in enumerate(best_mix_data['Allocation']):
            plt.text(i, v + 1000, f'${v:.0f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def visualize_markov_decision(self, markov_model):
        """Visualize Markov decision process with multiple separate files"""
        output_files = []
        
        actions = markov_model.actions
        states = markov_model.states
        
        # 1. Transition matrix heatmap for each action
        for action in actions:
            output_files.append(self._visualize_markov_transition_matrix(markov_model, action))
        
        # 2. Action value comparison
        output_files.append(self._visualize_markov_action_values(actions))
        
        # 3. State value distribution (simulated data)
        output_files.append(self._visualize_markov_state_values(states))
        
        # 4. Policy visualization (simulated data)
        output_files.append(self._visualize_markov_policy(actions, states))
        
        # 5. Transition probability distribution for each state
        for state in states:
            output_files.append(self._visualize_markov_state_transitions(markov_model, state))
        
        # 6. Action comparison radar chart
        output_files.append(self._visualize_markov_action_radar(actions))
        
        # 7. Markov chain sample path (simulated data)
        output_files.append(self._visualize_markov_sample_path(states))
        
        # 8. Transition matrix summary heatmap
        output_files.append(self._visualize_markov_transition_summary(markov_model, actions, states))
        
        return output_files
    
    def _visualize_markov_transition_matrix(self, markov_model, action):
        """Visualize transition matrix for a specific action"""
        filename = f'markov_transition_matrix_{action.lower().replace(" ", "_")}.png'
        states = markov_model.states
        
        # Build transition matrix
        transition_matrix = []
        for state in states:
            row = []
            for next_state in states:
                row.append(markov_model.transition_matrix[action][state].get(next_state, 0))
            transition_matrix.append(row)
        
        plt.figure(figsize=(12, 10))
        # Use more attractive heatmap style with custom colors
        sns.heatmap(
            transition_matrix, 
            annot=True, 
            fmt='.2f', 
            cmap='RdBu_r',  # Use red-blue gradient for better aesthetics
            xticklabels=states, 
            yticklabels=states,
            linewidths=0.5,  # Add grid lines
            square=True,  # Square cells
            cbar_kws={'label': 'Transition Probability'}  # Color bar label
        )
        plt.title(f'{action} State Transition Matrix', fontweight='bold')
        plt.xlabel('Next State')
        plt.ylabel('Current State')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_markov_action_values(self, actions):
        """Visualize action value comparison"""
        filename = 'markov_action_values.png'
        plt.figure(figsize=(12, 8))
        # Create simulated action values
        action_values = {action: np.random.uniform(0.1, 0.9) for action in actions}
        action_values_df = pd.DataFrame.from_dict(action_values, orient='index', columns=['Expected Value'])
        action_values_df = action_values_df.sort_values('Expected Value', ascending=False)
        
        sns.barplot(x=action_values_df.index, y='Expected Value', data=action_values_df, palette=COLOR_PALETTES['markov'][:len(action_values_df)])
        plt.title('Action Value Comparison', fontweight='bold')
        plt.xlabel('Action')
        plt.ylabel('Expected Value')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add data labels
        for i, v in enumerate(action_values_df['Expected Value']):
            plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_markov_state_values(self, states):
        """Visualize state value distribution"""
        filename = 'markov_state_values.png'
        plt.figure(figsize=(12, 8))
        # Create simulated state values
        state_values = {state: np.random.uniform(0.1, 0.9) for state in states}
        state_values_df = pd.DataFrame.from_dict(state_values, orient='index', columns=['Value'])
        state_values_df = state_values_df.sort_values('Value', ascending=False)
        
        sns.barplot(x=state_values_df.index, y='Value', data=state_values_df, palette=COLOR_PALETTES['markov'][:len(state_values_df)])
        plt.title('State Value Distribution', fontweight='bold')
        plt.xlabel('State')
        plt.ylabel('Value')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add data labels
        for i, v in enumerate(state_values_df['Value']):
            plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_markov_policy(self, actions, states):
        """Visualize optimal policy"""
        filename = 'markov_policy.png'
        plt.figure(figsize=(12, 10))
        
        # Create simulated policy data
        policy_data = []
        for state in states:
            for action in actions:
                # Simulate policy probability
                probability = np.random.uniform(0, 1)
                policy_data.append({
                    'State': state,
                    'Action': action,
                    'Probability': probability
                })
        
        policy_df = pd.DataFrame(policy_data)
        policy_pivot = policy_df.pivot(index='State', columns='Action', values='Probability')
        
        sns.heatmap(policy_pivot, annot=True, fmt='.2f', cmap='coolwarm', 
                    linewidths=0.5, cbar_kws={'label': 'Policy Probability'})
        plt.title('Optimal Policy Visualization', fontweight='bold')
        plt.xlabel('Action')
        plt.ylabel('State')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_markov_state_transitions(self, markov_model, state):
        """Visualize transition probabilities for a specific state"""
        filename = f'markov_state_transitions_{state.lower().replace(" ", "_")}.png'
        actions = markov_model.actions
        
        # Build transition data
        transition_data = []
        for action in actions:
            for next_state, prob in markov_model.transition_matrix[action][state].items():
                transition_data.append({
                    'Action': action,
                    'Next State': next_state,
                    'Probability': prob
                })
        
        transition_df = pd.DataFrame(transition_data)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Next State', y='Probability', hue='Action', data=transition_df, 
                    palette=COLOR_PALETTES['markov'][:len(actions)])
        plt.title(f'Transition Probabilities from State: {state}', fontweight='bold')
        plt.xlabel('Next State')
        plt.ylabel('Probability')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.legend(title='Action')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_markov_action_radar(self, actions):
        """Visualize action comparison using radar chart"""
        filename = 'markov_action_radar.png'
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        # Define metrics for radar chart
        metrics = ['Effectiveness', 'Efficiency', 'Risk', 'Reward', 'Consistency']
        
        # Create simulated data for each action
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        for i, action in enumerate(actions):
            # Simulate metric values
            values = [np.random.uniform(0.3, 0.9) for _ in metrics]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, 'o-', linewidth=2, label=action, 
                     color=COLOR_PALETTES['markov'][i % len(COLOR_PALETTES['markov'])])
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_title('Action Comparison Radar Chart', fontweight='bold', size=14, pad=20)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_markov_sample_path(self, states):
        """Visualize a sample path through the Markov chain"""
        filename = 'markov_sample_path.png'
        plt.figure(figsize=(14, 8))
        
        # Create simulated sample path
        n_steps = 50
        sample_path = [np.random.choice(states) for _ in range(n_steps)]
        steps = list(range(n_steps))
        
        # Convert states to numerical values for plotting
        state_to_num = {state: i for i, state in enumerate(states)}
        numerical_path = [state_to_num[state] for state in sample_path]
        
        plt.plot(steps, numerical_path, '-o', linewidth=2, markersize=6, 
                 color=COLOR_PALETTES['markov'][0])
        plt.yticks(ticks=range(len(states)), labels=states)
        plt.title('Markov Chain Sample Path', fontweight='bold')
        plt.xlabel('Step')
        plt.ylabel('State')
        plt.grid(axis='both', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_markov_transition_summary(self, markov_model, actions, states):
        """Visualize transition matrix summary heatmap"""
        filename = 'markov_transition_summary.png'
        plt.figure(figsize=(14, 12))
        
        # Build summary transition matrix (average over all actions)
        summary_matrix = np.zeros((len(states), len(states)))
        for action in actions:
            for i, state in enumerate(states):
                for j, next_state in enumerate(states):
                    summary_matrix[i, j] += markov_model.transition_matrix[action][state].get(next_state, 0)
        # Average over actions
        summary_matrix /= len(actions)
        
        sns.heatmap(summary_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    xticklabels=states, yticklabels=states,
                    linewidths=0.5, square=True, 
                    cbar_kws={'label': 'Average Transition Probability'})
        plt.title('Average Transition Probability Matrix', fontweight='bold')
        plt.xlabel('Next State')
        plt.ylabel('Current State')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def visualize_roster_optimization(self, selected_players):
        """Visualize team roster optimization results with multiple separate files"""
        output_files = []
        
        # 1. Player value vs salary comparison
        output_files.append(self._visualize_roster_value_vs_salary(selected_players))
        
        # 2. Player risk score distribution
        if 'Risk_Score' in selected_players.columns:
            output_files.append(self._visualize_roster_risk_distribution(selected_players))
        
        # 3. Value to cost ratio
        output_files.append(self._visualize_roster_value_to_cost(selected_players))
        
        # 4. Team salary distribution
        output_files.append(self._visualize_roster_salary_distribution(selected_players))
        
        # 5. Player positions distribution (if available)
        if 'Pos' in selected_players.columns:
            output_files.append(self._visualize_roster_position_distribution(selected_players))
        else:
            # Fallback to value distribution
            output_files.append(self._visualize_roster_value_distribution(selected_players))
        
        # 6. Team budget utilization
        output_files.append(self._visualize_roster_budget_utilization(selected_players))
        
        # 7. Player performance metrics (if available)
        if all(col in selected_players.columns for col in ['PTS', 'AST', 'TRB']):
            output_files.append(self._visualize_roster_performance_metrics(selected_players))
        
        # 8. Age distribution (if available)
        if 'Age' in selected_players.columns:
            output_files.append(self._visualize_roster_age_distribution(selected_players))
        
        # 9. Player efficiency radar chart (if available)
        if all(col in selected_players.columns for col in ['PTS', 'AST', 'TRB', 'PER']):
            output_files.append(self._visualize_roster_efficiency_radar(selected_players))
        
        # 10. Team composition heatmap
        output_files.append(self._visualize_roster_composition_heatmap(selected_players))
        
        return output_files
    
    def _visualize_roster_value_vs_salary(self, selected_players):
        """Visualize player value vs salary comparison"""
        filename = 'roster_value_vs_salary.png'
        plt.figure(figsize=(12, 10))
        
        # Sort by value index for clearer visualization
        sorted_players = selected_players.sort_values('Value_Index', ascending=False)
        players = sorted_players['Player'].tolist()
        values = sorted_players['Value_Index'].tolist()
        salaries = (sorted_players['2023/2024'] / 1000000).tolist()  # Convert to million dollars
        
        x = np.arange(len(players))
        width = 0.4
        
        # Use custom colors
        bars1 = plt.bar(x - width/2, values, width, label='Value Index', color=COLOR_PALETTES['roster'][0])
        bars2 = plt.bar(x + width/2, salaries, width, label='Salary (Million $)', color=COLOR_PALETTES['roster'][1])
        plt.xticks(x, players, rotation=45, ha='right')
        plt.title('Optimized Roster: Player Value vs Salary', fontweight='bold')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Display specific values on bars
        for bar in bars1:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}', ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_roster_risk_distribution(self, selected_players):
        """Visualize player risk score distribution"""
        filename = 'roster_risk_distribution.png'
        plt.figure(figsize=(12, 10))
        
        # Sort by risk score
        risk_sorted = selected_players.sort_values('Risk_Score', ascending=False)
        
        # Use horizontal box plot to show overall risk distribution
        sns.boxplot(y='Risk_Score', data=selected_players, color=COLOR_PALETTES['roster'][2], width=0.3)
        # Overlay scatter plot to show each player's specific risk
        sns.swarmplot(x='Player', y='Risk_Score', data=risk_sorted, color='black', size=8, alpha=0.7)
        plt.title('Player Risk Score Distribution', fontweight='bold')
        plt.xlabel('Player')
        plt.ylabel('Risk Score')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_roster_value_to_cost(self, selected_players):
        """Visualize player value to cost ratio"""
        filename = 'roster_value_to_cost.png'
        plt.figure(figsize=(12, 10))
        
        players = selected_players['Player']
        values = selected_players['Value_Index']
        salaries = selected_players['2023/2024'] / 1000000  # Convert to million dollars
        
        value_to_cost = values / salaries
        value_cost_df = pd.DataFrame({
            'Player': players,
            'Value to Cost Ratio': value_to_cost
        }).sort_values('Value to Cost Ratio', ascending=False)
        
        sns.barplot(x='Value to Cost Ratio', y='Player', data=value_cost_df, palette=COLOR_PALETTES['roster'][:len(value_cost_df)])
        plt.title('Player Value to Cost Ratio', fontweight='bold')
        plt.xlabel('Value to Cost Ratio')
        plt.ylabel('Player')
        plt.grid(axis='x', alpha=0.3)
        
        # Add data labels
        for i, v in enumerate(value_cost_df['Value to Cost Ratio']):
            plt.text(v + 0.05, i, f'{v:.2f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_roster_salary_distribution(self, selected_players):
        """Visualize team salary distribution"""
        filename = 'roster_salary_distribution.png'
        plt.figure(figsize=(12, 8))
        
        salaries = selected_players['2023/2024'] / 1000000  # Convert to million dollars
        
        sns.histplot(salaries, bins=8, kde=True, color=COLOR_PALETTES['roster'][3], edgecolor='black')
        plt.title('Team Salary Distribution', fontweight='bold')
        plt.xlabel('Salary (Million $)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_roster_position_distribution(self, selected_players):
        """Visualize player positions distribution"""
        filename = 'roster_position_distribution.png'
        plt.figure(figsize=(12, 8))
        
        position_counts = selected_players['Pos'].value_counts()
        sns.barplot(x=position_counts.index, y=position_counts.values, palette=COLOR_PALETTES['roster'][:len(position_counts)])
        plt.title('Team Position Distribution', fontweight='bold')
        plt.xlabel('Position')
        plt.ylabel('Count')
        plt.grid(axis='y', alpha=0.3)
        
        # Add data labels
        for i, v in enumerate(position_counts.values):
            plt.text(i, v + 0.1, f'{v}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_roster_value_distribution(self, selected_players):
        """Visualize player value distribution"""
        filename = 'roster_value_distribution.png'
        plt.figure(figsize=(12, 8))
        
        values = selected_players['Value_Index']
        sns.histplot(values, bins=8, kde=True, color=COLOR_PALETTES['roster'][4], edgecolor='black')
        plt.title('Player Value Distribution', fontweight='bold')
        plt.xlabel('Value Index')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_roster_budget_utilization(self, selected_players):
        """Visualize team budget utilization"""
        filename = 'roster_budget_utilization.png'
        plt.figure(figsize=(12, 10))
        
        salaries = selected_players['2023/2024'] / 1000000  # Convert to million dollars
        total_salary = sum(salaries)
        
        # Ensure remaining budget is non-negative
        remaining_budget = max(0, 100 - total_salary)  # Assuming budget is 100 million
        
        budget_utilization = pd.DataFrame({
            'Category': ['Total Salary', 'Remaining Budget'],
            'Amount': [total_salary, remaining_budget]
        })
        
        colors = [COLOR_PALETTES['roster'][5], COLOR_PALETTES['roster'][6]]
        wedges, texts, autotexts = plt.pie(
            budget_utilization['Amount'], 
            labels=budget_utilization['Category'], 
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
            textprops={'fontsize': 12}
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        plt.title('Team Budget Utilization', fontweight='bold')
        plt.axis('equal')
        plt.legend(wedges, budget_utilization['Category'], loc='best', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_roster_performance_metrics(self, selected_players):
        """Visualize player performance metrics"""
        filename = 'roster_performance_metrics.png'
        plt.figure(figsize=(14, 10))
        
        # Melt the data for grouped bar chart
        performance_data = pd.melt(selected_players, id_vars=['Player'], 
                                  value_vars=['PTS', 'AST', 'TRB'], 
                                  var_name='Metric', value_name='Value')
        
        sns.barplot(x='Player', y='Value', hue='Metric', data=performance_data, 
                    palette=COLOR_PALETTES['roster'][:3])
        plt.title('Player Performance Metrics', fontweight='bold')
        plt.xlabel('Player')
        plt.ylabel('Value')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.legend(title='Metric')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_roster_age_distribution(self, selected_players):
        """Visualize player age distribution"""
        filename = 'roster_age_distribution.png'
        plt.figure(figsize=(12, 8))
        
        sns.histplot(selected_players['Age'], bins=10, kde=True, color=COLOR_PALETTES['roster'][4], edgecolor='black')
        plt.title('Player Age Distribution', fontweight='bold')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_roster_efficiency_radar(self, selected_players):
        """Visualize player efficiency using radar chart"""
        filename = 'roster_efficiency_radar.png'
        plt.figure(figsize=(14, 12))
        
        # Get top 5 players by value index
        top_5_players = selected_players.nlargest(5, 'Value_Index')
        
        # Define metrics for radar chart
        metrics = ['PTS', 'AST', 'TRB', 'PER']
        
        # Create radar chart for each player
        for i, (_, player) in enumerate(top_5_players.iterrows()):
            ax = plt.subplot(3, 2, i+1, polar=True)
            
            # Normalize values for radar chart
            values = player[metrics].values
            max_values = selected_players[metrics].max().values
            normalized_values = values / max_values
            
            # Create angles for radar chart
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Close the loop
            normalized_values = np.append(normalized_values, normalized_values[0])  # Close the loop
            
            # Plot radar chart
            ax.plot(angles, normalized_values, 'o-', linewidth=2, 
                     label=player['Player'], color=COLOR_PALETTES['roster'][i % len(COLOR_PALETTES['roster'])])
            ax.fill(angles, normalized_values, alpha=0.25)
            
            ax.set_title(player['Player'], fontweight='bold', size=12, pad=10)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics, fontsize=10)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
            ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        
        plt.suptitle('Player Efficiency Radar Charts', fontweight='bold', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _visualize_roster_composition_heatmap(self, selected_players):
        """Visualize team composition heatmap"""
        filename = 'roster_composition_heatmap.png'
        plt.figure(figsize=(14, 10))
        
        # Create composition data based on available columns
        composition_data = []
        
        for _, player in selected_players.iterrows():
            player_data = {
                'Player': player['Player'],
                'Value': player['Value_Index'],
                'Salary': player['2023/2024'] / 1000000  # Convert to million dollars
            }
            
            # Add optional metrics if available
            if 'Risk_Score' in selected_players.columns:
                player_data['Risk'] = player['Risk_Score']
            if 'Age' in selected_players.columns:
                player_data['Age'] = player['Age']
            if 'PER' in selected_players.columns:
                player_data['PER'] = player['PER']
            
            composition_data.append(player_data)
        
        composition_df = pd.DataFrame(composition_data)
        composition_df = composition_df.set_index('Player')
        
        # Normalize data for heatmap
        normalized_df = (composition_df - composition_df.min()) / (composition_df.max() - composition_df.min())
        
        sns.heatmap(normalized_df, annot=True, fmt='.2f', cmap='coolwarm', 
                    linewidths=0.5, cbar_kws={'label': 'Normalized Value'})
        plt.title('Team Composition Heatmap', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def visualize_decision_tree(self, model, feature_names, filename='decision_tree.png'):
        """可视化决策树模型"""
        from sklearn.tree import plot_tree
        
        # 根据树的深度调整图表大小
        depth = model.get_depth()
        fig_width = max(20, depth * 3)
        fig_height = max(10, depth * 1.5)
        plt.figure(figsize=(fig_width, fig_height))
        
        # 使用更美观的颜色和样式
        plot_tree(
            model, 
            feature_names=feature_names, 
            filled=True, 
            rounded=True, 
            fontsize=12,
            impurity=True,
            node_ids=True,
            proportion=True,
            precision=2,
            # 使用更美观的颜色映射
            class_names=[str(i) for i in model.classes_] if hasattr(model, 'classes_') else None
        )
        plt.title('决策树模型', fontweight='bold', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        return os.path.join(self.output_dir, filename)
    
    def generate_dashboard(self, model_results):
        """Generate comprehensive dashboard with multiple separate files"""
        output_files = []
        
        # 1. Player value distribution
        if 'player_data' in model_results:
            output_files.append(self._dashboard_player_value_distribution(model_results['player_data']))
        
        # 2. Top players by value
        if 'player_data' in model_results:
            output_files.append(self._dashboard_top_players(model_results['player_data']))
        
        # 3. Team expansion evaluation
        if 'expansion_data' in model_results:
            output_files.append(self._dashboard_expansion_evaluation(model_results['expansion_data']))
        
        # 4. Ticket pricing strategy
        if 'pricing_data' in model_results:
            output_files.append(self._dashboard_ticket_pricing(model_results['pricing_data']))
        
        # 5. Media platform distribution
        if 'media_data' in model_results:
            output_files.append(self._dashboard_media_distribution(model_results['media_data']))
        
        # 6. Roster salary distribution
        if 'roster_data' in model_results:
            output_files.append(self._dashboard_roster_salary(model_results['roster_data']))
        
        # 7. Decision results
        if 'decision_data' in model_results:
            output_files.append(self._dashboard_decision_results(model_results['decision_data']))
        
        # 8. Team performance metrics
        output_files.append(self._dashboard_team_performance())
        
        # 9. Budget allocation
        output_files.append(self._dashboard_budget_allocation())
        
        # 10. Key performance indicators summary
        output_files.append(self._dashboard_kpi_summary(model_results))
        
        # 11. Data correlation analysis (if player data available)
        if 'player_data' in model_results:
            output_files.append(self._dashboard_correlation_analysis(model_results['player_data']))
        
        # 12. Trend analysis (simulated data)
        output_files.append(self._dashboard_trend_analysis())
        
        return output_files
    
    def _dashboard_player_value_distribution(self, player_data):
        """Dashboard: Player value distribution"""
        filename = 'dashboard_player_value_distribution.png'
        plt.figure(figsize=(12, 8))
        
        sns.histplot(player_data['Value_Index'], bins=20, kde=True, 
                     color=COLOR_PALETTES['dashboard'][0], edgecolor='black')
        plt.title('Player Value Index Distribution', fontweight='bold')
        plt.xlabel('Value Index')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _dashboard_top_players(self, player_data):
        """Dashboard: Top players by value"""
        filename = 'dashboard_top_players.png'
        plt.figure(figsize=(12, 10))
        
        top_10_players = player_data.nlargest(10, 'Value_Index')
        sns.barplot(x='Value_Index', y='Player', data=top_10_players, 
                    palette=COLOR_PALETTES['dashboard'][:10])
        plt.title('Top 10 Players by Value Index', fontweight='bold')
        plt.xlabel('Value Index')
        plt.ylabel('Player')
        plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _dashboard_expansion_evaluation(self, expansion_data):
        """Dashboard: Team expansion evaluation"""
        filename = 'dashboard_expansion_evaluation.png'
        plt.figure(figsize=(12, 10))
        
        expansion_data = expansion_data.sort_values('Evaluation_Score', ascending=True)
        sns.barplot(y='Location', x='Evaluation_Score', data=expansion_data, 
                    palette=COLOR_PALETTES['dashboard'][:len(expansion_data)])
        plt.title('Potential Expansion Location Evaluation', fontweight='bold')
        plt.xlabel('Evaluation Score')
        plt.ylabel('Location')
        plt.grid(axis='x', alpha=0.3)
        
        # Display specific values on bars
        for i, v in enumerate(expansion_data['Evaluation_Score']):
            plt.text(v + 0.1, i, f'{v:.2f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _dashboard_ticket_pricing(self, pricing_data):
        """Dashboard: Ticket pricing strategy"""
        filename = 'dashboard_ticket_pricing.png'
        plt.figure(figsize=(12, 8))
        
        pricing_df = pd.DataFrame.from_dict(
            pricing_data['pricing_strategy'], 
            orient='index', 
            columns=['Price']
        )
        pricing_df = pricing_df.sort_values('Price', ascending=False)
        
        sns.barplot(x=pricing_df.index, y='Price', data=pricing_df, 
                    palette=COLOR_PALETTES['dashboard'][:len(pricing_df)])
        plt.title('Ticket Pricing Strategy', fontweight='bold')
        plt.xlabel('Game Type')
        plt.ylabel('Price ($)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Display specific values on bars
        for i, v in enumerate(pricing_df['Price']):
            plt.text(i, v + 2, f'${v:.2f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _dashboard_media_distribution(self, media_data):
        """Dashboard: Media platform distribution"""
        filename = 'dashboard_media_distribution.png'
        plt.figure(figsize=(12, 10))
        
        platform_data = media_data['platform_strategy']
        colors = COLOR_PALETTES['dashboard'][:len(platform_data)]
        wedges, texts, autotexts = plt.pie(
            platform_data.values(), 
            labels=platform_data.keys(), 
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        plt.title('Media Platform Investment Distribution', fontweight='bold')
        plt.axis('equal')
        plt.legend(wedges, platform_data.keys(), loc='best', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _dashboard_roster_salary(self, roster_data):
        """Dashboard: Roster salary distribution"""
        filename = 'dashboard_roster_salary.png'
        plt.figure(figsize=(12, 8))
        
        # Convert salary to million dollars for better readability
        salaries_million = roster_data['2023/2024'] / 1000000
        sns.violinplot(y=salaries_million, inner='quartile', color=COLOR_PALETTES['dashboard'][4])
        sns.swarmplot(y=salaries_million, color='black', size=6, alpha=0.7)
        plt.title('Roster Salary Distribution', fontweight='bold')
        plt.xlabel('')
        plt.ylabel('Salary (Million $)')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _dashboard_decision_results(self, decision_data):
        """Dashboard: Decision results"""
        filename = 'dashboard_decision_results.png'
        plt.figure(figsize=(12, 8))
        
        outcomes = decision_data['Expected_Outcomes']
        outcome_df = pd.DataFrame.from_dict(outcomes, orient='index', columns=['Value'])
        outcome_df = outcome_df.sort_values('Value', ascending=False)
        
        sns.barplot(x=outcome_df.index, y='Value', data=outcome_df, 
                    palette=COLOR_PALETTES['dashboard'][:len(outcome_df)])
        plt.title('Expected Decision Results', fontweight='bold')
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Display specific values on bars
        for i, v in enumerate(outcome_df['Value']):
            plt.text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _dashboard_team_performance(self):
        """Dashboard: Team performance metrics"""
        filename = 'dashboard_team_performance.png'
        plt.figure(figsize=(12, 10))
        
        # Create simulated performance data
        performance_metrics = {
            'Win Rate': np.random.uniform(0.4, 0.7),
            'Avg. Attendance': np.random.uniform(15000, 20000),
            'Revenue Growth': np.random.uniform(0.03, 0.10),
            'Social Media Followers': np.random.uniform(1000000, 5000000)
        }
        performance_df = pd.DataFrame.from_dict(performance_metrics, orient='index', columns=['Value'])
        performance_df = performance_df.sort_values('Value', ascending=False)
        
        sns.barplot(x='Value', y=performance_df.index, data=performance_df, 
                    palette=COLOR_PALETTES['dashboard'][:len(performance_df)])
        plt.title('Team Performance Metrics', fontweight='bold')
        plt.xlabel('Value')
        plt.ylabel('Metric')
        plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _dashboard_budget_allocation(self):
        """Dashboard: Budget allocation"""
        filename = 'dashboard_budget_allocation.png'
        plt.figure(figsize=(12, 10))
        
        # Create simulated budget allocation data
        budget_allocation = {
            'Player Salaries': 60,
            'Marketing': 15,
            'Operations': 10,
            'Infrastructure': 10,
            'Other': 5
        }
        budget_df = pd.DataFrame.from_dict(budget_allocation, orient='index', columns=['Percentage'])
        
        colors = COLOR_PALETTES['dashboard'][:len(budget_df)]
        wedges, texts, autotexts = plt.pie(
            budget_df['Percentage'], 
            labels=budget_df.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        plt.title('Team Budget Allocation', fontweight='bold')
        plt.axis('equal')
        plt.legend(wedges, budget_df.index, loc='best', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _dashboard_kpi_summary(self, model_results):
        """Dashboard: Key performance indicators summary"""
        filename = 'dashboard_kpi_summary.png'
        plt.figure(figsize=(14, 10))
        
        # Create KPI summary data
        kpi_data = []
        
        # Add KPIs from available data
        if 'player_data' in model_results:
            avg_value = model_results['player_data']['Value_Index'].mean()
            kpi_data.append({'KPI': 'Average Player Value', 'Value': avg_value})
        
        if 'expansion_data' in model_results:
            top_location = model_results['expansion_data'].nlargest(1, 'Evaluation_Score')['Location'].iloc[0]
            top_score = model_results['expansion_data'].nlargest(1, 'Evaluation_Score')['Evaluation_Score'].iloc[0]
            kpi_data.append({'KPI': 'Top Expansion Location Score', 'Value': top_score})
        
        if 'pricing_data' in model_results:
            pricing_df = pd.DataFrame.from_dict(model_results['pricing_data']['pricing_strategy'], orient='index', columns=['Price'])
            avg_price = pricing_df['Price'].mean()
            kpi_data.append({'KPI': 'Average Ticket Price', 'Value': avg_price})
        
        if 'roster_data' in model_results:
            total_salary = model_results['roster_data']['2023/2024'].sum() / 1000000  # Convert to million
            kpi_data.append({'KPI': 'Total Roster Salary (Million $)', 'Value': total_salary})
        
        # Add simulated KPIs if data is limited
        if len(kpi_data) < 6:
            simulated_kpis = [
                {'KPI': 'Expected Revenue Growth', 'Value': np.random.uniform(0.05, 0.15)},
                {'KPI': 'Fan Engagement Score', 'Value': np.random.uniform(0.6, 0.9)},
                {'KPI': 'Sponsorship Value', 'Value': np.random.uniform(5, 15)},
                {'KPI': 'Merchandise Sales', 'Value': np.random.uniform(2, 8)}
            ]
            kpi_data.extend(simulated_kpis[:6 - len(kpi_data)])
        
        kpi_df = pd.DataFrame(kpi_data)
        kpi_df = kpi_df.sort_values('Value', ascending=False)
        
        sns.barplot(x='Value', y='KPI', data=kpi_df, palette=COLOR_PALETTES['dashboard'][:len(kpi_df)])
        plt.title('Key Performance Indicators Summary', fontweight='bold')
        plt.xlabel('Value')
        plt.ylabel('KPI')
        plt.grid(axis='x', alpha=0.3)
        
        # Add data labels
        for i, v in enumerate(kpi_df['Value']):
            plt.text(v + 0.01, i, f'{v:.2f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _dashboard_correlation_analysis(self, player_data):
        """Dashboard: Data correlation analysis"""
        filename = 'dashboard_correlation_analysis.png'
        plt.figure(figsize=(14, 12))
        
        # Select relevant numerical columns for correlation
        metrics_cols = ['Value_Index', '2023/2024']
        if 'PER' in player_data.columns:
            metrics_cols.append('PER')
        if 'Age' in player_data.columns:
            metrics_cols.append('Age')
        if 'Risk_Score' in player_data.columns:
            metrics_cols.append('Risk_Score')
        if 'PTS' in player_data.columns:
            metrics_cols.append('PTS')
        if 'AST' in player_data.columns:
            metrics_cols.append('AST')
        if 'TRB' in player_data.columns:
            metrics_cols.append('TRB')
        
        # Filter only available columns
        available_cols = [col for col in metrics_cols if col in player_data.columns]
        correlation_data = player_data[available_cols]
        
        # Calculate correlation matrix
        corr_matrix = correlation_data.corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    square=True, linewidths=0.5, cbar_kws={'label': 'Correlation'})
        plt.title('Key Metrics Correlation Analysis', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)
    
    def _dashboard_trend_analysis(self):
        """Dashboard: Trend analysis"""
        filename = 'dashboard_trend_analysis.png'
        plt.figure(figsize=(14, 10))
        
        # Create simulated trend data for 12 months
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        trend_data = []
        
        # Generate trends for different metrics
        metrics = ['Revenue', 'Attendance', 'Social Media', 'Merchandise Sales']
        
        for metric in metrics:
            # Generate base value and trend
            base_value = np.random.uniform(1000000, 5000000) if metric == 'Revenue' else np.random.uniform(10000, 20000)
            trend = np.random.uniform(-0.02, 0.05)  # Monthly trend
            
            for i, month in enumerate(months):
                # Add some seasonality and random variation
                seasonality = np.sin(i * np.pi / 6) * 0.1  # Seasonal pattern
                variation = np.random.uniform(-0.05, 0.05)  # Random variation
                value = base_value * (1 + trend) ** i * (1 + seasonality) * (1 + variation)
                
                trend_data.append({
                    'Month': month,
                    'Metric': metric,
                    'Value': value
                })
        
        trend_df = pd.DataFrame(trend_data)
        
        # Plot trends
        sns.lineplot(x='Month', y='Value', hue='Metric', data=trend_df, 
                     palette=COLOR_PALETTES['dashboard'][:len(metrics)], linewidth=2, marker='o')
        plt.title('Annual Trend Analysis', fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Value')
        plt.grid(axis='both', alpha=0.3)
        plt.legend(title='Metric')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.join(self.output_dir, filename)