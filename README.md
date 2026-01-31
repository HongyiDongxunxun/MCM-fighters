# 体育团队管理模型

## 项目概述

本项目是一个基于机器学习和统计模型的体育团队管理决策支持系统，专为MCM 2026 ICM Problem D设计。该系统集成了多种先进的分析方法，帮助体育团队管理层在追求竞技成功的同时最大化商业价值。

## 核心功能

### 1. 球员价值评估模块
- **球员效率评分(PER)计算**：基于球员表现数据计算综合效率指标
- **财务贡献评估**：考虑球员的商业价值和市场影响力
- **伤病影响分析**：使用贝叶斯网络评估伤病对球队的连锁影响
- **替补球员价值排序**：基于替代能力评分推荐最佳替补
- **随机森林价值预测**：使用机器学习模型预测球员价值

### 2. 球队扩张与选址策略模块
- **空间计量经济学分析**：基于地理距离计算市场侵蚀效应
- **系统动力学模拟**：分析扩军后的连锁反应
- **潜在位置评估**：综合考虑市场潜力、地理因素和竞争强度

### 3. 门票定价优化模块
- **价格歧视策略**：根据不同因素制定差异化票价
- **季票转化率预测**：基于球队表现和市场因素
- **收入优化**：比较不同票价策略的预期收入

### 4. 媒体曝光度调整模块
- **社交媒体影响力分析**：基于粉丝数量和互动数据
- **媒体投资回报率计算**：优化媒体策略投资
- **平台和内容策略推荐**：根据球队特点定制媒体策略

### 5. 马尔可夫决策过程
- **值迭代算法**：求解最优决策策略
- **风险调整机制**：考虑不同决策的风险水平
- **多因素决策支持**：综合球队表现、经济条件和风险偏好

### 6. 球队阵容优化
- **多目标优化**：平衡竞技表现、商业价值和风险
- **动态规划算法**：在预算约束下优化阵容
- **综合评分系统**：考虑球员的全面价值

## 技术架构

### 核心技术栈
- **Python 3.8+**：主要开发语言
- **Pandas**：数据处理和分析
- **NumPy**：数值计算
- **Scikit-learn**：机器学习模型
- **Matplotlib/Seaborn**：数据可视化
- **SciPy**：科学计算和优化

### 模块结构
```
MCM/
├── data_source/              # 数据源
│   ├── player_team_and_performance/  # 球员表现数据
│   ├── player_salaries/              # 球员薪资数据
│   ├── player_social_influence/       # 球员社交影响力数据
│   ├── team_markets/                 # 球队市场数据
│   └── tickets_gain/                 # 门票收入数据
├── target/                   # 核心模块
│   ├── player_value_evaluator.py     # 球员价值评估
│   ├── team_expansion_analyzer.py    # 球队扩张分析
│   ├── ticket_pricing_optimizer.py   # 门票定价优化
│   ├── media_exposure_adjuster.py    # 媒体曝光度调整
│   ├── main_model.py                 # 主模型集成
│   └── test_modules.py               # 测试脚本
├── mcm-problem-analysis/     # 问题分析工具
└── README.md                 # 项目说明
```

## 安装说明

### 依赖库安装
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

### 数据准备
1. 确保 `data_source` 目录包含所有必要的数据文件
2. 数据文件格式应与现有代码兼容
3. 对于CSV文件，确保编码正确（建议使用UTF-8或latin1）

## 使用方法

### 基本使用流程

1. **初始化模型**
```python
from target.main_model import SportsTeamManagementModel

model = SportsTeamManagementModel()
model.load_data()
model.preprocess_data()
```

2. **评估球员价值**
```python
# 计算球员平衡价值
value = model.player_value_evaluator.calculate_balanced_value('Precious Achiuwa')

# 评估伤病影响
injury_impact = model.player_value_evaluator.bayesian_injury_impact_analysis('Precious Achiuwa', injury_severity=0.5)
```

3. **分析球队扩张策略**
```python
potential_locations = ['Seattle', 'Las Vegas', 'Kansas City', 'Louisville']
evaluation = model.team_expansion_analyzer.evaluate_location_strategy(potential_locations)
```

4. **优化门票定价**
```python
pricing_strategy = model.ticket_pricing_optimizer.optimize_ticket_pricing('Lakers')
```

5. **优化球队阵容**
```python
optimal_roster = model.optimize_team_roster(100000000, max_players=12)
```

6. **制定最终决策**
```python
decision = model.make_final_decision(
    current_state='Average_Performance',
    team_performance={'win_rate': 0.55, 'avg_attendance': 18000},
    economic_conditions={'market_growth': 0.04},
    risk_aversion=0.1
)
```

### 运行测试
```bash
cd target
python test_modules.py
```

## 约束条件

### 财务约束
- **薪资硬帽**：球队总薪资不得超过预算限制
- **阵容人数**：球队阵容必须在8-12人之间
- **运营成本**：考虑球馆费用和媒体宣发成本

### 竞技约束
- **位置覆盖**：确保阵容涵盖所有必要位置
- **关键球能力**：保证阵容中有足够的关键球球员
- **深度要求**：维持合理的阵容深度以应对伤病

### 风险约束
- **伤病风险**：高风险球员薪资总额不得超过一定比例
- **财务风险**：考虑由于伤病、战绩波动等因素带来的财务风险
- **长期可持续性**：平衡短期投入和长期发展

## 数据需求

### 必要数据
- **球员表现数据**：得分、篮板、助攻等技术统计
- **球员薪资数据**：当前和历史薪资信息
- **社交媒体数据**：粉丝数量和互动指标
- **球队市场数据**：市场规模、人口数据等
- **门票收入数据**： attendance 和票价信息

### 数据格式
- **CSV文件**：主要数据格式
- **编码**：建议使用UTF-8或latin1
- **分隔符**：逗号或分号

## 模型输出

### 分析报告
- **球员价值评估报告**：详细的球员价值分析和建议
- **球队扩张分析报告**：潜在位置评估和策略建议
- **门票定价策略报告**：差异化票价和预期收入
- **媒体策略建议**：平台投资和内容策略
- **阵容优化报告**：最佳阵容组合和财务分析
- **决策建议**：基于马尔可夫链的详细决策建议

### 可视化输出
- **球员价值分布图**：展示球员价值分布
- **球队扩张影响地图**：地理可视化市场影响
- **门票收入预测图**：不同票价策略的收入预测
- **媒体投资回报分析**：投资回报率可视化

## 应用场景

### 球队管理层
- **阵容构建**：在预算约束下优化球队阵容
- **球员交易决策**：评估潜在交易的价值和风险
- **门票定价策略**：制定最大化收入的票价策略
- **媒体策略优化**：优化媒体投资和内容策略
- **长期发展规划**：制定可持续的球队发展战略

### 投资者和所有者
- **投资回报分析**：评估球队投资的预期回报
- **风险评估**：分析球队运营的财务风险
- **品牌价值管理**：最大化球队的品牌价值
- **市场扩张机会**：评估潜在的市场扩张机会

## 优势与创新

1. **多模型集成**：将各种机器学习和统计模型有机结合
2. **数据驱动**：基于实际数据做出决策
3. **风险感知**：考虑决策的风险和不确定性
4. **商业价值量化**：明确考虑球员的商业价值
5. **多约束优化**：在复杂约束条件下找到最优解
6. **可解释性**：提供详细的决策分析和理由

## 未来发展

### 计划功能
- **实时数据集成**：接入实时比赛和球员数据
- **预测模型改进**：使用更先进的机器学习模型
- **交互式决策界面**：开发用户友好的决策支持界面
- **多体育项目支持**：扩展到其他体育项目
- **联盟层面分析**：提供联盟层面的战略分析

### 技术路线图
1. **数据增强**：整合更多数据源和实时数据
2. **模型优化**：改进现有模型的准确性和效率
3. **界面开发**：构建交互式决策支持系统
4. **案例研究**：应用于实际球队管理决策
5. **学术发表**：基于项目成果发表研究论文

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

如有问题或建议，请联系项目维护者：
- 邮箱：contact@sportsteammanagement.com
- 网站：www.sportsteammanagement.com

---

**版本**：1.0.0
**最后更新**：2026年1月31日
**开发团队**：MCM Sports Analytics Team