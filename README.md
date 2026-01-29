# MCM 比赛建模与分析工具集

## 项目介绍

本项目是一个专门为数学建模竞赛（MCM/ICM）设计的综合工具集，提供从数据预处理、模型选择与构建到模型验证的完整解决方案。同时支持从MCM题目PDF中提取任务要求并生成研究流程，实现一站式建模分析。

## 目录结构

```
MCM/
├── mcm-data-preprocessing/       # 数据预处理技能
│   ├── scripts/                  # 预处理脚本
│   │   └── preprocess.py         # 数据预处理主脚本
│   └── SKILL.md                  # 技能定义和使用说明
├── mcm-model-selection/          # 模型选择与构建技能
│   ├── scripts/                  # 模型脚本
│   │   ├── dnn_model.py          # 深度学习模型
│   │   ├── ensemble_model.py     # 模型集成
│   │   ├── gray_model.py         # GM (1,1) 灰色预测模型
│   │   ├── lasso_model.py        # LASSO回归模型
│   │   ├── lightgbm_model.py     # LightGBM模型
│   │   ├── model_evaluator.py    # 模型评估比较
│   │   ├── model_selector.py     # 模型自动选择
│   │   ├── random_forest_model.py # 随机森林模型
│   │   ├── ridge_model.py        # Ridge回归模型
│   │   ├── svm_model.py          # SVM模型
│   │   ├── topsis_model.py       # PCA-TOPSIS模型
│   │   └── xgboost_model.py      # XGBoost+SHAP模型
│   └── SKILL.md                  # 技能定义和使用说明
├── mcm-model-validation/         # 模型效果检验技能
│   ├── scripts/                  # 验证脚本
│   │   ├── cross_validate.py     # 交叉验证
│   │   ├── sensitivity_analysis.py # 敏感性分析
│   │   ├── validate_classification.py # 分类模型验证
│   │   └── validate_regression.py # 回归模型验证
│   └── SKILL.md                  # 技能定义和使用说明
├── mcm-problem-analysis/         # 题目分析与研究流程生成技能
│   ├── scripts/                  # 分析脚本
│   │   ├── analyze_problem.py    # 主分析脚本
│   │   ├── pdf_processor.py      # PDF文件处理
│   │   ├── task_extractor.py     # 任务要求提取
│   │   ├── workflow_generator.py # 研究流程生成
│   │   └── skill_integrator.py   # 与其他技能集成
│   └── SKILL.md                  # 技能定义和使用说明
├── Skill.md                      # 项目技能总览
└── README.md                     # 项目说明文档
```

## 核心技能

### 1. 数据预处理技能 (mcm-data-preprocessing)

**功能**：
- 数据清洗（处理缺失值、异常值）
- 特征工程（生成比赛相关特征、时间特征、统计特征、交互特征）
- 特征选择（基于相关性、方差、模型的特征选择方法）
- 降维处理（使用PCA等方法减少特征维度）
- 数据可视化（数据分布、特征相关性、预处理效果）
- 指标生成（计算模型选择所需的关键指标）

**使用方法**：
```bash
python mcm-data-preprocessing/scripts/preprocess.py --input data.csv --output processed_data.csv --visualization True
```

### 2. 模型选择与构建技能 (mcm-model-selection)

**功能**：
- 自动模型选择（基于预处理指标匹配最优模型）
- 多模型实现（支持GM(1,1)、XGBoost、随机森林、SVM等多种模型）
- 深度学习支持（神经网络模型）
- 模型集成（Voting、Stacking等集成方法）
- 自动参数调优（网格搜索、随机搜索等）
- 模型评估比较（全面评估和比较不同模型的性能）

**使用方法**：
```bash
# 模型选择
python mcm-model-selection/scripts/model_selector.py --indicators indicators.json

# 模型训练（以XGBoost为例）
python mcm-model-selection/scripts/xgboost_model.py --input processed_data.csv --output predictions.csv
```

### 3. 模型效果检验技能 (mcm-model-validation)

**功能**：
- 回归模型验证（MSE、RMSE、MAE、R²等指标）
- 分类模型验证（准确率、精确率、召回率、F1分数等）
- 交叉验证（K折交叉验证，评估模型稳定性）
- 敏感性分析（分析模型对参数变化的敏感性）

**使用方法**：
```bash
# 回归模型验证
python mcm-model-validation/scripts/validate_regression.py --input predictions.csv

# 交叉验证
python mcm-model-validation/scripts/cross_validate.py --input processed_data.csv --model xgboost
```

### 4. 题目分析与研究流程生成技能 (mcm-problem-analysis)

**功能**：
- PDF文件处理（提取MCM题目文本内容）
- 任务要求提取（识别任务要求、约束条件、数据需求等）
- 研究流程生成（根据任务要求生成完整研究流程）
- 技能集成（调用其他三个技能完成研究）
- 流程可视化（生成研究流程的可视化图表）

**使用方法**：
```bash
# 完整流程执行
python mcm-problem-analysis/scripts/analyze_problem.py --input problem.pdf --data data.csv --output results/
```

## 环境配置

### 基本依赖

```bash
# 数据预处理依赖
pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels featuretools plotly

# 模型选择与构建依赖
pip install xgboost shap keras tensorflow optuna lightgbm

# 模型验证依赖
# 与数据预处理依赖相同

# 题目分析依赖
pip install PyPDF2 pdfplumber nltk spacy networkx

# 安装中文语言模型（可选）
python -m spacy download zh_core_web_sm
python -m nltk.downloader punkt
```

### 一键安装所有依赖

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels featuretools plotly xgboost shap keras tensorflow optuna lightgbm PyPDF2 pdfplumber nltk spacy networkx
```

## 使用流程

### 完整建模流程

1. **题目分析**：处理MCM题目PDF，提取任务要求并生成研究流程
   ```bash
   python mcm-problem-analysis/scripts/analyze_problem.py --input problem.pdf --data data.csv --output results/
   ```

2. **分步执行**：
   - 分析PDF并提取任务：
     ```bash
     python mcm-problem-analysis/scripts/pdf_processor.py --input problem.pdf --output extracted_tasks.json
     ```
   - 生成研究流程：
     ```bash
     python mcm-problem-analysis/scripts/workflow_generator.py --tasks extracted_tasks.json --output research_workflow.json
     ```
   - 数据预处理：
     ```bash
     python mcm-data-preprocessing/scripts/preprocess.py --input data.csv --output processed_data.csv --visualization True
     ```
   - 模型选择：
     ```bash
     python mcm-model-selection/scripts/model_selector.py --indicators indicators.json
     ```
   - 模型训练：
     ```bash
     python mcm-model-selection/scripts/xgboost_model.py --input processed_data.csv --output predictions.csv
     ```
   - 模型验证：
     ```bash
     python mcm-model-validation/scripts/validate_regression.py --input predictions.csv
     ```

## 示例应用

### 网球比赛动量分析示例

1. **数据预处理**：
   ```bash
   python mcm-data-preprocessing/scripts/preprocess.py --input tennis_data.csv --output processed_tennis_data.csv --visualization True
   ```

2. **模型选择**：
   ```bash
   python mcm-model-selection/scripts/model_selector.py --indicators indicators.json
   ```

3. **模型训练与预测**：
   ```bash
   python mcm-model-selection/scripts/xgboost_model.py --input processed_tennis_data.csv --output tennis_predictions.csv
   ```

4. **模型验证**：
   ```bash
   python mcm-model-validation/scripts/validate_regression.py --input tennis_predictions.csv
   python mcm-model-validation/scripts/sensitivity_analysis.py --input processed_tennis_data.csv --model xgboost
   ```

## 技术特点

1. **模块化设计**：各技能独立封装，可单独使用也可集成使用
2. **自动化流程**：从数据预处理到模型验证的全流程自动化
3. **多模型支持**：提供多种经典和现代建模方法
4. **深度学习集成**：支持神经网络等深度学习模型
5. **可视化分析**：丰富的数据和结果可视化功能
6. **PDF处理能力**：从MCM题目PDF中自动提取任务要求
7. **研究流程生成**：根据任务要求自动生成合理的研究流程
8. **跨技能集成**：各技能之间无缝集成，形成完整研究闭环

## 注意事项

1. **数据格式**：确保输入数据格式正确，包含必要的变量
2. **依赖安装**：使用前请安装所有必要的依赖库
3. **路径设置**：执行脚本时请确保路径设置正确
4. **参数调整**：根据具体数据集调整模型参数
5. **PDF质量**：确保输入的PDF文件清晰可读，不含扫描图像或加密内容

## 扩展建议

1. **数据增强**：添加更多数据预处理和特征工程方法
2. **模型扩展**：集成更多先进的机器学习和深度学习模型
3. **自动调优**：增强自动参数调优功能，支持更复杂的调优策略
4. **多语言支持**：增加对中文MCM题目的更好支持
5. **团队协作**：添加版本控制和团队协作功能
6. **报告生成**：根据研究结果自动生成初步报告

## 项目维护

- **版本**：1.0.0
- **作者**：MCM建模团队
- **更新日期**：2026-01-30

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

如有问题或建议，请联系项目维护者。

---

**祝大家在MCM/ICM比赛中取得优异成绩！**