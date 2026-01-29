---
name: mcm-problem-analysis
description: Comprehensive MCM contest problem analysis and research workflow generation. Use when analyzing MCM contest problems from PDF files, extracting task requirements, generating research workflows, and integrating with data preprocessing, model selection, and validation skills.
---

# MCM 比赛题目分析与研究流程生成技能

## 核心功能

本技能提供 **"PDF文件处理 + 任务要求提取 + 研究流程生成 + 多技能集成"** 的完整流程，帮助用户从MCM题目PDF中提取任务要求，生成研究流程，并集成其他三个技能完成研究。

### 功能概述

1. **PDF文件处理**：支持处理包含MCM题目的PDF文件，提取文本内容
2. **任务要求提取**：从PDF中识别并提取任务要求、约束条件、数据需求等关键信息
3. **研究流程生成**：根据提取的任务要求，结合相关信息生成完整的研究流程
4. **技能集成**：调用数据预处理、模型选择与构建、模型效果检验三个技能完成研究
5. **流程可视化**：生成研究流程的可视化图表，帮助理解整体研究思路

## 环境配置

```python
# 必备库安装
pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels PyPDF2 pdfplumber nltk spacy networkx

# 可选：安装中文语言模型
python -m spacy download zh_core_web_sm
python -m nltk.downloader punkt
```

## 使用方法

### 基本使用流程

1. **PDF文件分析**：处理MCM题目PDF文件，提取任务要求
2. **研究流程生成**：根据提取的任务要求生成研究流程
3. **技能集成执行**：调用其他三个技能完成具体研究任务

### 命令示例

```python
# 分析PDF文件并提取任务要求
python scripts/pdf_processor.py --input problem.pdf --output extracted_tasks.json

# 生成研究流程
python scripts/workflow_generator.py --tasks extracted_tasks.json --output research_workflow.json

# 集成其他技能执行研究
python scripts/skill_integrator.py --workflow research_workflow.json --data data.csv

# 完整流程执行
python scripts/analyze_problem.py --input problem.pdf --data data.csv --output final_results/
```

## 脚本实现

### 分析主脚本 (scripts/analyze_problem.py)

```python
import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description='MCM problem analysis and research workflow generation')
    parser.add_argument('--input', type=str, required=True, help='Input PDF file path with MCM problem')
    parser.add_argument('--data', type=str, required=True, help='Input data file path')
    parser.add_argument('--output', type=str, default='results', help='Output directory path')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 1. 处理PDF文件并提取任务要求
    print("Step 1: Processing PDF file and extracting tasks...")
    os.system(f'python scripts/pdf_processor.py --input {args.input} --output {args.output}/extracted_tasks.json')
    
    # 2. 生成研究流程
    print("Step 2: Generating research workflow...")
    os.system(f'python scripts/workflow_generator.py --tasks {args.output}/extracted_tasks.json --output {args.output}/research_workflow.json')
    
    # 3. 集成其他技能执行研究
    print("Step 3: Integrating with other skills and executing research...")
    os.system(f'python scripts/skill_integrator.py --workflow {args.output}/research_workflow.json --data {args.data} --output {args.output}')
    
    print("\nMCM problem analysis and research workflow execution completed!")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
```

### PDF处理脚本 (scripts/pdf_processor.py)

```python
import argparse
import PyPDF2
import pdfplumber
import json


def parse_args():
    parser = argparse.ArgumentParser(description='PDF file processor for MCM problem')
    parser.add_argument('--input', type=str, required=True, help='Input PDF file path')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path for extracted text')
    return parser.parse_args()


def extract_text_from_pdf(pdf_path):
    """从PDF文件中提取文本"""
    text = ""
    try:
        # 使用pdfplumber提取文本
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        # 备用：使用PyPDF2
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e2:
            print(f"Error extracting text: {e2}")
    return text


def main():
    args = parse_args()
    
    # 提取文本
    text = extract_text_from_pdf(args.input)
    
    # 保存提取的文本
    result = {
        'extracted_text': text,
        'file_path': args.input
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"PDF text extracted successfully and saved to {args.output}")


if __name__ == "__main__":
    main()
```

### 任务提取脚本 (scripts/task_extractor.py)

```python
import argparse
import json
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


def parse_args():
    parser = argparse.ArgumentParser(description='Task extractor from MCM problem text')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file path with extracted text')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path for extracted tasks')
    return parser.parse_args()


def extract_tasks(text):
    """从文本中提取任务要求"""
    tasks = {
        'main_tasks': [],
        'subtasks': [],
        'constraints': [],
        'data_requirements': [],
        'evaluation_criteria': []
    }
    
    # 分割句子
    sentences = sent_tokenize(text)
    
    # 关键词模式
    task_patterns = [r'task', r'problem', r'objective', r'goal', r'require', r'need', r'must']
    constraint_patterns = [r'constraint', r'limit', r'restrict', r'bound', r'condition']
    data_patterns = [r'data', r'information', r'variable', r'parameter', r'metric', r'indicator']
    evaluation_patterns = [r'evaluate', r'assess', r'measure', r'validate', r'criteria', r'metric']
    
    # 提取任务
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # 提取主任务
        if any(pattern in sentence_lower for pattern in task_patterns) and ('?' in sentence or '!' in sentence or '。' in sentence):
            tasks['main_tasks'].append(sentence)
        
        # 提取约束条件
        elif any(pattern in sentence_lower for pattern in constraint_patterns):
            tasks['constraints'].append(sentence)
        
        # 提取数据需求
        elif any(pattern in sentence_lower for pattern in data_patterns):
            tasks['data_requirements'].append(sentence)
        
        # 提取评估标准
        elif any(pattern in sentence_lower for pattern in evaluation_patterns):
            tasks['evaluation_criteria'].append(sentence)
    
    return tasks


def main():
    args = parse_args()
    
    # 加载提取的文本
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取任务
    text = data.get('extracted_text', '')
    tasks = extract_tasks(text)
    
    # 保存提取的任务
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    
    print(f"Tasks extracted successfully and saved to {args.output}")


if __name__ == "__main__":
    main()
```

### 研究流程生成脚本 (scripts/workflow_generator.py)

```python
import argparse
import json
import networkx as nx
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Research workflow generator for MCM problem')
    parser.add_argument('--tasks', type=str, required=True, help='Input JSON file path with extracted tasks')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path for research workflow')
    return parser.parse_args()


def generate_workflow(tasks):
    """根据任务要求生成研究流程"""
    workflow = {
        'title': 'MCM Research Workflow',
        'steps': [],
        'dependencies': []
    }
    
    # 基础步骤
    base_steps = [
        {
            'id': 1,
            'name': '问题理解与分析',
            'description': '深入理解MCM题目要求，明确研究目标',
            'tasks': tasks.get('main_tasks', [])
        },
        {
            'id': 2,
            'name': '数据收集与预处理',
            'description': '收集相关数据并进行清洗、特征工程等预处理',
            'tasks': tasks.get('data_requirements', [])
        },
        {
            'id': 3,
            'name': '模型选择与构建',
            'description': '根据问题特点选择合适的模型并进行构建',
            'constraints': tasks.get('constraints', [])
        },
        {
            'id': 4,
            'name': '模型训练与优化',
            'description': '训练模型并进行参数调优',
            'constraints': tasks.get('constraints', [])
        },
        {
            'id': 5,
            'name': '模型验证与评估',
            'description': '验证模型性能并评估结果',
            'tasks': tasks.get('evaluation_criteria', [])
        },
        {
            'id': 6,
            'name': '结果分析与可视化',
            'description': '分析模型结果并进行可视化展示'
        },
        {
            'id': 7,
            'name': '报告撰写',
            'description': '撰写完整的研究报告'
        }
    ]
    
    workflow['steps'] = base_steps
    
    # 定义依赖关系
    dependencies = [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7)
    ]
    
    workflow['dependencies'] = dependencies
    
    return workflow


def visualize_workflow(workflow, output_file):
    """可视化研究流程"""
    G = nx.DiGraph()
    
    # 添加节点
    for step in workflow['steps']:
        G.add_node(step['id'], label=step['name'])
    
    # 添加边
    for dep in workflow['dependencies']:
        G.add_edge(dep[0], dep[1])
    
    # 绘制图表
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'),
            node_size=3000, node_color='lightblue', font_size=10, font_weight='bold',
            edge_color='gray', arrowsize=20)
    plt.title('MCM Research Workflow')
    plt.tight_layout()
    plt.savefig(output_file)


def main():
    args = parse_args()
    
    # 加载提取的任务
    with open(args.tasks, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    # 生成研究流程
    workflow = generate_workflow(tasks)
    
    # 保存研究流程
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(workflow, f, ensure_ascii=False, indent=2)
    
    # 可视化研究流程
    visualize_workflow(workflow, 'workflow_visualization.png')
    
    print(f"Research workflow generated successfully and saved to {args.output}")
    print("Workflow visualization saved to workflow_visualization.png")


if __name__ == "__main__":
    main()
```

### 技能集成脚本 (scripts/skill_integrator.py)

```python
import argparse
import json
import os
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description='Skill integrator for MCM research workflow')
    parser.add_argument('--workflow', type=str, required=True, help='Input JSON file path with research workflow')
    parser.add_argument('--data', type=str, required=True, help='Input data file path')
    parser.add_argument('--output', type=str, default='results', help='Output directory path')
    return parser.parse_args()


def run_command(command, cwd=None):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=cwd)
        print(f"Command: {command}")
        print(f"Output: {result.stdout}")
        if result.stderr:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def integrate_skills(workflow, data_file, output_dir):
    """集成其他三个技能执行研究"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 数据预处理
    print("\n=== Step 1: Data Preprocessing ===")
    preprocess_cmd = f'python ../mcm-data-preprocessing/scripts/preprocess.py --input {data_file} --output {output_dir}/processed_data.csv --visualization True'
    success = run_command(preprocess_cmd, cwd='d:\\code\\MCM\\mcm-problem-analysis')
    
    if not success:
        print("Data preprocessing failed!")
        return False
    
    # 2. 模型选择与构建
    print("\n=== Step 2: Model Selection and Building ===")
    # 假设预处理生成了indicators.json文件
    model_select_cmd = f'python ../mcm-model-selection/scripts/model_selector.py --indicators {output_dir}/indicators.json'
    success = run_command(model_select_cmd, cwd='d:\\code\\MCM\\mcm-problem-analysis')
    
    if not success:
        print("Model selection failed!")
        return False
    
    # 3. 模型训练与预测
    print("\n=== Step 3: Model Training and Prediction ===")
    model_train_cmd = f'python ../mcm-model-selection/scripts/xgboost_model.py --input {output_dir}/processed_data.csv --output {output_dir}/predictions.csv'
    success = run_command(model_train_cmd, cwd='d:\\code\\MCM\\mcm-problem-analysis')
    
    if not success:
        print("Model training failed!")
        return False
    
    # 4. 模型验证
    print("\n=== Step 4: Model Validation ===")
    validate_cmd = f'python ../mcm-model-validation/scripts/validate_regression.py --input {output_dir}/predictions.csv'
    success = run_command(validate_cmd, cwd='d:\\code\\MCM\\mcm-problem-analysis')
    
    if not success:
        print("Model validation failed!")
        return False
    
    # 5. 交叉验证
    print("\n=== Step 5: Cross Validation ===")
    cross_validate_cmd = f'python ../mcm-model-validation/scripts/cross_validate.py --input {output_dir}/processed_data.csv --model xgboost'
    success = run_command(cross_validate_cmd, cwd='d:\\code\\MCM\\mcm-problem-analysis')
    
    if not success:
        print("Cross validation failed!")
        return False
    
    # 6. 敏感性分析
    print("\n=== Step 6: Sensitivity Analysis ===")
    sensitivity_cmd = f'python ../mcm-model-validation/scripts/sensitivity_analysis.py --input {output_dir}/processed_data.csv --model xgboost'
    success = run_command(sensitivity_cmd, cwd='d:\\code\\MCM\\mcm-problem-analysis')
    
    if not success:
        print("Sensitivity analysis failed!")
        return False
    
    return True


def main():
    args = parse_args()
    
    # 加载研究流程
    with open(args.workflow, 'r', encoding='utf-8') as f:
        workflow = json.load(f)
    
    # 集成其他技能
    success = integrate_skills(workflow, args.data, args.output)
    
    if success:
        print("\n=== Skill integration completed successfully! ===")
        print(f"All results saved to: {args.output}")
    else:
        print("\n=== Skill integration failed! ===")


if __name__ == "__main__":
    main()
```

## 注意事项

- **PDF文件格式**：确保输入的PDF文件清晰可读，不含扫描图像或加密内容
- **依赖库安装**：使用前请安装所有必要的依赖库
- **数据准备**：确保提供的数据文件格式正确，包含研究所需的变量
- **路径设置**：执行脚本时请确保路径设置正确，特别是在集成其他技能时

## 使用建议

1. **PDF预处理**：对于复杂的PDF文件，可以先转换为文本文件以提高提取效果
2. **任务验证**：提取任务要求后，请手动验证提取结果的准确性
3. **流程调整**：根据具体MCM题目的特点，调整生成的研究流程
4. **技能选择**：根据研究流程的需要，选择合适的子技能执行具体任务
5. **结果分析**：关注最终结果的物理意义和实际应用价值

## 扩展功能

- **多语言支持**：增加对中文MCM题目的支持
- **自动数据收集**：根据任务要求自动搜索和收集相关数据
- **流程优化**：根据历史数据优化研究流程
- **团队协作**：支持多用户协作完成研究任务
- **报告自动生成**：根据研究结果自动生成初步报告

## 输出文件

执行完成后，将生成以下文件：
1. **extracted_tasks.json**：提取的任务要求
2. **research_workflow.json**：生成的研究流程
3. **workflow_visualization.png**：研究流程可视化图表
4. **processed_data.csv**：预处理后的数据
5. **predictions.csv**：模型预测结果
6. **各种验证和分析结果文件**：如metrics.csv、sensitivity_analysis.csv等

## 集成说明

本技能通过调用以下三个技能完成完整的研究流程：
1. **mcm-data-preprocessing**：负责数据清洗、特征工程等预处理工作
2. **mcm-model-selection**：负责模型选择、构建和训练
3. **mcm-model-validation**：负责模型验证、评估和分析

通过这种集成方式，用户可以从MCM题目PDF开始，一站式完成整个研究过程，提高研究效率和质量。