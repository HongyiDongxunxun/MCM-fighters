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