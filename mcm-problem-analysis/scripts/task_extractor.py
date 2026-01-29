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