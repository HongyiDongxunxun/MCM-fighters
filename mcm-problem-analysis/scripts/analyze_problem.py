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