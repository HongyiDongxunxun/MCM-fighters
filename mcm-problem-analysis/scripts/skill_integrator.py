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