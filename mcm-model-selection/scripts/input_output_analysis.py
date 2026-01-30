import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Input-Output Analysis Model')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--model_type', type=str, default='leontief', choices=['leontief', 'ghosh'], help='Input-output model type')
    return parser.parse_args()


def load_data(file_path):
    """加载数据"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def create_direct_requirements_matrix(data):
    """创建直接消耗系数矩阵"""
    if 'sector' in data.columns:
        # 从数据中构建直接消耗系数矩阵
        sectors = data['sector'].unique()
        n_sectors = len(sectors)
        
        # 提取直接消耗数据
        A = np.zeros((n_sectors, n_sectors))
        
        for i, sector_i in enumerate(sectors):
            for j, sector_j in enumerate(sectors):
                # 查找i部门对j部门的直接消耗
                row = data[(data['sector'] == sector_i) & (data['input_sector'] == sector_j)]
                if not row.empty:
                    A[i, j] = row['direct_requirement'].values[0]
        
        return A, sectors
    else:
        # 默认直接消耗系数矩阵（体育产业示例）
        # 假设4个部门：体育赛事、体育媒体、体育用品、体育场馆
        A = np.array([
            [0.1, 0.2, 0.15, 0.3],  # 体育赛事对各部门的直接消耗
            [0.2, 0.1, 0.1, 0.15],  # 体育媒体对各部门的直接消耗
            [0.15, 0.05, 0.2, 0.1],  # 体育用品对各部门的直接消耗
            [0.25, 0.15, 0.1, 0.1]   # 体育场馆对各部门的直接消耗
        ])
        sectors = ['Sports Events', 'Sports Media', 'Sports Goods', 'Sports Venues']
        return A, sectors


def compute_leontief_inverse(A):
    """计算列昂惕夫逆矩阵"""
    n = A.shape[0]
    I = np.eye(n)
    try:
        # 计算 (I - A) 的逆矩阵
        leontief_inverse = np.linalg.inv(I - A)
        return leontief_inverse
    except np.linalg.LinAlgError:
        print("Error: Matrix is singular, cannot compute Leontief inverse.")
        return None


def compute_ghosh_inverse(B):
    """计算戈什逆矩阵"""
    n = B.shape[0]
    I = np.eye(n)
    try:
        # 计算 (I - B) 的逆矩阵
        ghosh_inverse = np.linalg.inv(I - B)
        return ghosh_inverse
    except np.linalg.LinAlgError:
        print("Error: Matrix is singular, cannot compute Ghosh inverse.")
        return None


def analyze_economic_impact(leontief_inverse, final_demand):
    """分析经济影响"""
    # 计算总产出
    total_output = leontief_inverse @ final_demand
    
    # 计算各部门的增加值
    value_added = total_output - np.sum(leontief_inverse * final_demand, axis=1)
    
    # 计算就业影响（假设就业系数）
    employment_coeff = np.array([0.1, 0.15, 0.2, 0.12])  # 每单位产出的就业人数
    employment_impact = employment_coeff * total_output
    
    return total_output, value_added, employment_impact


def main():
    args = parse_args()
    data = load_data(args.input)
    
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    print("Running Input-Output Analysis Model...")
    print(f"Model Type: {'Leontief' if args.model_type == 'leontief' else 'Ghosh'}")
    
    # 创建直接消耗系数矩阵
    A, sectors = create_direct_requirements_matrix(data)
    
    print("\nDirect Requirements Matrix (A):")
    print(pd.DataFrame(A, index=sectors, columns=sectors))
    
    if args.model_type == 'leontief':
        # 列昂惕夫模型分析
        leontief_inverse = compute_leontief_inverse(A)
        
        if leontief_inverse is not None:
            print("\nLeontief Inverse Matrix:")
            print(pd.DataFrame(leontief_inverse, index=sectors, columns=sectors))
            
            # 假设最终需求
            if 'final_demand' in data.columns:
                final_demand = data['final_demand'].values
            else:
                final_demand = np.array([100, 80, 120, 90])  # 默认最终需求
            
            print(f"\nFinal Demand:")
            print(pd.DataFrame({'Sector': sectors, 'Final Demand': final_demand}))
            
            # 分析经济影响
            total_output, value_added, employment_impact = analyze_economic_impact(leontief_inverse, final_demand)
            
            print("\nEconomic Impact Analysis:")
            impact_data = {
                'Sector': sectors,
                'Total Output': total_output,
                'Value Added': value_added,
                'Employment Impact': employment_impact
            }
            impact_df = pd.DataFrame(impact_data)
            print(impact_df)
            
            # 保存结果
            results = {
                'model_type': ['leontief'],
                'total_economic_impact': [np.sum(total_output)],
                'total_value_added': [np.sum(value_added)],
                'total_employment_impact': [np.sum(employment_impact)]
            }
            
            # 添加详细部门数据
            for i, sector in enumerate(sectors):
                results[f'{sector}_output'] = [total_output[i]]
                results[f'{sector}_value_added'] = [value_added[i]]
                results[f'{sector}_employment'] = [employment_impact[i]]
        
    else:
        # 戈什模型分析
        # 假设直接分配系数矩阵B（简化处理，使用A的转置）
        B = A.T
        
        ghosh_inverse = compute_ghosh_inverse(B)
        
        if ghosh_inverse is not None:
            print("\nGhosh Inverse Matrix:")
            print(pd.DataFrame(ghosh_inverse, index=sectors, columns=sectors))
            
            # 假设初始投入
            primary_input = np.array([50, 40, 60, 45])  # 默认初始投入
            
            print(f"\nPrimary Input:")
            print(pd.DataFrame({'Sector': sectors, 'Primary Input': primary_input}))
            
            # 计算总产出
            total_output = ghosh_inverse @ primary_input
            
            print("\nTotal Output:")
            output_df = pd.DataFrame({'Sector': sectors, 'Total Output': total_output})
            print(output_df)
            
            # 保存结果
            results = {
                'model_type': ['ghosh'],
                'total_output': [np.sum(total_output)]
            }
            
            # 添加详细部门数据
            for i, sector in enumerate(sectors):
                results[f'{sector}_output'] = [total_output[i]]
    
    # 保存结果
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output, index=False)
    
    print(f"\nResults saved to: {args.output}")
    print("\nInput-Output Analysis Model completed successfully!")


if __name__ == "__main__":
    main()
