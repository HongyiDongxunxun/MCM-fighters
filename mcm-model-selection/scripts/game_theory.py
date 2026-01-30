import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Game Theory Model')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--game_type', type=str, default='nash', choices=['nash', 'prisoners', 'bargaining'], help='Game type')
    return parser.parse_args()


def load_data(file_path):
    """加载数据"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def compute_nash_equilibrium(payoff_matrix):
    """计算纳什均衡"""
    # 假设是双人零和博弈或非零和博弈
    # 对于双人博弈，寻找纳什均衡
    
    n_players = 2
    n_strategies1 = payoff_matrix.shape[0]
    n_strategies2 = payoff_matrix.shape[1]
    
    # 寻找纯策略纳什均衡
    nash_equilibria = []
    
    for i in range(n_strategies1):
        for j in range(n_strategies2):
            # 检查玩家1是否不会偏离
            current_payoff1 = payoff_matrix[i, j, 0] if payoff_matrix.ndim == 3 else payoff_matrix[i, j]
            best_response1 = max([payoff_matrix[k, j, 0] if payoff_matrix.ndim == 3 else payoff_matrix[k, j] for k in range(n_strategies1)])
            
            # 检查玩家2是否不会偏离
            current_payoff2 = payoff_matrix[i, j, 1] if payoff_matrix.ndim == 3 else -payoff_matrix[i, j]
            best_response2 = max([payoff_matrix[i, k, 1] if payoff_matrix.ndim == 3 else -payoff_matrix[i, k] for k in range(n_strategies2)])
            
            if current_payoff1 == best_response1 and current_payoff2 == best_response2:
                nash_equilibria.append((i, j))
    
    return nash_equilibria


def prisoners_dilemma():
    """囚徒困境博弈"""
    # 支付矩阵: [玩家1, 玩家2]
    # 行: 玩家1的策略（合作, 背叛）
    # 列: 玩家2的策略（合作, 背叛）
    payoff_matrix = np.array([
        [[3, 3], [0, 5]],  # 玩家1合作
        [[5, 0], [1, 1]]   # 玩家1背叛
    ])
    
    # 计算纳什均衡
    nash_equilibria = compute_nash_equilibrium(payoff_matrix)
    
    return payoff_matrix, nash_equilibria


def bargaining_game(bargaining_power1=0.5, total_value=100):
    """讨价还价博弈"""
    # 纳什讨价还价解
    bargaining_power2 = 1 - bargaining_power1
    
    # 纳什讨价还价解
    allocation1 = bargaining_power1 * total_value
    allocation2 = bargaining_power2 * total_value
    
    return allocation1, allocation2


def analyze_game(data, game_type):
    """分析博弈"""
    if game_type == 'prisoners':
        # 囚徒困境
        payoff_matrix, nash_equilibria = prisoners_dilemma()
        
        print("Prisoner's Dilemma Game:")
        print("Payoff Matrix:")
        print("Player 1 \ Player 2 | Cooperate | Defect")
        print("Cooperate        | [3,3]     | [0,5]")
        print("Defect           | [5,0]     | [1,1]")
        print(f"Nash Equilibria: {nash_equilibria}")
        
        return {
            'game_type': 'prisoners_dilemma',
            'payoff_matrix': payoff_matrix.tolist(),
            'nash_equilibria': nash_equilibria,
            'analysis': 'The Nash equilibrium is (Defect, Defect), which is a dominant strategy equilibrium.'
        }
        
    elif game_type == 'bargaining':
        # 讨价还价博弈
        if 'bargaining_power' in data.columns:
            bargaining_power1 = data['bargaining_power'].values[0]
        else:
            bargaining_power1 = 0.5
        
        if 'total_value' in data.columns:
            total_value = data['total_value'].values[0]
        else:
            total_value = 100
        
        allocation1, allocation2 = bargaining_game(bargaining_power1, total_value)
        
        print("Bargaining Game:")
        print(f"Bargaining Power (Player 1): {bargaining_power1}")
        print(f"Total Value: {total_value}")
        print(f"Nash Bargaining Solution:")
        print(f"Player 1: {allocation1}")
        print(f"Player 2: {allocation2}")
        
        return {
            'game_type': 'bargaining',
            'bargaining_power1': bargaining_power1,
            'total_value': total_value,
            'allocation1': allocation1,
            'allocation2': allocation2,
            'analysis': 'Nash bargaining solution based on relative bargaining power.'
        }
        
    else:  # nash
        # 一般纳什均衡分析
        # 从数据中构建支付矩阵
        if 'payoff' in data.columns:
            # 假设数据格式为：player1_strategy, player2_strategy, payoff1, payoff2
            strategies1 = sorted(data['player1_strategy'].unique())
            strategies2 = sorted(data['player2_strategy'].unique())
            
            n_strategies1 = len(strategies1)
            n_strategies2 = len(strategies2)
            
            # 构建支付矩阵
            payoff_matrix = np.zeros((n_strategies1, n_strategies2, 2))
            
            for idx, row in data.iterrows():
                i = strategies1.index(row['player1_strategy'])
                j = strategies2.index(row['player2_strategy'])
                payoff_matrix[i, j, 0] = row['payoff1']
                payoff_matrix[i, j, 1] = row['payoff2']
        else:
            # 默认支付矩阵
            payoff_matrix = np.array([
                [[4, 3], [1, 5]],
                [[5, 1], [2, 2]]
            ])
        
        # 计算纳什均衡
        nash_equilibria = compute_nash_equilibrium(payoff_matrix)
        
        print("General Nash Equilibrium Analysis:")
        print("Payoff Matrix:")
        print(payoff_matrix)
        print(f"Nash Equilibria: {nash_equilibria}")
        
        return {
            'game_type': 'general_nash',
            'payoff_matrix': payoff_matrix.tolist(),
            'nash_equilibria': nash_equilibria,
            'analysis': 'Computed Nash equilibria for the given payoff matrix.'
        }


def main():
    args = parse_args()
    data = load_data(args.input)
    
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    print("Running Game Theory Model...")
    print(f"Game Type: {args.game_type}")
    
    # 分析博弈
    result = analyze_game(data, args.game_type)
    
    # 保存结果
    results = {
        'game_type': [result['game_type']],
        'analysis': [result['analysis']]
    }
    
    # 根据博弈类型添加特定结果
    if result['game_type'] == 'prisoners_dilemma':
        results['nash_equilibria'] = [str(result['nash_equilibria'])]
    elif result['game_type'] == 'bargaining':
        results['bargaining_power1'] = [result['bargaining_power1']]
        results['total_value'] = [result['total_value']]
        results['allocation1'] = [result['allocation1']]
        results['allocation2'] = [result['allocation2']]
    else:
        results['nash_equilibria'] = [str(result['nash_equilibria'])]
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output, index=False)
    
    print(f"\nResults saved to: {args.output}")
    print("\nGame Theory Model completed successfully!")


if __name__ == "__main__":
    main()
