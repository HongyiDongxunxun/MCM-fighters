import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


def parse_args():
    parser = argparse.ArgumentParser(description='Deep Neural Network model')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--hidden_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--neurons', type=int, default=64, help='Number of neurons per hidden layer')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    return parser.parse_args()


def create_dnn_model(input_dim, hidden_layers=2, neurons=64, dropout=0.2, learning_rate=0.001):
    """创建深度神经网络模型"""
    model = Sequential()
    
    # 输入层
    model.add(Dense(neurons, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(dropout))
    
    # 隐藏层
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons, activation='relu'))
        model.add(Dropout(dropout))
    
    # 输出层
    model.add(Dense(1))
    
    # 编译模型
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    
    # 准备特征和目标变量
    if 'score' in df.columns:
        # 选择数值型特征
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # 排除目标变量
        if 'score' in numeric_cols:
            numeric_cols.remove('score')
        X = df[numeric_cols]
        y = df['score']
        
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # 创建模型
        model = create_dnn_model(
            input_dim=X_train.shape[1],
            hidden_layers=args.hidden_layers,
            neurons=args.neurons,
            dropout=args.dropout,
            learning_rate=args.learning_rate
        )
        
        # 训练模型
        print("训练深度神经网络模型...")
        history = model.fit(
            X_train, y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=0.1,
            verbose=1
        )
        
        # 预测
        y_pred = model.predict(X_test).flatten()
        
        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("\n深度神经网络模型评估结果：")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        
        # 保存预测结果
        test_df = pd.DataFrame(X_test, columns=numeric_cols)
        test_df['actual_score'] = y_test
        test_df['predicted_score'] = y_pred
        test_df.to_csv(args.output, index=False)
        
        # 保存模型
        model.save('dnn_model.h5')
        
        print(f"\n预测结果已保存到 {args.output}")
        print("模型已保存到 dnn_model.h5")
    else:
        print("错误：数据中没有 'score' 列！")


if __name__ == "__main__":
    main()
