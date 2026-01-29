import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description='Classification model validation')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path with actual and predicted values')
    return parser.parse_args()


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """计算分类模型指标"""
    metrics = {}
    # 准确率
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    # 精确率
    metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    # 召回率
    metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    # F1分数
    metrics['f1_score'] = f1_score(y_true, y_pred, average='macro')
    # 混淆矩阵
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    # ROC AUC 分数
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, average='macro', multi_class='ovr')
        except:
            pass
    return metrics


def plot_confusion_matrix(cm, classes):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    # 检查必要的列
    if 'actual_label' in df.columns and 'predicted_label' in df.columns:
        y_true = df['actual_label'].values
        y_pred = df['predicted_label'].values
        # 检查是否有概率列
        y_pred_proba = None
        if 'predicted_prob' in df.columns:
            y_pred_proba = df['predicted_prob'].values
        # 计算指标
        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
        # 打印结果
        print("分类模型验证结果：")
        print("=" * 50)
        for key, value in metrics.items():
            if key != 'confusion_matrix':
                print(f"{key}: {value:.4f}")
        print("\n混淆矩阵：")
        print(metrics['confusion_matrix'])
        print("=" * 50)
        # 绘制混淆矩阵
        classes = np.unique(np.concatenate([y_true, y_pred]))
        plot_confusion_matrix(metrics['confusion_matrix'], classes)
        # 保存结果
        metrics_df = pd.DataFrame([{
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics.get('roc_auc', np.nan)
        }])
        metrics_df.to_csv('classification_metrics.csv', index=False)
        print("验证结果已保存到 classification_metrics.csv")
        print("混淆矩阵已保存到 confusion_matrix.png")
    else:
        print("错误：数据中没有 'actual_label' 或 'predicted_label' 列！")


if __name__ == "__main__":
    main()
