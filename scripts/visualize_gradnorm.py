import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wandb
from pathlib import Path
import torch

def visualize_gradnorm_weights(run_path=None, save_dir='./gradnorm_vis'):
    """
    可视化 GradNorm 训练过程中任务权重的变化
    
    Args:
        run_path: W&B run path 或本地日志目录
        save_dir: 保存图表的目录
    """
    Path(save_dir).mkdir(exist_ok=True)
    
    # 如果提供了 W&B run path，下载数据
    if run_path and run_path.startswith('wandb'):
        api = wandb.Api()
        run = api.run(run_path)
        history = run.history()
    else:
        # 从本地文件读取（需要实现）
        history = pd.read_csv(Path(run_path) / 'metrics.csv')
    
    # 提取任务权重数据
    task_names = ['num_atom', 'lattice', 'composition', 'coord', 'type', 'kld', 'energy', 'target']
    weight_columns = [f'task_weight/{task}' for task in task_names if f'task_weight/{task}' in history.columns]
    
    # 提取损失数据
    loss_columns = [f'train_{task}_loss' for task in task_names if f'train_{task}_loss' in history.columns]
    
    # 创建图表
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # 1. 任务权重随时间变化
    for col in weight_columns:
        task_name = col.split('/')[-1]
        ax1.plot(history.index, history[col], label=task_name, linewidth=2)
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Task Weight')
    ax1.set_title('Task Weights Evolution During Training')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. 任务损失随时间变化（对数刻度）
    for col in loss_columns:
        task_name = col.replace('train_', '').replace('_loss', '')
        ax2.semilogy(history.index, history[col], label=task_name, linewidth=2)
    
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Task Losses Evolution During Training')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. 任务权重与损失比率的关系
    final_step = history.index[-1]
    final_weights = []
    initial_losses = []
    final_losses = []
    task_labels = []
    
    for task in task_names:
        weight_col = f'task_weight/{task}'
        loss_col = f'train_{task}_loss'
        
        if weight_col in history.columns and loss_col in history.columns:
            final_weights.append(history[weight_col].iloc[-1])
            initial_losses.append(history[loss_col].iloc[0])
            final_losses.append(history[loss_col].iloc[-1])
            task_labels.append(task)
    
    # 计算损失比率
    loss_ratios = np.array(final_losses) / (np.array(initial_losses) + 1e-8)
    
    # 绘制散点图
    scatter = ax3.scatter(loss_ratios, final_weights, s=100, alpha=0.7)
    for i, task in enumerate(task_labels):
        ax3.annotate(task, (loss_ratios[i], final_weights[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    ax3.set_xlabel('Loss Ratio (Final/Initial)')
    ax3.set_ylabel('Final Task Weight')
    ax3.set_title('Task Weight vs Loss Ratio')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'gradnorm_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 额外图表：任务权重热力图
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 准备数据
    weight_data = history[weight_columns].T
    weight_data.index = [col.split('/')[-1] for col in weight_columns]
    
    # 下采样以减少数据点
    step_size = max(1, len(weight_data.columns) // 100)
    weight_data_sampled = weight_data.iloc[:, ::step_size]
    
    # 绘制热力图
    im = ax.imshow(weight_data_sampled, aspect='auto', cmap='RdBu_r', 
                   interpolation='nearest', vmin=0.5, vmax=2.0)
    
    ax.set_yticks(range(len(weight_data.index)))
    ax.set_yticklabels(weight_data.index)
    ax.set_xlabel('Training Step')
    ax.set_title('Task Weights Heatmap')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Task Weight')
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'gradnorm_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化结果已保存到 {save_dir}")
    
    # 生成摘要统计
    summary = {
        'task': task_labels,
        'initial_weight': [1.0] * len(task_labels),  # 初始权重都是1
        'final_weight': final_weights,
        'weight_change': [fw - 1.0 for fw in final_weights],
        'initial_loss': initial_losses,
        'final_loss': final_losses,
        'loss_ratio': loss_ratios.tolist(),
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(Path(save_dir) / 'gradnorm_summary.csv', index=False)
    
    # 打印摘要
    print("\n任务权重摘要:")
    print(summary_df.to_string(index=False))
    
    return summary_df


def compare_gradnorm_runs(run_paths, labels, save_dir='./gradnorm_comparison'):
    """
    比较多个 GradNorm 运行（例如不同的 alpha 值）
    
    Args:
        run_paths: 运行路径列表
        labels: 对应的标签列表
        save_dir: 保存目录
    """
    Path(save_dir).mkdir(exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    for run_path, label in zip(run_paths, labels):
        if run_path.startswith('wandb'):
            api = wandb.Api()
            run = api.run(run_path)
            history = run.history()
        else:
            history = pd.read_csv(Path(run_path) / 'metrics.csv')
        
        # 绘制总损失
        ax1.plot(history.index, history['train_loss'], label=label, linewidth=2)
        
        # 计算和绘制任务权重的标准差（作为平衡性的度量）
        weight_columns = [col for col in history.columns if col.startswith('task_weight/')]
        if weight_columns:
            weight_std = history[weight_columns].std(axis=1)
            ax2.plot(history.index, weight_std, label=label, linewidth=2)
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Task Weight Std Dev')
    ax2.set_title('Task Weight Balance (Lower = More Balanced)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'gradnorm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"比较结果已保存到 {save_dir}")


if __name__ == "__main__":
    # 示例使用
    
    # 可视化单个运行
    visualize_gradnorm_weights(
        run_path='./hydra/outputs/2024-03-15/gradnorm_run',
        save_dir='./visualizations/gradnorm_single'
    )
    
    # 比较多个运行
    alpha_values = [0.5, 1.0, 1.5, 2.0]
    run_paths = [f'./hydra/outputs/2024-03-15/gradnorm_alpha_{alpha}' for alpha in alpha_values]
    labels = [f'α={alpha}' for alpha in alpha_values]
    
    compare_gradnorm_runs(
        run_paths=run_paths,
        labels=labels,
        save_dir='./visualizations/gradnorm_comparison'
    )