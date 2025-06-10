import pandas as pd 
import numpy as np
import os
import json
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def stratified_sampling(df, target_column, n_bins=10):
    """
    基于目标属性进行分层采样，确保数据均匀性
    
    Args:
        df: 数据框
        target_column: 目标属性列名
        n_bins: 分层数量
    
    Returns:
        每层的索引字典
    """
    # 创建分层标签
    df_copy = df.copy()
    
    # 对目标属性进行分箱
    df_copy['target_bin'] = pd.cut(df_copy[target_column], bins=n_bins, labels=False)
    
    # 处理可能的NaN值（边界情况）
    df_copy['target_bin'] = df_copy['target_bin'].fillna(0)
    
    # 获取每层的索引
    stratified_indices = {}
    for bin_idx in range(n_bins):
        bin_mask = df_copy['target_bin'] == bin_idx
        bin_indices = df_copy[bin_mask].index.tolist()
        if len(bin_indices) > 0:
            stratified_indices[bin_idx] = bin_indices
    
    return stratified_indices

def balanced_split_with_fixed_train_size(
    df, 
    target_column, 
    train_size, 
    val_ratio=0.1, 
    test_ratio=0.1,
    stratify_bins=10,
    random_state=42
):
    """
    根据固定的训练集大小进行均匀分割
    
    Args:
        df: 数据框
        target_column: 目标属性列名用于分层
        train_size: 训练集大小（绝对数量）
        val_ratio: 验证集占总数据的比例
        test_ratio: 测试集占总数据的比例
        stratify_bins: 分层数量
        random_state: 随机种子
    
    Returns:
        train_df, val_df, test_df
    """
    np.random.seed(random_state)
    
    total_size = len(df)
    
    # 检查训练集大小是否合理
    if train_size >= total_size:
        raise ValueError(f"训练集大小 ({train_size}) 不能大于等于总数据量 ({total_size})")
    
    # 计算验证集和测试集的绝对大小
    val_size = int(total_size * val_ratio)
    test_size = int(total_size * test_ratio)
    
    # 确保总和不超过数据总量
    if train_size + val_size + test_size > total_size:
        # 按比例调整验证集和测试集大小
        remaining_size = total_size - train_size
        val_size = int(remaining_size * val_ratio / (val_ratio + test_ratio))
        test_size = remaining_size - val_size
    
    print(f"数据分割:")
    print(f"  总数据量: {total_size}")
    print(f"  训练集: {train_size} ({train_size/total_size*100:.1f}%)")
    print(f"  验证集: {val_size} ({val_size/total_size*100:.1f}%)")
    print(f"  测试集: {test_size} ({test_size/total_size*100:.1f}%)")
    print(f"  使用数据: {train_size + val_size + test_size} ({(train_size + val_size + test_size)/total_size*100:.1f}%)")
    
    # 进行分层采样
    stratified_indices = stratified_sampling(df, target_column, stratify_bins)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    # 从每一层按比例采样
    for bin_idx, indices in stratified_indices.items():
        n_indices = len(indices)
        if n_indices == 0:
            continue
            
        # 计算每层应该贡献的样本数
        layer_train_size = int(train_size * n_indices / total_size)
        layer_val_size = int(val_size * n_indices / total_size)
        layer_test_size = int(test_size * n_indices / total_size)
        
        # 确保不超过该层的总数
        layer_total = layer_train_size + layer_val_size + layer_test_size
        if layer_total > n_indices:
            # 按比例缩放
            scale_factor = n_indices / layer_total
            layer_train_size = int(layer_train_size * scale_factor)
            layer_val_size = int(layer_val_size * scale_factor)
            layer_test_size = n_indices - layer_train_size - layer_val_size
        
        # 随机采样
        np.random.shuffle(indices)
        
        train_indices.extend(indices[:layer_train_size])
        val_indices.extend(indices[layer_train_size:layer_train_size + layer_val_size])
        test_indices.extend(indices[layer_train_size + layer_val_size:layer_train_size + layer_val_size + layer_test_size])
    
    # 如果采样数量不足，从剩余数据中补充
    all_sampled = set(train_indices + val_indices + test_indices)
    remaining_indices = [i for i in df.index if i not in all_sampled]
    
    # 补充训练集
    while len(train_indices) < train_size and remaining_indices:
        train_indices.append(remaining_indices.pop(0))
    
    # 补充验证集
    while len(val_indices) < val_size and remaining_indices:
        val_indices.append(remaining_indices.pop(0))
    
    # 补充测试集
    while len(test_indices) < test_size and remaining_indices:
        test_indices.append(remaining_indices.pop(0))
    
    # 创建数据框
    train_df = df.loc[train_indices].copy()
    val_df = df.loc[val_indices].copy()
    test_df = df.loc[test_indices].copy()
    
    return train_df, val_df, test_df

def analyze_data_distribution(df, target_column, output_dir, dataset_name):
    """
    分析数据分布并生成可视化图表
    
    Args:
        df: 数据框
        target_column: 目标属性列名
        output_dir: 输出目录
        dataset_name: 数据集名称
    """
    # 创建可视化目录
    viz_dir = os.path.join(output_dir, 'distribution_analysis')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 设置图表样式
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. 目标属性分布
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 直方图
    axes[0, 0].hist(df[target_column], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title(f'{target_column} Distribution')
    axes[0, 0].set_xlabel(target_column)
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 箱线图
    axes[0, 1].boxplot(df[target_column])
    axes[0, 1].set_title(f'{target_column} Box Plot')
    axes[0, 1].set_ylabel(target_column)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q图（检查正态性）
    from scipy import stats
    stats.probplot(df[target_column], dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title(f'{target_column} Q-Q Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 累积分布
    sorted_values = np.sort(df[target_column])
    y = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    axes[1, 1].plot(sorted_values, y)
    axes[1, 1].set_title(f'{target_column} Cumulative Distribution')
    axes[1, 1].set_xlabel(target_column)
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'{dataset_name}_{target_column}_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 形成能与目标属性的关系
    if 'formation_energy_per_atom' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df['formation_energy_per_atom'], df[target_column], alpha=0.6)
        plt.xlabel('Formation Energy per Atom (eV)')
        plt.ylabel(target_column)
        plt.title(f'Formation Energy vs {target_column}')
        plt.grid(True, alpha=0.3)
        
        # 添加相关性系数
        correlation = df['formation_energy_per_atom'].corr(df[target_column])
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.savefig(os.path.join(viz_dir, f'{dataset_name}_energy_vs_{target_column}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def validate_split_uniformity(train_df, val_df, test_df, target_column, output_dir, dataset_name):
    """
    验证数据分割的均匀性
    
    Args:
        train_df, val_df, test_df: 分割后的数据框
        target_column: 目标属性列名
        output_dir: 输出目录
        dataset_name: 数据集名称
    """
    # 创建可视化目录
    viz_dir = os.path.join(output_dir, 'split_validation')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 统计信息
    stats_comparison = pd.DataFrame({
        'Dataset': ['Train', 'Validation', 'Test'],
        'Size': [len(train_df), len(val_df), len(test_df)],
        'Mean': [train_df[target_column].mean(), val_df[target_column].mean(), test_df[target_column].mean()],
        'Std': [train_df[target_column].std(), val_df[target_column].std(), test_df[target_column].std()],
        'Min': [train_df[target_column].min(), val_df[target_column].min(), test_df[target_column].min()],
        'Max': [train_df[target_column].max(), val_df[target_column].max(), test_df[target_column].max()],
        'Median': [train_df[target_column].median(), val_df[target_column].median(), test_df[target_column].median()]
    })
    
    # 保存统计信息
    stats_comparison.to_csv(os.path.join(viz_dir, f'{dataset_name}_split_statistics.csv'), index=False)
    
    # 可视化比较
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 分布比较（直方图）
    axes[0, 0].hist(train_df[target_column], bins=30, alpha=0.7, label='Train', density=True)
    axes[0, 0].hist(val_df[target_column], bins=30, alpha=0.7, label='Validation', density=True)
    axes[0, 0].hist(test_df[target_column], bins=30, alpha=0.7, label='Test', density=True)
    axes[0, 0].set_title(f'{target_column} Distribution Comparison')
    axes[0, 0].set_xlabel(target_column)
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 箱线图比较
    data_for_box = [train_df[target_column], val_df[target_column], test_df[target_column]]
    axes[0, 1].boxplot(data_for_box, labels=['Train', 'Val', 'Test'])
    axes[0, 1].set_title(f'{target_column} Box Plot Comparison')
    axes[0, 1].set_ylabel(target_column)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 统计信息条形图
    metrics = ['Mean', 'Std', 'Median']
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, dataset in enumerate(['Train', 'Validation', 'Test']):
        values = [stats_comparison.loc[i, metric] for metric in metrics]
        axes[1, 0].bar(x + i*width, values, width, label=dataset)
    
    axes[1, 0].set_xlabel('Metrics')
    axes[1, 0].set_ylabel('Values')
    axes[1, 0].set_title('Statistical Metrics Comparison')
    axes[1, 0].set_xticks(x + width)
    axes[1, 0].set_xticklabels(metrics)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 累积分布比较
    for df_split, label in [(train_df, 'Train'), (val_df, 'Validation'), (test_df, 'Test')]:
        sorted_values = np.sort(df_split[target_column])
        y = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        axes[1, 1].plot(sorted_values, y, label=label, linewidth=2)
    
    axes[1, 1].set_xlabel(target_column)
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Cumulative Distribution Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'{dataset_name}_split_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 计算KS测试统计量（检验分布一致性）
    from scipy.stats import ks_2samp
    
    ks_train_val = ks_2samp(train_df[target_column], val_df[target_column])
    ks_train_test = ks_2samp(train_df[target_column], test_df[target_column])
    ks_val_test = ks_2samp(val_df[target_column], test_df[target_column])
    
    ks_results = {
        'train_vs_val': {'statistic': ks_train_val.statistic, 'p_value': ks_train_val.pvalue},
        'train_vs_test': {'statistic': ks_train_test.statistic, 'p_value': ks_train_test.pvalue},
        'val_vs_test': {'statistic': ks_val_test.statistic, 'p_value': ks_val_test.pvalue}
    }
    
    # 保存KS测试结果
    with open(os.path.join(viz_dir, f'{dataset_name}_ks_test_results.json'), 'w') as f:
        json.dump(ks_results, f, indent=2)
    
    print(f"\n数据分割均匀性验证:")
    print(f"KS测试结果 (p值 > 0.05 表示分布相似):")
    for comparison, result in ks_results.items():
        print(f"  {comparison}: statistic={result['statistic']:.4f}, p_value={result['p_value']:.4f}")

def process_data(
    input_file, 
    output_dir, 
    target_property, 
    formation_energy_col='formation_energy_per_atom', 
    train_size=None,
    train_ratio=0.8, 
    val_ratio=0.1,
    test_ratio=0.1,
    stratify_bins=10,
    random_state=42,
    analyze_distribution=True,
    validate_split=True
):
    """
    处理材料数据，转换为CDVAE多目标(形成能&目标属性)格式，保留cif列
    支持固定训练集大小或按比例分割
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # 检查必要的列是否存在
    required_columns = ['material_id', formation_energy_col, target_property]
    if 'cif' not in df.columns:
        raise ValueError("输入数据中缺少 'cif' 列，无法继续。")

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"必要的列 '{col}' 不存在于输入数据中")
    
    # 将形成能列重命名为标准名称
    if formation_energy_col != 'formation_energy_per_atom':
        df = df.rename(columns={formation_energy_col: 'formation_energy_per_atom'})
    
    # 过滤掉包含NaN的行
    df = df.dropna(subset=['formation_energy_per_atom', target_property, 'cif'])
    print(f"数据集大小(过滤NaN后): {len(df)}")
    
    # 数据分布分析
    if analyze_distribution:
        print("正在分析数据分布...")
        dataset_name = os.path.basename(output_dir)
        analyze_data_distribution(df, target_property, output_dir, dataset_name)
    
    # 数据分割
    if train_size is not None:
        # 使用固定训练集大小
        print(f"使用固定训练集大小: {train_size}")
        train_df, val_df, test_df = balanced_split_with_fixed_train_size(
            df, target_property, train_size, val_ratio, test_ratio, stratify_bins, random_state
        )
    else:
        # 使用比例分割
        print(f"使用比例分割: train={train_ratio}, val={val_ratio}, test={test_ratio}")
        
        # 确保比例总和为1
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            print(f"警告: 比例总和为 {total_ratio}，将进行归一化")
            train_ratio /= total_ratio
            val_ratio /= total_ratio
            test_ratio /= total_ratio
        
        # 计算绝对大小
        n = len(df)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        test_size = n - train_size - val_size
        
        train_df, val_df, test_df = balanced_split_with_fixed_train_size(
            df, target_property, train_size, val_size/n, test_size/n, stratify_bins, random_state
        )
    
    print(f"最终数据分割:")
    print(f"  训练集大小: {len(train_df)}")
    print(f"  验证集大小: {len(val_df)}")
    print(f"  测试集大小: {len(test_df)}")
    
    # 验证分割均匀性
    if validate_split:
        print("正在验证数据分割均匀性...")
        dataset_name = os.path.basename(output_dir)
        validate_split_uniformity(train_df, val_df, test_df, target_property, output_dir, dataset_name)
    
    # 创建CDVAE格式的输出数据
    for name, subset in [('train', train_df), ('val', val_df), ('test', test_df)]:
        # 选取并调整列顺序：material_id, cif, formation_energy_per_atom, target_property
        output_data = subset[['material_id', 'cif', 'formation_energy_per_atom', target_property]].copy()
        
        # 保存到CSV
        output_path = os.path.join(output_dir, f"{name}.csv")
        output_data.to_csv(output_path, index=False)
        print(f"已保存{name}集到 {output_path}")
    
    # 创建增强的元数据文件
    metadata = {
        "dataset_name": os.path.basename(output_dir),
        "target_property": target_property,
        "preprocessing_params": {
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "stratify_bins": stratify_bins,
            "random_state": random_state
        },
        "formation_energy_stats": {
            "overall": {
                "mean": float(df['formation_energy_per_atom'].mean()),
                "std": float(df['formation_energy_per_atom'].std()),
                "min": float(df['formation_energy_per_atom'].min()),
                "max": float(df['formation_energy_per_atom'].max()),
            },
            "train": {
                "mean": float(train_df['formation_energy_per_atom'].mean()),
                "std": float(train_df['formation_energy_per_atom'].std()),
                "min": float(train_df['formation_energy_per_atom'].min()),
                "max": float(train_df['formation_energy_per_atom'].max()),
            },
            "val": {
                "mean": float(val_df['formation_energy_per_atom'].mean()),
                "std": float(val_df['formation_energy_per_atom'].std()),
                "min": float(val_df['formation_energy_per_atom'].min()),
                "max": float(val_df['formation_energy_per_atom'].max()),
            },
            "test": {
                "mean": float(test_df['formation_energy_per_atom'].mean()),
                "std": float(test_df['formation_energy_per_atom'].std()),
                "min": float(test_df['formation_energy_per_atom'].min()),
                "max": float(test_df['formation_energy_per_atom'].max()),
            }
        },
        "target_property_stats": {
            "overall": {
                "mean": float(df[target_property].mean()),
                "std": float(df[target_property].std()),
                "min": float(df[target_property].min()),
                "max": float(df[target_property].max()),
            },
            "train": {
                "mean": float(train_df[target_property].mean()),
                "std": float(train_df[target_property].std()),
                "min": float(train_df[target_property].min()),
                "max": float(train_df[target_property].max()),
            },
            "val": {
                "mean": float(val_df[target_property].mean()),
                "std": float(val_df[target_property].std()),
                "min": float(val_df[target_property].min()),
                "max": float(val_df[target_property].max()),
            },
            "test": {
                "mean": float(test_df[target_property].mean()),
                "std": float(test_df[target_property].std()),
                "min": float(test_df[target_property].min()),
                "max": float(test_df[target_property].max()),
            }
        }
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("处理完成！")
    
    # 输出能量分位点信息
    percentiles = np.percentile(df['formation_energy_per_atom'], [5, 10, 15])
    print("\n形成能分位点 (用于compute_metrics.py EnergyPercentiles配置):")
    print(f"{os.path.basename(output_dir)}: np.array([{percentiles[0]:.6f}, {percentiles[1]:.6f}, {percentiles[2]:.6f}]),")
    
    # 输出目标属性分位点
    target_percentiles = np.percentile(df[target_property], [5, 10, 15])
    print(f"\n目标属性分位点 (用于compute_metrics.py Percentiles配置):")
    print(f"{os.path.basename(output_dir)}: np.array([{target_percentiles[0]:.6f}, {target_percentiles[1]:.6f}, {target_percentiles[2]:.6f}]),")
    
    return len(train_df), len(val_df), len(test_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理材料数据为CDVAE多目标格式（支持固定训练集大小）')
    
    # 基本参数
    parser.add_argument('--input', required=True, help='输入CSV文件路径')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--target_property', required=True, help='目标属性列名')
    parser.add_argument('--formation_energy_col', default='formation_energy_per_atom', 
                       help='形成能列名(将被重命名为formation_energy_per_atom)')
    
    # 数据分割参数
    parser.add_argument('--train_size', type=int, default=None,
                       help='训练集大小（绝对数量）。如果指定，将忽略train_ratio')
    parser.add_argument('--train_ratio', type=float, default=0.8, 
                       help='训练集比例（仅在未指定train_size时有效）')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='测试集比例')
    
    # 高级参数
    parser.add_argument('--stratify_bins', type=int, default=10,
                       help='分层采样的分箱数量（确保数据均匀性）')
    parser.add_argument('--random_state', type=int, default=42,
                       help='随机种子（用于结果复现）')
    
    # 分析参数
    parser.add_argument('--no_analysis', action='store_true',
                       help='跳过数据分布分析')
    parser.add_argument('--no_validation', action='store_true',
                       help='跳过数据分割验证')
    
    args = parser.parse_args()
    
    # 参数验证
    if args.train_size is None and abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        print("警告: 比例总和不等于1，将进行归一化")
    
    # 运行处理
    try:
        train_size, val_size, test_size = process_data(
            args.input,
            args.output_dir,
            args.target_property,
            args.formation_energy_col,
            args.train_size,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.stratify_bins,
            args.random_state,
            not args.no_analysis,
            not args.no_validation
        )
        
        print(f"\n✅ 数据预处理成功完成!")
        print(f"   训练集: {train_size} 样本")
        print(f"   验证集: {val_size} 样本") 
        print(f"   测试集: {test_size} 样本")
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        raise