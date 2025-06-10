import pandas as pd 
import numpy as np
import os
import json
import argparse
from tqdm import tqdm

def process_data(input_file, output_dir, target_property, formation_energy_col='formation_energy_per_atom', train_ratio=0.8, val_ratio=0.1):
    """
    处理材料数据，转换为CDVAE多目标(形成能&目标属性)格式，保留cif列
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
    
    # 拆分数据集
    n = len(df)
    indices = np.random.permutation(n)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_df = df.iloc[train_indices]
    val_df = df.iloc[val_indices]
    test_df = df.iloc[test_indices]
    
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"测试集大小: {len(test_df)}")
    
    # 创建CDVAE格式的输出数据
    for name, subset in [('train', train_df), ('val', val_df), ('test', test_df)]:
        # 选取并调整列顺序：material_id, cif, formation_energy_per_atom, target_property
        output_data = subset[['material_id', 'cif', 'formation_energy_per_atom', target_property]].copy()
        
        # 保存到CSV
        output_path = os.path.join(output_dir, f"{name}.csv")
        output_data.to_csv(output_path, index=False)
        print(f"已保存{name}集到 {output_path}")
    
    # 创建元数据文件
    metadata = {
        "dataset_name": os.path.basename(output_dir),
        "target_property": target_property,
        "formation_energy_stats": {
            "mean": float(df['formation_energy_per_atom'].mean()),
            "std": float(df['formation_energy_per_atom'].std()),
            "min": float(df['formation_energy_per_atom'].min()),
            "max": float(df['formation_energy_per_atom'].max()),
        },
        "target_property_stats": {
            "mean": float(df[target_property].mean()),
            "std": float(df[target_property].std()),
            "min": float(df[target_property].min()),
            "max": float(df[target_property].max()),
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理材料数据为CDVAE多目标格式')
    parser.add_argument('--input', required=True, help='输入CSV文件路径')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--target_property', required=True, help='目标属性列名')
    parser.add_argument('--formation_energy_col', default='formation_energy_per_atom', 
                       help='形成能列名(将被重命名为formation_energy_per_atom)')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    
    args = parser.parse_args()
    
    process_data(
        args.input,
        args.output_dir,
        args.target_property,
        args.formation_energy_col,
        args.train_ratio,
        args.val_ratio
    )