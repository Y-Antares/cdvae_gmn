#!/usr/bin/env python3
"""
调试配置中的数据类型
"""

import sys
import os
from pathlib import Path
import yaml
from omegaconf import OmegaConf

# 设置项目路径
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(PROJECT_ROOT))

def check_data_types(config, path=""):
    """递归检查配置中的数据类型"""
    if isinstance(config, dict):
        for key, value in config.items():
            current_path = f"{path}.{key}" if path else key
            if isinstance(value, str) and value.isdigit():
                print(f"🔍 {current_path}: '{value}' (str, 应该是 int)")
            elif isinstance(value, str) and '.' in value and value.replace('.', '').isdigit():
                print(f"🔍 {current_path}: '{value}' (str, 应该是 float)")
            elif isinstance(value, (dict, list)):
                check_data_types(value, current_path)
    elif isinstance(config, list):
        for i, item in enumerate(config):
            check_data_types(item, f"{path}[{i}]")

def main():
    # 加载 perov_1k 数据配置
    data_config_path = PROJECT_ROOT / "conf/data/perov_1k.yaml"
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print("=== 检查数据配置中的类型问题 ===")
    check_data_types(data_config, "data")
    
    print("\n=== 关键参数检查 ===")
    key_params = [
        'preprocess_workers', 'num_targets', 'max_atoms',
        'train_max_epochs', 'early_stopping_patience'
    ]
    
    for param in key_params:
        if param in data_config:
            value = data_config[param]
            print(f"{param}: {value} ({type(value).__name__})")
    
    print("\n=== datamodule 配置检查 ===")
    if 'datamodule' in data_config:
        check_data_types(data_config['datamodule'], "datamodule")
    
    print("\n=== datasets 配置详细检查 ===")
    if 'datamodule' in data_config and 'datasets' in data_config['datamodule']:
        datasets = data_config['datamodule']['datasets']
        for dataset_name, dataset_config in datasets.items():
            print(f"\n{dataset_name} 数据集:")
            if isinstance(dataset_config, list):
                for i, ds_cfg in enumerate(dataset_config):
                    print(f"  [{i}] preprocess_workers: {ds_cfg.get('preprocess_workers', 'NOT_FOUND')} ({type(ds_cfg.get('preprocess_workers', None)).__name__})")
            else:
                print(f"  preprocess_workers: {dataset_config.get('preprocess_workers', 'NOT_FOUND')} ({type(dataset_config.get('preprocess_workers', None)).__name__})")
                check_data_types(dataset_config, f"datasets.{dataset_name}")

if __name__ == "__main__":
    main()