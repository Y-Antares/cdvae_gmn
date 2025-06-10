#!/usr/bin/env python3
"""
配置插值调试工具
用于分析配置文件中的插值引用和依赖关系
"""

import os
import sys
import re
import yaml
from pathlib import Path
from typing import Dict, Set, List

# 设置项目路径
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(PROJECT_ROOT))

def find_interpolations(text: str) -> Set[str]:
    """查找文本中的所有插值引用"""
    if not isinstance(text, str):
        return set()
    
    # 匹配 ${variable} 和 ${path.to.variable} 格式
    pattern = r'\$\{([^}]+)\}'
    matches = re.findall(pattern, text)
    
    interpolations = set()
    for match in matches:
        # 只保留变量名的最后一部分
        if '.' in match:
            # ${data.property} -> property
            interpolations.add(match.split('.')[-1])
        else:
            # ${variable} -> variable
            interpolations.add(match)
    
    return interpolations

def scan_config_file(file_path: Path) -> Dict[str, Set[str]]:
    """扫描配置文件，找出所有插值引用"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 分析原始文本中的插值
        text_interpolations = find_interpolations(content)
        
        # 解析YAML并递归查找插值
        yaml_data = yaml.safe_load(content)
        yaml_interpolations = find_interpolations_in_data(yaml_data)
        
        return {
            'text_based': text_interpolations,
            'yaml_based': yaml_interpolations,
            'all': text_interpolations | yaml_interpolations
        }
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {'text_based': set(), 'yaml_based': set(), 'all': set()}

def find_interpolations_in_data(data) -> Set[str]:
    """递归查找数据结构中的插值引用"""
    interpolations = set()
    
    if isinstance(data, dict):
        for key, value in data.items():
            interpolations.update(find_interpolations_in_data(value))
    elif isinstance(data, list):
        for item in data:
            interpolations.update(find_interpolations_in_data(item))
    elif isinstance(data, str):
        interpolations.update(find_interpolations(data))
    
    return interpolations

def analyze_config_dependencies():
    """分析配置文件的依赖关系"""
    print("=== 配置插值依赖分析 ===\n")
    
    # 要分析的配置文件
    config_files = {
        'data/perov_1k': PROJECT_ROOT / "conf" / "data" / "perov_1k.yaml",
        'model/enhanced_cdvae': PROJECT_ROOT / "conf" / "model" / "enhanced_cdvae.yaml",
        'default': PROJECT_ROOT / "conf" / "default.yaml",
        'train/default': PROJECT_ROOT / "conf" / "train" / "default.yaml",
        'optim/default': PROJECT_ROOT / "conf" / "optim" / "default.yaml",
        'logging/default': PROJECT_ROOT / "conf" / "logging" / "default.yaml",
    }
    
    all_interpolations = set()
    all_defined_vars = set()
    
    # 分析每个配置文件
    for name, file_path in config_files.items():
        if not file_path.exists():
            print(f"❌ {name}: 文件不存在 ({file_path})")
            continue
            
        print(f"📁 分析 {name}:")
        print(f"   路径: {file_path}")
        
        # 读取并分析文件
        interpolations = scan_config_file(file_path)
        
        # 读取原始内容查看定义的变量
        with open(file_path, 'r') as f:
            content = f.read()
            yaml_data = yaml.safe_load(content)
        
        # 收集这个文件中定义的变量
        defined_vars = collect_defined_variables(yaml_data)
        
        print(f"   🔍 插值引用: {sorted(interpolations['all'])}")
        print(f"   📝 定义变量: {sorted(defined_vars)}")
        
        all_interpolations.update(interpolations['all'])
        all_defined_vars.update(defined_vars)
        print()
    
    # 分析缺失的变量
    missing_vars = all_interpolations - all_defined_vars
    
    print("=== 总结 ===")
    print(f"🔍 所有插值引用: {sorted(all_interpolations)}")
    print(f"📝 所有定义变量: {sorted(all_defined_vars)}")
    print(f"❌ 缺失变量: {sorted(missing_vars)}")
    
    if missing_vars:
        print(f"\n🚨 发现 {len(missing_vars)} 个缺失的插值变量:")
        for var in sorted(missing_vars):
            print(f"   - {var}")
            
        print(f"\n💡 建议解决方案:")
        print(f"   1. 在 perov_1k.yaml 中添加缺失变量的定义")
        print(f"   2. 或在训练脚本中添加这些变量到顶层配置")
        
        # 生成建议的配置
        print(f"\n📝 建议添加到 perov_1k.yaml:")
        for var in sorted(missing_vars):
            default_value = get_default_value(var)
            print(f"   {var}: {default_value}")
    else:
        print("✅ 所有插值引用都有对应的定义!")

def collect_defined_variables(data, prefix="") -> Set[str]:
    """收集数据结构中定义的所有变量名"""
    variables = set()
    
    if isinstance(data, dict):
        for key, value in data.items():
            if key.startswith('_'):  # 跳过特殊键如 _target_
                continue
                
            # 添加当前键
            variables.add(key)
            
            # 递归处理嵌套结构
            if isinstance(value, (dict, list)):
                nested_vars = collect_defined_variables(value, f"{prefix}{key}.")
                variables.update(nested_vars)
    
    return variables

def get_default_value(var_name: str):
    """为变量提供建议的默认值"""
    defaults = {
        'gradnorm_lr': '0.025',
        'scaler_type': 'minmax',
        'energy_scaler_type': 'minmax',
        'optimization_method': 'weighted',
        'use_gradnorm': 'true',
        'gradnorm_alpha': '1.5',
        'property_weights': '[0.6, 0.4]',
        'optimization_direction': '[min, max]',
        'boundary_theta': '5.0',
        'init_ideal_points': '[999.0, 999.0]',
        'multi_target': 'true',
        'energy_weight': '0.4',
    }
    return defaults.get(var_name, 'TODO')

def check_specific_interpolation(config_name: str, interpolation: str):
    """检查特定插值在哪些文件中被引用"""
    print(f"\n=== 查找插值 '{interpolation}' 的使用情况 ===")
    
    config_dirs = [
        PROJECT_ROOT / "conf" / "data",
        PROJECT_ROOT / "conf" / "model", 
        PROJECT_ROOT / "conf" / "train",
        PROJECT_ROOT / "conf" / "optim",
        PROJECT_ROOT / "conf" / "logging",
    ]
    
    found_in = []
    
    for config_dir in config_dirs:
        if not config_dir.exists():
            continue
            
        for yaml_file in config_dir.rglob("*.yaml"):
            try:
                with open(yaml_file, 'r') as f:
                    content = f.read()
                    
                if f"${{{interpolation}}}" in content or f"${{data.{interpolation}}}" in content:
                    found_in.append(yaml_file)
                    print(f"📁 发现在: {yaml_file.relative_to(PROJECT_ROOT)}")
                    
                    # 显示相关行
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if interpolation in line and ('${' in line):
                            print(f"   第{i}行: {line.strip()}")
                    
            except Exception as e:
                print(f"❌ 读取 {yaml_file} 时出错: {e}")
    
    if not found_in:
        print(f"❌ 没有找到对 '{interpolation}' 的引用")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 检查特定插值
        interpolation = sys.argv[1]
        check_specific_interpolation("", interpolation)
    else:
        # 完整分析
        analyze_config_dependencies()