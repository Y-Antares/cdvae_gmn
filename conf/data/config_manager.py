#!/usr/bin/env python
"""
数据预处理配置管理器
简化数据预处理配置的使用和管理
"""

import os
import sys
import yaml
import argparse
import subprocess
from pathlib import Path

# 预定义配置
PREDEFINED_CONFIGS = {
    'small_test': {
        'description': '小规模快速测试配置',
        'train_size': 1000,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'stratify_bins': 5,
        'random_state': 42,
        'extra_args': ['--no_analysis', '--no_validation']
    },
    
    'medium_dev': {
        'description': '中等规模开发配置',
        'train_size': 5000,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'stratify_bins': 10,
        'random_state': 42,
        'extra_args': []
    },
    
    'large_production': {
        'description': '大规模生产配置',
        'train_size': 20000,
        'val_ratio': 0.08,
        'test_ratio': 0.12,
        'stratify_bins': 20,
        'random_state': 42,
        'extra_args': []
    },
    
    'perovskite_bandgap': {
        'description': '钙钛矿带隙优化专用配置',
        'train_size': 12000,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'stratify_bins': 15,
        'random_state': 42,
        'target_property': 'dir_gap',
        'formation_energy_col': 'formation_energy_per_atom',
        'extra_args': []
    },
    
    'carbon_elastic': {
        'description': '碳材料弹性模量预测配置',
        'train_size': 8000,
        'val_ratio': 0.12,
        'test_ratio': 0.13,
        'stratify_bins': 12,
        'random_state': 123,
        'target_property': 'bulk_modulus',
        'formation_energy_col': 'formation_energy_per_atom',
        'extra_args': []
    },
    
    'alloy_thermal': {
        'description': '金属合金热导率研究配置',
        'train_size': 6000,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'stratify_bins': 10,
        'random_state': 456,
        'target_property': 'thermal_conductivity',
        'formation_energy_col': 'formation_enthalpy',
        'extra_args': []
    },
    
    'traditional_split': {
        'description': '传统80/10/10比例分割',
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'stratify_bins': 10,
        'random_state': 42,
        'extra_args': []
    },
    
    'data_scarce': {
        'description': '数据稀缺情况配置（更多训练数据）',
        'train_ratio': 0.9,
        'val_ratio': 0.05,
        'test_ratio': 0.05,
        'stratify_bins': 8,
        'random_state': 42,
        'extra_args': []
    }
}

class DataPrepConfigManager:
    """数据预处理配置管理器"""
    
    def __init__(self, script_path='enhanced_prepare_multi_target_data.py'):
        """
        初始化配置管理器
        
        Args:
            script_path: 数据预处理脚本的路径
        """
        self.script_path = script_path
        self.configs = PREDEFINED_CONFIGS.copy()
        
    def list_configs(self):
        """列出所有可用的配置"""
        print("可用的预定义配置:")
        print("=" * 50)
        
        for name, config in self.configs.items():
            print(f"\n{name}:")
            print(f"  描述: {config['description']}")
            
            if 'train_size' in config:
                print(f"  训练集大小: {config['train_size']}")
            if 'train_ratio' in config:
                print(f"  训练集比例: {config['train_ratio']}")
                
            print(f"  验证集比例: {config['val_ratio']}")
            print(f"  测试集比例: {config['test_ratio']}")
            print(f"  分层数量: {config['stratify_bins']}")
            print(f"  随机种子: {config['random_state']}")
            
            if 'target_property' in config:
                print(f"  目标属性: {config['target_property']}")
            if 'formation_energy_col' in config:
                print(f"  形成能列: {config['formation_energy_col']}")
    
    def show_config(self, config_name):
        """显示特定配置的详细信息"""
        if config_name not in self.configs:
            print(f"错误: 配置 '{config_name}' 不存在")
            return
        
        config = self.configs[config_name]
        print(f"配置 '{config_name}' 详细信息:")
        print("=" * 40)
        print(f"描述: {config['description']}")
        print()
        
        # 显示所有参数
        for key, value in config.items():
            if key not in ['description', 'extra_args']:
                print(f"{key}: {value}")
        
        if config.get('extra_args'):
            print(f"额外参数: {' '.join(config['extra_args'])}")
    
    def generate_command(self, config_name, input_file, output_dir, **overrides):
        """
        生成数据预处理命令
        
        Args:
            config_name: 配置名称
            input_file: 输入文件路径
            output_dir: 输出目录
            **overrides: 覆盖配置参数
            
        Returns:
            生成的命令字符串
        """
        if config_name not in self.configs:
            raise ValueError(f"配置 '{config_name}' 不存在")
        
        config = self.configs[config_name].copy()
        
        # 应用覆盖参数
        config.update(overrides)
        
        # 构建命令
        cmd_parts = ['python', self.script_path]
        cmd_parts.extend(['--input', str(input_file)])
        cmd_parts.extend(['--output_dir', str(output_dir)])
        
        # 添加配置参数
        for key, value in config.items():
            if key in ['description', 'extra_args']:
                continue
                
            param_name = f"--{key}"
            cmd_parts.extend([param_name, str(value)])
        
        # 添加额外参数
        if 'extra_args' in config:
            cmd_parts.extend(config['extra_args'])
        
        return cmd_parts
    
    def run_config(self, config_name, input_file, output_dir, dry_run=False, **overrides):
        """
        运行指定配置的数据预处理
        
        Args:
            config_name: 配置名称
            input_file: 输入文件路径
            output_dir: 输出目录
            dry_run: 是否只打印命令而不执行
            **overrides: 覆盖配置参数
            
        Returns:
            如果不是dry_run，返回subprocess的结果
        """
        try:
            cmd = self.generate_command(config_name, input_file, output_dir, **overrides)
            
            print(f"使用配置 '{config_name}' 处理数据...")
            print(f"命令: {' '.join(cmd)}")
            
            if dry_run:
                print("(干运行模式 - 未实际执行)")
                return None
            
            # 执行命令
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ 数据预处理成功完成!")
                print(result.stdout)
            else:
                print("❌ 数据预处理失败!")
                print("错误输出:", result.stderr)
            
            return result
            
        except Exception as e:
            print(f"❌ 执行过程中出现错误: {e}")
            return None
    
    def create_custom_config(self, name, **params):
        """
        创建自定义配置
        
        Args:
            name: 配置名称
            **params: 配置参数
        """
        if 'description' not in params:
            params['description'] = f"自定义配置: {name}"
        
        self.configs[name] = params
        print(f"✅ 自定义配置 '{name}' 已创建")
    
    def save_config_to_file(self, filename):
        """将当前配置保存到YAML文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            yaml.dump(self.configs, f, default_flow_style=False, allow_unicode=True)
        print(f"✅ 配置已保存到 {filename}")
    
    def load_config_from_file(self, filename):
        """从YAML文件加载配置"""
        with open(filename, 'r', encoding='utf-8') as f:
            loaded_configs = yaml.safe_load(f)
        
        self.configs.update(loaded_configs)
        print(f"✅ 配置已从 {filename} 加载")
    
    def batch_process(self, config_name, input_files, output_base_dir, **overrides):
        """
        批量处理多个输入文件
        
        Args:
            config_name: 配置名称
            input_files: 输入文件列表
            output_base_dir: 输出基础目录
            **overrides: 覆盖参数
        """
        print(f"开始批量处理 {len(input_files)} 个文件...")
        
        results = []
        for i, input_file in enumerate(input_files):
            input_path = Path(input_file)
            output_dir = Path(output_base_dir) / input_path.stem
            
            print(f"\n处理 {i+1}/{len(input_files)}: {input_path.name}")
            
            result = self.run_config(
                config_name, 
                input_file, 
                output_dir, 
                **overrides
            )
            
            results.append({
                'input_file': input_file,
                'output_dir': str(output_dir),
                'success': result.returncode == 0 if result else False
            })
        
        # 输出批量处理结果总结
        print(f"\n批量处理完成:")
        print("=" * 40)
        successful = sum(1 for r in results if r['success'])
        print(f"成功: {successful}/{len(results)}")
        
        for result in results:
            status = "✅" if result['success'] else "❌"
            print(f"{status} {Path(result['input_file']).name} -> {result['output_dir']}")
        
        return results


def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(description='数据预处理配置管理器')
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 列出配置命令
    list_parser = subparsers.add_parser('list', help='列出所有可用配置')
    
    # 显示配置命令
    show_parser = subparsers.add_parser('show', help='显示特定配置详情')
    show_parser.add_argument('config_name', help='配置名称')
    
    # 运行配置命令
    run_parser = subparsers.add_parser('run', help='运行指定配置')
    run_parser.add_argument('config_name', help='配置名称')
    run_parser.add_argument('--input', required=True, help='输入文件路径')
    run_parser.add_argument('--output_dir', required=True, help='输出目录')
    run_parser.add_argument('--dry_run', action='store_true', help='只显示命令，不执行')
    run_parser.add_argument('--train_size', type=int, help='覆盖训练集大小')
    run_parser.add_argument('--target_property', help='覆盖目标属性')
    run_parser.add_argument('--formation_energy_col', help='覆盖形成能列名')
    run_parser.add_argument('--random_state', type=int, help='覆盖随机种子')
    
    # 生成命令
    gen_parser = subparsers.add_parser('generate', help='生成命令字符串')
    gen_parser.add_argument('config_name', help='配置名称')
    gen_parser.add_argument('--input', required=True, help='输入文件路径')
    gen_parser.add_argument('--output_dir', required=True, help='输出目录')
    gen_parser.add_argument('--train_size', type=int, help='覆盖训练集大小')
    gen_parser.add_argument('--target_property', help='覆盖目标属性')
    
    # 批量处理命令
    batch_parser = subparsers.add_parser('batch', help='批量处理多个文件')
    batch_parser.add_argument('config_name', help='配置名称')
    batch_parser.add_argument('--input_files', nargs='+', required=True, help='输入文件列表')
    batch_parser.add_argument('--output_base_dir', required=True, help='输出基础目录')
    batch_parser.add_argument('--train_size', type=int, help='覆盖训练集大小')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 创建配置管理器
    manager = DataPrepConfigManager()
    
    # 执行对应命令
    if args.command == 'list':
        manager.list_configs()
        
    elif args.command == 'show':
        manager.show_config(args.config_name)
        
    elif args.command == 'run':
        overrides = {}
        if args.train_size is not None:
            overrides['train_size'] = args.train_size
        if args.target_property:
            overrides['target_property'] = args.target_property
        if args.formation_energy_col:
            overrides['formation_energy_col'] = args.formation_energy_col
        if args.random_state is not None:
            overrides['random_state'] = args.random_state
            
        manager.run_config(
            args.config_name, 
            args.input, 
            args.output_dir, 
            dry_run=args.dry_run,
            **overrides
        )
        
    elif args.command == 'generate':
        overrides = {}
        if args.train_size is not None:
            overrides['train_size'] = args.train_size
        if args.target_property:
            overrides['target_property'] = args.target_property
            
        cmd = manager.generate_command(
            args.config_name, 
            args.input, 
            args.output_dir, 
            **overrides
        )
        print(' '.join(cmd))
        
    elif args.command == 'batch':
        overrides = {}
        if args.train_size is not None:
            overrides['train_size'] = args.train_size
            
        manager.batch_process(
            args.config_name,
            args.input_files,
            args.output_base_dir,
            **overrides
        )


if __name__ == '__main__':
    main()

# 使用示例
"""
# 列出所有配置
python config_manager.py list

# 显示特定配置
python config_manager.py show perovskite_bandgap

# 运行配置（干运行）
python config_manager.py run perovskite_bandgap --input data.csv --output_dir output --dry_run

# 运行配置并覆盖训练集大小
python config_manager.py run perovskite_bandgap --input data.csv --output_dir output --train_size 8000

# 生成命令字符串
python config_manager.py generate medium_dev --input data.csv --output_dir output

# 批量处理
python config_manager.py batch small_test --input_files data1.csv data2.csv data3.csv --output_base_dir outputs
"""