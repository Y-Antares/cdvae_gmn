#!/usr/bin/env python3
"""
生成项目目录的树状结构，只显示 Python 和 YAML 文件
"""

import os
import sys
from pathlib import Path
from typing import List, Set

def should_include_file(file_path: Path) -> bool:
    """判断是否应该包含这个文件"""
    # 只包含 .py 和 .yaml/.yml 文件
    extensions = {'.py', '.yaml', '.yml'}
    return file_path.suffix.lower() in extensions

def should_include_dir(dir_path: Path) -> bool:
    """判断是否应该包含这个目录"""
    # 排除的目录
    exclude_dirs = {
        '__pycache__', '.git', '.vscode', '.idea', 
        'node_modules', '.pytest_cache', '.mypy_cache',
        'dist', 'build', 'egg-info', '.tox', 'venv', 'env'
    }
    
    return dir_path.name not in exclude_dirs and not dir_path.name.startswith('.')

def generate_tree(root_path: Path, prefix: str = "", max_depth: int = 10, current_depth: int = 0) -> List[str]:
    """生成目录树"""
    if current_depth > max_depth:
        return []
    
    items = []
    
    try:
        # 获取目录下的所有项目
        all_items = list(root_path.iterdir())
        
        # 分离文件和目录
        dirs = [item for item in all_items if item.is_dir() and should_include_dir(item)]
        files = [item for item in all_items if item.is_file() and should_include_file(item)]
        
        # 排序
        dirs.sort(key=lambda x: x.name.lower())
        files.sort(key=lambda x: x.name.lower())
        
        all_valid_items = dirs + files
        
        for i, item in enumerate(all_valid_items):
            is_last = i == len(all_valid_items) - 1
            
            # 选择前缀符号
            if is_last:
                current_prefix = "└── "
                next_prefix = prefix + "    "
            else:
                current_prefix = "├── "
                next_prefix = prefix + "│   "
            
            # 添加当前项目
            if item.is_file():
                # 文件，显示大小
                size = item.stat().st_size
                size_str = format_size(size)
                items.append(f"{prefix}{current_prefix}{item.name} ({size_str})")
            else:
                # 目录
                items.append(f"{prefix}{current_prefix}{item.name}/")
                
                # 递归处理子目录
                if current_depth < max_depth:
                    sub_items = generate_tree(item, next_prefix, max_depth, current_depth + 1)
                    items.extend(sub_items)
    
    except PermissionError:
        items.append(f"{prefix}[Permission Denied]")
    
    return items

def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes == 0:
        return "0B"
    
    units = ['B', 'KB', 'MB', 'GB']
    unit_index = 0
    size = float(size_bytes)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    if unit_index == 0:
        return f"{int(size)}B"
    else:
        return f"{size:.1f}{units[unit_index]}"

def count_files(root_path: Path) -> dict:
    """统计文件数量"""
    counts = {'python': 0, 'yaml': 0, 'dirs': 0}
    
    for item in root_path.rglob('*'):
        if item.is_file() and should_include_file(item):
            if item.suffix == '.py':
                counts['python'] += 1
            elif item.suffix in ['.yaml', '.yml']:
                counts['yaml'] += 1
        elif item.is_dir() and should_include_dir(item):
            counts['dirs'] += 1
    
    return counts

def main():
    """主函数"""
    # 解析命令行参数
    if len(sys.argv) > 1:
        root_path = Path(sys.argv[1])
    else:
        root_path = Path('.')
    
    # 获取可选参数
    max_depth = 10
    if len(sys.argv) > 2:
        try:
            max_depth = int(sys.argv[2])
        except ValueError:
            print("Warning: Invalid depth, using default (10)")
    
    # 检查路径是否存在
    if not root_path.exists():
        print(f"Error: Path '{root_path}' does not exist")
        return
    
    if not root_path.is_dir():
        print(f"Error: Path '{root_path}' is not a directory")
        return
    
    # 生成树状结构
    print(f"📁 Project Structure: {root_path.absolute()}")
    print(f"📊 Showing Python (.py) and YAML (.yaml/.yml) files (max depth: {max_depth})")
    print("=" * 60)
    
    tree_lines = generate_tree(root_path, max_depth=max_depth)
    
    if not tree_lines:
        print("No Python or YAML files found.")
        return
    
    # 显示根目录
    print(f"{root_path.name}/")
    
    # 显示树状结构
    for line in tree_lines:
        print(line)
    
    # 显示统计信息
    print("\n" + "=" * 60)
    counts = count_files(root_path)
    print(f"📈 Statistics:")
    print(f"   🐍 Python files: {counts['python']}")
    print(f"   📄 YAML files: {counts['yaml']}")
    print(f"   📁 Directories: {counts['dirs']}")
    print(f"   📝 Total files: {counts['python'] + counts['yaml']}")

def generate_compact_tree(root_path: Path, patterns: List[str] = None) -> str:
    """生成紧凑的树状结构，用于复制粘贴"""
    if patterns is None:
        patterns = ['*.py', '*.yaml', '*.yml']
    
    lines = [f"{root_path.name}/"]
    
    def add_files_recursively(path: Path, prefix: str = "", depth: int = 0):
        if depth > 8:  # 限制深度
            return
        
        try:
            items = []
            
            # 收集文件和目录
            for item in path.iterdir():
                if item.is_file() and should_include_file(item):
                    items.append(('file', item))
                elif item.is_dir() and should_include_dir(item):
                    # 检查目录下是否有相关文件
                    has_relevant_files = any(
                        child.is_file() and should_include_file(child)
                        for child in item.rglob('*')
                    )
                    if has_relevant_files:
                        items.append(('dir', item))
            
            # 排序
            items.sort(key=lambda x: (x[0] == 'file', x[1].name.lower()))
            
            for i, (item_type, item) in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                next_prefix = prefix + ("    " if is_last else "│   ")
                
                if item_type == 'file':
                    lines.append(f"{prefix}{current_prefix}{item.name}")
                else:
                    lines.append(f"{prefix}{current_prefix}{item.name}/")
                    add_files_recursively(item, next_prefix, depth + 1)
        
        except PermissionError:
            lines.append(f"{prefix}[Permission Denied]")
    
    add_files_recursively(root_path)
    return '\n'.join(lines)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--compact':
        # 紧凑模式
        root_path = Path(sys.argv[2] if len(sys.argv) > 2 else '.')
        print(generate_compact_tree(root_path))
    else:
        # 详细模式
        main()