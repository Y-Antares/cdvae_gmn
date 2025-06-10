#!/usr/bin/env python
# 文件名: fix_matminer_yaml.py

import os
import sys
import importlib.util

def find_module_path(module_name):
    """查找模块的文件路径"""
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    return spec.origin

def patch_matminer_yaml():
    """修补matminer中的YAML加载问题"""
    # 查找fingerprint.py文件路径
    fingerprint_path = find_module_path("matminer.featurizers.site.fingerprint")
    if not fingerprint_path:
        print("无法找到matminer的fingerprint模块")
        return False
    
    print(f"正在修补文件: {fingerprint_path}")
    
    # 读取文件内容
    with open(fingerprint_path, 'r') as f:
        content = f.read()
    
    # 修补import语句
    if "from ruamel.yaml import YAML" not in content:
        if "import yaml" in content:
            # 替换导入语句
            content = content.replace(
                "import yaml", 
                "import yaml\nfrom ruamel.yaml import YAML"
            )
        else:
            # 添加导入语句
            content = "from ruamel.yaml import YAML\n" + content
    
    # 修补yaml.safe_load调用
    if "yaml.safe_load(f)" in content:
        content = content.replace(
            "yaml.safe_load(f)", 
            "YAML(typ='safe', pure=True).load(f)"
        )
    
    # 写回文件
    with open(fingerprint_path, 'w') as f:
        f.write(content)
    
    print("成功修补matminer库中的YAML问题")
    return True

if __name__ == "__main__":
    if patch_matminer_yaml():
        print("修补成功！请重新运行您的脚本。")
    else:
        print("修补失败，请手动修改代码。")