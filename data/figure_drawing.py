import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter

# 加载数据
df = pd.read_csv('D:\github\CDVAE\cdvae\data\mp_data_Ag2.csv')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.size'] = 24  # 设置更大的全局默认字体大小

# 函数：创建条形图
def create_bar_chart(data_counter, title, filename, figsize=(16, 12), color='skyblue'):
    # 排序数据（按值降序）
    sorted_data = sorted(data_counter.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_data)
    
    # 计算百分比
    total = sum(values)
    percentages = [v/total*100 for v in values]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(labels)), percentages, color=color, width=0.6)
    
    # 添加标签和标题
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=24)  # 增大x轴标签字体
    ax.set_ylabel('百分比 (%)', fontsize=24)  # 增大y轴标签字体
    ax.set_title(title, fontsize=36, pad=30)  # 增大标题字体
    
    # 设置轴刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=18)  # 增大刻度标签字体
    
    # 在条形图上添加百分比标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{percentages[i]:.1f}%\n({values[i]})',
                ha='center', va='bottom', fontsize=18, fontweight='bold')  # 增大并加粗条形图上的数字
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# 1. 分析晶体系统
crystal_systems = []
for sys in df['symmetry']:
    match = re.search(r"crystal_system=<CrystalSystem\.([^:]+): '([^']+)'", str(sys))
    if match:
        crystal_systems.append(match.group(2))
    else:
        crystal_systems.append('Unknown')

crystal_systems_counter = Counter(crystal_systems)
create_bar_chart(crystal_systems_counter, '晶系分布', 'crystal_systems.png')

# 2. 分析 total_magnetization
# 先处理 total_magnetization 字段，将其分类
def categorize_magnetization(mag_value):
    if pd.isna(mag_value):
        return 'Unknown'
    
    # 将字符串转换为浮点数
    try:
        mag_value = float(mag_value)
    except (ValueError, TypeError):
        return 'Unknown'
    
    # 对磁化强度进行分类
    if mag_value == 0:
        return '无磁性 (0)'
    elif 0 < abs(mag_value) < 0.1:
        return '弱磁性 (<0.1)'
    elif 0.1 <= abs(mag_value) < 1:
        return '中等磁性 (0.1-1)'
    elif 1 <= abs(mag_value) < 5:
        return '强磁性 (1-5)'
    else:
        return '超强磁性 (>5)'

# 应用分类函数
magnetization_categories = df['total_magnetization'].apply(categorize_magnetization)
magnetization_counter = Counter(magnetization_categories)

create_bar_chart(magnetization_counter, '总磁化强度分布', 'total_magnetization.png', color='lightgreen')

print("分析完成。结果已保存为图表图像。")