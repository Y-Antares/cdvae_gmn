import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
file_path = "D:\github\CDVAE\cdvae\data\perov_5/train.csv"
df = pd.read_csv(file_path)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.size'] = 24  # 设置全局默认字体大小

# 检查"heat_all"列是否存在
if 'heat_all' not in df.columns:
    print(f"注意: 'heat_all'列不存在于数据集中。可用的列有: {df.columns.tolist()}")
else:
    # 基本统计分析
    heat_stats = df['heat_all'].describe()
    print("heat_all 统计摘要:")
    print(heat_stats)
    
    # 绘制直方图
    plt.figure(figsize=(16, 10))
    n, bins, patches = plt.hist(df['heat_all'], bins=30, color='skyblue', alpha=0.7, edgecolor='black')
    
    # 添加平均线和中位数线
    plt.axvline(heat_stats['mean'], color='red', linestyle='dashed', linewidth=2, label=f"平均值: {heat_stats['mean']:.2f}")
    plt.axvline(heat_stats['50%'], color='green', linestyle='dashed', linewidth=2, label=f"中位数: {heat_stats['50%']:.2f}")
    
    # 添加标题和标签
    plt.title("heat_all 数值分布", fontsize=30)
    plt.xlabel("heat_all 值", fontsize=24)
    plt.ylabel("频率", fontsize=24)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=24)
    
    # 设置轴刻度字体大小
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig("heat_all_distribution.png", dpi=300, bbox_inches='tight')
    
    # 创建箱形图
    plt.figure(figsize=(12, 8))
    boxplot = plt.boxplot(df['heat_all'], patch_artist=True)
    
    # 设置箱形图颜色
    for patch in boxplot['boxes']:
        patch.set_facecolor('lightgreen')
        
    # 添加标题和标签
    plt.title("heat_all 箱形图", fontsize=24)
    plt.ylabel("heat_all 值", fontsize=20)
    plt.grid(True, axis='y', alpha=0.3)
    
    # 设置轴刻度字体大小
    plt.xticks([])  # 移除x轴刻度
    plt.yticks(fontsize=24)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig("heat_all_boxplot.png", dpi=300, bbox_inches='tight')
    
    # 探索可能的离群值
    q1 = heat_stats['25%']
    q3 = heat_stats['75%']
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = df[(df['heat_all'] < lower_bound) | (df['heat_all'] > upper_bound)]
    print(f"\n离群值数量: {len(outliers)}")
    print(f"离群值百分比: {(len(outliers) / len(df)) * 100:.2f}%")
    
    # 如果离群值很多，可以绘制去除离群值后的分布
    if len(outliers) > 0:
        plt.figure(figsize=(16, 10))
        filtered_data = df[(df['heat_all'] >= lower_bound) & (df['heat_all'] <= upper_bound)]
        n, bins, patches = plt.hist(filtered_data['heat_all'], bins=30, color='lightblue', alpha=0.7, edgecolor='black')
        
        # 添加标题和标签
        plt.title("heat_all 数值分布 (无离群值)", fontsize=24)
        plt.xlabel("heat_all 值", fontsize=20)
        plt.ylabel("频率", fontsize=20)
        plt.grid(True, alpha=0.3)
        
        # 设置轴刻度字体大小
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig("heat_all_distribution_no_outliers.png", dpi=300, bbox_inches='tight')

    # 计算并显示最大和最小值的位置
    min_idx = df['heat_all'].idxmin()
    max_idx = df['heat_all'].idxmax()
    
    print(f"\n最小 heat_all 值: {df['heat_all'].min():.4f} (索引: {min_idx})")
    print(f"最大 heat_all 值: {df['heat_all'].max():.4f} (索引: {max_idx})")
    
    # 绘制散点图，按索引展示heat_all的分布
    plt.figure(figsize=(16, 8))
    plt.scatter(range(len(df)), df['heat_all'], alpha=0.5, s=10, c='blue')
    
    # 高亮最大和最小值
    plt.scatter([min_idx], [df['heat_all'].min()], color='red', s=100, label=f'最小值: {df["heat_all"].min():.4f}')
    plt.scatter([max_idx], [df['heat_all'].max()], color='green', s=100, label=f'最大值: {df["heat_all"].max():.4f}')
    
    # 添加标题和标签
    plt.title("heat_all 散点分布", fontsize=24)
    plt.xlabel("数据点索引", fontsize=20)
    plt.ylabel("heat_all 值", fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=18)
    
    # 设置轴刻度字体大小
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig("heat_all_scatter.png", dpi=300, bbox_inches='tight')

print("分析完成。结果已保存为图表图像。")