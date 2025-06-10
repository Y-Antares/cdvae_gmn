import pandas as pd
import os

# 设置输入和输出路径
INPUT_CSV_PATH = "D:\github\CDVAE\cdvae\data\perov_bg/test.csv"  # 修改为你的实际输入文件路径
OUTPUT_CSV_PATH = "D:\github\CDVAE\cdvae\data\perov_bg/test1.csv"   # 修改为你想要的输出文件路径

def filter_zero_bandgap(input_csv_path=INPUT_CSV_PATH, output_csv_path=OUTPUT_CSV_PATH):
    """
    过滤CSV文件中band_gap=0的行，并保存为新的CSV文件
    
    参数:
    input_csv_path: 输入CSV文件的路径
    output_csv_path: 输出CSV文件的路径
    
    返回:
    output_csv_path: 保存的新CSV文件路径
    """
    print(f"读取CSV文件: {input_csv_path}")
    
    # 读取CSV文件
    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return None
    
    # 检查是否存在band_gap列
    band_gap_col = None
    for col in df.columns:
        if col.lower() in ['band_gap', 'bandgap', 'band gap', 'gap']:
            band_gap_col = col
            break
    
    if band_gap_col is None:
        print("警告: 在CSV文件中未找到band_gap列")
        print(f"可用的列名: {df.columns.tolist()}")
        band_gap_col = input("请输入正确的band_gap列名(按Enter取消): ")
        if not band_gap_col:
            return None
    
    # 统计原始行数
    original_count = len(df)
    print(f"原始数据行数: {original_count}")
    
    # 统计band_gap=0的行数
    zero_gap_count = len(df[df[band_gap_col] == 0])
    print(f"band_gap=0的行数: {zero_gap_count} ({zero_gap_count/original_count*100:.2f}%)")
    
    # 过滤band_gap=0的行
    df_filtered = df[df[band_gap_col] > 0]
    
    # 统计过滤后的行数
    filtered_count = len(df_filtered)
    print(f"过滤后的行数: {filtered_count}")
    
    # 保存过滤后的数据
    df_filtered.to_csv(output_csv_path, index=False)
    print(f"已将过滤后的数据保存到: {output_csv_path}")
    
    return output_csv_path

if __name__ == "__main__":
    # 直接调用函数，使用预先定义的路径
    filter_zero_bandgap()