import os
import numpy as np

def visualize_pareto_front(model_path, optimization_methods=['weighted', 'tchebycheff', 'boundary'], 
                          target_type='combined', eval_model=None, marker_size=50, 
                          save_data=True, show_plot=False):
    """
    可视化不同优化方法得到的帕累托前沿。
    
    参数:
        model_path (str): 模型结果路径
        optimization_methods (list): 要比较的优化方法列表
        target_type (str): 优化目标类型，默认为'combined'
        eval_model (str): 评估模型名称，如果为None则从配置文件中读取
        marker_size (int): 散点图标记大小
        save_data (bool): 是否保存提取的数据到CSV文件
        show_plot (bool): 是否显示图表(交互模式)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import os
    from tqdm import tqdm
    from pathlib import Path
    from eval_utils import load_config, prop_model_eval, get_crystals_list
    from scripts.compute_metrics import Crystal
    
    # 创建输出目录
    output_dir = os.path.join(model_path, 'pareto_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取评估模型
    if eval_model is None:
        cfg = load_config(model_path)
        eval_model = cfg.data.eval_model_name
    
    # 设置颜色和标记样式
    colors = {'weighted': 'blue', 'tchebycheff': 'red', 'boundary': 'green'}
    markers = {'weighted': 'o', 'tchebycheff': 's', 'boundary': '^'}
    
    # 准备存储所有结果的数据框
    all_data = []
    
    # 创建绘图
    plt.figure(figsize=(12, 10))
    
    # 记录帕累托边界点
    pareto_frontier_points = {}
    
    for method in optimization_methods:
        print(f"处理 {method} 方法的结果...")
        
        # 构建结果文件路径
        result_path = os.path.join(model_path, f'eval_opt_{target_type}_{method}.pt')
        if not os.path.exists(result_path):
            print(f"  未找到 {method} 方法的结果文件，尝试替代路径...")
            # 尝试替代文件命名格式
            alternative_paths = [
                os.path.join(model_path, f'eval_opt_{method}_{target_type}.pt'),
                os.path.join(model_path, f'eval_opt_{target_type}.pt')
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    result_path = alt_path
                    print(f"  找到替代文件: {alt_path}")
                    break
            
            if not os.path.exists(result_path):
                print(f"  跳过 {method} 方法，未找到结果文件")
                continue
        
        try:
            # 加载优化结果
            data = torch.load(result_path, map_location='cpu')
            
            # 检查是否包含属性数据
            if 'properties' in data:
                print(f"  发现预计算的属性数据，直接使用...")
                # 使用预计算的属性
                properties = data['properties'].squeeze(0)
                if properties.shape[1] >= 2:
                    # 假设第一列是formation_energy，第二列是target_energy
                    formation_energies = properties[:, 0].numpy()
                    target_properties = properties[:, 1].numpy()
                    
                    method_data = pd.DataFrame({
                        'formation_energy': formation_energies,
                        'target_property': target_properties,
                        'method': method
                    })
                    all_data.append(method_data)
                    
                    # 绘制所有点
                    plt.scatter(
                        formation_energies, target_properties, 
                        label=f'{method} (all points)', 
                        color=colors.get(method, 'gray'), 
                        marker=markers.get(method, 'o'),
                        alpha=0.5, s=marker_size/2
                    )
                    
                    # 计算帕累托边界
                    pareto_points = compute_pareto_frontier(formation_energies, target_properties)
                    pareto_frontier_points[method] = pareto_points
                    
                    # 绘制帕累托边界点
                    plt.scatter(
                        pareto_points[:, 0], pareto_points[:, 1],
                        label=f'{method} (pareto frontier)',
                        color=colors.get(method, 'gray'),
                        marker=markers.get(method, 'o'),
                        edgecolors='black',
                        s=marker_size, linewidths=2
                    )
                    
                    print(f"  {method} 方法共有 {len(formation_energies)} 个点，其中 {len(pareto_points)} 个在帕累托边界上")
                else:
                    print(f"  预计算属性维度不足，跳过 {method} 方法")
            else:
                print(f"  未找到预计算的属性数据，开始评估晶体...")
                # 提取晶体结构
                crys_array_list = get_crystals_list(
                    data['frac_coords'][0],
                    data['atom_types'][0],
                    data['lengths'][0],
                    data['angles'][0],
                    data['num_atoms'][0]
                )
                
                # 转换为Crystal对象并过滤无效结构
                print(f"  处理 {len(crys_array_list)} 个晶体结构...")
                crystals = []
                for crys_array in tqdm(crys_array_list):
                    crystal = Crystal(crys_array)
                    if crystal.valid and len(crystal.atom_types) <= 30:  # 过滤无效和过大的结构
                        crystals.append(crystal)
                
                if len(crystals) == 0:
                    print(f"  没有有效的晶体结构，跳过 {method} 方法")
                    continue
                    
                print(f"  评估 {len(crystals)} 个有效晶体的属性...")
                # 评估属性
                pred_results = prop_model_eval(eval_model, [c.dict for c in crystals])
                
                if pred_results is None or len(pred_results) == 0:
                    print(f"  未得到有效的预测结果，跳过 {method} 方法")
                    continue
                
                # 过滤None值
                pred_results = [p for p in pred_results if p is not None]
                
                # 检查是否为多目标预测结果
                if len(pred_results) > 0 and hasattr(pred_results[0], "__len__") and len(pred_results[0]) > 1:
                    # 多目标结果
                    formation_energies = np.array([p[0] for p in pred_results])
                    target_properties = np.array([p[1] for p in pred_results])
                else:
                    # 单目标结果
                    print(f"  警告: 仅获得单目标预测结果，无法绘制帕累托前沿")
                    continue
                
                method_data = pd.DataFrame({
                    'formation_energy': formation_energies,
                    'target_property': target_properties,
                    'method': method
                })
                all_data.append(method_data)
                
                # 绘制所有点
                plt.scatter(
                    formation_energies, target_properties, 
                    label=f'{method} (all points)', 
                    color=colors.get(method, 'gray'), 
                    marker=markers.get(method, 'o'),
                    alpha=0.5, s=marker_size/2
                )
                
                # 计算帕累托边界
                pareto_points = compute_pareto_frontier(formation_energies, target_properties)
                pareto_frontier_points[method] = pareto_points
                
                # 绘制帕累托边界点
                plt.scatter(
                    pareto_points[:, 0], pareto_points[:, 1],
                    label=f'{method} (pareto frontier)',
                    color=colors.get(method, 'gray'),
                    marker=markers.get(method, 'o'),
                    edgecolors='black',
                    s=marker_size, linewidths=2
                )
                
                print(f"  {method} 方法共有 {len(formation_energies)} 个点，其中 {len(pareto_points)} 个在帕累托边界上")
        except Exception as e:
            print(f"  处理 {method} 方法时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 合并所有数据
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # 保存数据到CSV
        if save_data:
            data_path = os.path.join(output_dir, f'pareto_data_{target_type}.csv')
            combined_data.to_csv(data_path, index=False)
            print(f"已保存数据到: {data_path}")
            
            # 保存帕累托边界数据
            for method, points in pareto_frontier_points.items():
                pareto_df = pd.DataFrame(points, columns=['formation_energy', 'target_property'])
                pareto_df['method'] = method
                pareto_path = os.path.join(output_dir, f'pareto_frontier_{method}_{target_type}.csv')
                pareto_df.to_csv(pareto_path, index=False)
                print(f"已保存 {method} 方法的帕累托边界数据到: {pareto_path}")
        
        # 计算超体积指标
        for method, points in pareto_frontier_points.items():
            if len(points) > 0:
                hypervolume = calculate_hypervolume(points)
                print(f"{method} 方法的超体积指标: {hypervolume:.6f}")
        
        # 完善图表
        plt.xlabel('formation_energy (eV/atom)', fontsize=14)
        plt.ylabel('target_energy', fontsize=14)
        plt.title(f'comparison ({target_type})', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # 保存图表
        plot_path = os.path.join(output_dir, f'pareto_front_{target_type}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"已保存帕累托前沿图到: {plot_path}")
        
        if show_plot:
            plt.show()
        plt.close()
        
        # 创建放大的帕累托边界图
        plt.figure(figsize=(12, 10))
        
        for method, points in pareto_frontier_points.items():
            if len(points) > 0:
                # 绘制帕累托边界点和连线
                plt.scatter(
                    points[:, 0], points[:, 1],
                    label=f'{method}',
                    color=colors.get(method, 'gray'),
                    marker=markers.get(method, 'o'),
                    s=marker_size, linewidths=2
                )
                
                # 按formation_energy排序连接帕累托点
                sorted_indices = np.argsort(points[:, 0])
                sorted_points = points[sorted_indices]
                plt.plot(
                    sorted_points[:, 0], sorted_points[:, 1], 
                    color=colors.get(method, 'gray'),
                    linestyle='-', linewidth=2, alpha=0.7
                )
        
        plt.xlabel('formation_energy (eV/atom)', fontsize=14)
        plt.ylabel('target property', fontsize=14)
        plt.title(f'pareto frontier comparison ({target_type})', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # 保存帕累托边界图
        pareto_path = os.path.join(output_dir, f'pareto_frontiers_{target_type}.png')
        plt.savefig(pareto_path, dpi=300, bbox_inches='tight')
        print(f"已保存仅含帕累托边界的图到: {pareto_path}")
        
        if show_plot:
            plt.show()
        plt.close()
        
        return combined_data, pareto_frontier_points
    else:
        print("没有找到任何有效数据进行可视化")
        return None, None

def compute_pareto_frontier(costs, values):
    """
    计算帕累托前沿点。
    对于多目标优化问题，目标是最小化成本(costs)，最大化收益(values)。
    
    参数:
        costs: formation_energy数组
        values: target_energy数组
        
    返回:
        帕累托前沿点的数组 [N, 2]
    """
    import numpy as np
    
    # 将数据组合成点
    points = np.column_stack([costs, values])
    
    # 创建帕累托前沿
    pareto_points = []
    
    # 对于最小化-最大化问题，我们需要转换值为负数（因为我们总是找最小化的帕累托前沿）
    # 如果目标是最小化-最小化，则不需要这一步
    points_min_min = np.column_stack([points[:, 0], -points[:, 1]])
    
    # 对每个点，检查是否被支配
    for i, point in enumerate(points_min_min):
        # 假设点是非支配的
        is_pareto = True
        
        # 检查是否被其他点支配
        for j, other_point in enumerate(points_min_min):
            if i != j:  # 不与自己比较
                # 如果other_point在所有维度上小于等于point，且至少一个维度严格小于
                if np.all(other_point <= point) and np.any(other_point < point):
                    is_pareto = False
                    break
        
        # 如果点是非支配的，添加到帕累托前沿
        if is_pareto:
            # 转换回原始坐标
            pareto_points.append(points[i])
    
    return np.array(pareto_points)

def calculate_hypervolume(pareto_points, reference_point=None):
    """
    计算帕累托前沿的超体积指标
    
    参数:
        pareto_points: 帕累托前沿点的数组 [N, 2]
        reference_point: 参考点，默认为None，会自动根据数据设置
        
    返回:
        超体积指标值
    """
    import numpy as np
    
    if len(pareto_points) == 0:
        return 0.0
    
    # 默认参考点，能量轴取最大值+10%余量，target_energy轴取最小值-10%余量
    if reference_point is None:
        max_energy = np.max(pareto_points[:, 0]) * 1.1
        min_property = np.min(pareto_points[:, 1]) * 0.9
        reference_point = np.array([max_energy, min_property])
    
    # 对于最小化-最大化问题，我们计算"贡献区域"
    # 首先按能量排序
    sorted_indices = np.argsort(pareto_points[:, 0])
    sorted_points = pareto_points[sorted_indices]
    
    # 计算超体积
    hypervolume = 0.0
    prev_energy = reference_point[0]
    prev_property = reference_point[1]
    
    for i, point in enumerate(sorted_points):
        energy, prop = point
        # 添加当前点的贡献
        area = (prev_energy - energy) * (prop - prev_property)
        hypervolume += area
        
        # 更新前一个属性值
        prev_property = prop
    
    return hypervolume

def calculate_metrics(pareto_frontier_points):
    """
    计算不同优化方法的性能指标
    
    参数:
        pareto_frontier_points: 不同方法的帕累托前沿点字典
        
    返回:
        包含各种指标的字典
    """
    import numpy as np
    
    metrics = {}
    
    # 1. 超体积指标
    metrics['hypervolume'] = {}
    for method, points in pareto_frontier_points.items():
        if len(points) > 0:
            metrics['hypervolume'][method] = calculate_hypervolume(points)
        else:
            metrics['hypervolume'][method] = 0.0
    
    # 2. 帕累托点数量
    metrics['pareto_count'] = {method: len(points) for method, points in pareto_frontier_points.items()}
    
    # 3. 最小能量点
    metrics['min_energy'] = {}
    for method, points in pareto_frontier_points.items():
        if len(points) > 0:
            min_energy_idx = np.argmin(points[:, 0])
            metrics['min_energy'][method] = {
                'energy': points[min_energy_idx, 0],
                'property': points[min_energy_idx, 1]
            }
    
    # 4. 最大属性点
    metrics['max_property'] = {}
    for method, points in pareto_frontier_points.items():
        if len(points) > 0:
            max_prop_idx = np.argmax(points[:, 1])
            metrics['max_property'][method] = {
                'energy': points[max_prop_idx, 0],
                'property': points[max_prop_idx, 1]
            }
    
    # 5. 计算覆盖度度量（一个方法的帕累托集对另一个方法的）
    if len(pareto_frontier_points) > 1:
        methods = list(pareto_frontier_points.keys())
        metrics['coverage'] = {}
        
        for i, method1 in enumerate(methods):
            metrics['coverage'][method1] = {}
            points1 = pareto_frontier_points[method1]
            
            for j, method2 in enumerate(methods):
                if i != j:
                    points2 = pareto_frontier_points[method2]
                    if len(points1) > 0 and len(points2) > 0:
                        dominated_count = 0
                        
                        for p2 in points2:
                            for p1 in points1:
                                # 检查p1是否支配p2
                                if p1[0] <= p2[0] and p1[1] >= p2[1] and (p1[0] < p2[0] or p1[1] > p2[1]):
                                    dominated_count += 1
                                    break
                        
                        metrics['coverage'][method1][method2] = dominated_count / len(points2)
                    else:
                        metrics['coverage'][method1][method2] = 0.0
    
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化不同优化方法的帕累托前沿')
    parser.add_argument('--model_path', required=True, help='模型结果路径')
    parser.add_argument('--methods', nargs='+', default=['weighted', 'tchebycheff', 'boundary'], 
                        help='要比较的优化方法')
    parser.add_argument('--target_type', default='combined', 
                        choices=['combined', 'energy', 'property', 'weighted'],
                        help='优化目标类型')
    parser.add_argument('--eval_model', default=None, help='评估模型名称')
    parser.add_argument('--marker_size', type=int, default=50, help='散点图标记大小')
    parser.add_argument('--save_data', action='store_true', help='保存提取的数据到CSV文件')
    parser.add_argument('--show_plot', action='store_true', help='显示图表(交互模式)')
    
    args = parser.parse_args()
    
    print(f"开始可视化以下方法的帕累托前沿: {args.methods}")
    combined_data, pareto_points = visualize_pareto_front(
        model_path=args.model_path,
        optimization_methods=args.methods,
        target_type=args.target_type,
        eval_model=args.eval_model,
        marker_size=args.marker_size,
        save_data=args.save_data,
        show_plot=args.show_plot
    )
    
    # 计算性能指标
    if pareto_points:
        print("\n优化性能指标:")
        metrics = calculate_metrics(pareto_points)
        
        # 打印超体积指标
        print("\n超体积指标 (越大越好):")
        for method, hv in metrics['hypervolume'].items():
            print(f"  {method}: {hv:.6f}")
        
        # 打印帕累托点数量
        print("\n帕累托点数量:")
        for method, count in metrics['pareto_count'].items():
            print(f"  {method}: {count}")
        
        # 打印最小能量点
        print("\n最小能量点:")
        for method, values in metrics['min_energy'].items():
            print(f"  {method}: 能量={values['energy']:.6f}, 属性={values['property']:.6f}")
        
        # 打印最大属性点
        print("\n最大属性点:")
        for method, values in metrics['max_property'].items():
            print(f"  {method}: 能量={values['energy']:.6f}, 属性={values['property']:.6f}")
        
        # 打印覆盖度
        if 'coverage' in metrics:
            print("\n覆盖度度量 (越高越好):")
            for method1, coverages in metrics['coverage'].items():
                for method2, value in coverages.items():
                    print(f"  {method1} 覆盖 {method2}: {value:.6f}")
        
        # 导出指标到JSON
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = os.path.join(args.model_path, 'pareto_analysis', f'metrics_{args.target_type}_{timestamp}.json')
        
        # 将numpy值转换为Python原生类型
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(i) for i in obj]
            else:
                return obj
        
        with open(metrics_path, 'w') as f:
            json.dump(convert_for_json(metrics), f, indent=2)
        
        print(f"\n已保存详细指标到: {metrics_path}")