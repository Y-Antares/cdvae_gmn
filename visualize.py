import torch
import os
from pymatgen.io.vasp import Poscar
from pymatgen.io.cif import CifWriter
import codecs
import smact
import smact.data_loader
from collections import Counter
import argparse
from tqdm import tqdm
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice

class Crystal:
    def __init__(self, frac_coords, atom_types, lengths, angles):
        # 直接接收各个组件
        self.frac_coords = frac_coords
        self.atom_types = atom_types
        self.lengths = lengths
        self.angles = angles
        self.constructed = True
        
        # 添加属性字段
        self.formation_energy = None
        self.target_property = None
        
        try:
            # 确保数据是标量而不是数组
            if isinstance(self.lengths, (list, np.ndarray)) and len(self.lengths) == 3:
                a, b, c = self.lengths[0], self.lengths[1], self.lengths[2]
            else:
                a, b, c = self.lengths
                
            if isinstance(self.angles, (list, np.ndarray)) and len(self.angles) == 3:
                alpha, beta, gamma = self.angles[0], self.angles[1], self.angles[2]
            else:
                alpha, beta, gamma = self.angles
            
            # 创建晶格对象
            self.lattice = Lattice.from_parameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
            
            # 创建结构对象
            self.structure = Structure(
                lattice=self.lattice,
                species=self.atom_types,
                coords=self.frac_coords,
                coords_are_cartesian=False
            )
        except Exception as e:
            self.constructed = False
            self.invalid_reason = str(e)

def load_tensor_format_structures(filepath):
    """从包含分离张量的PT文件加载结构数据"""
    data = torch.load(filepath)
    print(f"PT文件内容的键: {list(data.keys())}")
    
    # 定义张量转换为列表的函数
    def to_list(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().detach().numpy().tolist()
        return tensor
    
    # 检查是否有属性信息
    has_properties = 'properties' in data
    if has_properties:
        print("发现属性信息！")
        properties = to_list(data['properties'])
        props_shape = np.array(properties).shape if isinstance(properties, list) else 'unknown'
        print(f"属性形状: {props_shape}")
    
    if all(key in data for key in ['frac_coords', 'atom_types', 'lengths', 'angles']):
        # 将张量转换为Python列表
        frac_coords = to_list(data['frac_coords'])
        atom_types = to_list(data['atom_types'])
        lengths = to_list(data['lengths'])
        angles = to_list(data['angles'])
        
        # 打印数据形状以便调试
        print(f"frac_coords类型: {type(frac_coords)}")
        if isinstance(frac_coords, list):
            print(f"frac_coords长度: {len(frac_coords)}")
            if len(frac_coords) > 0 and isinstance(frac_coords[0], list):
                print(f"frac_coords[0]长度: {len(frac_coords[0])}")
        
        print(f"atom_types类型: {type(atom_types)}")
        print(f"lengths类型: {type(lengths)}")
        print(f"angles类型: {type(angles)}")
        
        # 处理批次大小
        batch_size = 1
        if 'num_atoms' in data:
            num_atoms = to_list(data['num_atoms'])
            print(f"num_atoms类型: {type(num_atoms)}")
            if isinstance(num_atoms, list):
                print(f"num_atoms值: {num_atoms}")
                # 确定实际结构数量（逗号分隔的列表）
                if len(num_atoms) > 0 and isinstance(num_atoms[0], list) and len(num_atoms[0]) > 0:
                    # 每个num_atoms条目代表一个结构
                    structures = []
                    
                    # 从第一个批次创建所有结构
                    coords_list = frac_coords[0]
                    atoms_list = atom_types[0]
                    batch_lengths = lengths[0] if isinstance(lengths[0], list) else lengths
                    batch_angles = angles[0] if isinstance(angles[0], list) else angles
                    
                    # 获取有效原子数
                    valid_atoms = num_atoms[0]
                    
                    # 创建结构
                    start_idx = 0
                    for i, n_atoms in enumerate(valid_atoms):
                        try:
                            # 提取当前结构的原子
                            end_idx = start_idx + n_atoms
                            current_coords = coords_list[start_idx:end_idx]
                            current_atoms = atoms_list[start_idx:end_idx]
                            
                            # 创建结构
                            crystal = Crystal(
                                current_coords,
                                current_atoms, 
                                batch_lengths if not isinstance(batch_lengths[0], list) else batch_lengths[i],
                                batch_angles if not isinstance(batch_angles[0], list) else batch_angles[i]
                            )
                            
                            # 如果有属性信息，添加到晶体对象
                            if has_properties and len(properties) > 0 and len(properties[0]) > i:
                                if isinstance(properties[0][i], list) or isinstance(properties[0][i], np.ndarray):
                                    if len(properties[0][i]) > 0:
                                        crystal.formation_energy = properties[0][i][0]
                                    if len(properties[0][i]) > 1:
                                        crystal.target_property = properties[0][i][1]
                                else:
                                    # 单目标情况
                                    crystal.target_property = properties[0][i]
                            
                            structures.append(crystal)
                            
                            # 更新起始索引
                            start_idx = end_idx
                        except Exception as e:
                            print(f"处理结构 {i} 时出错: {e}")
                            continue
                    
                    return structures
        
        # 如果上面的处理没有返回结构，则尝试简单地创建一个结构
        try:
            if isinstance(frac_coords, list) and len(frac_coords) > 0:
                if isinstance(frac_coords[0], list) and len(frac_coords[0]) > 0:
                    crystal = Crystal(
                        frac_coords[0], 
                        atom_types[0], 
                        lengths[0] if isinstance(lengths[0], list) else lengths, 
                        angles[0] if isinstance(angles[0], list) else angles
                    )
                    
                    # 如果有属性信息，添加到晶体对象
                    if has_properties and len(properties) > 0 and len(properties[0]) > 0:
                        if isinstance(properties[0][0], list) or isinstance(properties[0][0], np.ndarray):
                            if len(properties[0][0]) > 0:
                                crystal.formation_energy = properties[0][0][0]
                            if len(properties[0][0]) > 1:
                                crystal.target_property = properties[0][0][1]
                        else:
                            # 单目标情况
                            crystal.target_property = properties[0][0]
                            
                    return [crystal]
        except Exception as e:
            print(f"创建单个结构时出错: {e}")
        
        # 如果都失败了，返回一个空列表
        print("无法从数据中解析有效的结构")
        return []
    else:
        raise ValueError(f"文件不包含所需的所有张量键（frac_coords, atom_types, lengths, angles）")

def parse_args():
    parser = argparse.ArgumentParser(description='生成和可视化优化结构')
    parser.add_argument('--ptfile', type=str, default='/root/cdvae/hydra/singlerun/2025-03-15/Ag/eval_opt.pt',
                       help='优化结构的PT文件路径')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='结果保存目录。如果不指定，将使用结构名自动生成')
    parser.add_argument('--allowed_elements', type=str, default='',
                       help='允许的元素，用逗号分隔。默认为空，表示不筛选')
    parser.add_argument('--filter_formula', type=str, default=None,
                       help='要筛选的化学式格式，例如"ABC2"')
    parser.add_argument('--sort_by_energy', action='store_true',
                       help='按形成能从低到高排序')
    parser.add_argument('--sort_by_property', action='store_true',
                       help='按目标属性从低到高排序')
    return parser.parse_args()

def patch_smact_file_open():
    """修复smact文件打开方法，强制使用UTF-8编码。"""
    original_open = open
    def utf8_open(*args, **kwargs):
        kwargs['encoding'] = 'utf-8'
        return original_open(*args, **kwargs)
    smact.data_loader.open = utf8_open

def parse_allowed_elements(elements_str):
    """从逗号分隔的字符串解析允许的元素列表"""
    if not elements_str:
        return []
    elements = elements_str.split(',')
    return [elem.strip() for elem in elements if elem.strip()]

def main():
    args = parse_args()
    ptfile = args.ptfile
    
    # 获取当前目录
    current_directory = os.getcwd()
    print("当前默认路径:", current_directory)
    
    # 创建结果输出目录
    if args.output_dir:
        output_directory = args.output_dir
    else:
        # 从PT文件路径提取目录名
        pt_basename = os.path.basename(ptfile).split('.')[0]
        output_directory = os.path.join(current_directory, 'results', pt_basename)
    
    os.makedirs(output_directory, exist_ok=True)
    print(f"结果将保存至: {output_directory}")
    
    # 定义允许的元素组合
    allowed_elements = parse_allowed_elements(args.allowed_elements)
    if allowed_elements:
        print(f"允许的元素: {allowed_elements}")
    else:
        print("未指定元素筛选，将处理所有元素")
    
    # 应用补丁
    patch_smact_file_open()
    
    # 加载PT文件
    try:
        print(f"加载PT文件: {ptfile}")
        structures = load_tensor_format_structures(ptfile)
        print(f"成功加载 {len(structures)} 个结构")
    except Exception as e:
        print(f"加载结构时出错: {e}")
        raise
    
    # 检查是否有有效结构
    if not structures:
        print("未找到有效结构，处理终止")
        return
    
    # 打印第一个结构的信息用于检查
    print("\n第一个结构信息:")
    try:
        crystal = structures[0]
        print("晶格常数:", crystal.lengths)
        print("晶格角度:", crystal.angles)
        print("原子类型:", crystal.atom_types)
        print("分数坐标:", crystal.frac_coords)
        print("形成能:", crystal.formation_energy)
        print("目标属性:", crystal.target_property)
        if hasattr(crystal, 'constructed'):
            print("结构已构建:", crystal.constructed)
            if not crystal.constructed:
                print("无效原因:", crystal.invalid_reason)
    except Exception as e:
        print(f"检查第一个结构时出错: {e}")
    
    # 根据属性排序（如果需要）
    if args.sort_by_energy:
        print("按形成能排序...")
        valid_structures = [s for s in structures if s.constructed and s.formation_energy is not None]
        valid_structures.sort(key=lambda x: x.formation_energy)
        structures = valid_structures + [s for s in structures if s.constructed and s.formation_energy is None] + [s for s in structures if not s.constructed]
    elif args.sort_by_property:
        print("按目标属性排序...")
        valid_structures = [s for s in structures if s.constructed and s.target_property is not None]
        valid_structures.sort(key=lambda x: x.target_property)
        structures = valid_structures + [s for s in structures if s.constructed and s.target_property is None] + [s for s in structures if not s.constructed]
    
    # 转换并保存结构文件
    valid_structures = 0
    total_structures = len(structures)
    
    # 创建一个配置文件来记录有效结构的组成和属性
    composition_file = os.path.join(output_directory, 'compositions.txt')
    with open(composition_file, 'w') as f:
        f.write("index,composition,lattice,angle,formation energy(eV/atom),target\n")
    
    # 使用tqdm显示进度条
    for idx, crystal in enumerate(tqdm(structures, desc="处理结构")):
        try:
            if not crystal.constructed:
                print(f"结构 {idx} 构建失败: {crystal.invalid_reason}")
                continue
            
            # 打印当前组成以便调试
            elem_counter = Counter(crystal.atom_types)
            
            # 检查是否只包含允许的元素
            if allowed_elements and len(allowed_elements) > 0:
                all_allowed = all(str(elem) in allowed_elements for elem in elem_counter.keys())
                if not all_allowed:
                    print(f"跳过结构 {idx}，含有不允许的元素: {dict(elem_counter)}")
                    continue
            
            # 检查化学式过滤器（如果提供）
            if args.filter_formula:
                # 简单实现：计算不同元素的数量
                elem_types = len(elem_counter)
                formula_pattern = args.filter_formula.replace('A', '').replace('B', '').replace('C', '')
                try:
                    # 如果有数字，转换为整数
                    ratio = int(formula_pattern) if formula_pattern else 1
                    # 检查元素数量是否符合预期
                    total_atoms = sum(elem_counter.values())
                    if elem_types != len(args.filter_formula) - len(formula_pattern) or total_atoms % (elem_types + ratio - 1) != 0:
                        print(f"跳过结构 {idx}，不符合化学式格式 {args.filter_formula}")
                        continue
                except ValueError:
                    print(f"警告: 无法解析化学式格式 {args.filter_formula}，跳过过滤")
            
            # 转换为POSCAR格式
            poscar = Poscar(crystal.structure)
            poscar.write_file(os.path.join(output_directory, f'structure_{valid_structures}.vasp'))
            
            # 保存为CIF文件
            cif = CifWriter(crystal.structure)
            cif.write_file(os.path.join(output_directory, f'structure_{valid_structures}.cif'))
            
            # 记录组成信息和属性
            with open(composition_file, 'a') as f:
                formation_energy = crystal.formation_energy if crystal.formation_energy is not None else "N/A"
                target_property = crystal.target_property if crystal.target_property is not None else "N/A"
                f.write(f"{valid_structures},{dict(elem_counter)},{list(crystal.lengths)},{list(crystal.angles)},{formation_energy},{target_property}\n")
            
            valid_structures += 1
            if valid_structures % 10 == 0:
                print(f"已保存 {valid_structures} 个有效结构")
            
        except Exception as e:
            print(f"处理结构 {idx} 时出错: {e}")
    
    # 创建属性摘要文件 (排序后方便查看)
    property_summary_file = os.path.join(output_directory, 'property_summary.csv')
    with open(property_summary_file, 'w') as f:
        f.write("index,formation energy(eV/atom),target,composition\n")
        
        # 收集有效结构的属性信息
        property_data = []
        for idx, crystal in enumerate(structures):
            if not getattr(crystal, 'constructed', False):
                continue
                
            elem_counter = Counter(crystal.atom_types)
            formation_energy = crystal.formation_energy if hasattr(crystal, 'formation_energy') and crystal.formation_energy is not None else float('inf')
            target_property = crystal.target_property if hasattr(crystal, 'target_property') and crystal.target_property is not None else float('inf')
            
            property_data.append((idx, formation_energy, target_property, dict(elem_counter)))
        
        # 按形成能排序
        energy_sorted = sorted([p for p in property_data if p[1] != float('inf')], key=lambda x: x[1])
        for idx, formation_energy, target_property, composition in energy_sorted:
            f.write(f"{idx},{formation_energy},{target_property},{composition}\n")
        
        # 添加没有形成能的结构
        no_energy = [p for p in property_data if p[1] == float('inf')]
        for idx, formation_energy, target_property, composition in no_energy:
            f.write(f"{idx},N/A,{target_property if target_property != float('inf') else 'N/A'},{composition}\n")
    
    print(f"\n处理完成:")
    print(f"总结构数: {total_structures}")
    print(f"有效结构数: {valid_structures}")
    print(f"结果已保存至: {output_directory}")
    print(f"查看属性摘要: {property_summary_file}")

if __name__ == "__main__":
    main()