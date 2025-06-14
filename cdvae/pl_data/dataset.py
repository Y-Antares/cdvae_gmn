import hydra
import omegaconf
import torch
import pandas as pd
from omegaconf import ValueNode
from torch.utils.data import Dataset

from torch_geometric.data import Data

from cdvae.common.utils import PROJECT_ROOT
from cdvae.common.data_utils import (
    preprocess, preprocess_tensors, add_scaled_lattice_prop, lattice_params_to_matrix_torch)

class CrystDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode,
                 prop: ValueNode, niggli: ValueNode, primitive: ValueNode,
                 graph_method: ValueNode, preprocess_workers: ValueNode,
                 lattice_scale_method: ValueNode,
                 **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        # 预处理数据，确保形成能在属性列表中
        self.cached_data = preprocess(
            self.path,
            preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
            prop_list=[prop, 'formation_energy_per_atom'])  # 确保包含形成能

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        
        # 初始化标准化器为None，后续会设置
        self.lattice_scaler = None
        self.scaler = None
        self.energy_scaler = None  # 添加形成能标准化器

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, edge_indices,
        to_jimages, num_atoms) = data_dict['graph_arrays']
        
        # 导入您确认存在的函数
        from cdvae.common.data_utils import lattice_params_to_matrix_torch

        # 1. 【关键】将所有 numpy 数组转换为 torch 张量，并【立即】统一为 float32
        frac_coords = torch.from_numpy(frac_coords).float()
        atom_types = torch.from_numpy(atom_types).long()
        lengths = torch.from_numpy(lengths).float()
        angles = torch.from_numpy(angles).float()
        edge_indices = torch.from_numpy(edge_indices).long()
        to_jimages = torch.from_numpy(to_jimages).long()

        # 2. 因为所有输入都已是 float32，后续所有计算结果也都会是 float32
        #    增加 .view(1, -1) 来满足下游函数对“批次”维度的要求
        lattice = lattice_params_to_matrix_torch(lengths.view(1, -1), angles.view(1, -1)).squeeze(0)
        pos = torch.einsum('bi,ij->bj', frac_coords, lattice)

        # 3. 构建 PyG Data 对象，所有属性的数据类型都已统一
        data = Data(
            x=torch.ones(num_atoms, 1),
            pos=pos,
            frac_coords=frac_coords,
            atom_types=atom_types,
            lengths=lengths.view(1, -1),
            angles=angles.view(1, -1),
            cell=lattice,
            edge_index=edge_indices.t().contiguous(),
            to_jimages=to_jimages,
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,
        )

        # 4. 处理目标属性 y
        formation_energy = float(data_dict.get('formation_energy_per_atom', 0.0))
        target_prop_value = float(data_dict.get(self.prop, 0.0))
        
        data.y = torch.tensor([formation_energy, target_prop_value], dtype=torch.float32).view(1, 2)
        
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"


class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, prop=None, **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method
        self.prop = prop  # 可选的目标属性名称

        self.cached_data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        
        # 初始化标准化器
        self.lattice_scaler = None
        self.scaler = None
        self.energy_scaler = None  # 添加形成能标准化器

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # 构建基本数据对象
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        )
        
        # 使用安全的方式获取属性值
        formation_energy = 0.0
        target_prop_value = 0.0
        
        # 尝试获取形成能
        if 'formation_energy_per_atom' in data_dict:
            formation_energy = data_dict['formation_energy_per_atom']
            if self.energy_scaler:
                formation_energy = self.energy_scaler.transform(formation_energy)
        
        # 尝试获取目标属性（如果指定了属性）
        if self.prop and self.prop in data_dict:
            target_prop_value = data_dict[self.prop]
            if self.scaler:
                target_prop_value = self.scaler.transform(target_prop_value)
        
        # 只有在至少有一个属性存在时才添加 y 属性
        combined_y = torch.tensor([formation_energy, target_prop_value], dtype=torch.float).view(1, 2)
        data.y = combined_y
            
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3")
def main(cfg: omegaconf.DictConfig):
    from torch_geometric.data import Batch
    from cdvae.common.data_utils import get_scaler_from_data_list
    dataset: CrystDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )
    
    # 获取各项属性的标准化器
    lattice_scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key='scaled_lattice')
    scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key=dataset.prop)
    
    # 安全获取形成能标准化器
    try:
        energy_scaler = get_scaler_from_data_list(
            dataset.cached_data,
            key='formation_energy_per_atom')
    except KeyError:
        print("警告：找不到形成能数据，使用默认标准化器")
        from cdvae.common.data_utils import StandardScaler
        energy_scaler = StandardScaler(mean=0.0, std=1.0)
    
    # 设置标准化器
    dataset.energy_scaler = energy_scaler
    dataset.lattice_scaler = lattice_scaler
    dataset.scaler = scaler
    
    # 创建批处理数据
    data_list = [dataset[i] for i in range(len(dataset))]
    batch = Batch.from_data_list(data_list)
    
    # 输出批处理数据的形状，用于调试
    print(f"Batch y shape: {batch.y.shape}")  # 应该是 [batch_size, 1, 2]
    print(f"Formation energy (mean): {batch.y[:, :, 0].mean().item()}")
    print(f"Target property (mean): {batch.y[:, :, 1].mean().item()}")
    
    return batch


if __name__ == "__main__":
    main()