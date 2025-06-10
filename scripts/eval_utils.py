import itertools
import numpy as np
import torch
import hydra
import sys
import os
import yaml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from hydra import compose
from hydra import initialize_config_dir
from pathlib import Path
from cdvae.pl_modules.model import CDVAE

import smact
from smact.screening import pauling_test

import cdvae
from cdvae.common.constants import CompScalerMeans, CompScalerStds
from cdvae.common.data_utils import StandardScaler, chemical_symbols
from cdvae.pl_data.dataset import TensorCrystDataset
from cdvae.pl_data.datamodule import worker_init_fn

from torch_geometric.loader import DataLoader
from cdvae.common.data_utils import StandardScalerTorch

CompScaler = StandardScaler(
    means=np.array(CompScalerMeans),
    stds=np.array(CompScalerStds),
    replace_nan_token=0.)


def load_data(file_path):
    if file_path[-3:] == 'npy':
        data = np.load(file_path, allow_pickle=True).item()
        for k, v in data.items():
            if k == 'input_data_batch':
                for k1, v1 in data[k].items():
                    data[k][k1] = torch.from_numpy(v1)
            else:
                data[k] = torch.from_numpy(v).unsqueeze(0)
    else:
        data = torch.load(file_path, weights_only=False)
    return data


def get_model_path(eval_model_name):
    model_path = (
        Path(cdvae.__file__).parent / 'prop_models' / eval_model_name)
    return model_path


def load_config(model_path):
    with initialize_config_dir(str(model_path)):
        cfg = compose(config_name='hparams')
    return cfg

def load_model(model_path, load_data=False, testing=True):
    config_path = model_path / 'hparams.yaml'
    print(f"Loading config from: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print("Loaded config:", config)
    print("Model parameters:", config.get('model', {}))
    
    with initialize_config_dir(str(model_path)):
        cfg = compose(config_name='hparams')
        
        # 获取模型类
        model_cls = hydra.utils.get_class(cfg.model._target_)
        
        # 找到最新的检查点
        ckpts = list(model_path.glob('*.ckpt'))
        if len(ckpts) > 0:
            ckpt_epochs = np.array(
                [int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts])
            ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        
        # 加载检查点
        checkpoint = torch.load(ckpt, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        # 修复尺寸不匹配的权重
        for key in ['encoder.output_blocks.0.lin.weight',
                   'encoder.output_blocks.1.lin.weight',
                   'encoder.output_blocks.2.lin.weight',
                   'encoder.output_blocks.3.lin.weight',
                   'encoder.output_blocks.4.lin.weight']:
            if key in state_dict:
                old_weight = state_dict[key]  # [1, 256]
                new_weight = torch.zeros(256, 256)  # [256, 256]
                new_weight = old_weight.expand(256, -1)
                state_dict[key] = new_weight
        
        # 模型参数字典
        model_kwargs = {
            'hidden_dim': cfg.model.hidden_dim,
            'latent_dim': cfg.model.latent_dim,
            'encoder': cfg.model.encoder,
            'decoder': cfg.model.decoder,
            'max_atoms': cfg.data.max_atoms,
            'cost_natom': cfg.model.cost_natom,
            'cost_coord': cfg.model.cost_coord,
            'cost_type': cfg.model.cost_type,
            'cost_lattice': cfg.model.cost_lattice,
            'cost_composition': cfg.model.cost_composition,
            'cost_edge': cfg.model.cost_edge,
            'cost_property': cfg.model.cost_property,
            'beta': cfg.model.beta,
            'max_neighbors': cfg.model.max_neighbors,
            'radius': cfg.model.radius,
            'sigma_begin': cfg.model.sigma_begin,
            'sigma_end': cfg.model.sigma_end,
            'type_sigma_begin': cfg.model.type_sigma_begin,
            'type_sigma_end': cfg.model.type_sigma_end,
            'num_noise_level': cfg.model.num_noise_level,
            'predict_property': cfg.model.predict_property,
            'optim': cfg.optim,
            'data': cfg.data,
            'logging': cfg.logging
        }
        
        # 确保fc_num_layers存在于model_kwargs中
        if hasattr(cfg.model, 'fc_num_layers'):
            model_kwargs['fc_num_layers'] = cfg.model.fc_num_layers
        else:
            model_kwargs['fc_num_layers'] = 4  # 设置默认值
            print("Warning: fc_num_layers not found in config, using default value: 4")
            
        # 添加多目标支持配置
        if hasattr(cfg.model, 'property_weights'):
            model_kwargs['property_weights'] = cfg.model.property_weights
            print(f"Using property weights: {cfg.model.property_weights}")
        
        # 创建新模型
        model = model_cls(**model_kwargs)
        
        # 加载修改后的状态字典
        model.load_state_dict(state_dict, strict=False)
        
        # 初始化其他必要的组件
        if hasattr(model, 'init_sigmas'):
            model.init_sigmas()
        
        # 加载标准化器
        model.lattice_scaler = torch.load(model_path / 'lattice_scaler.pt', weights_only=False)
        model.scaler = torch.load(model_path / 'prop_scaler.pt', weights_only=False)
        
        # 检查并加载能量标准化器
        energy_scaler_path = model_path / 'energy_scaler.pt'
        if energy_scaler_path.exists():
            print("Loading energy scaler...")
            model.energy_scaler = torch.load(energy_scaler_path, weights_only=False)
        else:
            print("Warning: energy_scaler.pt not found")

        if load_data:
            datamodule = hydra.utils.instantiate(
                cfg.data.datamodule, _recursive_=False, scaler_path=model_path
            )
            if testing:
                datamodule.setup('test')
                test_loader = datamodule.test_dataloader()[0]
            else:
                datamodule.setup()
                test_loader = datamodule.val_dataloader()[0]
        else:
            test_loader = None

    return model, test_loader, cfg

def get_crystals_list(
        frac_coords, atom_types, lengths, angles, num_atoms):
    """
    args:
        frac_coords: (num_atoms, 3)
        atom_types: (num_atoms)
        lengths: (num_crystals)
        angles: (num_crystals)
        num_atoms: (num_crystals)
    """
    assert frac_coords.size(0) == atom_types.size(0) == num_atoms.sum()
    assert lengths.size(0) == angles.size(0) == num_atoms.size(0)

    start_idx = 0
    crystal_array_list = []
    for batch_idx, num_atom in enumerate(num_atoms.tolist()):
        cur_frac_coords = frac_coords.narrow(0, start_idx, num_atom)
        cur_atom_types = atom_types.narrow(0, start_idx, num_atom)
        cur_lengths = lengths[batch_idx]
        cur_angles = angles[batch_idx]

        crystal_array_list.append({
            'frac_coords': cur_frac_coords.detach().cpu().numpy(),
            'atom_types': cur_atom_types.detach().cpu().numpy(),
            'lengths': cur_lengths.detach().cpu().numpy(),
            'angles': cur_angles.detach().cpu().numpy(),
        })
        start_idx = start_idx + num_atom
    return crystal_array_list


def smact_validity(comp, count,
                   use_pauling_test=True,
                   include_alloys=True):
    elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                for ratio in cn_r:
                    compositions.append(
                        tuple([elem_symbols, ox_states, ratio]))
    compositions = [(i[0], i[2]) for i in compositions]
    compositions = list(set(compositions))
    if len(compositions) > 0:
        return True
    else:
        return False


def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(
        np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True


def get_fp_pdist(fp_array):
    if isinstance(fp_array, list):
        fp_array = np.array(fp_array)
    fp_pdists = pdist(fp_array)
    return fp_pdists.mean()


def prop_model_eval(eval_model_name, crystal_array_list):
    try:
        # 添加原子数检查
        filtered_list = []
        filtered_indices = []
        
        for i, crystal_dict in enumerate(crystal_array_list):
            num_atoms = None
            
            # 处理不同类型的输入结构
            if isinstance(crystal_dict, dict) and 'atom_types' in crystal_dict:
                num_atoms = len(crystal_dict['atom_types'])
            elif hasattr(crystal_dict, 'atom_types'):
                num_atoms = len(crystal_dict.atom_types)
            elif hasattr(crystal_dict, 'dict') and 'atom_types' in crystal_dict.dict:
                num_atoms = len(crystal_dict.dict['atom_types'])
            elif hasattr(crystal_dict, 'num_atoms'):
                num_atoms = crystal_dict.num_atoms
                
            # 过滤掉原子数超过30的结构
            if num_atoms is not None and num_atoms <= 30:
                filtered_list.append(crystal_dict)
                filtered_indices.append(i)
        
        if len(filtered_list) < len(crystal_array_list):
            print(f"Filtered {len(crystal_array_list) - len(filtered_list)} structures with >30 atoms in prop_model_eval")
        
        if not filtered_list:
            print("No valid structures left after filtering in prop_model_eval")
            return [None] * len(crystal_array_list)
        
        # 原有的模型评估代码
        model_path = get_model_path(eval_model_name)
        model, _, _ = load_model(model_path)
        cfg = load_config(model_path)

        dataset = TensorCrystDataset(
            filtered_list, cfg.data.niggli, cfg.data.primitive,
            cfg.data.graph_method, cfg.data.preprocess_workers,
            cfg.data.lattice_scale_method)

        dataset.scaler = model.scaler.copy() if hasattr(model.scaler, 'copy') else model.scaler
        
        # 添加能量标准化器传递
        if hasattr(model, 'energy_scaler') and model.energy_scaler is not None:
            dataset.energy_scaler = model.energy_scaler.copy() if hasattr(model.energy_scaler, 'copy') else model.energy_scaler
            print("Passed energy scaler to dataset")

        loader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=256,
            num_workers=0,
            worker_init_fn=worker_init_fn)

        model.eval()
        all_preds = []

        if isinstance(model.scaler, dict):
            print("Original model.scaler:", model.scaler)
            
            means = model.scaler.get('means', None)
            stds = model.scaler.get('stds', None)
            
            print("Extracted means type:", type(means))
            print("Extracted stds type:", type(stds))
            
            if isinstance(means, dict):
                print("means is a dict:", means)
                if 'data' in means:
                    means = means['data']
            
            if isinstance(stds, dict):
                print("stds is a dict:", stds)
                if 'data' in stds:
                    stds = stds['data']
            
            if means is not None and not isinstance(means, torch.Tensor):
                means = torch.tensor(means, dtype=torch.float)
            if stds is not None and not isinstance(stds, torch.Tensor):
                stds = torch.tensor(stds, dtype=torch.float)
            
            new_scaler = StandardScalerTorch(means=means, stds=stds)
            model.scaler = new_scaler
            
            print("Converted dictionary to StandardScalerTorch object")
            print("New scaler:", model.scaler)
            
        # 同样检查能量标准化器是否需要转换
        if hasattr(model, 'energy_scaler') and isinstance(model.energy_scaler, dict):
            print("Original model.energy_scaler:", model.energy_scaler)
            
            means = model.energy_scaler.get('means', None)
            stds = model.energy_scaler.get('stds', None)
            
            if isinstance(means, dict) and 'data' in means:
                means = means['data']
            
            if isinstance(stds, dict) and 'data' in stds:
                stds = stds['data']
            
            if means is not None and not isinstance(means, torch.Tensor):
                means = torch.tensor(means, dtype=torch.float)
            if stds is not None and not isinstance(stds, torch.Tensor):
                stds = torch.tensor(stds, dtype=torch.float)
            
            new_scaler = StandardScalerTorch(means=means, stds=stds)
            model.energy_scaler = new_scaler
            
            print("Converted energy_scaler dictionary to StandardScalerTorch object")

        for batch in loader:
            # 添加额外安全检查
            max_supported_atoms = None
            if hasattr(model, 'mlp_num_atoms') and isinstance(model.mlp_num_atoms, torch.nn.Sequential):
                final_layer = model.mlp_num_atoms[-1]
                if isinstance(final_layer, torch.nn.Linear):
                    max_supported_atoms = final_layer.out_features - 1
            elif hasattr(model, 'embeddings') and 'num_atom' in model.embeddings:
                max_supported_atoms = model.embeddings['num_atom'].num_embeddings - 1
            
            if max_supported_atoms is not None and hasattr(batch, 'num_atoms'):
                if batch.num_atoms.max() > max_supported_atoms:
                    print(f"Warning: Limiting num_atoms from {batch.num_atoms.max()} to {max_supported_atoms}")
                    batch.num_atoms = torch.clamp(batch.num_atoms, max=max_supported_atoms)
            
            try:
                with torch.no_grad():
                    preds = model(batch, teacher_forcing=False, training=False)
                
                # 处理多目标预测
                multi_target = hasattr(model, 'fc_property_shared') and model.fc_property_shared is not None
                
                if multi_target:
                    print("Model has multi-target prediction capability")
                    # 获取共享特征
                    shared_features = model.fc_property_shared(preds['z'])
                    
                    # 分别预测能量和目标属性
                    energy_pred = model.energy_head(shared_features)
                    target_pred = model.target_head(shared_features)
                    
                    # 应用标准化器
                    if hasattr(model, 'energy_scaler') and model.energy_scaler is not None:
                        model.energy_scaler.match_device(energy_pred)
                        scaled_energy_pred = model.energy_scaler.inverse_transform(energy_pred)
                    else:
                        scaled_energy_pred = energy_pred
                        
                    if hasattr(model, 'scaler') and model.scaler is not None:
                        model.scaler.match_device(target_pred)
                        scaled_target_pred = model.scaler.inverse_transform(target_pred)
                    else:
                        scaled_target_pred = target_pred
                    
                    # 这里我们只关注目标属性（第二个值）
                    scaled_preds = scaled_target_pred
                    
                else:
                    # 原始单目标处理
                    try:
                        model.scaler.match_device(preds)
                        scaled_preds = model.scaler.inverse_transform(preds)
                    except (AttributeError, TypeError) as e:
                        print(f"Warning: Error during scaling: {e}. Attempting fallback method.")
                        try:
                            scaled_preds = model.scaler.inverse_transform(preds)
                        except Exception as e2:
                            print(f"Fallback failed: {e2}. Using unscaled predictions.")
                            scaled_preds = preds
                
                if isinstance(scaled_preds, dict):
                    # 如果是字典，提取预测值
                    for key, value in scaled_preds.items():
                        if hasattr(value, 'detach'):
                            all_preds.append(value.detach().cpu().numpy())
                            break  # 只取第一个可用的预测值
                        else:
                            all_preds.append(value)
                            break  # 只取第一个可用的预测值
                else:
                    all_preds.append(scaled_preds.detach().cpu().numpy())
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue  # 跳过错误的批次

        if not all_preds:
            print("Warning: No predictions were generated")
            return [None] * len(crystal_array_list)
        
        try:
            all_preds = np.concatenate(all_preds, axis=0)
            if all_preds.ndim > 1 and all_preds.shape[1] == 1:
                all_preds = all_preds.squeeze(1)
            
            # 将预测结果转回原始长度
            all_preds_list = all_preds.tolist()
            
            # 创建与原始输入长度相同的结果数组，未评估的结构用None填充
            full_results = [None] * len(crystal_array_list)
            for i, original_idx in enumerate(filtered_indices):
                if i < len(all_preds_list):
                    full_results[original_idx] = all_preds_list[i]
            
            # 过滤掉None值，只返回成功评估的结果
            valid_results = [r for r in full_results if r is not None]
            if len(valid_results) == 0:
                print("Warning: All evaluations resulted in None")
                return [None] * len(crystal_array_list)
            
            return valid_results
        except Exception as e:
            print(f"Error processing predictions: {e}")
            return [None] * len(crystal_array_list)
            
    except Exception as e:
        print(f"Exception in prop_model_eval: {str(e)}")
        import traceback
        traceback.print_exc()
        return [None] * len(crystal_array_list)


def filter_fps(struc_fps, comp_fps):
    assert len(struc_fps) == len(comp_fps)

    filtered_struc_fps, filtered_comp_fps = [], []

    for struc_fp, comp_fp in zip(struc_fps, comp_fps):
        if struc_fp is not None and comp_fp is not None:
            filtered_struc_fps.append(struc_fp)
            filtered_comp_fps.append(comp_fp)
    return filtered_struc_fps, filtered_comp_fps


def compute_cov(crys, gt_crys,
                struc_cutoff, comp_cutoff, num_gen_crystals=None):
    struc_fps = [c.struct_fp for c in crys]
    comp_fps = [c.comp_fp for c in crys]
    gt_struc_fps = [c.struct_fp for c in gt_crys]
    gt_comp_fps = [c.comp_fp for c in gt_crys]

    assert len(struc_fps) == len(comp_fps)
    assert len(gt_struc_fps) == len(gt_comp_fps)

    # Use number of crystal before filtering to compute COV
    if num_gen_crystals is None:
        num_gen_crystals = len(struc_fps)

    struc_fps, comp_fps = filter_fps(struc_fps, comp_fps)

    comp_fps = CompScaler.transform(comp_fps)
    gt_comp_fps = CompScaler.transform(gt_comp_fps)

    struc_fps = np.array(struc_fps)
    gt_struc_fps = np.array(gt_struc_fps)
    comp_fps = np.array(comp_fps)
    gt_comp_fps = np.array(gt_comp_fps)

    struc_pdist = cdist(struc_fps, gt_struc_fps)
    comp_pdist = cdist(comp_fps, gt_comp_fps)

    struc_recall_dist = struc_pdist.min(axis=0)
    struc_precision_dist = struc_pdist.min(axis=1)
    comp_recall_dist = comp_pdist.min(axis=0)
    comp_precision_dist = comp_pdist.min(axis=1)

    cov_recall = np.mean(np.logical_and(
        struc_recall_dist <= struc_cutoff,
        comp_recall_dist <= comp_cutoff))
    cov_precision = np.sum(np.logical_and(
        struc_precision_dist <= struc_cutoff,
        comp_precision_dist <= comp_cutoff)) / num_gen_crystals

    metrics_dict = {
        'cov_recall': cov_recall,
        'cov_precision': cov_precision,
        'amsd_recall': np.mean(struc_recall_dist),
        'amsd_precision': np.mean(struc_precision_dist),
        'amcd_recall': np.mean(comp_recall_dist),
        'amcd_precision': np.mean(comp_precision_dist),
    }

    combined_dist_dict = {
        'struc_recall_dist': struc_recall_dist.tolist(),
        'struc_precision_dist': struc_precision_dist.tolist(),
        'comp_recall_dist': comp_recall_dist.tolist(),
        'comp_precision_dist': comp_precision_dist.tolist(),
    }

    return metrics_dict, combined_dist_dict