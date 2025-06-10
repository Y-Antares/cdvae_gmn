from collections import Counter
import argparse
import os
import json
import warnings
warnings.filterwarnings('ignore', message='No Pauling electronegativity')

current_directory = os.getcwd()

print("当前默认路径:", current_directory)
import sys
# 将scripts目录添加到Python路径
sys.path.append(os.path.join(os.getcwd(), 'scripts'))


import numpy as np
from pathlib import Path
from tqdm import tqdm
from p_tqdm import p_map
from scipy.stats import wasserstein_distance

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty

from eval_utils import (
    smact_validity, structure_validity, CompScaler, get_fp_pdist,
    load_config, load_data, get_crystals_list, prop_model_eval, compute_cov)

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

# 目标属性的分位点
Percentiles = {
    'mp20': np.array([-3.17562208, -2.82196882, -2.52814761]),
    'carbon': np.array([-154.527093, -154.45865733, -154.44206825]),
    'perovskite': np.array([0.43924842, 0.61202443, 0.7364607]),
}

# 形成能的分位点 (可以根据你的数据集调整)
EnergyPercentiles = {
    'mp20': np.array([-0.5, -0.3, -0.1]),  # 示例值，需要根据实际数据替换
    'carbon': np.array([-0.5, -0.3, -0.1]), # 示例值，需要根据实际数据替换
    'perovskite': np.array([-0.5, -0.3, -0.1]), # 示例值，需要根据实际数据替换
}

COV_Cutoffs = {
    'mp20': {'struc': 0.4, 'comp': 10.},
    'carbon': {'struc': 0.2, 'comp': 4.},
    'perovskite': {'struc': 0.2, 'comp': 4},
}


class Crystal(object):

    def __init__(self, crys_array_dict):
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.dict = crys_array_dict

        self.get_structure()
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()

    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = 'unrealistically_small_lattice'

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [CrystalNNFP.featurize(
                self.structure, i) for i in range(len(self.structure))]
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)


class RecEval(object):

    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3):
        assert len(pred_crys) == len(gt_crys)
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys

    def get_match_rate_and_rms(self):
        def process_one(pred, gt, is_valid):
            if not is_valid:
                return None
            try:
                rms_dist = self.matcher.get_rms_dist(
                    pred.structure, gt.structure)
                rms_dist = None if rms_dist is None else rms_dist[0]
                return rms_dist
            except Exception:
                return None
        validity = [c.valid for c in self.preds]

        rms_dists = []
        for i in tqdm(range(len(self.preds))):
            rms_dists.append(process_one(
                self.preds[i], self.gts[i], validity[i]))
        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds)
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        return {'match_rate': match_rate,
                'rms_dist': mean_rms_dist}

    def get_metrics(self):
        return self.get_match_rate_and_rms()


class GenEval(object):

    def __init__(self, pred_crys, gt_crys, n_samples=1000, eval_model_name=None):
        self.crys = pred_crys
        self.gt_crys = gt_crys
        self.n_samples = n_samples
        self.eval_model_name = eval_model_name

        valid_crys = [c for c in pred_crys if c.valid]
        if len(valid_crys) >= n_samples:
            sampled_indices = np.random.choice(
                len(valid_crys), n_samples, replace=False)
            self.valid_samples = [valid_crys[i] for i in sampled_indices]
        else:
            raise Exception(
                f'not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}')

    def get_validity(self):
        comp_valid = np.array([c.comp_valid for c in self.crys]).mean()
        struct_valid = np.array([c.struct_valid for c in self.crys]).mean()
        valid = np.array([c.valid for c in self.crys]).mean()
        return {'comp_valid': comp_valid,
                'struct_valid': struct_valid,
                'valid': valid}

    def get_comp_diversity(self):
        comp_fps = [c.comp_fp for c in self.valid_samples]
        comp_fps = CompScaler.transform(comp_fps)
        comp_div = get_fp_pdist(comp_fps)
        return {'comp_div': comp_div}

    def get_struct_diversity(self):
        return {'struct_div': get_fp_pdist([c.struct_fp for c in self.valid_samples])}

    def get_density_wdist(self):
        pred_densities = [c.structure.density for c in self.valid_samples]
        gt_densities = [c.structure.density for c in self.gt_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {'wdist_density': wdist_density}

    def get_num_elem_wdist(self):
        pred_nelems = [len(set(c.structure.species))
                       for c in self.valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.gt_crys]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {'wdist_num_elems': wdist_num_elems}

    # 在GenEval类的get_prop_wdist方法中进行修改
    def get_prop_wdist(self):
        if self.eval_model_name is not None:
            # 过滤掉原子数超过30的结构
            valid_samples_filtered = []
            for crystal in self.valid_samples:
                num_atoms = len(crystal.atom_types)
                if num_atoms <= 30:  # 设定最大支持30个原子
                    valid_samples_filtered.append(crystal)
            
            gt_crys_filtered = []
            for crystal in self.gt_crys:
                num_atoms = len(crystal.atom_types)
                if num_atoms <= 30:  # 设定最大支持30个原子
                    gt_crys_filtered.append(crystal)
            
            if len(valid_samples_filtered) == 0 or len(gt_crys_filtered) == 0:
                print(f"Warning: After filtering, no valid structures left. Original counts: valid_samples={len(self.valid_samples)}, gt_crys={len(self.gt_crys)}")
                return {'wdist_prop': None}
            
            print(f"Filtered structures: kept {len(valid_samples_filtered)}/{len(self.valid_samples)} generated and {len(gt_crys_filtered)}/{len(self.gt_crys)} ground truth")
            
            pred_props = prop_model_eval(self.eval_model_name, [c.dict for c in valid_samples_filtered])
            gt_props = prop_model_eval(self.eval_model_name, [c.dict for c in gt_crys_filtered])
            
            # 确保得到有效的预测结果
            if pred_props is None or gt_props is None or len(pred_props) == 0 or len(gt_props) == 0:
                print("Warning: Empty prediction results after evaluation")
                return {'wdist_prop': None}
            
            # 过滤掉None值
            pred_props = [p for p in pred_props if p is not None]
            gt_props = [p for p in gt_props if p is not None]
            
            if len(pred_props) == 0 or len(gt_props) == 0:
                print("Warning: No valid property predictions after filtering None values")
                return {'wdist_prop': None}
            
            wdist_prop = wasserstein_distance(pred_props, gt_props)
            return {'wdist_prop': wdist_prop}
        else:
            return {'wdist_prop': None}

    def get_coverage(self):
        cutoff_dict = COV_Cutoffs[self.eval_model_name]
        (cov_metrics_dict, combined_dist_dict) = compute_cov(
            self.crys, self.gt_crys,
            struc_cutoff=cutoff_dict['struc'],
            comp_cutoff=cutoff_dict['comp'])
        return cov_metrics_dict

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_validity())
        metrics.update(self.get_comp_diversity())
        metrics.update(self.get_struct_diversity())
        metrics.update(self.get_density_wdist())
        metrics.update(self.get_num_elem_wdist())
        metrics.update(self.get_prop_wdist())
        print(metrics)
        metrics.update(self.get_coverage())
        return metrics


class OptEval(object):

    def __init__(self, crys, num_opt=100, eval_model_name=None, target_type='combined', optimization_method='weighted'):
        """
        crys is a list of length (<step_opt> * <num_opt>),
        where <num_opt> is the number of different initialization for optimizing crystals,
        and <step_opt> is the number of saved crystals for each intialzation.
        
        参数:
            crys: 晶体列表
            num_opt: 不同初始化的晶体数量
            eval_model_name: 评估模型名称
            target_type: 优化目标类型，可以是 'combined', 'energy', 'property', 'weighted'
            optimization_method: 优化方法，可以是 'weighted', 'tchebycheff', 'boundary'
        """
        # 检查晶体数量是否能被num_opt整除，如果不能，调整num_opt
        total_crys = len(crys)
        if total_crys % num_opt != 0:
            # 查找最接近的能够整除总数的数字
            for possible_num_opt in [num_opt-1, num_opt+1, num_opt-2, num_opt+2]:
                if possible_num_opt > 0 and total_crys % possible_num_opt == 0:
                    print(f"Warning: Adjusting num_opt from {num_opt} to {possible_num_opt} to match crystal count ({total_crys})")
                    num_opt = possible_num_opt
                    break
            # 如果找不到合适的值，使用最大公约数
            if total_crys % num_opt != 0:
                import math
                factors = []
                for i in range(1, int(math.sqrt(total_crys)) + 1):
                    if total_crys % i == 0:
                        factors.append(i)
                        factors.append(total_crys // i)
                factors.sort()
                # 找到最接近原始num_opt的因数
                closest_factor = min(factors, key=lambda x: abs(x - num_opt))
                print(f"Warning: Adjusting num_opt from {num_opt} to {closest_factor} to match crystal count ({total_crys})")
                num_opt = closest_factor
                
        step_opt = total_crys // num_opt
        print(f"OptEval: total_crys={total_crys}, step_opt={step_opt}, num_opt={num_opt}")
        
        self.crys = crys
        self.step_opt = step_opt
        self.num_opt = num_opt
        self.eval_model_name = eval_model_name
        self.target_type = target_type
        self.optimization_method = optimization_method  # 新增参数

    def get_success_rate(self):
        valid_indices = np.array([c.valid for c in self.crys])
        
        # 检查数组大小是否匹配
        if len(valid_indices) != self.step_opt * self.num_opt:
            print(f"Warning: Array size mismatch. valid_indices size: {len(valid_indices)}, expected: {self.step_opt * self.num_opt}")
            # 截断或填充数组以匹配预期大小
            if len(valid_indices) > self.step_opt * self.num_opt:
                valid_indices = valid_indices[:self.step_opt * self.num_opt]
                print(f"Truncated valid_indices to size {len(valid_indices)}")
            else:
                # 填充额外的False值
                padding = np.zeros(self.step_opt * self.num_opt - len(valid_indices), dtype=bool)
                valid_indices = np.concatenate([valid_indices, padding])
                print(f"Padded valid_indices to size {len(valid_indices)}")
        
        try:
            valid_indices = valid_indices.reshape(self.step_opt, self.num_opt)
        except ValueError as e:
            print(f"Reshape error: {e}")
            print(f"Actual array shape: {valid_indices.shape}, trying to reshape to: ({self.step_opt}, {self.num_opt})")
            # 尝试调整step_opt
            if len(valid_indices) % self.num_opt == 0:
                self.step_opt = len(valid_indices) // self.num_opt
                print(f"Adjusted step_opt to {self.step_opt}")
                valid_indices = valid_indices.reshape(self.step_opt, self.num_opt)
            else:
                # 如果无法调整，可能需要裁剪数组
                trim_size = (len(valid_indices) // self.num_opt) * self.num_opt
                valid_indices = valid_indices[:trim_size]
                self.step_opt = len(valid_indices) // self.num_opt
                print(f"Trimmed array to size {len(valid_indices)} and adjusted step_opt to {self.step_opt}")
                valid_indices = valid_indices.reshape(self.step_opt, self.num_opt)
        
        valid_x, valid_y = valid_indices.nonzero()
        target_props = np.ones([self.step_opt, self.num_opt]) * np.inf
        energy_props = np.ones([self.step_opt, self.num_opt]) * np.inf
        
        valid_crys = [c for c in self.crys if c.valid]
        if len(valid_crys) == 0:
            target_sr_5, target_sr_10, target_sr_15 = 0, 0, 0
            energy_sr_5, energy_sr_10, energy_sr_15 = 0, 0, 0
        else:
            # 过滤掉原子数超过30的结构
            valid_crys_filtered = []
            for crystal in valid_crys:
                num_atoms = len(crystal.atom_types)
                if num_atoms <= 30:  # 设定最大支持30个原子
                    valid_crys_filtered.append(crystal)
            
            print(f"Optimization evaluation: kept {len(valid_crys_filtered)}/{len(valid_crys)} valid structures after atom count filtering")
            
            if len(valid_crys_filtered) == 0:
                target_sr_5, target_sr_10, target_sr_15 = 0, 0, 0
                energy_sr_5, energy_sr_10, energy_sr_15 = 0, 0, 0
            else:
                try:
                    # 进行属性预测
                    pred_results = prop_model_eval(self.eval_model_name, [c.dict for c in valid_crys_filtered])
                    
                    # 确保我们有足够的预测结果
                    if pred_results is None or len(pred_results) == 0 or all(p is None for p in pred_results):
                        print("Warning: No valid property predictions returned")
                        target_sr_5, target_sr_10, target_sr_15 = 0, 0, 0
                        energy_sr_5, energy_sr_10, energy_sr_15 = 0, 0, 0
                    else:
                        # 过滤掉None值
                        pred_results = [p for p in pred_results if p is not None]
                        
                        # 检查是否为多目标预测结果
                        multi_target = False
                        if len(pred_results) > 0:
                            if isinstance(pred_results[0], list) or (isinstance(pred_results[0], np.ndarray) and pred_results[0].size > 1):
                                multi_target = True
                                print("Detected multi-target prediction results")
                        
                        if multi_target:
                            # 多目标预测处理
                            valid_count = min(len(pred_results), len(valid_x))
                            for i in range(valid_count):
                                # 假设第一个元素是形成能，第二个是目标属性
                                energy_props[valid_x[i], valid_y[i]] = pred_results[i][0]
                                target_props[valid_x[i], valid_y[i]] = pred_results[i][1]
                        else:
                            # 单目标预测处理
                            valid_count = min(len(pred_results), len(valid_x))
                            for i in range(valid_count):
                                target_props[valid_x[i], valid_y[i]] = pred_results[i]
                                # 形成能值不可用，设为默认值
                                energy_props[valid_x[i], valid_y[i]] = 0
                        
                        # 根据优化目标类型计算成功率
                        if self.target_type == 'energy':
                            # 只优化形成能
                            best_props = energy_props.min(axis=0)
                            energy_percentiles = EnergyPercentiles.get(self.eval_model_name, np.array([0.0, 0.0, 0.0]))
                            target_sr_5 = (best_props <= energy_percentiles[0]).mean()
                            target_sr_10 = (best_props <= energy_percentiles[1]).mean()
                            target_sr_15 = (best_props <= energy_percentiles[2]).mean()
                            
                            # 目标属性的成功率设为0（因为没有优化）
                            energy_sr_5, energy_sr_10, energy_sr_15 = target_sr_5, target_sr_10, target_sr_15
                            target_sr_5, target_sr_10, target_sr_15 = 0, 0, 0
                        elif self.target_type == 'property':
                            # 只优化目标属性
                            best_props = target_props.min(axis=0)
                            target_percentiles = Percentiles.get(self.eval_model_name, np.array([0.0, 0.0, 0.0]))
                            target_sr_5 = (best_props <= target_percentiles[0]).mean()
                            target_sr_10 = (best_props <= target_percentiles[1]).mean()
                            target_sr_15 = (best_props <= target_percentiles[2]).mean()
                            
                            # 形成能的成功率设为0（因为没有优化）
                            energy_sr_5, energy_sr_10, energy_sr_15 = 0, 0, 0
                        elif multi_target:
                            # 计算两个目标的成功率
                            best_energy_props = energy_props.min(axis=0)
                            energy_percentiles = EnergyPercentiles.get(self.eval_model_name, np.array([0.0, 0.0, 0.0]))
                            energy_sr_5 = (best_energy_props <= energy_percentiles[0]).mean()
                            energy_sr_10 = (best_energy_props <= energy_percentiles[1]).mean()
                            energy_sr_15 = (best_energy_props <= energy_percentiles[2]).mean()
                            
                            best_target_props = target_props.min(axis=0)
                            target_percentiles = Percentiles.get(self.eval_model_name, np.array([0.0, 0.0, 0.0]))
                            target_sr_5 = (best_target_props <= target_percentiles[0]).mean()
                            target_sr_10 = (best_target_props <= target_percentiles[1]).mean()
                            target_sr_15 = (best_target_props <= target_percentiles[2]).mean()
                        else:
                            # 单目标情况，只有目标属性
                            best_props = target_props.min(axis=0)
                            target_percentiles = Percentiles.get(self.eval_model_name, np.array([0.0, 0.0, 0.0]))
                            target_sr_5 = (best_props <= target_percentiles[0]).mean()
                            target_sr_10 = (best_props <= target_percentiles[1]).mean()
                            target_sr_15 = (best_props <= target_percentiles[2]).mean()
                            energy_sr_5, energy_sr_10, energy_sr_15 = 0, 0, 0
                except Exception as e:
                    print(f"Error during property evaluation: {e}")
                    import traceback
                    traceback.print_exc()
                    target_sr_5, target_sr_10, target_sr_15 = 0, 0, 0
                    energy_sr_5, energy_sr_10, energy_sr_15 = 0, 0, 0
        
        # 构建指标字典，添加优化方法前缀
        metrics = {}
        method_prefix = f"{self.optimization_method.capitalize()}_" if self.optimization_method != 'weighted' else ""
        
        # 返回多目标优化评估结果
        if self.target_type == 'energy':
            metrics.update({
                f'{method_prefix}Energy_SR5': energy_sr_5, 
                f'{method_prefix}Energy_SR10': energy_sr_10, 
                f'{method_prefix}Energy_SR15': energy_sr_15
            })
        elif self.target_type == 'property':
            metrics.update({
                f'{method_prefix}SR5': target_sr_5, 
                f'{method_prefix}SR10': target_sr_10, 
                f'{method_prefix}SR15': target_sr_15
            })
        else:  # combined 或 weighted
            metrics.update({
                f'{method_prefix}SR5': target_sr_5, 
                f'{method_prefix}SR10': target_sr_10, 
                f'{method_prefix}SR15': target_sr_15,
                f'{method_prefix}Energy_SR5': energy_sr_5, 
                f'{method_prefix}Energy_SR10': energy_sr_10, 
                f'{method_prefix}Energy_SR15': energy_sr_15
            })
            
        return metrics

    def get_metrics(self):
        return self.get_success_rate()


def get_file_paths(root_path, task, label='', suffix='pt', target_type=None, optimization_method=None):
    """获取评估文件路径，支持不同优化方法"""
    if task == 'opt':
        # 构建文件名
        parts = [f'eval_{task}']
        
        if label:
            parts.append(label)
            
        if target_type:
            parts.append(target_type)
            
        if optimization_method and optimization_method != 'weighted':
            parts.append(optimization_method)
            
        out_name = f"{'_'.join(parts)}.{suffix}"
    else:
        if label == '':
            out_name = f'eval_{task}.{suffix}'
        else:
            out_name = f'eval_{task}_{label}.{suffix}'
            
    out_name = os.path.join(root_path, out_name)
    return out_name


def get_crystal_array_list(file_path, batch_idx=0):
    data = load_data(file_path)
    crys_array_list = get_crystals_list(
        data['frac_coords'][batch_idx],
        data['atom_types'][batch_idx],
        data['lengths'][batch_idx],
        data['angles'][batch_idx],
        data['num_atoms'][batch_idx])

    if 'input_data_batch' in data:
        batch = data['input_data_batch']
        if isinstance(batch, dict):
            true_crystal_array_list = get_crystals_list(
                batch['frac_coords'], batch['atom_types'], batch['lengths'],
                batch['angles'], batch['num_atoms'])
        else:
            true_crystal_array_list = get_crystals_list(
                batch.frac_coords, batch.atom_types, batch.lengths,
                batch.angles, batch.num_atoms)
    else:
        true_crystal_array_list = None

    return crys_array_list, true_crystal_array_list


def main(args):
    all_metrics = {}

    cfg = load_config(args.root_path)
    eval_model_name = cfg.data.eval_model_name

    if 'recon' in args.tasks:
        recon_file_path = get_file_paths(args.root_path, 'recon', args.label)
        crys_array_list, true_crystal_array_list = get_crystal_array_list(
            recon_file_path)
        pred_crys = p_map(lambda x: Crystal(x), crys_array_list)
        gt_crys = p_map(lambda x: Crystal(x), true_crystal_array_list)

        rec_evaluator = RecEval(pred_crys, gt_crys)
        recon_metrics = rec_evaluator.get_metrics()
        all_metrics.update(recon_metrics)

    if 'gen' in args.tasks:
        gen_file_path = get_file_paths(args.root_path, 'gen', args.label)
        recon_file_path = get_file_paths(args.root_path, 'recon', args.label)
        crys_array_list, _ = get_crystal_array_list(gen_file_path)
        gen_crys = p_map(lambda x: Crystal(x), crys_array_list)
        if 'recon' not in args.tasks:
            _, true_crystal_array_list = get_crystal_array_list(
                recon_file_path)
            gt_crys = p_map(lambda x: Crystal(x), true_crystal_array_list)

        gen_evaluator = GenEval(
            gen_crys, gt_crys, eval_model_name=eval_model_name)
        gen_metrics = gen_evaluator.get_metrics()
        all_metrics.update(gen_metrics)

    if 'opt' in args.tasks:
        # 检查是否有指定优化目标类型和优化方法
        target_type = getattr(args, 'target_type', 'combined')
        optimization_method = getattr(args, 'optimization_method', 'weighted')
        
        # 处理不同优化目标类型和优化方法的评估
        opt_file_path = get_file_paths(
            args.root_path, 'opt', args.label, 
            target_type=target_type, 
            optimization_method=optimization_method
        )
        
        print(f"Evaluating optimization for target type: {target_type}, method: {optimization_method}, file: {opt_file_path}")
        
        try:
            data = load_data(opt_file_path)
            # 从结果中获取实际使用的优化方法（如果存在）
            if 'optimization_method' in data:
                actual_method = data['optimization_method']
                if actual_method != optimization_method:
                    print(f"Note: File was optimized using '{actual_method}' method, not '{optimization_method}'")
                    optimization_method = actual_method
            
            crys_array_list, _ = get_crystal_array_list(opt_file_path)
            opt_crys = p_map(lambda x: Crystal(x), crys_array_list)
            
            # 获取估计的优化步数
            if 'eval_setting' in data and hasattr(data['eval_setting'], 'num_saved_crys'):
                num_saved_crys = data['eval_setting'].num_saved_crys
            else:
                num_saved_crys = 10  # 默认值
                
            opt_evaluator = OptEval(
                opt_crys, num_opt=num_saved_crys, 
                eval_model_name=eval_model_name,
                target_type=target_type,
                optimization_method=optimization_method
            )
            
            opt_metrics = opt_evaluator.get_metrics()
            all_metrics.update(opt_metrics)
            
        except FileNotFoundError:
            print(f"Warning: Optimization file not found: {opt_file_path}")
            print("Trying fallback to generic filename...")
            
            # 尝试使用不含优化方法的文件名
            fallback_path = get_file_paths(
                args.root_path, 'opt', args.label, 
                target_type=target_type
            )
            
            if fallback_path != opt_file_path and os.path.exists(fallback_path):
                print(f"Found fallback file: {fallback_path}")
                try:
                    crys_array_list, _ = get_crystal_array_list(fallback_path)
                    opt_crys = p_map(lambda x: Crystal(x), crys_array_list)
                    
                    # 获取估计的优化步数
                    data = load_data(fallback_path)
                    if 'eval_setting' in data and hasattr(data['eval_setting'], 'num_saved_crys'):
                        num_saved_crys = data['eval_setting'].num_saved_crys
                    else:
                        num_saved_crys = 10  # 默认值
                        
                    opt_evaluator = OptEval(
                        opt_crys, num_opt=num_saved_crys, 
                        eval_model_name=eval_model_name,
                        target_type=target_type,
                        optimization_method='weighted'  # 使用默认方法
                    )
                    
                    opt_metrics = opt_evaluator.get_metrics()
                    all_metrics.update(opt_metrics)
                except Exception as e:
                    print(f"Error processing fallback file: {e}")
            else:
                print(f"No suitable optimization file found for {target_type} with {optimization_method} method")

    print('Metrics:')
    for k, v in all_metrics.items():
        print(f'    {k}: {v}')

    # Save metrics
    if args.label == '':
        if hasattr(args, 'optimization_method') and args.optimization_method != 'weighted':
            metrics_file = os.path.join(args.root_path, f'metrics_{args.optimization_method}.json')
        else:
            metrics_file = os.path.join(args.root_path, 'metrics.json')
    else:
        if hasattr(args, 'optimization_method') and args.optimization_method != 'weighted':
            metrics_file = os.path.join(args.root_path, f'metrics_{args.label}_{args.optimization_method}.json')
        else:
            metrics_file = os.path.join(args.root_path, f'metrics_{args.label}.json')

    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--tasks', nargs='+', default=['recon', 'gen', 'opt'])
    parser.add_argument('--label', default='')
    # 添加优化目标类型参数
    parser.add_argument('--target_type', default='combined', type=str,
                        choices=['combined', 'energy', 'property', 'weighted'],
                        help='优化目标类型: combined(综合), energy(形成能), property(目标属性), weighted(加权)')
    # 添加优化方法参数
    parser.add_argument('--optimization_method', default='weighted', type=str,
                        choices=['weighted', 'tchebycheff', 'boundary'],
                        help='优化方法: weighted(线性加权), tchebycheff(切比雪夫), boundary(边界交叉法)')
    
    args = parser.parse_args()
    main(args)