import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch

from eval_utils import load_model


def reconstructon(loader, model, ld_kwargs, num_evals,
                  force_num_atoms=False, force_atom_types=False, down_sample_traj_step=1):
    """
    reconstruct the crystals in <loader>.
    """
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []
    input_data_list = []

    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()
        print(f'batch {idx} in {len(loader)}')
        batch_all_frac_coords = []
        batch_all_atom_types = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

        # only sample one z, multiple evals for stoichaticity in langevin dynamics
        _, _, z = model.encode(batch)

        for eval_idx in range(num_evals):
            gt_num_atoms = batch.num_atoms if force_num_atoms else None
            gt_atom_types = batch.atom_types if force_atom_types else None
            outputs = model.langevin_dynamics(
                z, ld_kwargs, gt_num_atoms, gt_atom_types)

            # collect sampled crystals in this batch.
            batch_frac_coords.append(outputs['frac_coords'].detach().cpu())
            batch_num_atoms.append(outputs['num_atoms'].detach().cpu())
            batch_atom_types.append(outputs['atom_types'].detach().cpu())
            batch_lengths.append(outputs['lengths'].detach().cpu())
            batch_angles.append(outputs['angles'].detach().cpu())
            if ld_kwargs.save_traj:
                batch_all_frac_coords.append(
                    outputs['all_frac_coords'][::down_sample_traj_step].detach().cpu())
                batch_all_atom_types.append(
                    outputs['all_atom_types'][::down_sample_traj_step].detach().cpu())
        # collect sampled crystals for this z.
        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lengths.append(torch.stack(batch_lengths, dim=0))
        angles.append(torch.stack(batch_angles, dim=0))
        if ld_kwargs.save_traj:
            all_frac_coords_stack.append(
                torch.stack(batch_all_frac_coords, dim=0))
            all_atom_types_stack.append(
                torch.stack(batch_all_atom_types, dim=0))
        # Save the ground truth structure
        input_data_list = input_data_list + batch.to_data_list()

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    input_data_batch = Batch.from_data_list(input_data_list)

    return (
        frac_coords, num_atoms, atom_types, lengths, angles,
        all_frac_coords_stack, all_atom_types_stack, input_data_batch)


def generation(model, ld_kwargs, num_batches_to_sample, num_samples_per_z,
               batch_size=512, down_sample_traj_step=1):
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []

    for z_idx in range(num_batches_to_sample):
        batch_all_frac_coords = []
        batch_all_atom_types = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

        z = torch.randn(batch_size, model.hparams.hidden_dim,
                        device=model.device)

        for sample_idx in range(num_samples_per_z):
            samples = model.langevin_dynamics(z, ld_kwargs)

            # collect sampled crystals in this batch.
            batch_frac_coords.append(samples['frac_coords'].detach().cpu())
            batch_num_atoms.append(samples['num_atoms'].detach().cpu())
            batch_atom_types.append(samples['atom_types'].detach().cpu())
            batch_lengths.append(samples['lengths'].detach().cpu())
            batch_angles.append(samples['angles'].detach().cpu())
            if ld_kwargs.save_traj:
                batch_all_frac_coords.append(
                    samples['all_frac_coords'][::down_sample_traj_step].detach().cpu())
                batch_all_atom_types.append(
                    samples['all_atom_types'][::down_sample_traj_step].detach().cpu())

        # collect sampled crystals for this z.
        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lengths.append(torch.stack(batch_lengths, dim=0))
        angles.append(torch.stack(batch_angles, dim=0))
        if ld_kwargs.save_traj:
            all_frac_coords_stack.append(
                torch.stack(batch_all_frac_coords, dim=0))
            all_atom_types_stack.append(
                torch.stack(batch_all_atom_types, dim=0))

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    return (frac_coords, num_atoms, atom_types, lengths, angles,
            all_frac_coords_stack, all_atom_types_stack)

def optimization(model, ld_kwargs, data_loader,
                 num_starting_points=100, num_gradient_steps=5000,
                 lr=1e-3, num_saved_crys=10, target_type='combined'):
    """
    执行属性优化，支持针对不同目标类型和优化方法的优化。
    
    参数:
        model: 模型实例
        ld_kwargs: langevin dynamics参数
        data_loader: 数据加载器
        num_starting_points: 起始点数量
        num_gradient_steps: 梯度步数
        lr: 学习率
        num_saved_crys: 保存的晶体数量
        target_type: 优化目标类型，可选值有：
            - 'combined': 综合优化（默认）
            - 'energy': 只优化形成能
            - 'property': 只优化目标属性
            - 'weighted': 使用模型中设置的权重进行加权优化
    """
    if data_loader is not None:
        batch = next(iter(data_loader)).to(model.device)
        _, _, z = model.encode(batch)
        z = z[:num_starting_points].detach().clone()
        z.requires_grad = True
    else:
        z = torch.randn(num_starting_points, model.hparams.hidden_dim,
                        device=model.device)
        z.requires_grad = True

    opt = Adam([z], lr=lr)
    model.freeze()

    # 检查模型是否支持多目标优化
    multi_target = hasattr(model, 'fc_property_shared') and model.fc_property_shared is not None
    
    # 获取优化方法
    optimization_method = getattr(model, 'optimization_method', 'weighted')
    print(f"使用优化方法: {optimization_method}")
    
    # 获取优化方向，确保它是一个有效的值
    optimization_direction = getattr(model, 'optimization_direction', ['min', 'min'])
    if not isinstance(optimization_direction, list):
        optimization_direction = ['min', 'min']
    print(f"优化方向: {optimization_direction}")
    
    # 初始化理想点（如果使用Tchebycheff或边界交叉法）
    if optimization_method in ['tchebycheff', 'boundary'] and not hasattr(model, 'ideal_points'):
        # 尝试从配置中获取初始理想点
        init_ideal_points = getattr(model.hparams, 'init_ideal_points', [float('inf'), float('inf')])
        model.register_buffer('ideal_points', torch.tensor(init_ideal_points, device=model.device))
        print(f"初始化理想点: {model.ideal_points}")
    
    all_crystals = []
    all_properties = []  # 存储属性值
    interval = num_gradient_steps // (num_saved_crys-1)
    
    for i in tqdm(range(num_gradient_steps)):
        opt.zero_grad()
        
        if multi_target:
            # 使用共享特征层
            shared_features = model.fc_property_shared(z)
            
            # 计算能量和目标属性
            energy_pred = model.energy_head(shared_features)
            target_pred = model.target_head(shared_features)
            
            # 根据优化方向调整目标属性
            target_pred_for_opt = target_pred
            if len(optimization_direction) > 1 and optimization_direction[1] == 'max':
                target_pred_for_opt = -target_pred  # 翻转最大化目标
                print(f"目标属性最大化: {i==0}") if i == 0 else None
            
            # 更新理想点（对于Tchebycheff和边界交叉法）
            if optimization_method in ['tchebycheff', 'boundary']:
                with torch.no_grad():
                    current_min_energy = energy_pred.min().item()
                    current_min_target = target_pred_for_opt.min().item()
                    model.ideal_points[0] = min(model.ideal_points[0].item(), current_min_energy)
                    model.ideal_points[1] = min(model.ideal_points[1].item(), current_min_target)
                    
                    if i == 0 or i == num_gradient_steps - 1:
                        print(f"当前理想点: {model.ideal_points}")
            
            # 根据不同的优化目标类型和优化方法计算损失
            if target_type == 'energy':
                # 只优化形成能
                loss = energy_pred.mean()
            elif target_type == 'property':
                # 只优化目标属性
                loss = target_pred_for_opt.mean()
            elif optimization_method == 'tchebycheff':
                # 使用Tchebycheff分解方法
                weights = getattr(model, 'property_weights', [0.5, 0.5])
                if not isinstance(weights, torch.Tensor):
                    weights = torch.tensor(weights, device=energy_pred.device)
                
                # 计算每个目标与理想点的加权距离
                energy_term = weights[0] * torch.abs(energy_pred - model.ideal_points[0])
                target_term = weights[1] * torch.abs(target_pred_for_opt - model.ideal_points[1])
                
                # 每个样本取最大值，然后求平均
                loss = torch.max(torch.stack([energy_term, target_term], dim=1), dim=1)[0].mean()
                if i == 0:
                    print(f"使用Tchebycheff分解法，权重: {weights}")
            elif optimization_method == 'boundary':
                # 使用边界交叉法
                weights = getattr(model, 'property_weights', [0.5, 0.5])
                if not isinstance(weights, torch.Tensor):
                    weights = torch.tensor(weights, device=energy_pred.device)
                    
                theta = getattr(model, 'boundary_theta', 5.0)
                
                # 构建当前解向量和理想点向量
                f_z = torch.cat([energy_pred, target_pred_for_opt], dim=1)
                z_star = model.ideal_points.to(f_z.device)
                
                # 计算向量差
                diff = f_z - z_star
                
                # 计算范数
                norm = torch.norm(diff, dim=1)
                
                # 计算夹角余弦值
                lambda_norm = torch.sqrt(weights[0]**2 + weights[1]**2)
                cos_angle = (weights[0] * diff[:, 0] + weights[1] * diff[:, 1]) / (norm * lambda_norm + 1e-8)
                
                # 计算d1和d2
                d1 = norm * cos_angle
                d2 = norm * torch.sqrt(1 - cos_angle**2 + 1e-8)
                
                loss = (d1 + theta * d2).mean()
                if i == 0:
                    print(f"使用边界交叉法，权重: {weights}, theta: {theta}")
            else:
                # 默认：线性加权
                if target_type == 'weighted' and hasattr(model, 'property_weights'):
                    # 使用模型中的权重
                    weights = model.property_weights
                    if not isinstance(weights, torch.Tensor):
                        weights = torch.tensor(weights, device=energy_pred.device)
                    loss = weights[0] * energy_pred.mean() + weights[1] * target_pred_for_opt.mean()
                    if i == 0:
                        print(f"使用加权法，权重: {weights}")
                else:
                    # 默认等权重
                    loss = 0.5 * energy_pred.mean() + 0.5 * target_pred_for_opt.mean()
                    if i == 0:
                        print("使用等权重加权法")
        else:
            # 原始单目标优化
            loss = model.fc_property(z).mean()
            if i == 0:
                print("使用单目标优化")
            
        loss.backward()
        opt.step()

        if i % interval == 0 or i == (num_gradient_steps-1):
            crystals = model.langevin_dynamics(z, ld_kwargs)
            all_crystals.append(crystals)
            
            # 添加这部分：预测每个晶体的属性值
            with torch.no_grad():
                if multi_target:
                    # 获取共享特征
                    shared_features = model.fc_property_shared(z)
                    
                    # 计算能量和目标属性
                    energy_pred = model.energy_head(shared_features)
                    target_pred = model.target_head(shared_features)
                    
                    # 应用逆变换（如果有标准化器）
                    if hasattr(model, 'energy_scaler') and model.energy_scaler is not None:
                        model.energy_scaler.match_device(energy_pred)
                        energy_pred = model.energy_scaler.inverse_transform(energy_pred)
                        
                    if hasattr(model, 'scaler') and model.scaler is not None:
                        model.scaler.match_device(target_pred)
                        target_pred = model.scaler.inverse_transform(target_pred)
                    
                    # 合并属性
                    properties = torch.cat([energy_pred, target_pred], dim=1).cpu()
                    all_properties.append(properties)
                else:
                    # 原始单目标预测
                    if hasattr(model, 'fc_property'):
                        props = model.fc_property(z)
                        if hasattr(model, 'scaler') and model.scaler is not None:
                            model.scaler.match_device(props)
                            props = model.scaler.inverse_transform(props)
                        all_properties.append(props.cpu())
                    else:
                        print("警告：模型没有fc_property属性，跳过属性预测")
    
    # 构建基本结果
    result = {k: torch.cat([d[k] for d in all_crystals]).unsqueeze(0) for k in
              ['frac_coords', 'atom_types', 'num_atoms', 'lengths', 'angles']}
    
    # 添加属性到结果
    if all_properties:
        properties_tensor = torch.cat(all_properties, dim=0)
        result['properties'] = properties_tensor.unsqueeze(0)
        print(f"已添加属性信息，形状：{result['properties'].shape}")
        
    # 添加优化方法信息
    result['optimization_method'] = optimization_method
    result['optimization_direction'] = optimization_direction
    
    return result

def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, test_loader, cfg = load_model(
        model_path, load_data=('recon' in args.tasks) or
        ('opt' in args.tasks and args.start_from == 'data'))
        
    # 设置优化方法和优化方向
    if 'opt' in args.tasks:
        # 从命令行参数设置优化方法（如果指定）
        if args.optimization_method is not None:
            model.optimization_method = args.optimization_method
            print(f"从命令行设置优化方法: {model.optimization_method}")
        # 否则从配置文件中获取
        elif hasattr(cfg.data, 'optimization_method'):
            model.optimization_method = cfg.data.optimization_method
            print(f"从配置文件设置优化方法: {model.optimization_method}")
            
        # 从配置文件获取优化方向
        if hasattr(cfg.data, 'optimization_direction'):
            model.optimization_direction = cfg.data.optimization_direction
            print(f"从配置文件设置优化方向: {model.optimization_direction}")
        
        # 获取边界交叉法的theta参数（如果存在）
        if hasattr(cfg.data, 'boundary_theta'):
            model.boundary_theta = cfg.data.boundary_theta
            print(f"从配置文件设置boundary_theta: {model.boundary_theta}")
            
        # 获取初始理想点（如果存在）
        if hasattr(cfg.data, 'init_ideal_points'):
            model.init_ideal_points = cfg.data.init_ideal_points
            print(f"从配置文件设置初始理想点: {model.init_ideal_points}")
        
        # 确保模型有多目标优化所需的权重属性
        if hasattr(cfg.data, 'property_weights'):
            model.property_weights = cfg.data.property_weights
            print(f"从配置文件设置多目标权重: {model.property_weights}")
            
    ld_kwargs = SimpleNamespace(n_step_each=args.n_step_each,
                                step_lr=args.step_lr,
                                min_sigma=args.min_sigma,
                                save_traj=args.save_traj,
                                disable_bar=args.disable_bar)

    if torch.cuda.is_available():
        model.to('cuda')

    if 'recon' in args.tasks:
        print('Evaluate model on the reconstruction task.')
        start_time = time.time()
        (frac_coords, num_atoms, atom_types, lengths, angles,
         all_frac_coords_stack, all_atom_types_stack, input_data_batch) = reconstructon(
            test_loader, model, ld_kwargs, args.num_evals,
            args.force_num_atoms, args.force_atom_types, args.down_sample_traj_step)

        if args.label == '':
            recon_out_name = 'eval_recon.pt'
        else:
            recon_out_name = f'eval_recon_{args.label}.pt'

        torch.save({
            'eval_setting': args,
            'input_data_batch': input_data_batch,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles,
            'all_frac_coords_stack': all_frac_coords_stack,
            'all_atom_types_stack': all_atom_types_stack,
            'time': time.time() - start_time
        }, model_path / recon_out_name)

    if 'gen' in args.tasks:
        print('Evaluate model on the generation task.')
        start_time = time.time()

        (frac_coords, num_atoms, atom_types, lengths, angles,
         all_frac_coords_stack, all_atom_types_stack) = generation(
            model, ld_kwargs, args.num_batches_to_samples, args.num_evals,
            args.batch_size, args.down_sample_traj_step)

        if args.label == '':
            gen_out_name = 'eval_gen.pt'
        else:
            gen_out_name = f'eval_gen_{args.label}.pt'

        torch.save({
            'eval_setting': args,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles,
            'all_frac_coords_stack': all_frac_coords_stack,
            'all_atom_types_stack': all_atom_types_stack,
            'time': time.time() - start_time
        }, model_path / gen_out_name)

    if 'opt' in args.tasks:
        opt_method = getattr(model, 'optimization_method', 'weighted')
        print(f'Evaluate model on the property optimization task. Target type: {args.target_type}, Method: {opt_method}')
        start_time = time.time()
        if args.start_from == 'data':
            loader = test_loader
        else:
            loader = None
        optimized_crystals = optimization(
            model, ld_kwargs, loader, 
            num_starting_points=args.num_starting_points,
            num_gradient_steps=args.num_gradient_steps,
            lr=args.lr,
            num_saved_crys=args.num_saved_crys,
            target_type=args.target_type
        )
        optimized_crystals.update({'eval_setting': args,
                                   'time': time.time() - start_time})

        # 在输出文件名中包含优化方法
        if args.label == '':
            if opt_method == 'weighted':
                gen_out_name = f'eval_opt_{args.target_type}.pt'
            else:
                gen_out_name = f'eval_opt_{args.target_type}_{opt_method}.pt'
        else:
            if opt_method == 'weighted':
                gen_out_name = f'eval_opt_{args.label}_{args.target_type}.pt'
            else:
                gen_out_name = f'eval_opt_{args.label}_{args.target_type}_{opt_method}.pt'
        
        print(f"保存优化结果到: {gen_out_name}")
        torch.save(optimized_crystals, model_path / gen_out_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--tasks', nargs='+', default=['recon', 'gen', 'opt'])
    parser.add_argument('--n_step_each', default=100, type=int)
    parser.add_argument('--step_lr', default=1e-4, type=float)
    parser.add_argument('--min_sigma', default=0, type=float)
    parser.add_argument('--save_traj', default=False, type=bool)
    parser.add_argument('--disable_bar', default=False, type=bool)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--num_batches_to_samples', default=20, type=int)
    parser.add_argument('--start_from', default='data', type=str)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--force_num_atoms', action='store_true')
    parser.add_argument('--force_atom_types', action='store_true')
    parser.add_argument('--down_sample_traj_step', default=10, type=int)
    parser.add_argument('--label', default='')
    
    # 优化相关参数
    parser.add_argument('--num_starting_points', default=100, type=int, help='优化的起始点数量')
    parser.add_argument('--num_gradient_steps', default=5000, type=int, help='梯度优化的步数')
    parser.add_argument('--lr', default=1e-3, type=float, help='优化的学习率')
    parser.add_argument('--num_saved_crys', default=10, type=int, help='保存的晶体数量')
    parser.add_argument('--target_type', default='combined', type=str, 
                        choices=['combined', 'energy', 'property', 'weighted'],
                        help='优化目标类型: combined(综合), energy(形成能), property(目标属性), weighted(加权)')
    # 添加优化方法参数                    
    parser.add_argument('--optimization_method', default=None, type=str,
                        choices=['weighted', 'tchebycheff', 'boundary'],
                        help='多目标优化方法: weighted(线性加权), tchebycheff(切比雪夫), boundary(边界交叉法)')

    args = parser.parse_args()
    main(args)