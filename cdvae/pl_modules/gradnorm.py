import torch
import torch.nn as nn
import numpy as np
from typing import Dict

class GradNorm:
    """
    基于 GradNorm 算法实现动态任务权重平衡
    参考论文: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Learning
    """
    def __init__(self, 
                 num_tasks: int,
                 alpha: float = 1.5,
                 initial_losses: Dict[str, float] = None):
        """
        Args:
            num_tasks: 任务数量
            alpha: 不对称超参数，控制训练率的程度。默认 1.5
            initial_losses: 各任务的初始损失值字典，用于计算损失比率
        """
        self.num_tasks = num_tasks
        self.alpha = alpha
        
        # 初始化任务权重（全部为1）
        self.task_weights = nn.Parameter(torch.ones(num_tasks).float())
        
        # 存储初始损失用于归一化
        self.initial_losses = initial_losses or {}
        self.last_losses = {}
        
        # 记录训练步数
        self.train_step = 0
        
    def compute_grad_norm_loss(self, losses: Dict[str, torch.Tensor], 
                             shared_params: nn.Parameter) -> torch.Tensor:
        """
        计算 GradNorm 损失
        
        Args:
            losses: 各任务的损失字典 {'task_name': loss_tensor}
            shared_params: 共享层的参数（用于计算梯度）
        
        Returns:
            gradnorm_loss: GradNorm 损失，用于更新任务权重
        """
        # 确保 task_weights 梯度清零
        if self.task_weights.grad is not None:
            self.task_weights.grad.zero_()
            
        task_names = list(losses.keys())
        assert len(task_names) == self.num_tasks
        
        # 计算加权损失
        weighted_losses = []
        for i, task_name in enumerate(task_names):
            weighted_loss = self.task_weights[i] * losses[task_name]
            weighted_losses.append(weighted_loss)
        
        # 计算总损失
        total_loss = sum(weighted_losses)
        
        # 获取每个任务相对于共享参数的梯度
        task_gradients = []
        grad_norms = []
        
        for i, weighted_loss in enumerate(weighted_losses):
            # 计算当前任务损失相对于共享参数的梯度
            grads = torch.autograd.grad(weighted_loss, shared_params, 
                                       retain_graph=True, create_graph=True)[0]
            # 计算梯度范数
            grad_norm = torch.norm(grads)
            task_gradients.append(grads)
            grad_norms.append(grad_norm)
        
        # 计算平均梯度范数
        mean_grad_norm = torch.stack(grad_norms).mean()
        
        # 计算损失比率和逆训练率
        loss_ratios = []
        for i, task_name in enumerate(task_names):
            if self.train_step == 0:
                # 第一步时初始化
                self.initial_losses[task_name] = float(losses[task_name].detach())
                loss_ratio = 1.0
            else:
                initial_loss = self.initial_losses.get(task_name, float(losses[task_name].detach()))
                current_loss = float(losses[task_name].detach())
                loss_ratio = current_loss / (initial_loss + 1e-8)
            
            loss_ratios.append(loss_ratio)
        
        # 计算相对逆训练率
        mean_loss_ratio = np.mean(loss_ratios)
        relative_inverse_rates = [(loss_ratio / mean_loss_ratio) ** self.alpha 
                                  for loss_ratio in loss_ratios]
        
        # 计算 GradNorm 损失
        gradnorm_loss = 0
        for i, grad_norm in enumerate(grad_norms):
            target_grad_norm = mean_grad_norm * relative_inverse_rates[i]
            gradnorm_loss += torch.abs(grad_norm - target_grad_norm)
        
        self.train_step += 1
        return gradnorm_loss / self.num_tasks
        
    def normalize_weights(self):
        """
        归一化任务权重，使其和为任务数量
        """
        with torch.no_grad():
            self.task_weights.data = self.task_weights.data * self.num_tasks / self.task_weights.data.sum()


class CDVAE_with_GradNorm(nn.Module):
    """
    集成 GradNorm 的 CDVAE 模型
    """
    def __init__(self, original_cdvae_config, gradnorm_config=None):
        super().__init__()
        
        # 从原始配置创建 CDVAE 模型
        self.cdvae = hydra.utils.instantiate(original_cdvae_config)
        
        # GradNorm 配置
        gradnorm_config = gradnorm_config or {}
        self.use_gradnorm = gradnorm_config.get('use_gradnorm', True)
        
        if self.use_gradnorm:
            # 计算任务数量
            num_tasks = 0
            task_names = []
            
            # 基础损失任务
            base_tasks = ['num_atom', 'lattice', 'composition', 'coord', 'type', 'kld']
            num_tasks += len(base_tasks)
            task_names.extend(base_tasks)
            
            # 属性预测任务（如果启用）
            if hasattr(self.cdvae, 'predict_property') and self.cdvae.predict_property:
                property_tasks = ['energy', 'target']
                num_tasks += len(property_tasks)
                task_names.extend(property_tasks)
            
            self.task_names = task_names
            
            # 初始化 GradNorm
            self.gradnorm = GradNorm(
                num_tasks=num_tasks,
                alpha=gradnorm_config.get('alpha', 1.5)
            )
            
            # 定义共享参数（编码器的第一层）
            self.shared_params = self.cdvae.encoder.encoder.layers[0].weight
            
            # 优化器配置
            self.gradnorm_lr = gradnorm_config.get('lr', 0.025)
            
    def forward(self, batch, teacher_forcing=False, training=True):
        """前向传播"""
        return self.cdvae(batch, teacher_forcing, training)
    
    def compute_weighted_loss(self, outputs: Dict[str, torch.Tensor], 
                            use_gradnorm: bool = True) -> torch.Tensor:
        """
        计算加权总损失
        
        Args:
            outputs: CDVAE 输出的损失字典
            use_gradnorm: 是否使用 GradNorm 权重
        
        Returns:
            weighted_loss: 加权总损失
        """
        # 构建损失字典
        losses = {}
        
        # 基础损失
        losses['num_atom'] = outputs['num_atom_loss']
        losses['lattice'] = outputs['lattice_loss']
        losses['composition'] = outputs['composition_loss']
        losses['coord'] = outputs['coord_loss']
        losses['type'] = outputs['type_loss']
        losses['kld'] = outputs['kld_loss']
        
        # 属性损失（如果存在）
        if 'energy_loss' in outputs:
            losses['energy'] = outputs['energy_loss']
            losses['target'] = outputs['target_loss']
        
        if use_gradnorm and self.use_gradnorm:
            # 使用 GradNorm 权重
            weighted_loss = 0
            for i, task_name in enumerate(self.task_names):
                if task_name in losses:
                    weighted_loss += self.gradnorm.task_weights[i] * losses[task_name]
            
            # 计算 GradNorm 损失并返回
            gradnorm_loss = self.gradnorm.compute_grad_norm_loss(losses, self.shared_params)
            
            return weighted_loss, gradnorm_loss
        else:
            # 使用固定权重
            weights = {
                'num_atom': 1.0,
                'lattice': 1.0,
                'composition': 1.0,
                'coord': 10.0,
                'type': 1.0,
                'kld': 0.1,
                'energy': 0.5,
                'target': 0.5
            }
            
            weighted_loss = 0
            for task_name, loss in losses.items():
                weighted_loss += weights.get(task_name, 1.0) * loss
                
            return weighted_loss, None
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        outputs = self(batch, teacher_forcing=True, training=True)
        
        # 计算加权损失
        weighted_loss, gradnorm_loss = self.compute_weighted_loss(outputs, use_gradnorm=True)
        
        # 创建日志字典
        log_dict = {
            'train_loss': weighted_loss,
            'train_num_atom_loss': outputs['num_atom_loss'],
            'train_lattice_loss': outputs['lattice_loss'],
            'train_composition_loss': outputs['composition_loss'],
            'train_coord_loss': outputs['coord_loss'],
            'train_type_loss': outputs['type_loss'],
            'train_kld_loss': outputs['kld_loss'],
        }
        
        if 'energy_loss' in outputs:
            log_dict['train_energy_loss'] = outputs['energy_loss']
            log_dict['train_target_loss'] = outputs['target_loss']
        
        if self.use_gradnorm:
            log_dict['gradnorm_loss'] = gradnorm_loss
            
            # 记录任务权重
            for i, task_name in enumerate(self.task_names):
                log_dict[f'weight_{task_name}'] = self.gradnorm.task_weights[i]
        
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True)
        
        return weighted_loss + (gradnorm_loss if gradnorm_loss is not None else 0)
    
    def configure_optimizers(self):
        """配置优化器"""
        # 主模型参数
        main_params = [p for n, p in self.named_parameters() 
                       if 'task_weights' not in n]
        
        # 任务权重参数
        weight_params = [self.gradnorm.task_weights] if self.use_gradnorm else []
        
        # 主优化器（从原始配置）
        main_optimizer = hydra.utils.instantiate(
            self.hparams.optim.optimizer, 
            params=main_params, 
            _convert_="partial"
        )
        
        optimizers = [main_optimizer]
        schedulers = []
        
        # 如果使用调度器
        if self.hparams.optim.use_lr_scheduler:
            main_scheduler = hydra.utils.instantiate(
                self.hparams.optim.lr_scheduler, 
                optimizer=main_optimizer
            )
            schedulers.append(main_scheduler)
        
        # GradNorm 权重优化器
        if self.use_gradnorm and weight_params:
            weight_optimizer = torch.optim.Adam(weight_params, lr=self.gradnorm_lr)
            optimizers.append(weight_optimizer)
        
        if schedulers:
            return optimizers, schedulers
        else:
            return optimizers
    
    def on_after_backward(self):
        """在反向传播后归一化权重"""
        if self.use_gradnorm:
            self.gradnorm.normalize_weights()


# 使用示例配置
gradnorm_config = {
    'use_gradnorm': True,
    'alpha': 1.5,  # 不对称超参数
    'lr': 0.025    # 任务权重学习率
}

def create_model_with_gradnorm(cfg):
    """创建带有 GradNorm 的模型"""
    # 使用原始配置创建 CDVAE 配置
    original_config = cfg.model
    
    # 创建 GradNorm 增强模型
    model = CDVAE_with_GradNorm(
        original_cdvae_config=original_config,
        gradnorm_config=gradnorm_config
    )
    
    return model

