import torch
import torch.nn as nn
from typing import Dict, Optional
import hydra
import pytorch_lightning as pl
from cdvae.pl_modules.model import CDVAE, BaseModule

class GradNormCDVAE(CDVAE):
    """
        CDVAE 模型集成 GradNorm 动态任务平衡
    """
    def __init__(self, *args, **kwargs):
        # 提取配置
        self.gradnorm_config = kwargs.pop('gradnorm', {})
        super().__init__(*args, **kwargs)
        
        # 初始化 GradNorm
        self.use_gradnorm = self.gradnorm_config.get('enable', True)
        self.alpha = self.gradnorm_config.get('alpha', 1.5)
        self.gradnorm_lr = self.gradnorm_config.get('lr', 0.025)
        
        if self.use_gradnorm:
            self.task_names = ['num_atom', 'lattice', 'composition', 'coord', 'type', 'kld']
            
            if self.predict_property:
                self.task_names.extend(['energy', 'target'])
            
            self.num_tasks = len(self.task_names)
            
            self.task_weights = nn.Parameter(torch.ones(self.num_tasks))
            
            self.initial_losses = {}
            self.train_step = 0
            
            # **：获取共享层参数（编码器第一层）
            # 假设：编码器有一个 'layers' 属性
            if hasattr(self.encoder, 'int_blocks'):
                # 对于 GemNetT 或 DimeNetPlusPlus
                self.shared_layer = self.encoder.int_blocks[0]
            else:
                # 其他架构：使用编码器本身
                self.shared_layer = self.encoder
    
    def compute_gradnorm_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
            计算 GradNorm 损失
        """
        if not self.use_gradnorm:
            return torch.tensor(0.0, device=self.device)
        
        # 获取共享层的第一个参数
        shared_params = next(self.shared_layer.parameters())
        
        # 加权损失
        weighted_losses = []
        for i, task_name in enumerate(self.task_names):
            if task_name in losses:
                weighted_loss = self.task_weights[i] * losses[task_name]
                weighted_losses.append(weighted_loss)
        
        # 计算每个任务的梯度范数
        grad_norms = []
        for i, weighted_loss in enumerate(weighted_losses):
            grad = torch.autograd.grad(
                weighted_loss, shared_params, 
                retain_graph=True, create_graph=True
            )[0]
            grad_norm = torch.norm(grad)
            grad_norms.append(grad_norm)
        
        grad_norms = torch.stack(grad_norms)
        mean_grad_norm = grad_norms.mean()
        
        # 计算损失比率和逆训练率
        loss_ratios = []
        for i, task_name in enumerate(self.task_names):
            if task_name not in losses:
                continue
                
            current_loss = float(losses[task_name].detach())
            
            if self.train_step == 0:
                self.initial_losses[task_name] = current_loss
                loss_ratio = 1.0
            else:
                initial_loss = self.initial_losses.get(task_name, current_loss)
                loss_ratio = current_loss / (initial_loss + 1e-8)
            
            loss_ratios.append(loss_ratio)
        
        # 计算相对逆训练率
        loss_ratios = torch.tensor(loss_ratios, device=self.device)
        mean_loss_ratio = loss_ratios.mean()
        relative_inverse_rates = (loss_ratios / mean_loss_ratio) ** self.alpha
        
        # 计算 GradNorm 损失
        gradnorm_loss = 0
        for i in range(len(grad_norms)):
            target_grad_norm = mean_grad_norm * relative_inverse_rates[i]
            gradnorm_loss += torch.abs(grad_norms[i] - target_grad_norm)
        
        return gradnorm_loss / len(grad_norms)
    
    def compute_stats(self, batch, outputs, prefix):
        """
            重写父类方法；加入 GradNorm 逻辑
        """
        # 构建损失字典
        losses = {
            'num_atom': outputs['num_atom_loss'],
            'lattice': outputs['lattice_loss'],
            'composition': outputs['composition_loss'],
            'coord': outputs['coord_loss'],
            'type': outputs['type_loss'],
            'kld': outputs['kld_loss']
        }
        
        # 添加属性损失（如果存在）
        if self.predict_property:
            losses['energy'] = outputs['energy_loss']
            losses['target'] = outputs['target_loss']
        
        # 计算加权损失
        if self.use_gradnorm and self.training:
            # 使用 GradNorm 权重
            total_loss = 0
            for i, task_name in enumerate(self.task_names):
                if task_name in losses:
                    total_loss += self.task_weights[i] * losses[task_name]
            
            # 计算 GradNorm 损失
            gradnorm_loss = self.compute_gradnorm_loss(losses)
            total_loss += gradnorm_loss
            
            # 归一化权重
            with torch.no_grad():
                self.task_weights.data = (
                    self.task_weights.data * self.num_tasks / 
                    self.task_weights.data.sum()
                )
            
            self.train_step += 1
        else:
            # 使用固定权重（原始实现）
            total_loss = super().compute_stats(batch, outputs, prefix)[1]
            gradnorm_loss = torch.tensor(0.0)
        
        # 构建日志字典
        log_dict = {
            f'{prefix}_loss': total_loss,
            f'{prefix}_num_atom_loss': outputs['num_atom_loss'],
            f'{prefix}_lattice_loss': outputs['lattice_loss'],
            f'{prefix}_composition_loss': outputs['composition_loss'],
            f'{prefix}_coord_loss': outputs['coord_loss'],
            f'{prefix}_type_loss': outputs['type_loss'],
            f'{prefix}_kld_loss': outputs['kld_loss'],
        }
        
        if self.predict_property:
            log_dict[f'{prefix}_property_loss'] = outputs['property_loss']
            log_dict[f'{prefix}_energy_loss'] = outputs['energy_loss']
            log_dict[f'{prefix}_target_loss'] = outputs['target_loss']
        
        # 添加 GradNorm 相关日志
        if self.use_gradnorm and self.training:
            log_dict[f'{prefix}_gradnorm_loss'] = gradnorm_loss
            
            # 记录任务权重
            for i, task_name in enumerate(self.task_names):
                log_dict[f'{prefix}_weight_{task_name}'] = self.task_weights[i]
        
        return log_dict, total_loss
    
    def configure_optimizers(self):
        """
        配置优化器，为任务权重添加单独的优化器
        """
        # 获取主模型优化器配置
        optimizers = super().configure_optimizers()
        
        if self.use_gradnorm:
            # 为任务权重创建单独的优化器
            weight_optimizer = torch.optim.Adam(
                [self.task_weights], 
                lr=self.gradnorm_lr
            )
            
            if isinstance(optimizers, list):
                optimizers.append(weight_optimizer)
            elif isinstance(optimizers, dict):
                optimizers['optimizer'].append(weight_optimizer)
            else:
                optimizers = [optimizers, weight_optimizer]
        
        return optimizers


# 配置
def update_config_for_gradnorm(cfg):
    """
    更新配置以使用 GradNorm
    """
    # 添加 GradNorm 配置
    cfg.model.gradnorm = {
        'enable': True,
        'alpha': 1.5,
        'lr': 0.025
    }
    
    # 更改模型类为 GradNormCDVAE
    cfg.model._target_ = 'cdvae.pl_modules.model.GradNormCDVAE'
    
    return cfg


# 训练脚本中的使用
if __name__ == "__main__":
    import omegaconf
    from cdvae.common.utils import PROJECT_ROOT
    
    # 加载基础配置
    base_config = omegaconf.OmegaConf.load(
        PROJECT_ROOT / "conf" / "model" / "cdvae.yaml"
    )
    
    # 更新配置以使用 GradNorm
    config = update_config_for_gradnorm(base_config)
    
    # 创建模型
    model = hydra.utils.instantiate(config.model)
    
    print(f"创建了带有 GradNorm 的 CDVAE 模型")
    print(f"任务数量: {model.num_tasks}")
    print(f"任务名称: {model.task_names}")
    print(f"初始权重: {model.task_weights}")