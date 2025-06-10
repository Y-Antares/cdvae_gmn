#!/usr/bin/env python3
"""
基于原始 run.py 的训练脚本，使用 argparse 替换 Hydra 命令行
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen.io.cif")

import sys
import os
import argparse
from pathlib import Path

# 设置环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置 PROJECT_ROOT
if 'PROJECT_ROOT' not in os.environ:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ['PROJECT_ROOT'] = project_root

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
import hydra
from cdvae.common.utils import log_hyperparameters, PROJECT_ROOT

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CDVAE Training')
    
    parser.add_argument('--data', type=str, default='perov_1k', help='Data config name')
    parser.add_argument('--model', type=str, default='cdvae', help='Model config name')
    parser.add_argument('--max_epochs', type=int, default=300, help='Maximum training epochs')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--output_dir', type=str, default='results/default', help='Output directory')
    
    return parser.parse_args()

def fix_numeric_types(config: dict) -> dict:
    """修复配置中的数值类型问题"""
    if isinstance(config, dict):
        fixed_config = {}
        for key, value in config.items():
            if isinstance(value, dict):
                fixed_config[key] = fix_numeric_types(value)
            elif isinstance(value, list):
                # 处理列表中的数值
                fixed_list = []
                for item in value:
                    if isinstance(item, str):
                        try:
                            if '.' in item:
                                fixed_list.append(float(item))
                            else:
                                fixed_list.append(int(item))
                        except ValueError:
                            fixed_list.append(item)
                    else:
                        fixed_list.append(item)
                fixed_config[key] = fixed_list
            elif isinstance(value, str):
                # 尝试转换字符串为数值
                try:
                    # 处理科学计数法
                    if 'e-' in value.lower() or 'e+' in value.lower():
                        fixed_config[key] = float(value)
                    elif '.' in value:
                        fixed_config[key] = float(value)
                    elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                        fixed_config[key] = int(value)
                    else:
                        fixed_config[key] = value
                except ValueError:
                    fixed_config[key] = value
            else:
                fixed_config[key] = value
        return fixed_config
    return config

def load_hydra_config(data_name, model_name):
    """加载 Hydra 配置文件"""
    import yaml
    from datetime import datetime
    
    # 设置 Hydra 特殊变量
    current_date = datetime.now().strftime('%Y-%m-%d')
    hydra_vars = {
        'now:%Y-%m-%d': current_date,
        'id': '0',
        'num': '0',
        'expname': 'enhanced_cdvae_training'
    }
    
    # 基础配置结构
    cfg_dict = {
        'data': {},
        'model': {},
        'optim': {
            'optimizer': {
                '_target_': 'torch.optim.AdamW',
                'lr': 1e-4,
                'weight_decay': 1e-6,
            },
            'lr_scheduler': {
                '_target_': 'torch.optim.lr_scheduler.ReduceLROnPlateau',
                'mode': 'min',
                'factor': 0.6,
                'patience': 30,
                'min_lr': 1e-7,
            }
        },
        'train': {
            'deterministic': True,
            'random_seed': 42,
            'monitor_metric': 'val_loss',
            'monitor_metric_mode': 'min',
            'pl_trainer': {
                'max_epochs': 300,
                'fast_dev_run': False,
            }
        },
        'logging': {
            'val_check_interval': 1,
            'wandb': {
                'project': 'enhanced-cdvae',
                'mode': 'online',
            },
            'wandb_watch': {
                'log': 'gradients',
                'log_freq': 100,
            },
            'lr_monitor': {
                'logging_interval': 'step',
                'log_momentum': True,
            }
        },
        'core': {
            'tags': ['enhanced-cdvae'],
        }
    }
    
    # 添加 Hydra 变量到顶层
    cfg_dict.update(hydra_vars)
    
    # 加载数据配置
    data_config_path = PROJECT_ROOT / "conf" / "data" / f"{data_name}.yaml"
    if data_config_path.exists():
        with open(data_config_path, 'r') as f:
            data_config = yaml.safe_load(f)
        cfg_dict['data'].update(data_config)
        print(f"Loaded data config from {data_config_path}")
        print(f"Data config keys: {list(data_config.keys())}")
        
        # 将数据配置中的所有变量添加到顶层（解决插值问题）
        for key, value in data_config.items():
            if not key.startswith('_') and key not in cfg_dict:
                cfg_dict[key] = value
        
    else:
        print(f"Data config not found at {data_config_path}")
        return None
    
    # 加载模型配置
    model_config_path = PROJECT_ROOT / "conf" / "model" / f"{model_name}.yaml"
    if model_config_path.exists():
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        
        # 修复 _target_ 路径 - 重新启用 Enhanced CDVAE
        if '_target_' in model_config:
            if 'enhanced_cdvae' in model_config['_target_']:
                # 使用 Enhanced CDVAE（修补后应该可以工作）
                model_config['_target_'] = 'cdvae.pl_modules.enhanced_cdvae.EnhancedCDVAE'
                print("Using Enhanced CDVAE model (patched version)")
            print(f"Model target: {model_config['_target_']}")
        
        # 加载编码器和解码器配置
        if 'defaults' in model_config:
            for default in model_config['defaults']:
                if isinstance(default, dict) and 'encoder' in default:
                    encoder_name = default['encoder']
                    encoder_path = PROJECT_ROOT / "conf" / "model" / "encoder" / f"{encoder_name}.yaml"
                    if encoder_path.exists():
                        with open(encoder_path, 'r') as f:
                            encoder_config = yaml.safe_load(f)
                        model_config['encoder'] = encoder_config
                        print(f"Loaded encoder config: {encoder_name}")
                
                if isinstance(default, dict) and 'decoder' in default:
                    decoder_name = default['decoder']
                    decoder_path = PROJECT_ROOT / "conf" / "model" / "decoder" / f"{decoder_name}.yaml"
                    if decoder_path.exists():
                        with open(decoder_path, 'r') as f:
                            decoder_config = yaml.safe_load(f)
                        model_config['decoder'] = decoder_config
                        print(f"Loaded decoder config: {decoder_name}")
        
        cfg_dict['model'] = model_config
        print(f"Loaded model config from {model_config_path}")
        print(f"Model config keys: {list(model_config.keys())}")
        
        # 确保模型配置中的插值变量也存在于顶层
        model_interpolation_vars = [
            'max_atoms', 'teacher_forcing_max_epoch', 'property_weights',
            'optimization_direction', 'boundary_theta', 'init_ideal_points'
        ]
        for var in model_interpolation_vars:
            if var in cfg_dict['data'] and var not in cfg_dict:
                cfg_dict[var] = cfg_dict['data'][var]
    else:
        print(f"Model config not found at {model_config_path}")
        return None
    
    # 加载默认的训练、优化器、日志配置
    for config_type in ['train', 'optim', 'logging']:
        config_path = PROJECT_ROOT / "conf" / f"{config_type}" / "default.yaml"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # 确保数值类型正确
                if config_type == 'optim':
                    print(f"Raw optim config: {config_data}")
                    config_data = fix_numeric_types(config_data)
                    print(f"Fixed optim config: {config_data}")
                    
                    # 额外确保关键参数是数值类型
                    if 'optimizer' in config_data:
                        opt_config = config_data['optimizer']
                        numeric_params = ['lr', 'eps', 'weight_decay']
                        for param in numeric_params:
                            if param in opt_config and isinstance(opt_config[param], str):
                                try:
                                    opt_config[param] = float(opt_config[param])
                                    print(f"Force converted {param} to float: {opt_config[param]}")
                                except ValueError:
                                    print(f"Warning: Could not convert {param} = {opt_config[param]} to float")
                        
                        # 处理 betas 参数
                        if 'betas' in opt_config and isinstance(opt_config['betas'], list):
                            betas = []
                            for beta in opt_config['betas']:
                                if isinstance(beta, str):
                                    try:
                                        betas.append(float(beta))
                                    except ValueError:
                                        betas.append(beta)
                                else:
                                    betas.append(beta)
                            opt_config['betas'] = betas
                    
                    print(f"Final optim config: {config_data}")
                
                cfg_dict[config_type].update(config_data)
                print(f"✅ Loaded {config_type} config from {config_path}")
            except Exception as e:
                print(f"❌ Error loading {config_type} config: {e}")
                return None
        else:
            print(f"⚠️  {config_type} config not found at {config_path}")
    
    print(f"Final config keys: {list(cfg_dict.keys())}")
    
    # 打印调试信息
    print(f"Data config has datamodule: {'datamodule' in cfg_dict['data']}")
    print(f"Model config has _target_: {'_target_' in cfg_dict['model']}")
    
    # 验证关键配置存在
    if 'datamodule' not in cfg_dict['data']:
        print("❌ Missing datamodule in data config")
        return None
    
    if '_target_' not in cfg_dict['model']:
        print("❌ Missing _target_ in model config")
        return None
    
    try:
        omega_cfg = OmegaConf.create(cfg_dict)
        print("✅ Successfully created OmegaConf configuration")
        return omega_cfg
    except Exception as e:
        print(f"❌ Error creating OmegaConf: {e}")
        import traceback
        traceback.print_exc()
        return None

def build_callbacks(cfg: DictConfig, output_dir: Path):
    """构建训练回调"""
    callbacks = []

    if "lr_monitor" in cfg.logging:
        print("Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.train:
        print("Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.train:
        print("Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                dirpath=output_dir,
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                verbose=cfg.train.model_checkpoints.verbose,
            )
        )

    return callbacks

def main():
    """主训练函数"""
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"CDVAE Training")
    print(f"Data: {args.data}")
    print(f"Model: {args.model}")
    print(f"Output directory: {output_dir}")
    print(f"Max epochs: {args.max_epochs}")
    
    # 加载配置
    try:
        cfg = load_hydra_config(args.data, args.model)
        if cfg is None:
            print("❌ Configuration loading failed")
            return
        print("✅ Configuration loaded successfully")
        print(f"Config keys: {list(cfg.keys())}")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 设置 max_epochs - 检查配置结构
    try:
        if hasattr(cfg, 'train') and hasattr(cfg.train, 'pl_trainer'):
            cfg.train.pl_trainer.max_epochs = args.max_epochs
        else:
            print("Warning: Unable to set max_epochs, config structure issue")
    except Exception as e:
        print(f"Warning: Error setting max_epochs: {e}")
    
    # 设置随机种子
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)
    
    # 实例化数据模块
    print(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)
    
    # 实例化模型
    print(f"Instantiating <{cfg.model._target_}>")
    model = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )
    
    # 修复 MinMaxScalerTorch 的 copy 方法
    from cdvae.common.data_utils import MinMaxScalerTorch
    
    def patched_copy(self):
        """修复后的 copy 方法，确保 ranges 属性被正确复制"""
        new_scaler = MinMaxScalerTorch(
            min_val=self.min_val,
            max_val=self.max_val,
            mins=self.mins.clone().detach() if self.mins is not None else None,
            maxs=self.maxs.clone().detach() if self.maxs is not None else None
        )
        # 确保 ranges 属性也被复制
        if hasattr(self, 'ranges') and self.ranges is not None:
            new_scaler.ranges = self.ranges.clone().detach()
        elif hasattr(self, 'mins') and hasattr(self, 'maxs') and self.mins is not None and self.maxs is not None:
            # 如果原对象没有 ranges 但有 mins 和 maxs，重新计算
            new_scaler.ranges = self.maxs - self.mins
            new_scaler.ranges = torch.where(new_scaler.ranges == 0, torch.ones_like(new_scaler.ranges), new_scaler.ranges)
        return new_scaler
    
    # 应用猴子补丁
    MinMaxScalerTorch.copy = patched_copy
    
    # 修复 enhanced_cdvae.py 中的 compute_losses 方法调用
    def patch_enhanced_cdvae():
        """修复 EnhancedCDVAE 中的方法调用问题"""
        from cdvae.pl_modules.enhanced_cdvae import EnhancedCDVAE
        
        # 检查模型是否有 compute_losses 方法
        if hasattr(model, 'compute_losses'):
            original_compute_losses = model.compute_losses
            
            def patched_compute_losses(batch, outputs, kld_weight=1.0):
                """修复后的 compute_losses 方法"""
                try:
                    # 首先尝试调用父类方法（如果存在）
                    if hasattr(super(EnhancedCDVAE, model), 'compute_losses'):
                        return super(EnhancedCDVAE, model).compute_losses(batch, outputs, kld_weight)
                    else:
                        # 如果父类没有这个方法，实现基本的损失计算
                        losses = {}
                        
                        # 基础 VAE 损失
                        if 'z_mu' in outputs and 'z_log_var' in outputs:
                            kld_loss = -0.5 * torch.sum(1 + outputs['z_log_var'] - outputs['z_mu'].pow(2) - outputs['z_log_var'].exp())
                            losses['kld'] = kld_loss * kld_weight
                        
                        # 重建损失（这里需要根据具体实现调整）
                        if 'pred_frac_coords' in outputs and 'frac_coords' in batch:
                            coord_loss = torch.nn.functional.mse_loss(outputs['pred_frac_coords'], batch['frac_coords'])
                            losses['coord'] = coord_loss
                        
                        # 属性预测损失
                        if 'property_pred' in outputs and 'y' in batch:
                            prop_loss = torch.nn.functional.mse_loss(outputs['property_pred'], batch['y'])
                            losses['property'] = prop_loss
                        
                        return losses
                        
                except Exception as e:
                    print(f"Error in compute_losses: {e}")
                    # 返回基本损失结构
                    return {'total': torch.tensor(0.0, device=next(model.parameters()).device)}
            
            # 应用补丁
            model.compute_losses = patched_compute_losses.__get__(model, type(model))
            print("Applied compute_losses patch to EnhancedCDVAE")
    
    # 应用修复
    patch_enhanced_cdvae()
    
    # 传递标准化器
    print("Passing scalers from datamodule to model")
    model.lattice_scaler = datamodule.lattice_scaler.copy()
    model.scaler = datamodule.scaler.copy()
    
    # 传递能量标准化器
    if hasattr(datamodule, 'energy_scaler') and datamodule.energy_scaler is not None:
        print("Passing energy scaler from datamodule to model")
        model.energy_scaler = datamodule.energy_scaler.copy()
        torch.save(datamodule.energy_scaler, output_dir / 'energy_scaler.pt')
    else:
        print("Warning: No energy_scaler found in datamodule")
        from cdvae.common.data_utils import StandardScaler
        model.energy_scaler = StandardScaler(mean=0.0, std=1.0)
        torch.save(model.energy_scaler, output_dir / 'energy_scaler.pt')
    
    # 保存标准化器
    torch.save(datamodule.lattice_scaler, output_dir / 'lattice_scaler.pt')
    torch.save(datamodule.scaler, output_dir / 'prop_scaler.pt')
    
    # 构建回调
    callbacks = build_callbacks(cfg, output_dir)
    
    # 设置logger
    wandb_logger = None
    if args.wandb and "wandb" in cfg.logging:
        print("Instantiating <WandbLogger>")
        wandb_config = cfg.logging.wandb
        wandb_logger = WandbLogger(
            **wandb_config,
            tags=cfg.core.tags,
            save_dir=output_dir,
        )
        print(f"W&B is now watching <{cfg.logging.wandb_watch.log}>!")
        wandb_logger.watch(
            model,
            log=cfg.logging.wandb_watch.log,
            log_freq=cfg.logging.wandb_watch.log_freq,
        )
    
    # 保存配置
    yaml_conf = OmegaConf.to_yaml(cfg=cfg)
    (output_dir / "hparams.yaml").write_text(yaml_conf)
    
    # 检查检查点
    ckpts = list(output_dir.glob('*.ckpt'))
    if len(ckpts) > 0:
        ckpt_epochs = np.array([int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts])
        ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        print(f"Found checkpoint: {ckpt}")
    else:
        ckpt = None
    
    # 实例化训练器
    print("Instantiating the Trainer")
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        check_val_every_n_epoch=cfg.logging.val_check_interval,
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
    )
    
    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)
    
    # 开始训练
    print("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt)
    
    print("Starting testing!")
    trainer.test(datamodule=datamodule)
    
    print(f"Training completed! Results saved to: {output_dir}")
    
    # 关闭logger
    if wandb_logger is not None:
        wandb_logger.experiment.finish()

if __name__ == "__main__":
    main()