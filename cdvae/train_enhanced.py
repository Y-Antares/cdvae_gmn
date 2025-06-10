#!/usr/bin/env python3
"""
Enhanced CDVAE Training Script with argparse
支持传统命令行参数格式
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen.io.cif")

import sys
import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any

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

# 设置环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Enhanced CDVAE Training')
    
    # 基础配置
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='perovskite',
                       help='Dataset name')
    parser.add_argument('--output_dir', type=str, default='results/default',
                       help='Output directory for results')
    
    # 训练参数
    parser.add_argument('--max_epochs', type=int, default=300,
                       help='Maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # 多目标优化参数
    parser.add_argument('--gradnorm', action='store_true',
                       help='Enable GradNorm')
    parser.add_argument('--gradnorm_alpha', type=float, default=1.5,
                       help='GradNorm alpha parameter')
    parser.add_argument('--gradnorm_lr', type=float, default=0.025,
                       help='GradNorm learning rate')
    
    parser.add_argument('--multi_obj_method', type=str, default='weighted',
                       choices=['weighted', 'tchebycheff', 'boundary'],
                       help='Multi-objective optimization method')
    parser.add_argument('--property_weights', type=float, nargs='+', 
                       default=[0.5, 0.5],
                       help='Property weights for multi-objective optimization')
    parser.add_argument('--boundary_theta', type=float, default=5.0,
                       help='Boundary theta parameter')
    
    # 标准化器配置
    parser.add_argument('--scaler_type', type=str, default='minmax',
                       choices=['standard', 'minmax'],
                       help='Scaler type')
    parser.add_argument('--energy_scaler_type', type=str, default='minmax',
                       choices=['standard', 'minmax'],
                       help='Energy scaler type')
    
    # GPU配置
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    
    # 日志配置
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='enhanced-cdvae',
                       help='W&B project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='W&B run name')
    
    # 早停配置
    parser.add_argument('--early_stopping', action='store_true',
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_hydra_config(args, base_config: Dict[str, Any]) -> DictConfig:
    """将argparse参数转换为Hydra兼容的配置"""
    
    # 创建基础配置结构
    cfg_dict = {
        'data': base_config.copy(),
        'model': {
            '_target_': 'cdvae.pl_modules.enhanced_cdvae.EnhancedCDVAE',
        },
        'optim': {
            'optimizer': {
                '_target_': 'torch.optim.AdamW',
                'lr': args.lr,
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
            'random_seed': args.seed,
            'monitor_metric': 'val_loss',
            'monitor_metric_mode': 'min',
            'pl_trainer': {
                'max_epochs': args.max_epochs,
                'fast_dev_run': False,
            }
        },
        'logging': {
            'val_check_interval': 1,
        },
        'core': {
            'tags': [args.dataset, args.multi_obj_method],
        }
    }
    
    # 添加多目标优化配置
    cfg_dict['use_gradnorm'] = args.gradnorm
    cfg_dict['gradnorm_alpha'] = args.gradnorm_alpha
    cfg_dict['gradnorm_lr'] = args.gradnorm_lr
    cfg_dict['multi_obj_method'] = args.multi_obj_method
    cfg_dict['property_weights'] = args.property_weights
    cfg_dict['boundary_theta'] = args.boundary_theta
    
    # 添加标准化器配置
    cfg_dict['data']['scaler_type'] = args.scaler_type
    cfg_dict['data']['energy_scaler_type'] = args.energy_scaler_type
    cfg_dict['data']['optimization_method'] = args.multi_obj_method
    cfg_dict['data']['use_gradnorm'] = args.gradnorm
    cfg_dict['data']['gradnorm_alpha'] = args.gradnorm_alpha
    
    # 添加batch size配置
    if 'datamodule' in cfg_dict['data'] and 'batch_size' in cfg_dict['data']['datamodule']:
        cfg_dict['data']['datamodule']['batch_size']['train'] = args.batch_size
    
    # 添加W&B配置
    if args.wandb:
        cfg_dict['logging']['wandb'] = {
            'project': args.wandb_project,
            'name': args.wandb_name or f"{args.dataset}_{args.multi_obj_method}",
            'mode': 'online',
        }
        cfg_dict['logging']['wandb_watch'] = {
            'log': 'gradients',
            'log_freq': 100,
        }
    
    # 添加早停配置
    if args.early_stopping:
        cfg_dict['train']['early_stopping'] = {
            'patience': args.patience,
            'verbose': True,
        }
        cfg_dict['train']['model_checkpoints'] = {
            'save_top_k': 3,
            'verbose': True,
        }
    
    return OmegaConf.create(cfg_dict)

def build_callbacks(cfg: DictConfig, output_dir: Path) -> list:
    """构建训练回调"""
    callbacks = []
    
    # 学习率监控
    callbacks.append(LearningRateMonitor(
        logging_interval='step',
        log_momentum=True,
    ))
    
    # 早停
    if "early_stopping" in cfg.train:
        callbacks.append(EarlyStopping(
            monitor=cfg.train.monitor_metric,
            mode=cfg.train.monitor_metric_mode,
            patience=cfg.train.early_stopping.patience,
            verbose=cfg.train.early_stopping.verbose,
        ))
    
    # 模型检查点
    if "model_checkpoints" in cfg.train:
        callbacks.append(ModelCheckpoint(
            dirpath=output_dir,
            monitor=cfg.train.monitor_metric,
            mode=cfg.train.monitor_metric_mode,
            save_top_k=cfg.train.model_checkpoints.save_top_k,
            verbose=cfg.train.model_checkpoints.verbose,
            filename='epoch={epoch:02d}-val_loss={val_loss:.3f}',
        ))
    
    return callbacks

def main():
    """主训练函数"""
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Enhanced CDVAE Training")
    print(f"Dataset: {args.dataset}")
    print(f"Multi-objective method: {args.multi_obj_method}")
    print(f"GradNorm: {args.gradnorm}")
    print(f"Output directory: {output_dir}")
    print(f"Max epochs: {args.max_epochs}")
    
    # 加载基础配置
    base_config = load_config(args.config)
    
    # 创建Hydra兼容配置
    cfg = create_hydra_config(args, base_config)
    
    # 设置随机种子
    if cfg.train.deterministic:
        seed_everything(args.seed)
    
    # 实例化数据模块
    print("Instantiating datamodule...")
    from cdvae.pl_data.datamodule import CrystDataModule
    
    datamodule = CrystDataModule(
        datasets=cfg.data.datamodule.datasets,
        num_workers=cfg.data.datamodule.num_workers,
        batch_size=cfg.data.datamodule.batch_size,
    )
    
    # 实例化Enhanced CDVAE模型
    print("Instantiating Enhanced CDVAE model...")
    
    # 构建模型配置
    model_config = cfg.model.copy()
    
    # 添加GradNorm配置
    if args.gradnorm:
        model_config['gradnorm'] = {
            'enable': True,
            'alpha': args.gradnorm_alpha,
            'lr': args.gradnorm_lr
        }
    else:
        model_config['gradnorm'] = {'enable': False}
    
    # 添加多目标优化配置
    model_config['multi_objective'] = {
        'method': args.multi_obj_method,
        'weights': args.property_weights,
        'direction': ['min', 'max'],
        'boundary_theta': args.boundary_theta,
        'init_ideal_points': [float('inf'), float('inf')]
    }
    
    # 创建模型
    from cdvae.pl_modules.enhanced_cdvae import EnhancedCDVAE
    model = EnhancedCDVAE(
        **model_config,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
    )
    
    # 传递标准化器
    print("Setting up scalers...")
    model.lattice_scaler = datamodule.lattice_scaler.copy()
    model.scaler = datamodule.scaler.copy()
    
    # 传递能量标准化器
    if hasattr(datamodule, 'energy_scaler') and datamodule.energy_scaler is not None:
        model.energy_scaler = datamodule.energy_scaler.copy()
        torch.save(datamodule.energy_scaler, output_dir / 'energy_scaler.pt')
    else:
        from cdvae.common.data_utils import StandardScaler
        model.energy_scaler = StandardScaler(mean=0.0, std=1.0)
        torch.save(model.energy_scaler, output_dir / 'energy_scaler.pt')
    
    # 保存标准化器
    torch.save(datamodule.lattice_scaler, output_dir / 'lattice_scaler.pt')
    torch.save(datamodule.scaler, output_dir / 'prop_scaler.pt')
    
    # 构建回调
    callbacks = build_callbacks(cfg, output_dir)
    
    # 设置logger
    logger = None
    if args.wandb and "wandb" in cfg.logging:
        logger = WandbLogger(
            **cfg.logging.wandb,
            tags=cfg.core.tags,
        )
        logger.watch(model, log='gradients', log_freq=100)
    
    # 保存配置
    config_save_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_save_path)
    
    # 实例化训练器
    print("Instantiating trainer...")
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        logger=logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=cfg.logging.val_check_interval,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
    )
    
    # 开始训练
    print("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)
    
    print("Starting testing!")
    trainer.test(datamodule=datamodule)
    
    print(f"Training completed! Results saved to: {output_dir}")
    
    # 关闭loggers
    if logger is not None:
        logger.experiment.finish()

if __name__ == "__main__":
    main()