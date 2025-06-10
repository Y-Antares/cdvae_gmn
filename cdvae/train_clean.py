#!/usr/bin/env python3
"""
Enhanced CDVAE Training Script with argparse
干净版本，无语法错误
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen.io.cif")

import sys
import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List

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

# 设置环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Enhanced CDVAE Training')
    
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='perovskite', help='Dataset name')
    parser.add_argument('--output_dir', type=str, default='results/default', help='Output directory')
    parser.add_argument('--max_epochs', type=int, default=300, help='Maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gradnorm', action='store_true', help='Enable GradNorm')
    parser.add_argument('--gradnorm_alpha', type=float, default=1.5, help='GradNorm alpha')
    parser.add_argument('--gradnorm_lr', type=float, default=0.025, help='GradNorm learning rate')
    parser.add_argument('--multi_obj_method', type=str, default='weighted', 
                       choices=['weighted', 'tchebycheff', 'boundary'], help='Multi-objective method')
    parser.add_argument('--property_weights', type=float, nargs='+', default=[0.5, 0.5],
                       help='Property weights')
    parser.add_argument('--boundary_theta', type=float, default=5.0, help='Boundary theta')
    parser.add_argument('--scaler_type', type=str, default='minmax', 
                       choices=['standard', 'minmax'], help='Scaler type')
    parser.add_argument('--energy_scaler_type', type=str, default='minmax',
                       choices=['standard', 'minmax'], help='Energy scaler type')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--wandb_project', type=str, default='enhanced-cdvae', help='W&B project')
    parser.add_argument('--wandb_name', type=str, default=None, help='W&B run name')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    
    return parser.parse_args()

def main():
    """主训练函数"""
    args = parse_args()
    
    # 设置环境变量
    if 'PROJECT_ROOT' not in os.environ:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.environ['PROJECT_ROOT'] = project_root
        print(f"Setting PROJECT_ROOT to: {project_root}")
    
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
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # 设置随机种子
    seed_everything(args.seed)
    
    # 构建完整配置
    cfg_dict = {
        'scaler_type': args.scaler_type,
        'energy_scaler_type': args.energy_scaler_type,
        'optimization_method': args.multi_obj_method,
        'use_gradnorm': args.gradnorm,
        'gradnorm_alpha': args.gradnorm_alpha,
        'data': base_config.copy(),
        'model': {
            '_target_': 'cdvae.pl_modules.model.CDVAE',
            'encoder': {
                'atom_embedding_dim': 92,
                'num_message_passing': 3,
                'num_attention_heads': 4,
                'hidden_dim': 256,
                'act_fn': 'silu',
                'dis_emb': True,
            },
            'decoder': {
                'atom_embedding_dim': 92,
                'num_message_passing': 3,
                'num_attention_heads': 4,
                'hidden_dim': 256,
                'act_fn': 'silu',
            },
            'latent_dim': 256,
            'beta': 0.5,
            'gamma': 1.0,
            'hidden_dim': 256,
            'fc_num_layers': 3,
            'num_targets': base_config.get('num_targets', 2),
            'max_atoms': base_config.get('max_atoms', 200),
            'teacher_forcing_max_epoch': base_config.get('teacher_forcing_max_epoch', 150),
            'predict_property': True,
            'property_predictor': {
                'hidden_dim': 256,
                'fc_num_layers': 3,
                'num_targets': base_config.get('num_targets', 2),
            },
            'lattice_predictor': {
                'hidden_dim': 256,
                'fc_num_layers': 3,
            },
            'composition_predictor': {
                'hidden_dim': 256,
                'fc_num_layers': 3,
            },
            # 添加缺失的扩散模型参数
            'sigma_begin': 1e-4,
            'sigma_end': 2e-2,
            'num_noise_level': 50,
            'noise_type': 'uniform',
            # 添加其他可能需要的参数
            'cost_lattice': 1.0,
            'cost_coord': 1.0,
            'cost_type': 1.0,
            'cost_length': 1.0,
            'cost_angle': 1.0,
            'cost_composition': 1.0,
            'cost_property': 1.0,
            'beta_kl': 1e-3,
            'lattice_loss_type': 'mse',
            'coord_loss_type': 'mse',
            'type_loss_type': 'ce',
            'length_loss_type': 'mse',
            'angle_loss_type': 'mse',
            'composition_loss_type': 'mse',
            'property_loss_type': 'mse',
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
    
    # 更新数据配置
    cfg_dict['data']['scaler_type'] = args.scaler_type
    cfg_dict['data']['energy_scaler_type'] = args.energy_scaler_type
    cfg_dict['data']['optimization_method'] = args.multi_obj_method
    cfg_dict['data']['use_gradnorm'] = args.gradnorm
    cfg_dict['data']['gradnorm_alpha'] = args.gradnorm_alpha
    
    # 更新datamodule配置
    if 'datamodule' in cfg_dict['data']:
        if 'batch_size' not in cfg_dict['data']['datamodule']:
            cfg_dict['data']['datamodule']['batch_size'] = {}
        cfg_dict['data']['datamodule']['batch_size']['train'] = args.batch_size
        cfg_dict['data']['datamodule']['batch_size']['val'] = args.batch_size // 2
        cfg_dict['data']['datamodule']['batch_size']['test'] = args.batch_size // 2
        
        if 'num_workers' not in cfg_dict['data']['datamodule']:
            cfg_dict['data']['datamodule']['num_workers'] = {}
        cfg_dict['data']['datamodule']['num_workers']['train'] = args.num_workers
        cfg_dict['data']['datamodule']['num_workers']['val'] = args.num_workers // 2
        cfg_dict['data']['datamodule']['num_workers']['test'] = args.num_workers // 2
    
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
        cfg_dict['logging']['lr_monitor'] = {
            'logging_interval': 'step',
            'log_momentum': True,
        }
    
    # 转换为OmegaConf
    cfg = OmegaConf.create(cfg_dict)
    
    print("Configuration created successfully")
    
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
    callbacks = []
    
    if "lr_monitor" in cfg.logging:
        print("Adding callback <LearningRateMonitor>")
        callbacks.append(LearningRateMonitor(
            logging_interval=cfg.logging.lr_monitor.logging_interval,
            log_momentum=cfg.logging.lr_monitor.log_momentum,
        ))
    
    if "early_stopping" in cfg.train:
        print("Adding callback <EarlyStopping>")
        callbacks.append(EarlyStopping(
            monitor=cfg.train.monitor_metric,
            mode=cfg.train.monitor_metric_mode,
            patience=cfg.train.early_stopping.patience,
            verbose=cfg.train.early_stopping.verbose,
        ))
    
    if "model_checkpoints" in cfg.train:
        print("Adding callback <ModelCheckpoint>")
        callbacks.append(ModelCheckpoint(
            dirpath=output_dir,
            monitor=cfg.train.monitor_metric,
            mode=cfg.train.monitor_metric_mode,
            save_top_k=cfg.train.model_checkpoints.save_top_k,
            verbose=cfg.train.model_checkpoints.verbose,
            filename='epoch={epoch:02d}-val_loss={val_loss:.3f}',
        ))
    
    # 设置logger
    wandb_logger = None
    if "wandb" in cfg.logging:
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
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # 记录超参数
    from cdvae.common.utils import log_hyperparameters
    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)
    
    # 开始训练
    print("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)
    
    print("Starting testing!")
    trainer.test(datamodule=datamodule)
    
    print(f"Training completed! Results saved to: {output_dir}")
    
    # 关闭logger
    if wandb_logger is not None:
        wandb_logger.experiment.finish()

if __name__ == "__main__":
    main()