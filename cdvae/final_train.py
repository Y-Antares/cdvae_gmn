#!/usr/bin/env python3
"""
Enhanced CDVAE Training Script with argparse
解决OmegaConf插值问题
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
    # ======================= 新增参数 =======================
    parser.add_argument('--wandb_offline', action='store_true', help='Enable W&B offline mode for environments without internet')
    # =======================================================
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    
    return parser.parse_args()

def resolve_config_interpolations(config_dict: Dict, args) -> Dict:
    """解决配置中的插值问题"""
    # 添加所有可能需要的变量到顶层
    config_dict['scaler_type'] = args.scaler_type
    config_dict['energy_scaler_type'] = args.energy_scaler_type
    config_dict['optimization_method'] = args.multi_obj_method
    config_dict['use_gradnorm'] = args.gradnorm
    config_dict['gradnorm_alpha'] = args.gradnorm_alpha
    
    # 如果配置中有data字段，也添加到data下
    if 'data' in config_dict:
        config_dict['data'].update({
            'scaler_type': args.scaler_type,
            'energy_scaler_type': args.energy_scaler_type,
            'optimization_method': args.multi_obj_method,
            'use_gradnorm': args.gradnorm,
            'gradnorm_alpha': args.gradnorm_alpha,
        })
    
    return config_dict

# 在 cdvae/final_train.py 文件中，请用下面的函数替换旧的 create_hydra_style_config 函数

def create_hydra_style_config(args, base_config: Dict[str, Any]) -> DictConfig:
    """创建类似Hydra的配置结构，解决插值问题"""
    
    resolved_base_config = resolve_config_interpolations(base_config, args)
    
    model_config_path = "conf/model/enhanced_cdvae.yaml"
    model_config = {}
    
    try:
        if os.path.exists(model_config_path):
            with open(model_config_path, 'r') as f:
                model_config = yaml.safe_load(f)
            print(f"Loaded model config from {model_config_path}")

            if 'defaults' in model_config and isinstance(model_config['defaults'], list):
                print("Processing defaults from model config...")
                project_root = Path(__file__).resolve().parent.parent 
                conf_dir = project_root / 'conf'
                
                for default_item in model_config['defaults']:
                    if isinstance(default_item, dict):
                        for key, value in default_item.items():
                            default_config_path = conf_dir / f"model/{key}/{value}.yaml"
                            if default_config_path.exists():
                                with open(default_config_path, 'r') as df:
                                    default_config = yaml.safe_load(df)
                                    model_config[key] = default_config
                                    print(f"  - Loaded and merged config for '{key}' from {default_config_path}")
                            else:
                                print(f"  - CRITICAL WARNING: Default config file not found: {default_config_path}")
                del model_config['defaults']
    except Exception as e:
        print(f"Failed to load model config: {e}, using defaults")
    
    cfg_dict = {
        'scaler_type': args.scaler_type,
        'energy_scaler_type': args.energy_scaler_type,
        'optimization_method': args.multi_obj_method,
        'use_gradnorm': args.gradnorm,
        'gradnorm_alpha': args.gradnorm_alpha,
        
        'data': resolved_base_config.copy(),
        'model': model_config, 
        
        'optim': {
            # ======================= 添加下面这一行 =======================
            'use_lr_scheduler': True,
            # ==========================================================
            'optimizer': {'_target_': 'torch.optim.AdamW', 'lr': args.lr, 'weight_decay': 1e-6},
            'lr_scheduler': {'_target_': 'torch.optim.lr_scheduler.ReduceLROnPlateau', 'mode': 'min', 'factor': 0.6, 'patience': 30, 'min_lr': 1e-7}
        },
        
        'train': {
            'deterministic': True,
            'random_seed': args.seed,
            'monitor_metric': 'val_loss',
            'monitor_metric_mode': 'min',
            'pl_trainer': {'max_epochs': args.max_epochs, 'fast_dev_run': False}
        },
        
        'logging': {'val_check_interval': 1},
        
        'core': {'tags': [args.dataset, args.multi_obj_method]}
    }
    
    cfg_dict['model']['gradnorm'] = {
        'enable': args.gradnorm,
        'alpha': args.gradnorm_alpha,
        'lr': args.gradnorm_lr
    }
    
    cfg_dict['model']['multi_objective'] = {
        'method': args.multi_obj_method,
        'weights': args.property_weights,
        'direction': base_config.get('optimization_direction', ['min', 'max']),
        'boundary_theta': args.boundary_theta,
        'init_ideal_points': base_config.get('init_ideal_points', [float('inf'), float('inf')])
    }

    for key in model_config:
        if key not in ['_target_', 'encoder', 'decoder', 'gradnorm', 'multi_objective']:
            cfg_dict['model'][key] = model_config[key]

    cfg_dict['data'].update({
        'scaler_type': args.scaler_type,
        'energy_scaler_type': args.energy_scaler_type,
    })
    
    if 'datamodule' in cfg_dict['data']:
        if 'batch_size' not in cfg_dict['data']['datamodule']:
            cfg_dict['data']['datamodule']['batch_size'] = {}
        cfg_dict['data']['datamodule']['batch_size']['train'] = args.batch_size
        cfg_dict['data']['datamodule']['batch_size']['val'] = args.batch_size
        cfg_dict['data']['datamodule']['batch_size']['test'] = args.batch_size
        
        if 'num_workers' not in cfg_dict['data']['datamodule']:
            cfg_dict['data']['datamodule']['num_workers'] = {}
        cfg_dict['data']['datamodule']['num_workers']['train'] = args.num_workers
        cfg_dict['data']['datamodule']['num_workers']['val'] = args.num_workers
        cfg_dict['data']['datamodule']['num_workers']['test'] = args.num_workers
    
    if args.early_stopping:
        cfg_dict['train']['early_stopping'] = {'patience': args.patience, 'verbose': True}
        cfg_dict['train']['model_checkpoints'] = {'save_top_k': 3, 'verbose': True}
    
    if args.wandb:
        cfg_dict['logging']['wandb'] = {
            'project': args.wandb_project,
            'name': args.wandb_name or f"{args.dataset}_{args.multi_obj_method}",
            'mode': 'offline' if args.wandb_offline else 'online',
        }
        cfg_dict['logging']['wandb_watch'] = {'log': 'gradients', 'log_freq': 100}
        cfg_dict['logging']['lr_monitor'] = {'logging_interval': 'step', 'log_momentum': True}
    
    return OmegaConf.create(cfg_dict)

def build_callbacks(cfg: DictConfig, output_dir: Path) -> List[Callback]:
    callbacks = []
    
    if "lr_monitor" in cfg.logging:
        callbacks.append(LearningRateMonitor(
            logging_interval=cfg.logging.lr_monitor.get('logging_interval', 'step'),
        ))
    
    if "early_stopping" in cfg.train:
        callbacks.append(EarlyStopping(
            monitor=cfg.train.monitor_metric,
            mode=cfg.train.monitor_metric_mode,
            patience=cfg.train.early_stopping.patience,
            verbose=cfg.train.early_stopping.verbose,
        ))
    
    if "model_checkpoints" in cfg.train:
        callbacks.append(ModelCheckpoint(
            dirpath=str(output_dir),
            monitor=cfg.train.monitor_metric,
            mode=cfg.train.monitor_metric_mode,
            save_top_k=cfg.train.model_checkpoints.save_top_k,
            verbose=cfg.train.model_checkpoints.verbose,
        ))
    
    return callbacks

def main():
    args = parse_args()
    
    if 'PROJECT_ROOT' not in os.environ:
        project_root = Path(__file__).resolve().parent.parent
        os.environ['PROJECT_ROOT'] = str(project_root)
        print(f"Setting PROJECT_ROOT to: {project_root}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Enhanced CDVAE Training on Dataset: {args.dataset}")
    
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    cfg = create_hydra_style_config(args, base_config)
    
    print("Configuration created successfully.")
    
    if cfg.train.get('deterministic', True):
        seed_everything(args.seed, workers=True)
    
    print(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)
    
    print(f"Instantiating <{cfg.model._target_}>")
    model = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        _recursive_=False,
    )
    
    model.lattice_scaler = datamodule.lattice_scaler.copy()
    model.scaler = datamodule.scaler.copy()
    
    if hasattr(datamodule, 'energy_scaler') and datamodule.energy_scaler is not None:
        model.energy_scaler = datamodule.energy_scaler.copy()
        torch.save(datamodule.energy_scaler, output_dir / 'energy_scaler.pt')
    
    torch.save(datamodule.lattice_scaler, output_dir / 'lattice_scaler.pt')
    torch.save(datamodule.scaler, output_dir / 'prop_scaler.pt')
    
    callbacks = build_callbacks(cfg, output_dir)
    
    wandb_logger = None
    if "wandb" in cfg.logging and args.wandb:
        wandb_logger = WandbLogger(
            project=cfg.logging.wandb.project,
            name=cfg.logging.wandb.name,
            save_dir=str(output_dir),
            tags=cfg.core.tags,
            mode=cfg.logging.wandb.mode
        )
        wandb_logger.watch(model, log=cfg.logging.wandb_watch.get('log', 'gradients'), log_freq=cfg.logging.wandb_watch.get('log_freq', 100))
    
    OmegaConf.save(cfg, output_dir / "hparams.yaml")
    
    ckpts = sorted(output_dir.glob('*.ckpt'), key=os.path.getmtime, reverse=True)
    ckpt_path = str(ckpts[0]) if ckpts else None
    
    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=cfg.train.get('deterministic', True),
        check_val_every_n_epoch=cfg.logging.val_check_interval,
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
    )
    
    print("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    
    print("Starting testing!")
    trainer.test(datamodule=datamodule, ckpt_path='best')
    
    if wandb_logger:
        wandb_logger.experiment.finish()

if __name__ == "__main__":
    main()