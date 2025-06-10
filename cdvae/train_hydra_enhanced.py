#!/usr/bin/env python3
"""
基于原始 Hydra 的增强训练脚本
使用 Hydra 的 compose API 避免插值问题
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen.io.cif")

import sys
import os
import argparse
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import compose, initialize_config_dir

# 设置环境
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PROJECT_ROOT'] = str(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))

def parse_args():
    parser = argparse.ArgumentParser(description='Enhanced CDVAE Training')
    parser.add_argument('--data', type=str, default='perov_1k', help='Data config name')
    parser.add_argument('--model', type=str, default='enhanced_cdvae', help='Model config name')
    parser.add_argument('--max_epochs', type=int, default=300, help='Maximum training epochs')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--output_dir', type=str, default='results/default', help='Output directory')
    parser.add_argument('--clean_start', action='store_true', help='Start training from scratch')
    return parser.parse_args()

def patch_minmax_scaler():
    """修补 MinMaxScalerTorch 的 copy 方法"""
    from cdvae.common.data_utils import MinMaxScalerTorch
    
    original_copy = MinMaxScalerTorch.copy
    
    def fixed_copy(self):
        new_scaler = MinMaxScalerTorch(
            min_val=self.min_val,
            max_val=self.max_val,
            mins=self.mins.clone().detach() if self.mins is not None else None,
            maxs=self.maxs.clone().detach() if self.maxs is not None else None
        )
        # 确保复制 ranges 属性
        if hasattr(self, 'ranges') and self.ranges is not None:
            new_scaler.ranges = self.ranges.clone().detach()
        return new_scaler
    
    MinMaxScalerTorch.copy = fixed_copy
    print("✅ 已修补 MinMaxScalerTorch.copy 方法")

def main():
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置必要的环境变量
    os.environ.setdefault('HYDRA_JOBS', str(output_dir.parent / 'hydra_jobs'))
    os.environ.setdefault('WANDB_DIR', str(output_dir.parent / 'wandb'))
    
    print(f"Enhanced CDVAE Training (Hydra Mode)")
    print(f"Data: {args.data}")
    print(f"Model: {args.model}")
    print(f"Output directory: {output_dir}")
    print(f"Max epochs: {args.max_epochs}")
    
    # 处理检查点
    if args.clean_start:
        ckpts = list(output_dir.glob('*.ckpt'))
        if ckpts:
            backup_dir = output_dir / "backup_checkpoints"
            backup_dir.mkdir(exist_ok=True)
            for ckpt in ckpts:
                ckpt.rename(backup_dir / ckpt.name)
            print(f"Moved {len(ckpts)} checkpoints to backup")
    
    # 应用补丁
    patch_minmax_scaler()
    
    try:
        # 使用 Hydra compose API
        config_dir = str(PROJECT_ROOT / "conf")
        
        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            # 组合配置，使用完整的 Hydra 系统
            cfg = compose(
                config_name="default",
                overrides=[
                    f"data={args.data}",
                    f"model={args.model}",
                    f"train.pl_trainer.max_epochs={args.max_epochs}",
                    f"expname={args.data}_{args.model}",
                    f"hydra.run.dir={output_dir}",  # 覆盖输出目录
                    "hydra.job.chdir=False",  # 不改变工作目录
                ]
            )
            
            print("✅ Hydra 配置创建成功")
            print(f"模型配置键: {list(cfg.model.keys())}")
            print(f"包含 encoder: {'encoder' in cfg.model}")
            print(f"包含 decoder: {'decoder' in cfg.model}")
            
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
            
            print("✅ 模型创建成功")
            print(f"模型类型: {type(model)}")
            
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
            
            # 保存其他标准化器
            torch.save(datamodule.lattice_scaler, output_dir / 'lattice_scaler.pt')
            torch.save(datamodule.scaler, output_dir / 'prop_scaler.pt')
            
            # 构建回调
            callbacks = []
            
            if "lr_monitor" in cfg.logging:
                callbacks.append(LearningRateMonitor(
                    logging_interval=cfg.logging.lr_monitor.logging_interval,
                    log_momentum=cfg.logging.lr_monitor.log_momentum,
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
                    dirpath=output_dir,
                    monitor=cfg.train.monitor_metric,
                    mode=cfg.train.monitor_metric_mode,
                    save_top_k=cfg.train.model_checkpoints.save_top_k,
                    verbose=cfg.train.model_checkpoints.verbose,
                    filename='epoch={epoch:02d}-val_loss={val_loss:.3f}',
                ))
            
            # 手动添加基础回调（如果配置中没有）
            if not any(isinstance(cb, LearningRateMonitor) for cb in callbacks):
                callbacks.append(LearningRateMonitor(logging_interval='step', log_momentum=True))
            
            if not any(isinstance(cb, EarlyStopping) for cb in callbacks):
                callbacks.append(EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=True))
            
            if not any(isinstance(cb, ModelCheckpoint) for cb in callbacks):
                callbacks.append(ModelCheckpoint(
                    dirpath=output_dir,
                    monitor='val_loss',
                    mode='min',
                    save_top_k=3,
                    verbose=True,
                    filename='epoch={epoch:02d}-val_loss={val_loss:.3f}',
                ))
            
            # 设置logger
            wandb_logger = None
            if args.wandb:
                wandb_logger = WandbLogger(
                    project='enhanced-cdvae',
                    name=f"{args.data}_{args.model}",
                    save_dir=output_dir,
                    tags=['enhanced', 'gradnorm', 'multi-objective'],
                )
                wandb_logger.watch(model, log='gradients', log_freq=100)
            
            # 保存配置
            yaml_conf = OmegaConf.to_yaml(cfg=cfg)
            (output_dir / "hparams.yaml").write_text(yaml_conf)
            
            # 检查是否有检查点
            ckpts = list(output_dir.glob('*.ckpt'))
            ckpt = None
            if ckpts and not args.clean_start:
                ckpt_epochs = np.array([int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts])
                ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
                print(f"Found checkpoint: {ckpt}")
            
            # 实例化训练器
            print("Instantiating the Trainer")
            trainer = pl.Trainer(
                default_root_dir=output_dir,
                logger=wandb_logger,
                callbacks=callbacks,
                deterministic=cfg.train.deterministic,
                max_epochs=args.max_epochs,
                check_val_every_n_epoch=cfg.logging.val_check_interval,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                enable_progress_bar=True,
                enable_model_summary=True,
            )
            
            # 记录超参数
            from cdvae.common.utils import log_hyperparameters
            log_hyperparameters(trainer=trainer, model=model, cfg=cfg)
            
            # 开始训练
            print("🚀 Starting training!")
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt)
            
            print("🧪 Starting testing!")
            trainer.test(datamodule=datamodule)
            
            print(f"🎉 Training completed! Results saved to: {output_dir}")
            
            # 关闭logger
            if wandb_logger is not None:
                wandb_logger.experiment.finish()
    
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)