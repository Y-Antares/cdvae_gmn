#!/usr/bin/env python3
"""
最终修复版训练脚本
解决所有配置加载和插值问题
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen.io.cif")

import sys
import os
import argparse
import yaml
from pathlib import Path
import re

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
import hydra

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

def resolve_interpolations_in_text(text, context):
    """
    手动解析文本中的插值变量
    将 ${key} 或 ${path.to.key} 替换为实际值
    """
    if not isinstance(text, str):
        return text
    
    # 匹配插值模式
    pattern = r'\$\{([^}]+)\}'
    
    def replace_interpolation(match):
        var_path = match.group(1)
        
        # 特殊处理环境变量
        if var_path.startswith('oc.env:'):
            env_var = var_path.replace('oc.env:', '')
            if env_var in os.environ:
                return os.environ[env_var]
            else:
                print(f"⚠️ 环境变量未设置: {env_var}")
                return match.group(0)
        
        # 解析路径
        if '.' in var_path:
            keys = var_path.split('.')
            value = context
            try:
                for key in keys:
                    value = value[key]
                return str(value)
            except (KeyError, TypeError):
                print(f"⚠️ 无法解析插值变量: {var_path}")
                return match.group(0)  # 返回原始文本
        else:
            # 简单变量名
            if var_path in context:
                return str(context[var_path])
            else:
                print(f"⚠️ 无法解析插值变量: {var_path}")
                return match.group(0)  # 返回原始文本
    
    return re.sub(pattern, replace_interpolation, text)

def resolve_interpolations_in_config(config, context):
    """
    递归解析配置中的所有插值变量
    """
    if isinstance(config, dict):
        resolved = {}
        for key, value in config.items():
            resolved[key] = resolve_interpolations_in_config(value, context)
        return resolved
    elif isinstance(config, list):
        return [resolve_interpolations_in_config(item, context) for item in config]
    elif isinstance(config, str):
        return resolve_interpolations_in_text(config, context)
    else:
        return config

def load_complete_config(data_name, model_name):
    """
    加载完整配置并解决所有插值问题
    """
    print(f"加载数据配置: {data_name}")
    data_config_path = PROJECT_ROOT / "conf/data" / f"{data_name}.yaml"
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"加载模型配置: {model_name}")
    model_config_path = PROJECT_ROOT / "conf/model" / f"{model_name}.yaml"
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    # 创建更完整的插值上下文
    interpolation_context = {
        'data': data_config,
        'model': model_config,
        # 添加环境变量
        'oc': {
            'env': {
                'PROJECT_ROOT': str(PROJECT_ROOT)
            }
        },
        # 添加 Hydra 相关变量
        'expname': f"{data_name}_{model_name}",
        'now': {
            '%Y-%m-%d': '2024-06-10',
            '%H-%M-%S': '12-00-00'
        },
        'id': '001',
        'num': '1',
        # 将data和model中的所有变量提升到顶层
        **data_config,
        # 添加一些model的关键变量到顶层
        'latent_dim': model_config.get('latent_dim', 256),
        'max_neighbors': model_config.get('max_neighbors', 20),
        'radius': model_config.get('radius', 7.0),
    }
    
    print("解析模型配置中的 defaults...")
    
    # 处理 defaults 中的 encoder 和 decoder
    if 'defaults' in model_config:
        for default_item in model_config['defaults']:
            if isinstance(default_item, dict):
                if 'encoder' in default_item:
                    encoder_name = default_item['encoder']
                    encoder_path = PROJECT_ROOT / "conf/model/encoder" / f"{encoder_name}.yaml"
                    print(f"  加载编码器: {encoder_name}")
                    
                    with open(encoder_path, 'r') as f:
                        encoder_config = yaml.safe_load(f)
                    
                    # 解析编码器配置中的插值
                    print("  解析编码器插值...")
                    encoder_config = resolve_interpolations_in_config(encoder_config, interpolation_context)
                    model_config['encoder'] = encoder_config
                
                if 'decoder' in default_item:
                    decoder_name = default_item['decoder']
                    decoder_path = PROJECT_ROOT / "conf/model/decoder" / f"{decoder_name}.yaml"
                    print(f"  加载解码器: {decoder_name}")
                    
                    with open(decoder_path, 'r') as f:
                        decoder_config = yaml.safe_load(f)
                    
                    # 解析解码器配置中的插值
                    print("  解析解码器插值...")
                    decoder_config = resolve_interpolations_in_config(decoder_config, interpolation_context)
                    model_config['decoder'] = decoder_config
    
    # 解析模型配置中剩余的插值
    print("解析模型配置中的其他插值...")
    model_config = resolve_interpolations_in_config(model_config, interpolation_context)
    
    # 解析数据配置中的插值
    print("解析数据配置中的插值...")
    data_config = resolve_interpolations_in_config(data_config, interpolation_context)
    
    # 移除 defaults 键，因为我们已经手动处理了
    if 'defaults' in model_config:
        del model_config['defaults']
    
    # 加载其他默认配置
    default_configs = {}
    
    # 加载训练配置
    train_config_path = PROJECT_ROOT / "conf/train/default.yaml"
    if train_config_path.exists():
        with open(train_config_path, 'r') as f:
            train_config = yaml.safe_load(f)
        default_configs['train'] = train_config
    
    # 加载优化器配置
    optim_config_path = PROJECT_ROOT / "conf/optim/default.yaml"
    if optim_config_path.exists():
        with open(optim_config_path, 'r') as f:
            optim_config = yaml.safe_load(f)
        default_configs['optim'] = optim_config
    
    # 加载日志配置
    logging_config_path = PROJECT_ROOT / "conf/logging/default.yaml"
    if logging_config_path.exists():
        with open(logging_config_path, 'r') as f:
            logging_config = yaml.safe_load(f)
        default_configs['logging'] = logging_config
    
    print("配置加载完成，组装最终配置...")
    
    # 组装最终配置
    final_config = {
        'data': data_config,
        'model': model_config,
        **default_configs
    }
    
    return OmegaConf.create(final_config)

def fix_numeric_types(config):
    """修复配置中的数值类型"""
    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, str):
                # 尝试转换为数值
                try:
                    if '.' in value:
                        config[key] = float(value)
                    else:
                        config[key] = int(value)
                except ValueError:
                    pass  # 保持字符串
            elif isinstance(value, (dict, list)):
                fix_numeric_types(value)
    elif isinstance(config, list):
        for item in config:
            fix_numeric_types(item)

def fix_all_numeric_types(cfg):
    """修复整个配置中的数值类型"""
    print("修复数值类型...")
    
    # 修复优化器配置
    if hasattr(cfg, 'optim'):
        fix_numeric_types(cfg.optim)
    
    # 修复数据配置
    if hasattr(cfg, 'data'):
        fix_numeric_types(cfg.data)
        # 特别注意这些关键的数值参数
        numeric_keys = [
            'preprocess_workers', 'num_targets', 'max_atoms', 
            'train_max_epochs', 'early_stopping_patience', 
            'teacher_forcing_max_epoch'
        ]
        for key in numeric_keys:
            if hasattr(cfg.data, key) and isinstance(getattr(cfg.data, key), str):
                try:
                    setattr(cfg.data, key, int(getattr(cfg.data, key)))
                    print(f"  转换 {key}: {getattr(cfg.data, key)} (int)")
                except ValueError:
                    pass
    
    # 修复模型配置
    if hasattr(cfg, 'model'):
        fix_numeric_types(cfg.model)
    
    # 修复训练配置
    if hasattr(cfg, 'train'):
        fix_numeric_types(cfg.train)

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
    
    print(f"Enhanced CDVAE Training")
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
        # 加载完整配置
        cfg = load_complete_config(args.data, args.model)
        
        # 修复所有数值类型
        fix_all_numeric_types(cfg)
        
        # 设置训练参数
        cfg.train.pl_trainer.max_epochs = args.max_epochs
        
        print("✅ 配置加载成功")
        print(f"模型配置键: {list(cfg.model.keys())}")
        print(f"包含 encoder: {'encoder' in cfg.model}")
        print(f"包含 decoder: {'decoder' in cfg.model}")
        
        # 设置随机种子
        seed_everything(42)
        
        # 实例化数据模块之前，确保数据集配置中的数值类型正确
        print("修复数据集配置中的数值类型...")
        if hasattr(cfg.data.datamodule, 'datasets'):
            for dataset_name in ['train', 'val', 'test']:
                if hasattr(cfg.data.datamodule.datasets, dataset_name):
                    dataset_cfg = getattr(cfg.data.datamodule.datasets, dataset_name)
                    if isinstance(dataset_cfg, list):
                        # val 和 test 可能是列表
                        for i, ds_cfg in enumerate(dataset_cfg):
                            fix_numeric_types(ds_cfg)
                            # 特别处理 preprocess_workers
                            if hasattr(ds_cfg, 'preprocess_workers') and isinstance(ds_cfg.preprocess_workers, str):
                                ds_cfg.preprocess_workers = int(ds_cfg.preprocess_workers)
                                print(f"  修复 {dataset_name}[{i}].preprocess_workers: {ds_cfg.preprocess_workers}")
                    else:
                        # train 通常是单个配置
                        fix_numeric_types(dataset_cfg)
                        # 特别处理 preprocess_workers
                        if hasattr(dataset_cfg, 'preprocess_workers') and isinstance(dataset_cfg.preprocess_workers, str):
                            dataset_cfg.preprocess_workers = int(dataset_cfg.preprocess_workers)
                            print(f"  修复 {dataset_name}.preprocess_workers: {dataset_cfg.preprocess_workers}")
        
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
        
        callbacks.append(LearningRateMonitor(
            logging_interval='step',
            log_momentum=True,
        ))
        
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=50,
            verbose=True,
        ))
        
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
            deterministic=True,
            max_epochs=args.max_epochs,
            check_val_every_n_epoch=1,
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