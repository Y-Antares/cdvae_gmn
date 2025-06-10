#!/usr/bin/env python3
"""
修复配置加载问题，确保 encoder 配置正确传递
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
from omegaconf import OmegaConf
import hydra

# 设置环境
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PROJECT_ROOT'] = str(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))

def debug_config_loading():
    """调试配置加载过程"""
    
    print("=== 配置加载调试 ===")
    
    # 1. 检查 enhanced_cdvae.yaml
    model_config_path = PROJECT_ROOT / "conf/model/enhanced_cdvae.yaml"
    print(f"检查模型配置: {model_config_path}")
    
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    print("Model config keys:", list(model_config.keys()))
    print("Has 'encoder'?", 'encoder' in model_config)
    print("Has 'defaults'?", 'defaults' in model_config)
    
    if 'defaults' in model_config:
        print("Defaults:", model_config['defaults'])
    
    # 2. 检查 encoder 配置文件
    if 'defaults' in model_config:
        for default_item in model_config['defaults']:
            if isinstance(default_item, dict) and 'encoder' in default_item:
                encoder_name = default_item['encoder']
                encoder_path = PROJECT_ROOT / "conf/model/encoder" / f"{encoder_name}.yaml"
                print(f"\n检查编码器配置: {encoder_path}")
                
                if encoder_path.exists():
                    with open(encoder_path, 'r') as f:
                        encoder_config = yaml.safe_load(f)
                    print("Encoder config keys:", list(encoder_config.keys()))
                    print("Encoder config:", encoder_config)
                else:
                    print(f"❌ 编码器配置文件不存在: {encoder_path}")
    
    # 3. 检查父类需要什么参数
    print("\n=== 检查父类构造函数 ===")
    model_file = PROJECT_ROOT / "cdvae/pl_modules/model.py"
    
    # 查找 __init__ 方法
    with open(model_file, 'r') as f:
        lines = f.readlines()
    
    in_init = False
    for i, line in enumerate(lines):
        if 'def __init__' in line:
            in_init = True
            print(f"第{i+1}行: {line.strip()}")
        elif in_init and line.strip().startswith('def '):
            break
        elif in_init and ('encoder' in line or 'self.hparams.encoder' in line):
            print(f"第{i+1}行: {line.strip()}")

def fix_config_and_load():
    """修复配置并尝试加载"""
    
    print("\n=== 修复配置 ===")
    
    # 手动构建完整配置
    def load_complete_config():
        # 加载基础模型配置
        model_config_path = PROJECT_ROOT / "conf/model/enhanced_cdvae.yaml"
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        
        # 加载编码器配置
        encoder_config_path = PROJECT_ROOT / "conf/model/encoder/dimenet.yaml"
        if encoder_config_path.exists():
            with open(encoder_config_path, 'r') as f:
                encoder_config = yaml.safe_load(f)
            model_config['encoder'] = encoder_config
            print("✅ 成功加载编码器配置")
        else:
            # 使用默认编码器配置
            model_config['encoder'] = {
                '_target_': 'cdvae.pl_modules.encoder.DimeNetEncoder',
                'atom_embedding_dim': 92,
                'num_message_passing': 3,
                'num_attention_heads': 4,
                'hidden_dim': 256,
                'act_fn': 'silu',
                'dis_emb': True,
            }
            print("⚠️ 使用默认编码器配置")
        
        # 加载解码器配置
        decoder_config_path = PROJECT_ROOT / "conf/model/decoder/gemnet.yaml"
        if decoder_config_path.exists():
            with open(decoder_config_path, 'r') as f:
                decoder_config = yaml.safe_load(f)
            model_config['decoder'] = decoder_config
            print("✅ 成功加载解码器配置")
        else:
            # 使用默认解码器配置
            model_config['decoder'] = {
                '_target_': 'cdvae.pl_modules.decoder.GemNetDecoder',
                'atom_embedding_dim': 92,
                'num_message_passing': 3,
                'num_attention_heads': 4,
                'hidden_dim': 256,
                'act_fn': 'silu',
            }
            print("⚠️ 使用默认解码器配置")
        
        return model_config
    
    model_config = load_complete_config()
    
    print("\n最终模型配置包含的键:")
    for key in sorted(model_config.keys()):
        print(f"  - {key}")
    
    # 测试实例化
    print("\n=== 测试模型实例化 ===")
    try:
        cfg_model = OmegaConf.create(model_config)
        
        # 添加必需的其他配置
        cfg_optim = OmegaConf.create({
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
        })
        
        cfg_data = OmegaConf.create({
            'scaler_type': 'minmax',
            'energy_scaler_type': 'minmax',
        })
        
        cfg_logging = OmegaConf.create({
            'val_check_interval': 1,
        })
        
        print("尝试实例化模型...")
        model = hydra.utils.instantiate(
            cfg_model,
            optim=cfg_optim,
            data=cfg_data,
            logging=cfg_logging,
            _recursive_=False,
        )
        print("✅ 模型实例化成功!")
        print(f"模型类型: {type(model)}")
        
        return cfg_model
        
    except Exception as e:
        print(f"❌ 模型实例化失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_fixed_training_script():
    """创建修复后的训练脚本"""
    
    script_content = '''#!/usr/bin/env python3
"""
修复后的训练脚本，确保配置正确加载
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen.io.cif")

import sys
import os
import argparse
import yaml
from pathlib import Path

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

def load_complete_model_config():
    """加载完整的模型配置，确保所有必需组件都存在"""
    
    # 加载基础模型配置
    model_config_path = PROJECT_ROOT / "conf/model/enhanced_cdvae.yaml"
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    # 确保有编码器配置
    if 'encoder' not in model_config:
        encoder_config_path = PROJECT_ROOT / "conf/model/encoder/dimenet.yaml"
        if encoder_config_path.exists():
            with open(encoder_config_path, 'r') as f:
                encoder_config = yaml.safe_load(f)
            model_config['encoder'] = encoder_config
        else:
            model_config['encoder'] = {
                '_target_': 'cdvae.pl_modules.encoder.DimeNetEncoder',
                'atom_embedding_dim': 92,
                'num_message_passing': 3,
                'num_attention_heads': 4,
                'hidden_dim': 256,
                'act_fn': 'silu',
                'dis_emb': True,
            }
    
    # 确保有解码器配置
    if 'decoder' not in model_config:
        decoder_config_path = PROJECT_ROOT / "conf/model/decoder/gemnet.yaml"
        if decoder_config_path.exists():
            with open(decoder_config_path, 'r') as f:
                decoder_config = yaml.safe_load(f)
            model_config['decoder'] = decoder_config
        else:
            model_config['decoder'] = {
                '_target_': 'cdvae.pl_modules.decoder.GemNetDecoder',
                'atom_embedding_dim': 92,
                'num_message_passing': 3,
                'num_attention_heads': 4,
                'hidden_dim': 256,
                'act_fn': 'silu',
            }
    
    return model_config

def main():
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Enhanced CDVAE Training")
    print(f"Data: {args.data}")
    print(f"Model: {args.model}")
    print(f"Output directory: {output_dir}")
    
    # 处理检查点
    if args.clean_start:
        ckpts = list(output_dir.glob('*.ckpt'))
        if ckpts:
            backup_dir = output_dir / "backup_checkpoints"
            backup_dir.mkdir(exist_ok=True)
            for ckpt in ckpts:
                ckpt.rename(backup_dir / ckpt.name)
            print(f"Moved {len(ckpts)} checkpoints to backup")
    
    # 加载数据配置
    data_config_path = PROJECT_ROOT / "conf/data" / f"{args.data}.yaml"
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # 加载完整模型配置
    model_config = load_complete_model_config()
    
    # 创建配置
    cfg_dict = {
        'data': data_config,
        'model': model_config,
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
                'max_epochs': args.max_epochs,
            }
        },
        'logging': {
            'val_check_interval': 1,
        },
    }
    
    cfg = OmegaConf.create(cfg_dict)
    
    print("Configuration created successfully")
    print(f"Model config keys: {list(cfg.model.keys())}")
    print(f"Has encoder: {'encoder' in cfg.model}")
    print(f"Has decoder: {'decoder' in cfg.model}")
    
    # 设置随机种子
    seed_everything(42)
    
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
    print("Setting up scalers...")
    model.lattice_scaler = datamodule.lattice_scaler.copy()
    model.scaler = datamodule.scaler.copy()
    
    # 构建回调
    callbacks = []
    callbacks.append(LearningRateMonitor(logging_interval='step', log_momentum=True))
    callbacks.append(EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=True))
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
        )
    
    # 实例化训练器
    print("Instantiating the Trainer")
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=True,
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
    )
    
    # 开始训练
    print("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
'''
    
    script_path = PROJECT_ROOT / "cdvae/train_fixed.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"✅ 创建修复后的训练脚本: {script_path}")
    return script_path

if __name__ == "__main__":
    # 运行调试
    debug_config_loading()
    
    # 尝试修复配置
    fixed_config = fix_config_and_load()
    
    if fixed_config:
        # 创建修复后的训练脚本
        script_path = create_fixed_training_script()
        print(f"\n使用修复后的脚本运行:")
        print(f"python {script_path} --data perov_1k --model enhanced_cdvae --max_epochs 300 --output_dir results/test --clean_start")