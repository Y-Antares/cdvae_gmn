# Enhanced CDVAE Quick Start Guide
## 快速入门指南

### 🚀 5分钟快速开始

#### 第一步: 环境准备
```bash
# 克隆项目
git clone <your-repo-url>
cd cdvae

# 安装依赖
conda create -n cdvae python=3.8
conda activate cdvae
pip install -r requirements.txt

# 设置环境变量
export PROJECT_ROOT="/path/to/enhanced-cdvae"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
```

#### 第二步: 数据准备 (2分钟)
```bash
# 使用示例数据或准备你的数据
# CSV格式: material_id, cif, formation_energy_per_atom, target_property

# 快速处理示例数据
python config_manager.py run small_test \
  --input examples/sample_data.csv \
  --output_dir data/quick_test \
  --train_size 1000

# 检查结果
ls data/quick_test/
# 应该看到: train.csv, val.csv, test.csv, metadata.json
```

#### 第三步: 模型训练 (主要时间)
```bash
# 快速训练 (小规模测试)
python train_enhanced_cdvae.py \
  --config conf/data/quick_test.yaml \
  --dataset quick_test \
  --gradnorm \
  --multi_obj_method weighted \
  --max_epochs 50 \
  --output_dir results/quick_test

# 生产级训练 (完整训练)
python train_enhanced_cdvae.py \
  --config conf/data/your_dataset.yaml \
  --dataset your_dataset \
  --gradnorm \
  --multi_obj_method tchebycheff \
  --max_epochs 300 \
  --output_dir results/production
```

#### 第四步: 评估和分析 (2分钟)
```bash
# 快速评估
python enhanced_compute_metrics.py \
  --root_path results/quick_test \
  --methods weighted \
  --target_types combined

# 可视化结果
python visualize_pareto.py \
  --model_path results/quick_test \
  --methods weighted \
  --target_type combined \
  --save_data \
  --show_plot
```

---

### 📋 常用命令速查

#### 数据预处理命令
```bash
# 列出预定义配置
python config_manager.py list

# 查看特定配置
python config_manager.py show perovskite_bandgap

# 使用配置处理数据
python config_manager.py run [CONFIG_NAME] \
  --input [INPUT.csv] \
  --output_dir [OUTPUT_DIR] \
  --train_size [SIZE]
```

#### 训练命令模板
```bash
python train_enhanced_cdvae.py \
  --config [CONFIG.yaml] \
  --dataset [DATASET_NAME] \
  [--gradnorm] \
  --multi_obj_method [weighted|tchebycheff|boundary] \
  --max_epochs [EPOCHS] \
  --output_dir [OUTPUT_DIR]
```

#### 评估命令模板
```bash
# 标准评估
python evaluate.py \
  --model_path [MODEL_PATH] \
  --tasks recon gen opt \
  --target_type combined \
  --optimization_method [METHOD]

# 增强评估
python enhanced_compute_metrics.py \
  --root_path [MODEL_PATH] \
  --methods [METHOD1] [METHOD2] \
  --target_types combined
```

---

### 🎯 典型使用场景

#### 场景1: 钙钛矿太阳能材料
```bash
# 1. 数据准备
python config_manager.py run perovskite_bandgap \
  --input data/perovskite_solar.csv \
  --output_dir data/perov_processed \
  --train_size 12000

# 2. 训练模型
python train_enhanced_cdvae.py \
  --config conf/data/perov_processed.yaml \
  --dataset perovskite \
  --gradnorm \
  --multi_obj_method tchebycheff \
  --max_epochs 300 \
  --output_dir results/perov_solar

# 3. 分析结果
python enhanced_compute_metrics.py \
  --root_path results/perov_solar \
  --methods tchebycheff \
  --target_types combined
```

#### 场景2: 高强度材料发现
```bash
# 1. 数据准备
python config_manager.py run carbon_elastic \
  --input data/carbon_materials.csv \
  --output_dir data/carbon_processed \
  --train_size 8000

# 2. 训练模型
python train_enhanced_cdvae.py \
  --config conf/data/carbon_processed.yaml \
  --dataset carbon \
  --gradnorm \
  --multi_obj_method boundary \
  --max_epochs 350 \
  --output_dir results/carbon_strength

# 3. 可视化结果
python visualize_pareto.py \
  --model_path results/carbon_strength \
  --methods boundary \
  --target_type combined \
  --save_data
```

#### 场景3: 方法对比研究
```bash
# 批量训练不同方法
methods=("weighted" "tchebycheff" "boundary")
for method in "${methods[@]}"; do
  for gradnorm in "" "--gradnorm"; do
    suffix=$([ -n "$gradnorm" ] && echo "_gradnorm" || echo "_fixed")
    
    python train_enhanced_cdvae.py \
      --config conf/data/dataset.yaml \
      --dataset comparison \
      $gradnorm \
      --multi_obj_method $method \
      --max_epochs 300 \
      --output_dir results/${method}${suffix}
  done
done

# 综合分析
python scripts/compare_all_methods.py \
  --result_dirs results/*/ \
  --output_dir comparison_analysis
```

---

### ⚙️ 配置文件快速设置

#### 数据配置模板 (conf/data/your_dataset.yaml)
```yaml
root_path: ${oc.env:PROJECT_ROOT}/data/your_processed_data
prop: target_property
target_property: target_property
num_targets: 2
niggli: true
primitive: false
graph_method: crystalnn
lattice_scale_method: scale_length
preprocess_workers: 16
readout: mean
max_atoms: 200
otf_graph: false
eval_model_name: your_dataset

# 多目标配置
multi_target: true
energy_weight: 0.6
property_weights: [0.6, 0.4]
optimization_method: "tchebycheff"
optimization_direction: [min, max]

train_max_epochs: 300
early_stopping_patience: 50
teacher_forcing_max_epoch: 150

datamodule:
  _target_: cdvae.pl_data.datamodule.CrystDataModule
  datasets:
    train:
      _target_: cdvae.pl_data.dataset.CrystDataset
      name: Your dataset train
      path: ${data.root_path}/train.csv
      prop: ${data.prop}
      # ... 其他配置

  num_workers:
    train: 8
    val: 4
    test: 4

  batch_size:
    train: 256
    val: 128
    test: 128
```

#### 模型配置模板 (conf/model/enhanced_cdvae.yaml)
```yaml
_target_: enhanced_cdvae.EnhancedCDVAE
hidden_dim: 256
latent_dim: 256
fc_num_layers: 1
max_atoms: ${data.max_atoms}

# 基础损失权重
cost_natom: 1.0
cost_coord: 10.0
cost_type: 1.0
cost_lattice: 10.0
cost_composition: 1.0
cost_edge: 10.0
cost_property: 1.0
beta: 0.01

# 多目标优化配置
multi_objective:
  method: "tchebycheff"
  weights: [0.6, 0.4]
  direction: [min, max]
  boundary_theta: 5.0
  init_ideal_points: [999.0, 999.0]

# GradNorm配置
gradnorm:
  enable: true
  alpha: 1.5
  lr: 0.025

# 其他训练参数
teacher_forcing_lattice: true
teacher_forcing_max_epoch: ${data.teacher_forcing_max_epoch}
predict_property: true

defaults:
  - encoder: dimenet
  - decoder: gemnet
```

---

### 🔧 故障排除

#### 常见问题及解决方案

**问题1: 内存不足**
```bash
# 解决方案: 减少批大小
# 在配置文件中修改:
batch_size:
  train: 128  # 从256减少到128
  val: 64
  test: 64
```

**问题2: 训练不收敛**
```bash
# 解决方案: 调整学习率和GradNorm参数
# 在配置文件中修改:
gradnorm:
  alpha: 1.0    # 从1.5减少到1.0
  lr: 0.01      # 从0.025减少到0.01
```

**问题3: 数据处理失败**
```bash
# 检查数据格式
head -5 your_data.csv
# 确保包含必要列: material_id, cif, formation_energy_per_atom, target_property

# 检查数据质量
python -c "
import pandas as pd
df = pd.read_csv('your_data.csv')
print('Data shape:', df.shape)
print('Missing values:', df.isnull().sum())
print('Columns:', list(df.columns))
"
```

**问题4: GPU内存不足**
```bash
# 监控GPU使用
watch -n 1 nvidia-smi

# 减少模型参数
# 在配置文件中修改:
hidden_dim: 128    # 从256减少到128
latent_dim: 128    # 从256减少到128
```

---

### 📊 性能监控

#### 训练过程监控
```bash
# 查看训练日志
tail -f results/your_experiment/training.log

# 监控GPU使用
watch -n 1 nvidia-smi

# 检查任务权重演化 (如果使用GradNorm)
python -c "
import torch
import matplotlib.pyplot as plt

# 加载检查点
ckpt = torch.load('results/your_experiment/last.ckpt')
if 'gradnorm.task_weights' in ckpt['state_dict']:
    weights = ckpt['state_dict']['gradnorm.task_weights']
    print('Current task weights:', weights)
"
```

#### 评估结果检查
```bash
# 检查评估结果
cat results/your_experiment/enhanced_evaluation/evaluation_summary.json

# 查看帕累托前沿
ls results/your_experiment/enhanced_evaluation/*/
# 应该包含: pareto_front.png, pareto_data.csv, detailed_metrics.json
```

---

### 🎓 学习路径建议

#### 初学者 (第1-2周)
1. 运行快速示例，理解基本流程
2. 阅读算法介绍，理解核心概念
3. 使用预定义配置处理自己的数据
4. 训练第一个模型并查看结果

#### 进阶用户 (第3-4周)
1. 对比不同多目标优化方法
2. 实验GradNorm vs 固定权重
3. 调优超参数提升性能
4. 分析帕累托前沿质量

#### 高级用户 (第5-8周)
1. 深入理解各算法原理
2. 修改模型架构适应特定问题
3. 开发新的评估指标
4. 集成额外的物理约束

#### 研究者 (持续)
1. 发表方法改进和应用论文
2. 贡献代码到开源社区
3. 扩展到新的材料体系
4. 开发下一代算法

---

### 📚 推荐资源

#### 理论学习
- **多目标优化**: Miettinen - "Nonlinear Multiobjective Optimization"
- **深度学习**: Goodfellow - "Deep Learning"
- **图神经网络**: Hamilton - "Graph Representation Learning"
- **材料信息学**: Butler - "Machine learning for molecular and materials science"

#### 实践工具
- **PyTorch Geometric**: 图神经网络框架
- **Pymatgen**: 材料科学Python库
- **ASE**: 原子模拟环境
- **VESTA**: 晶体结构可视化

#### 数据资源
- **Materials Project**: 材料数据库
- **OQMD**: 开放量子材料数据库
- **ICSD**: 无机晶体结构数据库
- **COD**: 晶体学开放数据库

这个快速入门指南为用户提供了从安装到高级应用的完整路径，确保能够快速上手并充分利用Enhanced CDVAE系统的强大功能。