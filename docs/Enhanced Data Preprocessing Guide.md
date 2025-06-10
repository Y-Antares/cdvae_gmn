# Enhanced Data Preprocessing Guide
## 多目标数据预处理系统

### 概述

主要功能：

1. **固定训练集大小控制** - 可以精确指定训练集的样本数量
2. **智能分层采样** - 确保数据分割的均匀性
3. **数据分布分析** - 自动生成数据分布图表
4. **分割质量验证** - 使用统计检验验证数据分割质量
5. **配置管理系统** - 便于管理不同的预处理配置

### 主要改进

#### 1. 固定训练集大小
```bash
# 固定训练集为10000个样本，验证集和测试集按比例缩放
python enhanced_prepare_multi_target_data.py \
  --input data.csv \
  --output_dir processed_data \
  --target_property band_gap \
  --train_size 10000 \
  --val_ratio 0.1 \
  --test_ratio 0.1
```

#### 2. 分层采样保证均匀性
- 基于目标属性值进行分层
- 每层按比例分配到训练/验证/测试集
- 防止数据分布偏差

#### 3. 自动数据分析
生成的分析包括：
- 目标属性分布图
- 形成能与目标属性关系图
- 数据分割质量对比图
- KS统计检验结果

### 使用方法

#### 方法1: 直接使用脚本

```bash
# 基本用法 - 固定训练集大小
python enhanced_prepare_multi_target_data.py \
  --input raw_data/perovskite.csv \
  --output_dir data/perov_10k \
  --target_property dir_gap \
  --train_size 10000 \
  --val_ratio 0.1 \
  --test_ratio 0.1

# 传统比例分割
python enhanced_prepare_multi_target_data.py \
  --input raw_data/materials.csv \
  --output_dir data/materials_80_10_10 \
  --target_property target_property \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1
```

#### 方法2: 使用配置管理器

```bash
# 列出预定义配置
python config_manager.py list

# 使用预定义配置
python config_manager.py run perovskite_bandgap \
  --input data/perovskite.csv \
  --output_dir data/processed

# 覆盖配置参数
python config_manager.py run medium_dev \
  --input data/test.csv \
  --output_dir data/custom \
  --train_size 8000
```

### 关键参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--train_size` | 训练集绝对大小（优先级高于train_ratio） | `10000` |
| `--train_ratio` | 训练集比例（仅当未指定train_size时使用） | `0.8` |
| `--val_ratio` | 验证集比例（相对于总数据） | `0.1` |
| `--test_ratio` | 测试集比例（相对于总数据） | `0.1` |
| `--stratify_bins` | 分层数量（越多越均匀，但需要足够数据） | `10-20` |
| `--random_state` | 随机种子（确保结果可复现） | `42` |

### 输出文件结构

```
output_directory/
├── train.csv                           # 训练集
├── val.csv                            # 验证集
├── test.csv                           # 测试集
├── metadata.json                      # 元数据信息
├── distribution_analysis/             # 数据分布分析
│   ├── dataset_target_distribution.png
│   └── dataset_energy_vs_target.png
└── split_validation/                  # 分割质量验证
    ├── dataset_split_comparison.png
    ├── dataset_split_statistics.csv
    └── dataset_ks_test_results.json
```

### 预定义配置

系统提供以下预定义配置：

1. **small_test** - 快速测试（1000个训练样本）
2. **medium_dev** - 中等开发（5000个训练样本）
3. **large_production** - 大规模生产（20000个训练样本）
4. **perovskite_bandgap** - 钙钛矿带隙专用
5. **carbon_elastic** - 碳材料弹性模量
6. **alloy_thermal** - 合金热导率
7. **traditional_split** - 传统80/10/10分割
8. **data_scarce** - 数据稀缺情况（90/5/5分割）

### 数据质量验证

系统自动进行以下验证：

1. **分布一致性检查** - KS统计检验
2. **统计量对比** - 均值、标准差、中位数
3. **可视化对比** - 直方图、箱线图、累积分布

**KS检验结果解释：**
- p值 > 0.05：分布相似，分割质量好
- p值 ≤ 0.05：分布有显著差异，需要调整参数

### 最佳实践建议

#### 1. 训练集大小选择
- 深度学习模型：至少5000-10000个样本
- 简单模型：1000-5000个样本可能足够
- 复杂任务：可能需要20000+个样本

#### 2. 分层参数调整
- 小数据集（<5000）：`stratify_bins=5-8`
- 中等数据集（5000-20000）：`stratify_bins=10-15`
- 大数据集（>20000）：`stratify_bins=15-25`

#### 3. 验证集和测试集比例
- 充足数据：各10-15%
- 数据稀缺：各5-10%
- 确保验证集和测试集至少有几百个样本

#### 4. 质量检查
- 始终检查生成的分析图表
- 关注KS检验的p值
- 验证训练/验证/测试集的统计量相似性

### 常见问题解决

#### 1. 训练集大小超过总数据量
```
错误: 训练集大小 (15000) 不能大于等于总数据量 (10000)
解决: 减少train_size或增加数据量
```

#### 2. 分层采样失败
```
警告: 某些分层为空
解决: 减少stratify_bins数量
```

#### 3. KS检验p值过低
```
警告: 数据分布不均匀
解决: 增加stratify_bins或检查数据质量
```

### 批量处理示例

```bash
# 处理不同大小的训练集进行对比研究
for size in 1000 2000 5000 10000; do
  python enhanced_prepare_multi_target_data.py \
    --input raw_data/full_dataset.csv \
    --output_dir data/size_study/train_${size} \
    --target_property target_property \
    --train_size ${size} \
    --val_ratio 0.1 \
    --test_ratio 0.1
done
```

### 与CDVAE训练的集成

处理完成后，更新数据配置文件：

```yaml
# conf/data/your_dataset.yaml
root_path: ${oc.env:PROJECT_ROOT}/data/your_processed_data
prop: target_property
target_property: target_property
num_targets: 2
# ... 其他配置
```

然后使用增强的CDVAE进行训练：

```bash
python train_enhanced_cdvae.py \
  --config conf/data/your_dataset.yaml \
  --dataset your_dataset \
  --gradnorm \
  --multi_obj_method tchebycheff \
  --max_epochs 300
```
