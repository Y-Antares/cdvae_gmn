
# ================================
# 使用示例脚本
# ================================

#!/bin/bash
# evaluation_examples.sh
# Enhanced Model Evaluation Usage Examples

echo "Enhanced Model Evaluation Examples"
echo "=================================="

# 示例1: 基础评估
echo "示例1: 基础模型评估"
python enhanced_model_evaluator.py \
  --model_path results/perov_gradnorm_tchebycheff \
  --checkpoint results/perov_gradnorm_tchebycheff/best.ckpt \
  --test_data results/perov_gradnorm_tchebycheff/test_results.pt \
  --output_dir evaluation/basic_eval \
  --batch_size 64

# 示例2: 包含域外数据的完整评估
echo "示例2: 完整评估（包含域外数据）"
python enhanced_model_evaluator.py \
  --model_path results/perov_gradnorm_tchebycheff \
  --checkpoint results/perov_gradnorm_tchebycheff/best.ckpt \
  --test_data results/perov_gradnorm_tchebycheff/test_results.pt \
  --ood_data data/ood_perovskites.pt \
  --output_dir evaluation/complete_eval \
  --device cuda \
  --batch_size 32

# 示例3: 批量评估多个模型
echo "示例3: 批量评估"
models=(
  "results/perov_gradnorm_weighted"
  "results/perov_gradnorm_tchebycheff" 
  "results/perov_gradnorm_boundary"
  "results/perov_fixed_weighted"
  "results/perov_fixed_tchebycheff"
)

for model in "${models[@]}"; do
  model_name=$(basename "$model")
  echo "评估模型: $model_name"
  
  python enhanced_model_evaluator.py \
    --model_path "$model" \
    --checkpoint "$model/best.ckpt" \
    --test_data "$model/test_results.pt" \
    --output_dir "evaluation/batch_eval/$model_name" \
    --batch_size 64
done

# 示例4: 自定义评估配置
echo "示例4: 使用自定义配置"
python enhanced_model_evaluator.py \
  --model_path results/carbon_elastic \
  --checkpoint results/carbon_elastic/best.ckpt \
  --test_data results/carbon_elastic/test_results.pt \
  --config evaluation_configs/carbon_materials.yaml \
  --output_dir evaluation/carbon_eval

# 示例5: 快速评估（仅核心指标）
echo "示例5: 快速评估"
python enhanced_model_evaluator.py \
  --model_path results/quick_test \
  --checkpoint results/quick_test/last.ckpt \
  --test_data results/quick_test/test_results.pt \
  --output_dir evaluation/quick_eval \
  --tasks reconstruction generation property_prediction \
  --batch_size 128

---

# ================================
# Python使用示例
# ================================

"""
Enhanced Model Evaluator Python Usage Examples
增强模型评估器Python使用示例
"""

import torch
import numpy as np
from pathlib import Path
from enhanced_model_evaluator import ComprehensiveModelEvaluator

# 示例1: 基础Python API使用
def example_basic_evaluation():
    """基础评估示例"""
    
    # 创建评估器
    evaluator = ComprehensiveModelEvaluator(
        model_path='results/perov_gradnorm_tchebycheff',
        data_config={
            'max_atoms': 200,
            'property_names': ['formation_energy', 'band_gap']
        },
        device='cuda',
        output_dir='evaluation/python_basic'
    )
    
    # 运行评估
    results = evaluator.run_comprehensive_evaluation(
        model_checkpoint='results/perov_gradnorm_tchebycheff/best.ckpt',
        test_data_path='results/perov_gradnorm_tchebycheff/test_results.pt',
        save_detailed_results=True
    )
    
    # 查看结果
    print(f"Overall Score: {results['overall_score']:.4f}")
    
    if 'reconstruction' in results:
        print(f"Match Rate: {results['reconstruction']['match_rate']:.4f}")
        print(f"Mean RMSD: {results['reconstruction']['mean_rmsd']:.4f}")
    
    if 'generation' in results:
        print(f"Validity Rate: {results['generation']['validity_rate']:.4f}")
        print(f"Diversity: {results['generation']['comp_diversity']:.4f}")
    
    return results

# 示例2: 分步评估
def example_stepwise_evaluation():
    """分步评估示例"""
    
    evaluator = ComprehensiveModelEvaluator(
        model_path='results/carbon_elastic',
        data_config={},
        output_dir='evaluation/python_stepwise'
    )
    
    # 模拟数据（实际使用中从文件加载）
    from pymatgen.core import Structure, Lattice
    
    # 创建示例结构
    lattice = Lattice.cubic(4.0)
    structures = [
        Structure(lattice, ['C', 'C'], [[0, 0, 0], [0.5, 0.5, 0.5]])
        for _ in range(100)
    ]
    
    properties = np.random.rand(100, 2)  # [formation_energy, elastic_modulus]
    
    # 1. 重构质量评估
    recon_results = evaluator.evaluate_reconstruction_quality(
        pred_structures=structures[:50],
        true_structures=structures[50:]
    )
    print("Reconstruction Results:", recon_results)
    
    # 2. 生成质量评估
    gen_results = evaluator.evaluate_generation_quality(structures)
    print("Generation Results:", gen_results)
    
    # 3. 属性预测评估
    prop_results = evaluator.evaluate_property_prediction(
        pred_properties=properties[:50],
        true_properties=properties[50:],
        property_names=['formation_energy', 'elastic_modulus']
    )
    print("Property Prediction Results:", prop_results)
    
    # 4. 多目标优化评估
    mo_results = evaluator.evaluate_multi_objective_optimization(
        optimized_structures=structures[:30],
        optimized_properties=properties[:30]
    )
    print("Multi-objective Results:", mo_results)
    
    return {
        'reconstruction': recon_results,
        'generation': gen_results,
        'property_prediction': prop_results,
        'multi_objective': mo_results
    }

# 示例3: 批量对比评估
def example_batch_comparison():
    """批量对比评估示例"""
    
    models = [
        'results/perov_gradnorm_weighted',
        'results/perov_gradnorm_tchebycheff',
        'results/perov_gradnorm_boundary',
        'results/perov_fixed_weighted'
    ]
    
    all_results = {}
    
    for model_path in models:
        model_name = Path(model_path).name
        
        evaluator = ComprehensiveModelEvaluator(
            model_path=model_path,
            data_config={},
            output_dir=f'evaluation/comparison/{model_name}'
        )
        
        results = evaluator.run_comprehensive_evaluation(
            model_checkpoint=f'{model_path}/best.ckpt',
            test_data_path=f'{model_path}/test_results.pt',
            save_detailed_results=True
        )
        
        all_results[model_name] = results
    
    # 对比分析
    comparison_analysis = analyze_model_comparison(all_results)
    save_comparison_report(comparison_analysis, 'evaluation/comparison_report.md')
    
    return all_results

def analyze_model_comparison(all_results):
    """分析模型对比结果"""
    
    comparison = {
        'model_names': list(all_results.keys()),
        'overall_scores': [],
        'best_reconstruction': None,
        'best_generation': None,
        'best_property_prediction': None,
        'best_multi_objective': None
    }
    
    best_scores = {
        'reconstruction': 0,
        'generation': 0, 
        'property_prediction': 0,
        'multi_objective': 0
    }
    
    for model_name, results in all_results.items():
        # 收集综合分数
        overall_score = results.get('overall_score', 0)
        comparison['overall_scores'].append(overall_score)
        
        # 找出各类别最佳模型
        for category in best_scores.keys():
            if category in results:
                if category == 'reconstruction':
                    score = results[category].get('match_rate', 0)
                elif category == 'generation': 
                    score = results[category].get('validity_rate', 0)
                elif category == 'property_prediction':
                    if 'overall' in results[category]:
                        score = results[category]['overall'].get('overall_r2', 0)
                    else:
                        score = 0
                elif category == 'multi_objective':
                    score = results[category].get('hypervolume', 0)
                else:
                    score = 0
                
                if score > best_scores[category]:
                    best_scores[category] = score
                    comparison[f'best_{category}'] = model_name
    
    # 统计分析
    comparison['mean_overall_score'] = np.mean(comparison['overall_scores'])
    comparison['std_overall_score'] = np.std(comparison['overall_scores'])
    comparison['best_overall'] = comparison['model_names'][
        np.argmax(comparison['overall_scores'])
    ]
    
    return comparison

def save_comparison_report(comparison, output_path):
    """保存对比报告"""
    
    with open(output_path, 'w') as f:
        f.write("# Model Comparison Report\n\n")
        
        f.write("## Overall Performance\n\n")
        f.write(f"**Best Overall Model:** {comparison['best_overall']}\n")
        f.write(f"**Mean Overall Score:** {comparison['mean_overall_score']:.4f}\n")
        f.write(f"**Std Overall Score:** {comparison['std_overall_score']:.4f}\n\n")
        
        f.write("## Category Winners\n\n")
        f.write(f"- **Best Reconstruction:** {comparison['best_reconstruction']}\n")
        f.write(f"- **Best Generation:** {comparison['best_generation']}\n") 
        f.write(f"- **Best Property Prediction:** {comparison['best_property_prediction']}\n")
        f.write(f"- **Best Multi-objective:** {comparison['best_multi_objective']}\n\n")
        
        f.write("## Detailed Scores\n\n")
        f.write("| Model | Overall Score |\n")
        f.write("|-------|---------------|\n")
        for model, score in zip(comparison['model_names'], comparison['overall_scores']):
            f.write(f"| {model} | {score:.4f} |\n")

# 示例4: 自定义评估流程
def example_custom_evaluation():
    """自定义评估流程示例"""
    
    class CustomEvaluator(ComprehensiveModelEvaluator):
        """自定义评估器"""
        
        def evaluate_physical_constraints(self, structures):
            """评估物理约束满足度"""
            constraints_satisfied = 0
            
            for struct in structures:
                # 检查最小原子间距
                distances = struct.distance_matrix
                np.fill_diagonal(distances, np.inf)
                min_distance = np.min(distances)
                
                if min_distance > 1.0:  # 最小距离大于1埃
                    constraints_satisfied += 1
            
            return {
                'physical_constraint_rate': constraints_satisfied / len(structures)
            }
        
        def evaluate_chemical_feasibility(self, structures):
            """评估化学可行性"""
            feasible_count = 0
            
            for struct in structures:
                # 简化的化学可行性检查
                # 实际应用中可以使用更复杂的规则
                species = [str(site.specie) for site in struct]
                unique_species = set(species)
                
                # 检查元素组合是否合理
                if len(unique_species) <= 4:  # 不超过4种元素
                    feasible_count += 1
            
            return {
                'chemical_feasibility_rate': feasible_count / len(structures)
            }
        
        def run_enhanced_evaluation(self, structures):
            """运行增强评估"""
            results = {}
            
            # 标准评估
            results['generation'] = self.evaluate_generation_quality(structures)
            
            # 自定义评估
            results['physical_constraints'] = self.evaluate_physical_constraints(structures)
            results['chemical_feasibility'] = self.evaluate_chemical_feasibility(structures)
            
            return results
    
    # 使用自定义评估器
    evaluator = CustomEvaluator(
        model_path='results/custom_model',
        data_config={},
        output_dir='evaluation/custom'
    )
    
    # 创建示例数据
    from pymatgen.core import Structure, Lattice
    lattice = Lattice.cubic(4.0)
    structures = [
        Structure(lattice, ['C', 'C'], [[0, 0, 0], [0.5, 0.5, 0.5]])
        for _ in range(50)
    ]
    
    # 运行增强评估
    results = evaluator.run_enhanced_evaluation(structures)
    
    print("Enhanced Evaluation Results:")
    for category, metrics in results.items():
        print(f"{category}: {metrics}")
    
    return results

# 示例5: 实时评估监控
def example_realtime_monitoring():
    """实时评估监控示例"""
    
    import time
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    class RealtimeEvaluator:
        """实时评估监控器"""
        
        def __init__(self, model_path):
            self.evaluator = ComprehensiveModelEvaluator(
                model_path=model_path,
                data_config={},
                output_dir='evaluation/realtime'
            )
            self.metrics_history = {
                'timestamps': [],
                'overall_scores': [],
                'validity_rates': [],
                'diversity_scores': []
            }
        
        def evaluate_checkpoint(self, checkpoint_path, test_data):
            """评估单个检查点"""
            # 简化评估（仅生成质量）
            gen_results = self.evaluator.evaluate_generation_quality(test_data)
            
            current_time = time.time()
            self.metrics_history['timestamps'].append(current_time)
            self.metrics_history['validity_rates'].append(
                gen_results.get('validity_rate', 0)
            )
            self.metrics_history['diversity_scores'].append(
                gen_results.get('comp_diversity', 0)
            )
            
            # 计算简化的整体分数
            overall_score = (
                gen_results.get('validity_rate', 0) * 0.6 +
                min(gen_results.get('comp_diversity', 0), 1.0) * 0.4
            )
            self.metrics_history['overall_scores'].append(overall_score)
            
            return gen_results
        
        def plot_metrics(self):
            """绘制指标趋势"""
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            times = self.metrics_history['timestamps']
            
            # 整体分数
            axes[0, 0].plot(times, self.metrics_history['overall_scores'])
            axes[0, 0].set_title('Overall Score')
            axes[0, 0].set_ylabel('Score')
            
            # 有效性
            axes[0, 1].plot(times, self.metrics_history['validity_rates'])
            axes[0, 1].set_title('Validity Rate')
            axes[0, 1].set_ylabel('Rate')
            
            # 多样性
            axes[1, 0].plot(times, self.metrics_history['diversity_scores'])
            axes[1, 0].set_title('Diversity Score')
            axes[1, 0].set_ylabel('Score')
            
            # 综合视图
            axes[1, 1].plot(times, self.metrics_history['validity_rates'], label='Validity')
            axes[1, 1].plot(times, self.metrics_history['diversity_scores'], label='Diversity')
            axes[1, 1].set_title('Combined View')
            axes[1, 1].legend()
            
            plt.tight_layout()
            plt.savefig('evaluation/realtime/metrics_trend.png')
            plt.show()
    
    # 使用实时监控器
    monitor = RealtimeEvaluator('results/training_model')
    
    # 模拟检查点评估
    from pymatgen.core import Structure, Lattice
    lattice = Lattice.cubic(4.0)
    test_data = [
        Structure(lattice, ['C', 'C'], [[0, 0, 0], [0.5, 0.5, 0.5]])
        for _ in range(20)
    ]
    
    # 模拟多个训练周期的评估
    checkpoint_paths = [
        'epoch_10.ckpt', 'epoch_20.ckpt', 'epoch_30.ckpt',
        'epoch_40.ckpt', 'epoch_50.ckpt'
    ]
    
    for checkpoint in checkpoint_paths:
        print(f"Evaluating {checkpoint}...")
        results = monitor.evaluate_checkpoint(checkpoint, test_data)
        time.sleep(1)  # 模拟评估间隔
    
    # 绘制趋势图
    monitor.plot_metrics()
    
    return monitor.metrics_history

# 主函数
if __name__ == "__main__":
    print("Running Enhanced Model Evaluator Examples...")
    
    # 运行示例（根据需要取消注释）
    # results1 = example_basic_evaluation()
    # results2 = example_stepwise_evaluation()
    # results3 = example_batch_comparison()
    # results4 = example_custom_evaluation()
    # results5 = example_realtime_monitoring()
    
    print("Examples completed!")

---

# ================================
# 集成脚本 - 完整评估流程
# ================================

#!/bin/bash
# integrated_evaluation.sh
# 完整的模型评估集成脚本

set -e  # 遇到错误立即退出

echo "======================================"
echo "Enhanced CDVAE Model Evaluation Pipeline"
echo "======================================"

# 配置参数
PROJECT_ROOT=${PROJECT_ROOT:-"/path/to/enhanced-cdvae"}
RESULTS_DIR=${RESULTS_DIR:-"$PROJECT_ROOT/results"}
EVAL_DIR=${EVAL_DIR:-"$PROJECT_ROOT/evaluation"}

# 创建评估目录
mkdir -p "$EVAL_DIR"

# 函数：评估单个模型
evaluate_single_model() {
    local model_path=$1
    local model_name=$(basename "$model_path")
    local output_dir="$EVAL_DIR/$model_name"
    
    echo "评估模型: $model_name"
    echo "模型路径: $model_path"
    echo "输出目录: $output_dir"
    
    # 检查模型文件是否存在
    if [ ! -d "$model_path" ]; then
        echo "警告: 模型目录不存在: $model_path"
        return 1
    fi
    
    # 查找最佳检查点
    checkpoint_file="$model_path/best.ckpt"
    if [ ! -f "$checkpoint_file" ]; then
        checkpoint_file="$model_path/last.ckpt"
        if [ ! -f "$checkpoint_file" ]; then
            echo "警告: 未找到检查点文件: $model_path"
            return 1
        fi
    fi
    
    # 查找测试数据
    test_data_file="$model_path/test_results.pt"
    if [ ! -f "$test_data_file" ]; then
        # 尝试其他可能的文件名
        for filename in "eval_gen.pt" "eval_opt.pt" "evaluation_results.pt"; do
            if [ -f "$model_path/$filename" ]; then
                test_data_file="$model_path/$filename"
                break
            fi
        done
    fi
    
    # 运行评估
    python "$PROJECT_ROOT/enhanced_model_evaluator.py" \
        --model_path "$model_path" \
        --checkpoint "$checkpoint_file" \
        --test_data "$test_data_file" \
        --output_dir "$output_dir" \
        --device auto \
        --batch_size 64
    
    if [ $? -eq 0 ]; then
        echo "✓ 模型 $model_name 评估完成"
        return 0
    else
        echo "✗ 模型 $model_name 评估失败"
        return 1
    fi
}

# 函数：批量评估
batch_evaluate() {
    echo "开始批量评估..."
    
    # 查找所有模型目录
    local success_count=0
    local total_count=0
    
    for model_dir in "$RESULTS_DIR"/*; do
        if [ -d "$model_dir" ]; then
            total_count=$((total_count + 1))
            
            if evaluate_single_model "$model_dir"; then
                success_count=$((success_count + 1))
            fi
        fi
    done
    
    echo "批量评估完成: $success_count/$total_count 成功"
}

# 函数：生成对比报告
generate_comparison_report() {
    echo "生成对比报告..."
    
    python - << EOF
import os
import json
import pandas as pd
from pathlib import Path

eval_dir = Path("$EVAL_DIR")
results = {}

# 收集所有评估结果
for model_dir in eval_dir.iterdir():
    if model_dir.is_dir():
        result_file = model_dir / "evaluation_results.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                data = json.load(f)
                results[model_dir.name] = data

# 创建对比表格
comparison_data = []
for model_name, data in results.items():
    row = {
        'Model': model_name,
        'Overall_Score': data.get('overall_score', 0),
    }
    
    # 添加各类别分数
    if 'reconstruction' in data:
        row['Match_Rate'] = data['reconstruction'].get('match_rate', 0)
        row['Mean_RMSD'] = data['reconstruction'].get('mean_rmsd', float('inf'))
    
    if 'generation' in data:
        row['Validity_Rate'] = data['generation'].get('validity_rate', 0)
        row['Diversity'] = data['generation'].get('comp_diversity', 0)
    
    if 'property_prediction' in data and 'overall' in data['property_prediction']:
        row['Property_R2'] = data['property_prediction']['overall'].get('overall_r2', 0)
    
    if 'multi_objective' in data:
        row['Hypervolume'] = data['multi_objective'].get('hypervolume', 0)
        row['Pareto_Size'] = data['multi_objective'].get('pareto_size', 0)
    
    comparison_data.append(row)

# 转换为DataFrame并保存
df = pd.DataFrame(comparison_data)
df = df.round(4)

# 保存CSV
df.to_csv(eval_dir / "model_comparison.csv", index=False)

# 生成Markdown报告
with open(eval_dir / "comparison_report.md", 'w') as f:
    f.write("# Model Comparison Report\n\n")
    f.write("## Summary\n\n")
    f.write(f"Total Models Evaluated: {len(results)}\n\n")
    
    if len(comparison_data) > 0:
        best_overall = max(comparison_data, key=lambda x: x['Overall_Score'])
        f.write(f"**Best Overall Model:** {best_overall['Model']} (Score: {best_overall['Overall_Score']:.4f})\n\n")
    
    f.write("## Detailed Comparison\n\n")
    f.write(df.to_markdown(index=False))
    f.write("\n\n")
    
    f.write("## Performance Analysis\n\n")
    for col in ['Overall_Score', 'Validity_Rate', 'Hypervolume']:
        if col in df.columns:
            f.write(f"- **{col}:**\n")
            f.write(f"  - Mean: {df[col].mean():.4f}\n")
            f.write(f"  - Std: {df[col].std():.4f}\n")
            f.write(f"  - Best: {df[col].max():.4f}\n\n")

print("对比报告生成完成")
print(f"CSV文件: {eval_dir}/model_comparison.csv")
print(f"Markdown报告: {eval_dir}/comparison_report.md")
EOF
}

# 主函数
main() {
    echo "开始完整评估流程..."
    
    # 检查项目根目录
    if [ ! -d "$PROJECT_ROOT" ]; then
        echo "错误: 项目根目录不存在: $PROJECT_ROOT"
        exit 1
    fi
    
    # 检查结果目录
    if [ ! -d "$RESULTS_DIR" ]; then
        echo "错误: 结果目录不存在: $RESULTS_DIR"
        exit 1
    fi
    
    # 设置Python路径
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    
    # 根据参数选择评估模式
    case "${1:-batch}" in
        "single")
            if [ -z "$2" ]; then
                echo "使用方法: $0 single <model_path>"
                exit 1
            fi
            evaluate_single_model "$2"
            ;;
        "batch")
            batch_evaluate
            generate_comparison_report
            ;;
        "compare")
            generate_comparison_report
            ;;
        *)
            echo "使用方法: $0 [single|batch|compare] [model_path]"
            echo "  single <model_path>: 评估单个模型"
            echo "  batch: 批量评估所有模型"
            echo "  compare: 仅生成对比报告"
            exit 1
            ;;
    esac
    
    echo "评估流程完成!"
    echo "结果保存在: $EVAL_DIR"
}

# 运行主函数
main "$@"