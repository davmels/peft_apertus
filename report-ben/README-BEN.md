# LoRA Fine-Tuning Analysis: Swiss Legal Domain

Systematic validation of LoRA hyperparameter selection on domain-specific fine-tuning tasks.

## Abstract

We conduct a comprehensive grid search over LoRA hyperparameters (learning rate, rank) on Apertus models (8B, 70B parameters) fine-tuned for Swiss legal judgment prediction. Our experiments validate key findings from recent LoRA literature while revealing critical trade-offs between task performance and knowledge retention. We observe that optimal training hyperparameters can induce catastrophic forgetting of general domain knowledge, suggesting the need for multi-objective optimization in production deployments.

**Experimental Setup**: 52 training runs across 3 learning rates {1e-5, 1e-4, 1e-3} and 10 ranks {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}, plus 6 full fine-tuning baselines. Task: Binary classification on Swiss Judgment Prediction dataset (85K training cases). Evaluation: Task accuracy and LEXam legal knowledge benchmark.

---

## 1. Learning Rate Scaling

![LR Sweep Combined](plots/figure2_style_combined.png)
*Figure 1: Final training loss vs learning rate for LoRA and full fine-tuning. LoRA achieves optimal convergence at 10-100× higher learning rates than full fine-tuning, with the ratio depending on model size.*

### Observations

**8B Model**: LoRA optimal at LR=1e-3, full fine-tuning optimal at LR=1e-4 (10× ratio).

**70B Model**: LoRA optimal at LR=1e-3, full fine-tuning optimal at LR=1e-5 (100× ratio). Note: Full fine-tuning at LR=1e-4 exhibits training instability (final loss 0.81 vs 0.67 at LR=1e-5).

### Analysis

The learning rate ratio scales with model size, suggesting that the effective learning rate for low-rank updates requires greater compensation in larger models. This is consistent with the reduced parameter space of LoRA requiring larger step sizes to achieve comparable gradient descent trajectories.

---

## 2. Rank Independence of Optimal Learning Rate

![Learning Curves 8B](plots/all_learning_curves_8B.png)
*Figure 2: 8B training dynamics across ranks and learning rates. All ranks converge optimally at LR=1e-3, with ranks ≥32 showing nearly identical trajectories.*

![Learning Curves 70B](plots/all_learning_curves_70B.png)
*Figure 3: 70B training dynamics. Optimal learning rate remains constant across ranks 16, 64, and 512. Rank=1 at LR=1e-3 diverges (final loss=9.99) and is excluded from analysis.*

### Observations

For both model sizes, the optimal learning rate is invariant to LoRA rank across the tested range (r ∈ {2, 4, 8, 16, 32, 64, 128, 256, 512}). Higher ranks (r≥32 for 8B, r≥64 for 70B) exhibit convergence to identical final loss values, suggesting rank saturation for this task.

### Implications

Hyperparameter tuning can be performed at a single representative rank (e.g., r=64), then applied across the rank spectrum. This significantly reduces the search space for LoRA deployment.

---

## 3. Task Performance vs Rank

![Swiss Judgment Accuracy](plots/swiss_judgment_accuracy_vs_rank.png)
*Figure 4: Swiss Judgment Prediction accuracy vs LoRA rank. Performance plateaus at moderate ranks, with diminishing returns beyond r=64.*

### Quantitative Results

| Model | Optimal Config | Accuracy | Rank=64 | Δ from Optimal |
|-------|---------------|----------|---------|----------------|
| 8B    | LR=1e-3, r=8  | 83.6%    | 83.3%   | -0.3%          |
| 70B   | LR=1e-4, r=1  | 83.1%    | 82.6%   | -0.5%          |

### Analysis

Rank 64 achieves 99.4-99.6% of optimal task performance across both model sizes while providing substantially more adaptation capacity than minimal ranks. This suggests r=64 as a robust default for production systems where model capacity for future adaptation is valued.

---

## 4. Full Fine-Tuning vs LoRA

### Training Loss Comparison

Across all configurations, full fine-tuning achieves 6-14% lower final training loss than LoRA:

- **8B**: Full FT (LR=1e-4) reaches 0.58, LoRA (LR=1e-3, r=512) reaches 0.62 (6.9% higher)
- **70B**: Full FT (LR=1e-5) reaches 0.67, LoRA (LR=1e-3, r=512) reaches 0.76 (13.4% higher)

### Interpretation

The low-rank constraint imposes a fundamental expressiveness limitation. For the Swiss legal domain (85K training cases), full fine-tuning's unrestricted parameter updates yield measurably better training loss convergence. This diverges from some literature claims that LoRA matches full fine-tuning performance on small-to-medium datasets.

### Parameter Efficiency Trade-off

LoRA updates <1% of model parameters while achieving 99%+ of task performance, representing a favorable efficiency-performance trade-off for most production scenarios.

---

## 5. Catastrophic Forgetting of Domain Knowledge

![Investigation 8B](plots/alignment_investigation_8B.png)
*Figure 5: 8B model - Training loss vs LEXam knowledge retention across learning rates and ranks. LR=1e-3 optimizes training loss but destroys general legal knowledge.*

![Investigation 70B](plots/alignment_investigation_70B.png)
*Figure 6: 70B model - Same catastrophic forgetting pattern. Higher learning rates induce complete knowledge collapse despite optimal task loss.*

### Quantitative Observations

**8B Model (LEXam Accuracy)**:
- Baseline (pre-fine-tuning): 27.1%
- LR=1e-5: 26-28% (knowledge preserved)
- LR=1e-4: 12-18% (partial degradation)
- LR=1e-3: 0-4% (catastrophic forgetting)

**70B Model (LEXam Accuracy)**:
- Baseline: 33.1%
- LR=1e-5: 29-31% (slight degradation)
- LR=1e-4: 8-15% (severe degradation)
- LR=1e-3: 0% (complete knowledge loss)

### Analysis

The learning rate that optimizes task-specific training loss (LR=1e-3) induces catastrophic forgetting of general legal knowledge. This effect is consistent across model sizes and LoRA ranks, suggesting a fundamental tension between task specialization and knowledge retention.

The LEXam benchmark evaluates legal knowledge through multiple-choice questions, representing capabilities orthogonal to the binary classification task. The dramatic performance collapse indicates that aggressive fine-tuning overwrites pre-trained knowledge rather than augmenting it.

### Implications for Deployment

Production systems must balance task performance against knowledge retention. For applications requiring robust general capabilities alongside task specialization, conservative learning rates (LR=1e-5) are essential despite suboptimal training loss.

---

## Recommendations

### Hyperparameter Selection

Based on 52 training runs and comprehensive evaluation:

```yaml
learning_rate: 1e-5              # Preserves knowledge retention
lora_r: 64                       # Near-optimal performance + capacity
lora_alpha: 32                   # Standard 2:1 ratio
lora_target_modules: all-linear  # Attention + MLP layers
batch_size: 16
num_train_epochs: 1              # ~500 steps sufficient
```

### Rationale

- **LR=1e-5**: Achieves 79-80% task accuracy while maintaining 26-31% LEXam accuracy (vs 0-4% at LR=1e-3)
- **Rank=64**: 99.5% of optimal task performance with 64× more capacity than minimal rank
- **All-linear modules**: Literature shows attention-only LoRA significantly underperforms

### Infrastructure Notes

- **8B**: ZeRO-0 sufficient, 2 nodes minimum, linear scaling to 8 nodes
- **70B**: ZeRO-3 required, 4 nodes minimum (memory constraints)
- **Multi-node scaling**: Near-perfect speedup observed (training time halves per doubling)

---

## Conclusions

1. **Learning rate scaling**: LoRA requires 10-100× higher learning rates than full fine-tuning, with the ratio increasing with model size.

2. **Rank independence**: Optimal learning rate is invariant to LoRA rank, enabling efficient hyperparameter search.

3. **Performance gap**: Full fine-tuning achieves 6-14% lower training loss than LoRA, indicating fundamental expressiveness limitations of low-rank updates.

4. **Catastrophic forgetting**: Optimal training hyperparameters induce severe degradation of general domain knowledge, revealing a critical trade-off for production deployment.

5. **Practical recommendation**: LR=1e-5 with rank=64 balances task performance, knowledge retention, and adaptation capacity for production systems.

### Divergence from Literature

Our results partially contradict claims that "LoRA matches full fine-tuning performance" on small-medium datasets. For the Swiss legal domain (85K cases), full fine-tuning consistently outperforms LoRA in training loss, though the gap in task accuracy is smaller (0.3-0.5%).

---

## Reproducibility

### Data

Training: Swiss Judgment Prediction (85K cases, binary classification)  
Evaluation: LEXam legal knowledge benchmark (multiple-choice questions)  
Models: Apertus-8B-Instruct, Apertus-70B-Instruct

### Code

```bash
# Fetch experimental data
cd report-ben
export WANDB_API_KEY=<key>
python scripts/01_fetch_wandb_data.py

# Download training histories
python scripts/02_download_history.py

# Generate analysis plots
python scripts/03_generate_plots.py
```

### Repository Structure

```
peft_apertus/
├── configs/          # Training configurations (LoRA, Full FT, ZeRO)
├── lexam/            # LEXam evaluation scripts
└── report-ben/       # Analysis pipeline
    ├── scripts/      # Data fetching and plot generation
    ├── analysis/     # Core utilities (data loading, plotting)
    ├── data/         # Training histories and evaluation results
    └── plots/        # Figures 1-6
```

---

**Author**: Ben Bullinger  
**Project**: LSAI-2025 Swiss AI Initiative  
**Date**: December 2025
