# Alignment Breaking Investigation

## Question
Is the catastrophic alignment breaking at LR=1e-3 real, or is it an artifact/error?

## Investigation Methods

1. **Heatmap Analysis**: Visualized LEXam accuracy and training loss across all 30 configurations (3 LRs × 10 ranks)
2. **Scatter Plot**: Direct comparison of training loss vs LEXam accuracy
3. **Rank Analysis**: Checked if the pattern holds across all ranks
4. **Cross-Model Validation**: Verified the pattern on both 8B and 70B models
5. **Error Checking**: Inspected WandB logs for evaluation errors or warnings

## Results

### 8B Model (30 configurations evaluated)

| Learning Rate | Training Loss | LEXam Accuracy | Change from Baseline |
|---------------|---------------|----------------|---------------------|
| 1e-5 | 0.823 | 27.0% | ±0% (maintains) |
| 1e-4 | 0.773 | 12.1% | -15.0% (degradation) |
| 1e-3 | 0.728 | 0.9% | -26.2% (catastrophic) |

**Baseline**: 27.1% (no fine-tuning)

### 70B Model (11 configurations evaluated)

| Learning Rate | LEXam Accuracy | Change from Baseline |
|---------------|----------------|---------------------|
| 1e-5 | 30.0% | -3.1% (slight degradation) |
| 1e-4 | 28.3% | -4.8% (moderate degradation) |
| 1e-3 | 0.0% | -33.1% (complete failure) |

**Baseline**: 33.1% (no fine-tuning)

### Key Observations

1. **Pattern is consistent across ranks**: All 10 ranks at LR=1e-3 show <5% accuracy
2. **Pattern is consistent across models**: Both 8B and 70B show catastrophic failure at LR=1e-3
3. **No evaluation errors**: WandB logs show successful completion with no errors
4. **Inverse relationship**: Lower training loss correlates with worse LEXam accuracy

## Conclusion

**The result is REAL, not an artifact.**

The learning rate that optimizes training loss on Swiss legal data (LR=1e-3) completely destroys the model's ability to follow instructions on general tasks. This is a critical finding about the trade-off between domain-specific optimization and general capabilities.

### Hypothesis

At LR=1e-3, the model is overfitting so aggressively to the Swiss legal task that it:
1. Loses its instruction-following capabilities
2. Becomes unable to parse MCQ question formats
3. May be outputting Swiss legal text regardless of input

This suggests that **training loss is not a reliable metric** for model quality when fine-tuning on narrow domains.

### Practical Recommendation

For Swiss legal fine-tuning:
- **Use LR=1e-5** to maintain general capabilities
- **Monitor held-out task performance**, not just training loss
- **Consider multi-task training** to preserve general abilities

## Supporting Evidence

See `plots/alignment_investigation_8B.png` for comprehensive visualization showing:
- Heatmaps of LEXam accuracy and training loss
- Scatter plot showing inverse relationship
- Line plots showing consistency across ranks
- Bar chart summarizing the trade-off
