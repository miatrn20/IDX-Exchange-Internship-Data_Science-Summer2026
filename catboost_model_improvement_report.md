# CatBoost Model Improvement Report

## Executive summary

The property-price regression model was improved by replacing the existing XGBoost model with CatBoost and increasing CatBoost's maximum number of boosting iterations from 1,500 to 4,000. The final CatBoost model achieved a test R² of **0.9044**, exceeding the project target of 0.90. It also reduced both MAE and RMSE compared with XGBoost and the initial CatBoost run.

## Objective

The objective was to improve test R² from approximately 0.89 to at least 0.90 without changing the existing processed features, training data, or test data.

## Data and evaluation method

The experiment used the processed sparse feature matrices generated in the earlier preprocessing workflow:

- Training observations: 128,568
- Test observations: 11,914
- Model features: 4,646

For model selection, 20% of the training observations were assigned to a validation set using `random_state=42`. CatBoost monitored validation RMSE during training. After the iteration count was selected, the model was retrained on all 128,568 training observations and evaluated on the held-out test set.

The models were evaluated with three regression metrics:

- R²: proportion of variation in property prices explained by the model; higher is better.
- MAE: average absolute prediction error in dollars; lower is better.
- RMSE: prediction error that gives greater weight to large mistakes; lower is better.

## Model configuration

The final CatBoost configuration was:

```python
CatBoostRegressor(
    iterations=4000,
    depth=8,
    learning_rate=0.05,
    loss_function="RMSE",
    random_seed=42,
    thread_count=-1,
    allow_writing_files=False
)
```

During validation training, early stopping was set to 75 rounds. The best validation result occurred at iteration 3,997. A final CatBoost model was therefore trained on the complete training set for 3,997 iterations.

## Results

| Model | Test R² | Test MAE | Test RMSE |
|---|---:|---:|---:|
| XGBoost | 0.889482 | $169,190 | $301,127 |
| CatBoost — 1,500 iterations | 0.894022 | $165,796 | $294,878 |
| CatBoost — 3,997 iterations | **0.904403** | **$152,842** | **$280,063** |

Compared with XGBoost, the final CatBoost model:

- Increased R² by 0.014921, from 0.889482 to 0.904403.
- Reduced MAE by approximately $16,349, to $152,842.
- Reduced RMSE by approximately $21,064, to $280,063.

Compared with the initial 1,500-iteration CatBoost model, the final model:

- Increased R² by 0.010381.
- Reduced MAE by approximately $12,954.
- Reduced RMSE by approximately $14,815.

## Why performance improved

At 1,500 iterations, CatBoost's validation RMSE was still decreasing. This indicated that the model had not yet converged and could benefit from additional boosting rounds. Raising the maximum from 1,500 to 4,000 allowed CatBoost to continue learning smaller residual patterns at the existing learning rate of 0.05.

The improvement did not come from adding features, changing the data split, or modifying the test set. The principal change was a larger iteration budget. CatBoost's ordered boosting and regularization also provided a stronger fit for this high-dimensional tabular dataset than the existing XGBoost configuration.

## Interpretation

The final R² of 0.9044 means the model explains approximately 90.4% of the variation in test-set property prices. The lower MAE shows that the typical prediction became more accurate, while the lower RMSE indicates a reduction in larger prediction errors.

Although the improvement is meaningful, the MAE of approximately $152,842 shows that prediction errors remain material in dollar terms. Performance should therefore be considered alongside the price distribution and errors within individual price bands, not only the overall R².

## Limitations and next steps

The selected iteration count was very close to the 4,000-iteration limit, so the model may benefit from a larger training limit or additional tuning. However, further decisions should be based on validation or cross-validation results rather than repeated optimization against the test set.

Recommended next steps are:

1. Tune depth, learning rate, L2 regularization, and row or feature sampling with cross-validation.
2. Evaluate performance separately across property-price bands and geographic districts.
3. Inspect residuals for systematic underprediction or overprediction.
4. Compare CatBoost with an ensemble that blends CatBoost and XGBoost predictions.
5. Preserve a final untouched holdout set for an unbiased estimate after all tuning is complete.

## Conclusion

Increasing CatBoost's training budget successfully moved the model above the target. The final model achieved **R² = 0.9044**, outperforming both XGBoost and the initial CatBoost model while also producing lower MAE and RMSE. Based on the current evaluation, CatBoost is the strongest model tested for this workflow.
