# California Home Sale Price Prediction

A data science internship project focused on predicting residential single-family home sale prices in California using historical CRMLS listing data, feature engineering, and machine learning.

## Overview

This project builds a predictive model for estimating the closing price of residential single-family homes in California. The workflow covers data acquisition, cleaning, leakage-aware preprocessing, model benchmarking, geographic feature engineering, gradient boosting optimization, and final evaluation.

The final model is an XGBoost regressor trained on a time-based split, with emphasis on realistic generalization to future transactions rather than random data leakage.

## Project goals

- Predict `ClosePrice` for residential single-family homes in California
- Compare multiple modeling approaches on a realistic temporal validation setup
- Engineer spatial and property features that improve predictive power
- Deliver a deployable, user-facing prediction app for price estimation

## Key results

The final model delivered strong out-of-sample performance on the May 2026 test month:

- R²: 0.8895
- MAE: $169,190
- RMSE: $301,127
- MAPE: 17.30%
- MdAPE: 9.81%

This result was achieved with a tuned XGBoost model using district-related features and a no-outlier training setup.

## Data

The project uses California Regional Multiple Listing Service (CRMLS) sold-listing data for the period January 2022 through May 2026. The source files live in the `california/` directory and are named like `CRMLSSold*.csv`.

The raw data is not included in the repository because it is large and may be subject to data-access restrictions. To reproduce the full pipeline, obtain the CRMLS export files and place them in the folder structure below:

```text
california/
├── CRMLSSold20220101_20231231_filled.csv
├── CRMLSSold202401_filled.csv
├── ...
└── CRMLSSold202605.csv
```

The prediction target is `ClosePrice`. A reference for the main CRMLS fields is provided in [`column_dictionary.md`](column_dictionary.md).

## Methodology

The project follows a structured modeling pipeline:

1. Data ingestion and filtering
   - Keep only `PropertyType == "Residential"`
   - Keep only `PropertySubType == "SingleFamilyResidence"`
   - Remove invalid references to sale price or closing date

2. Preprocessing and leakage control
   - Sort by closing date and remove duplicate listing records by `ListingKey`
   - Exclude identifiers and target-derived fields
   - Remove sparse or high-cardinality variables that would be unstable in production
   - Add missing-value indicators and learn imputation values from training data only
   - Encode categorical variables with unknown-category handling

3. Feature engineering
   - Home age
   - Log-transformed living area and lot size
   - Property ratios such as bedroom-to-bathroom and lot-to-living-area ratios
   - School-district boundary features derived from California geographic data

4. Time-based modeling setup
   - Train on the 12 months immediately preceding the latest reporting month
   - Test on the newest month (May 2026)
   - Filter extreme high-price outliers based on the training distribution

5. Model benchmarking
   - Linear regression
   - Decision tree
   - Random forest
   - XGBoost

6. Final evaluation
   - Assess performance using R², MAE, RMSE, MAPE, and MdAPE
   - Review error behavior across price bands and property segments

## Model performance

| Model / experiment | Test R² | MAE | RMSE |
|---|---:|---:|---:|
| Linear regression, all prices | 0.2872 | $452,569 | $1,416,770 |
| Linear regression, no outliers | 0.8415 | $220,163 | $360,590 |
| Decision tree, all prices | -0.2278 | $538,131 | $1,859,419 |
| Tuned decision tree, no outliers | 0.8143 | $211,439 | $390,298 |
| Random forest, all prices | 0.0340 | $371,849 | $1,649,257 |
| Tuned random forest, no outliers | 0.8656 | $173,670 | $332,015 |
| Tuned XGBoost, no outliers + district features | 0.8895 | $169,190 | $301,127 |

The best-performing configuration used a tuned XGBoost regressor with district features, selected through validation and retrained on the full training set before final evaluation.

## Repository structure

```text
.
├── 01_exploration.ipynb
├── 02_preprocessing.ipynb
├── 03_baseline_model.ipynb
├── 04_model_comparison.ipynb
├── 05_advanced_models.ipynb
├── 06_evaluation.ipynb
├── app.py
├── README.md
├── requirements.txt
├── column_dictionary.md
├── metrics_summary.csv
├── model_metadata.json
├── model_metadata.pkl
├── deployment_defaults.pkl
├── district_preprocessor.pkl
├── xgboost_model.pkl
├── california/
├── train_residential_single_family_week3.csv
├── test_residential_single_family_week3.csv
├── train_residential_single_family_week3_no_outliers.csv
├── test_residential_single_family_week3_no_outliers.csv
├── week6_X_train_district_processed.npz
├── week6_X_test_district_processed.npz
├── week6_y_train.npy
├── week6_y_test.npy
├── week7_xgb_predictions.npy
├── week7_y_test.npy
├── eda.ipynb
└── streamlit_app.ipynb
```

## Setup

This project is designed for Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you are reproducing the notebook workflow from scratch, make sure the source CRMLS files are placed in `california/` before running the notebooks.

## Reproducing the analysis

Run the notebook workflow from the repository root in order:

```bash
jupyter lab
```

Then execute the notebooks in sequence:

1. `01_exploration.ipynb`
2. `02_preprocessing.ipynb`
3. `03_baseline_model.ipynb`
4. `04_model_comparison.ipynb`
5. `05_advanced_models.ipynb`
6. `06_evaluation.ipynb`

For a non-interactive run:

```bash
jupyter nbconvert --to notebook --execute --inplace 01_exploration.ipynb --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute --inplace 02_preprocessing.ipynb --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute --inplace 03_baseline_model.ipynb --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute --inplace 04_model_comparison.ipynb --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute --inplace 05_advanced_models.ipynb --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute --inplace 06_evaluation.ipynb --ExecutePreprocessor.timeout=-1
```

Note: `04_model_comparison.ipynb` requires internet access for school-district GeoJSON data. If the processed matrices already exist, you can begin at the model-tuning stage instead of rerunning the full pipeline.

## Streamlit app

A deployable prediction app is included in `app.py`.

### Run locally

```bash
streamlit run app.py
```

Then open the local URL displayed in the terminal, typically:

```text
http://localhost:8501
```

### App features

- Input property attributes such as size, bedrooms, bathrooms, year built, and location
- Capture additional housing characteristics such as flooring, garage, pool, and view status
- Include school-district details when available
- Reconstruct engineered features used during model training
- Produce a sale-price estimate in U.S. dollars

## Deployment artifacts

The app expects the following serialized files to exist in the project root:

- `xgboost_model.pkl`
- `district_preprocessor.pkl`
- `model_metadata.pkl`
- `deployment_defaults.pkl`

These artifacts are produced during the model-training workflow and are required for inference in the app.

## Notes

- `eda.ipynb` is an earlier exploratory notebook retained for project history.
- The numbered notebooks represent the primary reproducible modeling workflow.
- The project is intended for portfolio and internship use, with a focus on clean methodology, reproducible ML engineering, and practical deployment readiness.

## Contact

For questions or collaboration opportunities, feel free to reach out through the project repository or professional contact channels.
