# California Home Sale Price Prediction

This project predicts the closing price of residential single-family homes in California. It covers data exploration, leakage-aware preprocessing, time-based validation, baseline and tree-based model comparisons, geographic feature engineering, XGBoost tuning, and final error analysis.

## Dataset source

The primary data consists of California Regional Multiple Listing Service (CRMLS) sold-listing CSV files supplied for the IDX Exchange Data Science internship. The raw files are named `CRMLSSold*.csv` and belong in the `california/` directory. They cover January 2022 through May 2026; the initial combined dataset contains 794,271 rows and 83 columns.

The raw listing data is not committed to Git because it is large and may be subject to data-access restrictions. To reproduce the project from the beginning, obtain the CRMLS exports through the internship/data provider and place them as follows:

```text
california/
├── CRMLSSold20220101_20231231_filled.csv
├── CRMLSSold202401_filled.csv
├── ...
└── CRMLSSold202605.csv
```

School-district boundary data is loaded during feature engineering from the [California State Geoportal](https://gis.data.ca.gov/api/download/v1/items/b0e3b936426a47ce9d9a2e77e2bb86cc/geojson?layers=0).

The prediction target is `ClosePrice`. See [`column_dictionary.md`](column_dictionary.md) for descriptions of the principal CRMLS columns.

## Preprocessing

The preprocessing pipeline in `02_preprocessing.ipynb`:

1. Combines all `california/CRMLSSold*.csv` files.
2. Keeps records where `PropertyType == "Residential"` and `PropertySubType == "SingleFamilyResidence"`.
3. Converts date fields to datetimes and removes records with a missing/non-positive `ClosePrice` or missing `CloseDate`.
4. Sorts by closing date and removes duplicate `ListingKey` values, keeping the latest record.
5. Excludes identifiers, target-derived variables, very sparse columns, and extremely high-cardinality categories from the predictors.
6. Adds missing-value flags, learns numeric medians from the training data only, and fills missing categorical values with `"Unknown"`.
7. Engineers home age, log-transformed living/lot area, property ratios, and California elementary, high-school, and unified school-district boundary features.
8. Uses the 12 months immediately before the newest month for training and the newest month for testing. The final split trains on May 2025 through April 2026 and tests on May 2026; the saved no-outlier data contains 128,568 training rows and 11,914 test rows.
9. Runs the primary model comparison on homes at or below the training set's 99th-percentile sale-price cutoff (approximately $6.42 million). The cutoff is learned from training data and then applied to both sets.
10. One-hot encodes categorical features with unknown-category handling. Numeric scaling is used for linear regression; tree models use unscaled numeric features.

Target-derived fields such as `log_ClosePrice` and reference-only fields such as `price_per_sqft`, `ListPrice`, and `OriginalListPrice` are excluded from model inputs to prevent leakage or unrealistic evaluation.

## Models tested

| Model / experiment | Test R² | MAE | RMSE |
|---|---:|---:|---:|
| Linear regression, all prices | 0.2872 | $452,569 | $1,416,770 |
| Linear regression, no outliers | 0.8415 | $220,163 | $360,590 |
| Decision tree, all prices | -0.2278 | $538,131 | $1,859,419 |
| Tuned decision tree, no outliers | 0.8143 | $211,439 | $390,298 |
| Random forest, all prices | 0.0340 | $371,849 | $1,649,257 |
| Tuned random forest, no outliers | 0.8656 | $173,670 | $332,015 |
| **Tuned XGBoost, no outliers + district features** | **0.8895** | **$169,190** | **$301,127** |

The strongest XGBoost configuration uses `max_depth=8`, `learning_rate=0.10`, `n_estimators=300`, the squared-error objective, and `random_state=42`. It was selected using an 80/20 split of the training set and then retrained on all training rows before evaluation on the untouched May 2026 test month.

## Best results

On 11,914 test properties, the final XGBoost model achieved:

- R²: **0.8895**
- MAE: **$169,190**
- RMSE: **$301,127**
- MAPE: **17.30%**
- Median absolute percentage error (MdAPE): **9.81%**

The median percentage error is notably lower than the mean, indicating that a smaller group of difficult properties produces disproportionately large errors. Results by price band are saved in `metrics_summary.csv`; within-band R² values should not be compared directly with overall R² because each band has a much narrower target range.

## Repository structure

```text
01_exploration.ipynb       Exploratory analysis and distribution checks
02_preprocessing.ipynb     Cleaning, feature selection, and time split
03_baseline_model.ipynb    Linear regression baseline
04_model_comparison.ipynb  Decision tree, random forest, and geo features
05_advanced_models.ipynb   XGBoost tuning and final predictions
06_evaluation.ipynb        Overall and price-band evaluation
column_dictionary.md       Data-field reference
metrics_summary.csv        Final XGBoost evaluation metrics
week6_*.npz / week6_*.npy  Saved processed matrices and targets
week7_*.npy                Saved test targets and XGBoost predictions
```

`eda.ipynb` is an earlier data-quality notebook retained for project history. The numbered notebooks are the main reproducible workflow.

## Re-run the project

Python 3.10 or newer is recommended. From the repository root, create an environment and install the notebook dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jupyter pandas numpy scipy scikit-learn matplotlib geopandas shapely xgboost
```

Place the raw CRMLS files in `california/`, then start Jupyter:

```bash
jupyter lab
```

Open and run the numbered notebooks in order from `01_exploration.ipynb` through `06_evaluation.ipynb`. Run each notebook from the repository root because the notebooks use relative paths. `04_model_comparison.ipynb` also needs internet access to download the school-district GeoJSON.

For a non-interactive run, execute:

```bash
jupyter nbconvert --to notebook --execute --inplace 01_exploration.ipynb --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute --inplace 02_preprocessing.ipynb --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute --inplace 03_baseline_model.ipynb --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute --inplace 04_model_comparison.ipynb --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute --inplace 05_advanced_models.ipynb --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute --inplace 06_evaluation.ipynb --ExecutePreprocessor.timeout=-1
```

Model training—especially the random-forest grid search, spatial joins, and XGBoost search—can take substantial time and memory. If the saved Week 6 matrices already exist, start at `05_advanced_models.ipynb`. If the Week 7 prediction arrays exist, run only `06_evaluation.ipynb` to regenerate `metrics_summary.csv`.

## Launching the app

There is currently no Streamlit, Flask, Gradio, or other application entry point in this repository, and the fitted preprocessor/model are not serialized for inference. Therefore, there is no valid app launch command yet. The current deliverable is a notebook-based modeling workflow; use `jupyter lab` to explore and run it.

To add a deployable app later, save a single fitted preprocessing-and-model pipeline (for example with `joblib`), create an app entry point such as `app.py`, and document its launch command here (for example, `streamlit run app.py`).
