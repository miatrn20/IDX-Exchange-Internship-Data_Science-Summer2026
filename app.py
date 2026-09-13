from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="California Home Price Predictor",
    page_icon="🏠",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "xgboost_model.pkl")
    preprocessor = joblib.load(BASE_DIR / "district_preprocessor.pkl")
    metadata = joblib.load(BASE_DIR / "model_metadata.pkl")
    defaults = joblib.load(BASE_DIR / "deployment_defaults.pkl")

    return model, preprocessor, metadata, defaults


model, preprocessor, metadata, defaults = load_artifacts()

numeric_features = metadata["numeric_features"]
categorical_features = metadata["categorical_features"]

st.title("California Home Price Predictor")
st.write(
    "Enter the property information below to estimate its closing price."
)

with st.form("prediction_form"):
    st.subheader("Property details")

    column1, column2, column3 = st.columns(3)

    with column1:
        living_area = st.number_input(
            "Living area (square feet)",
            min_value=100.0,
            max_value=30000.0,
            value=1810.0,
            step=50.0,
        )

        bedrooms = st.number_input(
            "Bedrooms",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
        )

        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1,
            max_value=20,
            value=2,
            step=1,
        )

        lot_square_feet = st.number_input(
            "Lot size (square feet)",
            min_value=100.0,
            max_value=2_000_000.0,
            value=7000.0,
            step=100.0,
        )

        year_built = st.number_input(
            "Year built",
            min_value=1800,
            max_value=date.today().year,
            value=1980,
            step=1,
        )

    with column2:
        garage_spaces = st.number_input(
            "Garage spaces",
            min_value=0,
            max_value=20,
            value=2,
            step=1,
        )

        parking_total = st.number_input(
            "Total parking spaces",
            min_value=0,
            max_value=30,
            value=2,
            step=1,
        )

        stories = st.number_input(
            "Stories",
            min_value=1.0,
            max_value=10.0,
            value=1.0,
            step=0.5,
        )

        main_level_bedrooms = st.number_input(
            "Main-level bedrooms",
            min_value=0,
            max_value=20,
            value=1,
            step=1,
        )

        association_fee = st.number_input(
            "Monthly association fee ($)",
            min_value=0.0,
            max_value=10000.0,
            value=0.0,
            step=25.0,
        )

    with column3:
        latitude = st.number_input(
            "Latitude",
            min_value=32.0,
            max_value=42.5,
            value=34.0522,
            format="%.6f",
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-124.5,
            max_value=-114.0,
            value=-118.2437,
            format="%.6f",
        )

        county = st.text_input(
            "County",
            value="Los Angeles",
        )

        city = st.text_input(
            "City",
            value="Los Angeles",
        )

        postal_code = st.text_input(
            "ZIP code",
            value="90001",
        )

    st.subheader("Additional features")

    column4, column5, column6 = st.columns(3)

    with column4:
        flooring = st.text_input(
            "Flooring",
            value=str(defaults["categorical"]["Flooring"]),
        )

        levels = st.text_input(
            "Levels",
            value=str(defaults["categorical"]["Levels"]),
        )

        attached_garage = st.selectbox(
            "Attached garage",
            ["True", "False"],
        )

    with column5:
        private_pool = st.selectbox(
            "Private pool",
            ["False", "True"],
        )

        new_construction = st.selectbox(
            "New construction",
            ["False", "True"],
        )

        has_view = st.selectbox(
            "View",
            ["False", "True"],
        )

    with column6:
        has_fireplace = st.selectbox(
            "Fireplace",
            ["False", "True"],
        )

    with st.expander("Optional school-district information"):
        high_school_district = st.text_input(
            "Listing high-school district",
            value=str(
                defaults["categorical"]["HighSchoolDistrict"]
            ),
        )

        elementary_boundary = st.text_input(
            "Elementary school district boundary",
            value=str(
                defaults["categorical"][
                    "ElementarySchoolDistrict"
                ]
            ),
        )

        high_school_boundary = st.text_input(
            "High-school district boundary",
            value=str(
                defaults["categorical"][
                    "HighSchoolDistrictBoundary"
                ]
            ),
        )

        unified_district = st.text_input(
            "Unified school district",
            value=str(
                defaults["categorical"]["UnifiedSchoolDistrict"]
            ),
        )

    submitted = st.form_submit_button("Predict price")


if submitted:
    # Begin with training-data defaults for every required feature.
    row = {}

    for feature in numeric_features:
        row[feature] = defaults["numeric"].get(feature, 0.0)

    for feature in categorical_features:
        row[feature] = str(
            defaults["categorical"].get(feature, "Unknown")
        )

    # Replace defaults with values entered by the user.
    row.update(
        {
            "LivingArea": float(living_area),
            "BedroomsTotal": float(bedrooms),
            "BathroomsTotalInteger": float(bathrooms),
            "LotSizeSquareFeet": float(lot_square_feet),
            "LotSizeAcres": float(lot_square_feet) / 43560.0,
            "LotSizeArea": float(lot_square_feet),
            "YearBuilt": float(year_built),
            "GarageSpaces": float(garage_spaces),
            "ParkingTotal": float(parking_total),
            "Latitude": float(latitude),
            "Longitude": float(longitude),
            "Stories": float(stories),
            "AssociationFee": float(association_fee),
            "MainLevelBedrooms": float(main_level_bedrooms),
            "CountyOrParish": county.strip(),
            "City": city.strip(),
            "PostalCode": postal_code.strip(),
            "HighSchoolDistrict": high_school_district.strip(),
            "Flooring": flooring.strip(),
            "AttachedGarageYN": attached_garage,
            "Levels": levels.strip(),
            "PoolPrivateYN": private_pool,
            "NewConstructionYN": new_construction,
            "ViewYN": has_view,
            "FireplaceYN": has_fireplace,
            "ElementarySchoolDistrict": (
                elementary_boundary.strip()
            ),
            "HighSchoolDistrictBoundary": (
                high_school_boundary.strip()
            ),
            "UnifiedSchoolDistrict": unified_district.strip(),
        }
    )

    # Reproduce the feature engineering used during training.
    row["home_age"] = max(
        date.today().year - int(year_built),
        0,
    )

    row["log_LivingArea"] = np.log1p(living_area)
    row["log_LotSizeSquareFeet"] = np.log1p(
        lot_square_feet
    )

    row["bed_bath_ratio"] = bedrooms / bathrooms
    row["living_area_per_bedroom"] = (
        living_area / bedrooms
    )
    row["lot_to_living_area_ratio"] = (
        lot_square_feet / living_area
    )
    row["garage_per_bedroom"] = (
        garage_spaces / bedrooms
    )

    # All form inputs are present, so their missing flags are zero.
    for feature in numeric_features:
        if feature.endswith("_missing_flag"):
            row[feature] = 0

    input_data = pd.DataFrame(
        [row],
        columns=numeric_features + categorical_features,
    )

    try:
        transformed_data = preprocessor.transform(input_data)

        if transformed_data.shape[1] != model.n_features_in_:
            raise ValueError(
                "The preprocessor produced "
                f"{transformed_data.shape[1]} features, but "
                f"the model expects {model.n_features_in_}."
            )

        prediction = float(
            model.predict(transformed_data)[0]
        )

        st.success(
            f"Estimated closing price: ${prediction:,.0f}"
        )

        st.caption(
            "This estimate is based on historical CRMLS sales "
            "and is not a professional appraisal."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")