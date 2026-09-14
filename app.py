from datetime import date
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from catboost import Pool
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="California Home Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background: var(--background-color);
            color: var(--text-color);
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .hero {
            padding: 2rem 2.2rem;
            border-radius: 22px;
            color: white;
            background: linear-gradient(120deg, #12355b 0%, #1976a3 100%);
            box-shadow: 0 14px 34px rgba(18, 53, 91, 0.18);
            margin-bottom: 1.6rem;
        }
        .hero h1 {
            color: white;
            font-size: clamp(2rem, 4vw, 3.1rem);
            line-height: 1.08;
            margin: 0 0 0.65rem 0;
        }
        .hero p {
            color: #e8f4fb;
            font-size: 1.05rem;
            margin: 0;
            max-width: 760px;
        }
        .company-line {
            color: #9fddff;
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            margin-bottom: 0.7rem;
        }
        .section-intro {
            color: color-mix(in srgb, var(--text-color) 76%, transparent);
            font-size: 1rem;
            line-height: 1.6;
            margin: -0.3rem 0 1.2rem;
        }
        .step-label {
            color: color-mix(in srgb, #21b99a 82%, var(--text-color));
            font-size: 0.79rem;
            font-weight: 800;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        div[data-testid="stMetric"] {
            padding: 1rem 1.1rem;
            border: 1px solid color-mix(in srgb, var(--text-color) 18%, transparent);
            border-radius: 14px;
            background: var(--secondary-background-color);
            box-shadow: 0 6px 18px rgba(32, 58, 86, 0.06);
        }
        div[data-testid="stForm"] {
            background: var(--secondary-background-color);
            border: 1px solid color-mix(in srgb, var(--text-color) 18%, transparent);
            border-radius: 20px;
            padding: 1.4rem 1.5rem 1.6rem;
            box-shadow: 0 10px 28px rgba(32, 58, 86, 0.08);
        }
        div[data-testid="stForm"] h3 {
            color: var(--text-color);
            margin-top: 0.25rem;
        }
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input {
            border-radius: 10px;
        }
        div[data-testid="stFormSubmitButton"] button {
            min-height: 3rem;
            border: 0;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 700;
            background: linear-gradient(90deg, #146b8c, #159b83);
            box-shadow: 0 7px 18px rgba(20, 107, 140, 0.24);
        }
        .prediction-card {
            padding: 1.5rem 1.7rem;
            border: 1px solid #ad79;
            border-left: 6px solid #138a72;
            border-radius: 16px;
            background: var(--secondary-background-color);
            box-shadow: 0 9px 24px rgba(19, 138, 114, 0.12);
            margin-top: 1.1rem;
        }
        .prediction-label {
            color: color-mix(in srgb, var(--text-color) 76%, transparent);
            font-size: 0.88rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .prediction-value {
            color: var(--text-color);
            font-size: clamp(2rem, 5vw, 3rem);
            font-weight: 800;
            line-height: 1.15;
            margin-top: 0.3rem;
        }
        .prediction-note {
            color: color-mix(in srgb, var(--text-color) 76%, transparent);
            font-size: 0.88rem;
            margin-top: 0.55rem;
        }
        .range-track {
            position: relative;
            height: 10px;
            border-radius: 999px;
            background: linear-gradient(90deg, #8fd5c4, #138a72, #8fd5c4);
            margin: 1rem 0 0.35rem;
        }
        .range-marker {
            position: absolute;
            left: 50%;
            top: 50%;
            width: 18px;
            height: 18px;
            border: 3px solid white;
            border-radius: 50%;
            background: #0f604f;
            box-shadow: 0 2px 7px rgba(15, 96, 79, 0.35);
            transform: translate(-50%, -50%);
        }
        .range-labels {
            display: flex;
            justify-content: space-between;
            color: color-mix(in srgb, var(--text-color) 76%, transparent);
            font-size: 0.78rem;
        }
        .developer-card {
            padding: 1.6rem;
            border: 1px solid color-mix(in srgb, var(--text-color) 18%, transparent);
            border-radius: 18px;
            background: var(--secondary-background-color);
            box-shadow: 0 8px 24px rgba(32, 58, 86, 0.07);
        }
        .developer-card h3 {
            color: var(--text-color);
            margin: 0 0 0.25rem;
        }
        .developer-card p {
            color: color-mix(in srgb, var(--text-color) 82%, transparent);
            line-height: 1.55;
            margin: 0.25rem 0;
        }
        .developer-links a {
            display: inline-block;
            color: color-mix(in srgb, #22c7a3 78%, var(--text-color));
            font-weight: 700;
            margin: 0.4rem 1rem 0 0;
            text-decoration: none;
        }
        .developer-links a:hover {
            text-decoration: underline;
        }
        [data-testid="stSidebar"] {
            background: var(--secondary-background-color);
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
        }
        div[data-testid="stImage"] {
            padding: 0.75rem 1rem;
            border-radius: 14px;
            background: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "catboost_model.pkl")
    preprocessor = joblib.load(BASE_DIR / "district_preprocessor.pkl")
    metadata = joblib.load(BASE_DIR / "model_metadata.pkl")
    defaults = joblib.load(BASE_DIR / "deployment_defaults.pkl")

    return model, preprocessor, metadata, defaults


model, preprocessor, metadata, defaults = load_artifacts()

numeric_features = metadata["numeric_features"]
categorical_features = metadata["categorical_features"]
transformed_feature_names = preprocessor.get_feature_names_out()

FRIENDLY_FEATURE_GROUPS = {
    "LivingArea": "Living area",
    "log_LivingArea": "Living area",
    "BedroomsTotal": "Bedrooms",
    "BathroomsTotalInteger": "Bathrooms",
    "LotSizeSquareFeet": "Lot size",
    "LotSizeAcres": "Lot size",
    "LotSizeArea": "Lot size",
    "log_LotSizeSquareFeet": "Lot size",
    "YearBuilt": "Property age and year built",
    "home_age": "Property age and year built",
    "GarageSpaces": "Garage and parking",
    "ParkingTotal": "Garage and parking",
    "Latitude": "Location",
    "Longitude": "Location",
    "CountyOrParish": "Location",
    "City": "Location",
    "PostalCode": "Location",
    "Stories": "Number of stories",
    "AssociationFee": "Association fee",
    "MainLevelBedrooms": "Main-level bedrooms",
    "bed_bath_ratio": "Bedroom and bathroom balance",
    "living_area_per_bedroom": "Space per bedroom",
    "lot_to_living_area_ratio": "Lot-to-home size",
    "garage_per_bedroom": "Garage space per bedroom",
    "HighSchoolDistrict": "School district",
    "ElementarySchoolDistrict": "School district",
    "HighSchoolDistrictBoundary": "School district",
    "UnifiedSchoolDistrict": "School district",
}


@st.cache_resource
def load_comparable_index():
    """Lazy-load historical training features and build a neighbor index."""
    feature_path = BASE_DIR / "week6_X_train_district_processed.npz"
    price_path = BASE_DIR / "week6_y_train.npy"
    if not feature_path.exists() or not price_path.exists():
        return None, None

    historical_features = sparse.load_npz(feature_path)
    historical_prices = np.load(price_path)
    index = NearestNeighbors(
        n_neighbors=25,
        metric="euclidean",
        algorithm="brute",
        n_jobs=-1,
    )
    index.fit(historical_features)
    return index, historical_prices


def source_feature_name(transformed_name):
    """Map a transformed column back to a user-facing source feature."""
    prefix, name = transformed_name.split("__", 1)
    if prefix == "num":
        return name

    for feature in sorted(categorical_features, key=len, reverse=True):
        if name == feature or name.startswith(f"{feature}_"):
            return feature
    return name


def friendly_feature_group(source_name):
    """Return a plain-language label and hide internal missing flags."""
    clean_name = source_name.removesuffix("_missing_flag")
    return FRIENDLY_FEATURE_GROUPS.get(
        clean_name,
        clean_name.replace("_", " ").title(),
    )


@st.cache_data(show_spinner=False, max_entries=200)
def predict_with_explanation(row_json):
    """Cache identical predictions and calculate local SHAP contributions."""
    row = json.loads(row_json)
    input_data = pd.DataFrame(
        [row],
        columns=numeric_features + categorical_features,
    )
    transformed_data = preprocessor.transform(input_data)
    expected_feature_count = len(model.feature_names_)

    if transformed_data.shape[1] != expected_feature_count:
        raise ValueError(
            f"The pipeline produced {transformed_data.shape[1]} features, "
            f"but the model expects {expected_feature_count}."
        )

    prediction = float(model.predict(transformed_data)[0])
    shap_values = model.get_feature_importance(
        Pool(transformed_data),
        type="ShapValues",
    )[0, :-1]

    contributions = {}
    for feature_name, contribution in zip(
        transformed_feature_names,
        shap_values,
    ):
        source_name = friendly_feature_group(source_feature_name(feature_name))
        contributions[source_name] = (
            contributions.get(source_name, 0.0) + float(contribution)
        )

    top_contributions = sorted(
        contributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:5]

    comparable_summary = None
    comparable_index, historical_prices = load_comparable_index()
    if comparable_index is not None:
        _, neighbor_indices = comparable_index.kneighbors(transformed_data)
        comparable_prices = historical_prices[neighbor_indices[0]]
        comparable_summary = {
            "count": int(len(comparable_prices)),
            "median": float(np.median(comparable_prices)),
            "lower_quartile": float(np.percentile(comparable_prices, 25)),
            "upper_quartile": float(np.percentile(comparable_prices, 75)),
            "nearest_prices": [
                float(value) for value in comparable_prices[:5]
            ],
        }

    return prediction, top_contributions, comparable_summary


st.markdown(
    """
    <section class="hero">
        <div class="company-line">IDX Exchange · Data Science Internship</div>
        <h1>California Home Price Predictor</h1>
        <p>A decision-support tool developed during the 12-week internship
        project to estimate the final sale price of a California
        single-family home.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.caption("IDX EXCHANGE INTERNSHIP PROJECT")
    st.header("About the project")
    st.write(
        "This application is the deployment deliverable for the California "
        "Property Close Price Prediction project. It uses historical CRMLS "
        "sales to estimate a property's final closing price."
    )
    st.divider()
    st.subheader("How it works")
    st.markdown(
        "1. Enter the property details.\n"
        "2. The app prepares those details for the price model.\n"
        "3. The model returns an estimated closing price."
    )
    st.divider()
    st.subheader("Input guidance")
    st.markdown(
        "- Use the property's current physical details.\n"
        "- Enter coordinates within California.\n"
        "- Optional district fields can improve location context."
    )
    st.info("For educational use only—not a professional appraisal.")

st.markdown("#### Estimate in three clear steps")
step1, step2, step3 = st.columns(3)
step1.info("**1 · Location**\n\nSet coordinates, county, city, and ZIP code.")
step2.info("**2 · Property details**\n\nAdd size, layout, age, and parking.")
step3.info("**3 · Features**\n\nDescribe amenities and optional districts.")

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

with st.form("prediction_form"):
    st.markdown(
        '<div class="step-label">Step 1 of 3</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Describe the property")
    st.caption("Enter the home's current physical and location details.")

    column1, column2, column3 = st.columns(3)

    with column1:
        st.markdown("**Size and layout**")
        st.selectbox(
            "Property type",
            ["Single-family residence"],
            disabled=True,
            help="The internship model was trained for this property type.",
        )
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
        st.markdown("**Parking and amenities**")
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
        st.markdown("**Location**")
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

    st.divider()
    st.markdown(
        '<div class="step-label">Step 2 of 3</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Add property characteristics")
    st.caption("Add characteristics that may influence market value.")

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

    st.markdown(
        '<div class="step-label">Step 3 of 3</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Add school-district context (optional)"):
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

    submitted = st.form_submit_button(
        "Estimate closing price",
        type="primary",
        width="stretch",
    )


if submitted:
    validation_errors = []
    if not county.strip():
        validation_errors.append("Enter a county.")
    if not city.strip():
        validation_errors.append("Enter a city.")
    if not postal_code.strip():
        validation_errors.append("Enter a ZIP code.")
    elif not postal_code.strip().isdigit():
        validation_errors.append("ZIP code must contain numbers only.")

    if validation_errors:
        st.error("Please correct the following before estimating:")
        for message in validation_errors:
            st.write(f"- {message}")
        st.stop()

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

    try:
        with st.spinner("Analyzing the property and local feature effects…"):
            (
                prediction,
                top_contributions,
                comparable_summary,
            ) = predict_with_explanation(json.dumps(row, sort_keys=True))

        lower_estimate = prediction * (1 - 0.086)
        upper_estimate = prediction * (1 + 0.086)

        st.markdown(
            f"""
            <section class="prediction-card" role="status">
                <div class="prediction-label">Estimated closing price</div>
                <div class="prediction-value">${prediction:,.0f}</div>
                <div class="prediction-note">
                    Typical prediction range based on past model results:
                    <strong>${lower_estimate:,.0f}–${upper_estimate:,.0f}</strong>
                </div>
                <div class="range-track">
                    <div class="range-marker"></div>
                </div>
                <div class="range-labels">
                    <span>${lower_estimate:,.0f}</span>
                    <span>Estimate</span>
                    <span>${upper_estimate:,.0f}</span>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

        result_left, result_right = st.columns([1.05, 0.95])
        with result_left:
            st.subheader("What pushed the estimate up or down")
            st.caption(
                "This chart explains the estimate for this home. Bars to the "
                "right raised the price; bars to the left lowered it."
            )
            influence_df = pd.DataFrame(
                top_contributions,
                columns=["Feature", "Price impact"],
            )
            influence_df["Feature"] = (
                influence_df["Feature"]
                .str.replace("_", " ", regex=False)
                .str.title()
            )
            st.bar_chart(
                influence_df.set_index("Feature"),
                horizontal=True,
                color="#146b8c",
            )

        with result_right:
            st.subheader("Estimate details")
            st.metric("Model estimate", f"${prediction:,.0f}")
            st.metric(
                "Typical prediction range",
                f"${lower_estimate:,.0f} – ${upper_estimate:,.0f}",
            )
            st.caption(
                "This practical range is based on past prediction errors. "
                "The actual sale price can still fall outside it."
            )
            st.code(
                f"IDX Exchange estimate: ${prediction:,.0f}\n"
                f"Typical range: ${lower_estimate:,.0f}–${upper_estimate:,.0f}",
                language=None,
            )
            st.download_button(
                "Download estimate",
                data=(
                    f"IDX Exchange property estimate\n"
                    f"Location: {city.strip()}, {county.strip()}\n"
                    f"Property: {int(bedrooms)} bedrooms, "
                    f"{int(bathrooms)} bathrooms, {living_area:,.0f} sq ft\n"
                    f"Estimated closing price: ${prediction:,.0f}\n"
                    f"Typical range: ${lower_estimate:,.0f}–${upper_estimate:,.0f}\n"
                ),
                file_name="idx_exchange_property_estimate.txt",
                mime="text/plain",
                width="stretch",
            )
            st.caption("To refine the result, adjust any input and estimate again.")

        if comparable_summary is not None:
            st.subheader("Compare with similar past home sales")
            st.caption(
                "The app found 25 past sales with property details most "
                "similar to the ones you entered."
            )
            comparable_median = comparable_summary["median"]
            difference_percentage = (
                (prediction - comparable_median) / comparable_median
            ) * 100
            comp1, comp2, comp3 = st.columns(3)
            comp1.metric(
                "Your estimated price",
                f"${prediction:,.0f}",
            )
            comp2.metric(
                "Typical price of similar homes",
                f"${comparable_median:,.0f}",
                delta=f"{difference_percentage:+.1f}% vs. similar homes",
                delta_color="off",
            )
            comp3.metric(
                "Common range for similar homes",
                (
                    f"${comparable_summary['lower_quartile']:,.0f} – "
                    f"${comparable_summary['upper_quartile']:,.0f}"
                ),
            )

            if (
                comparable_summary["lower_quartile"]
                <= prediction
                <= comparable_summary["upper_quartile"]
            ):
                st.success(
                    "This estimate is inside the common price range of similar "
                    "homes sold in the past."
                )
            else:
                st.warning(
                    "This estimate is outside the common price range of similar "
                    "past sales. Review any unusual property details before "
                    "using the result."
                )

            with st.expander("See five of the most similar past sale prices"):
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Similar home": [
                                f"Past sale {number}"
                                for number in range(1, 6)
                            ],
                            "Actual closing price": [
                                f"${price:,.0f}"
                                for price in comparable_summary["nearest_prices"]
                            ],
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "These homes were matched using the property details "
                    "entered above. They are references, not appraisals."
                )

        history_item = {
            "Location": f"{city.strip()}, {county.strip()}",
            "Property": f"{int(bedrooms)} bd · {int(bathrooms)} ba · {living_area:,.0f} sq ft",
            "Estimate": f"${prediction:,.0f}",
        }
        if (
            not st.session_state.prediction_history
            or st.session_state.prediction_history[0] != history_item
        ):
            st.session_state.prediction_history.insert(0, history_item)
            st.session_state.prediction_history = (
                st.session_state.prediction_history[:5]
            )

        with st.expander("How to interpret this estimate"):
            st.write(
                "The model's test MAE was approximately $152,842 and its "
                "median percentage error was 8.6%. Individual errors can be "
                "larger for unusual, low-priced, or luxury homes."
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

if st.session_state.prediction_history:
    st.subheader("Recent estimates")
    st.caption("Your five most recent estimates in this browser session.")
    st.dataframe(
        pd.DataFrame(st.session_state.prediction_history),
        hide_index=True,
        width="stretch",
    )

st.subheader("About the developer")
logo_left, logo_center, logo_right = st.columns([1, 3, 1])
with logo_center:
    st.image(
        BASE_DIR / "assets" / "idx-exchange-ucsd-logos.png",
        width="stretch",
    )
st.markdown(
    """
    <section class="developer-card">
        <div>
            <h3>Anh Tran</h3>
            <p><strong>Data Scientist Intern at IDX Exchange</strong> ·
            B.S. Data Science student with a minor in Business Analytics at
            the University of California, San Diego.</p>
            <p>Anh developed this California property-price workflow using
            Python, machine learning, feature engineering, geographic data,
            and Streamlit as part of the IDX Exchange internship.</p>
            <div class="developer-links">
                <a href="https://mail.google.com/mail/?view=cm&amp;fs=1&amp;to=nnt003%40ucsd.edu"
                   target="_blank" rel="noopener noreferrer">nnt003@ucsd.edu</a>
                <a href="https://www.linkedin.com/in/anh-tran-0a347018a/"
                   target="_blank" rel="noopener noreferrer">LinkedIn</a>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)
st.write("")

st.subheader("About the model")
about_left, about_right = st.columns(2)
with about_left:
    with st.expander("What data does the model use?"):
        st.write(
            "The project uses historical CRMLS sales for California "
            "single-family homes. It considers details such as location, "
            "living area, bedrooms, bathrooms, lot size, age, and amenities."
        )
    with st.expander("Why was this model selected?"):
        st.write(
            "CatBoost was the most accurate of the five methods tested. It "
            "explained 90.4% of price differences and produced smaller errors "
            "than the previous best model."
        )
with about_right:
    with st.expander("How accurate is the estimate?"):
        st.write(
            "Half of the test estimates were within 8.6% of the actual sale "
            "price. The average dollar difference was about $152,842, but "
            "accuracy varies by home and price range."
        )
    with st.expander("Can this replace an appraisal?"):
        st.write(
            "No. The app is an internship project and educational decision-"
            "support tool. It does not account for inspections, renovations, "
            "or every current market condition."
        )

st.divider()
st.caption(
    "Developed for the IDX Exchange Data Science Internship · "
    "California Property Close Price Prediction · Powered by CatBoost"
)
