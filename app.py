import streamlit as st
import os
import requests
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    explained_variance_score,
)
from sklearn.model_selection import train_test_split, cross_val_score
import pycountry


def get_country_code(country_name):
    try:
        country = pycountry.countries.get(name=country_name)
        return country.alpha_3
    except:
        return None


def fetch_historical_climate_data_for_country(country_code):
    if not os.path.exists(f"data/{country_code}_historical_climate_data.csv"):
        url = f"https://cckpapi.worldbank.org/cckp/v1/cmip6-x0.25_timeseries-smooth_tas_timeseries_annual_1950-2014_median_historical_ensemble_all_mean/{country_code}?_format=json"
        response = requests.get(url)
        data = response.json()
        country_data = data["data"][country_code]
        df = pd.DataFrame(list(country_data.items()), columns=["Year", "Value"])
        df["Year"] = df["Year"].apply(lambda x: int(x.split("-")[0]))
        df.to_csv(f"data/{country_code}_historical_climate_data.csv", index=False)
    else:
        df = pd.read_csv(f"data/{country_code}_historical_climate_data.csv")
    return df


def fetch_climate_data(scenario, country_code):
    if not os.path.exists(f"data/{country_code}_{scenario}_climate_data.csv"):
        url = f"https://cckpapi.worldbank.org/cckp/v1/cmip6-x0.25_timeseries-smooth_tas_timeseries_annual_2015-2100_median_{scenario}_ensemble_all_mean/{country_code}?_format=json"
        response = requests.get(url)
        data = response.json()
        country_data = data["data"][country_code]
        df = pd.DataFrame(list(country_data.items()), columns=["Year", "Value"])
        df["Year"] = df["Year"].apply(lambda x: int(x.split("-")[0]))
        df.to_csv(f"data/{country_code}_{scenario}_climate_data.csv", index=False)
    else:
        df = pd.read_csv(f"data/{country_code}_{scenario}_climate_data.csv")
    return df


def plot_comparison(historical_data, predictions, future_years, climate_data):
    fig = px.line()
    fig.add_scatter(
        x=historical_data["Year"],
        y=historical_data["Value"],
        mode="lines",
        name="Measured Data",
        line=dict(color="blue"),
        hovertemplate="Year: %{x}<br>Temperature: %{y}<br>Data: Measured Data",
    )
    prediction_df = future_years.copy()
    prediction_df["Predicted Value"] = predictions
    fig.add_scatter(
        x=prediction_df["Year"],
        y=prediction_df["Predicted Value"],
        mode="lines",
        name="Model Predictions",
        line=dict(color="green"),
        hovertemplate="Year: %{x}<br>Temperature: %{y}<br>Data: Model Prediction",
    )
    fig.add_scatter(
        x=climate_data["Year"],
        y=climate_data["Value"],
        mode="lines",
        name="Actual Projections",
        line=dict(color="red"),
        hovertemplate="Year: %{x}<br>Temperature: %{y}<br>Data: Actual Projection",
    )
    fig.update_layout(
        title="Model Predictions vs Actual Climate Projections",
        xaxis_title="Year",
        yaxis_title="Temperature (°C)",
    )
    st.plotly_chart(fig)


def evaluate_models(X, y):
    models = []
    models.append(("Linear Regression", LinearRegression(), {}))
    for degree in range(2, 6):
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)
        models.append(
            (
                f"Polynomial Regression (Degree {degree})",
                LinearRegression(),
                {"X": X_poly, "poly_features": poly_features},
            )
        )
    models.append(("Ridge Regression", Ridge(), {}))
    models.append(("Lasso Regression", Lasso(), {}))
    results = []
    for name, model, params in models:
        if "X" in params:
            X_model = params["X"]
        else:
            X_model = X
        cv_scores = cross_val_score(model, X_model, y, cv=5, scoring="r2")
        X_train, X_test, y_train, y_test = train_test_split(
            X_model, y, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        evs = explained_variance_score(y_test, predictions)
        results.append(
            {
                "Model": name,
                "R² Score": r2,
                "CV R² Score Mean": cv_scores.mean(),
                "CV R² Score Std": cv_scores.std(),
                "MSE": mse,
                "MAE": mae,
                "EVS": evs,
                "Model_Instance": model,
                "Params": params,
            }
        )
    return results


def evaluate_against_actual_projections(models, climate_data):
    results = []
    for model_info in models:
        model = model_info["Model_Instance"]
        if "poly_features" in model_info["Params"]:
            X_future = model_info["Params"]["poly_features"].fit_transform(
                climate_data["Year"].values.reshape(-1, 1)
            )
        else:
            X_future = climate_data["Year"].values.reshape(-1, 1)
        predictions = model.predict(X_future)
        r2 = r2_score(climate_data["Value"], predictions)
        mse = mean_squared_error(climate_data["Value"], predictions)
        mae = mean_absolute_error(climate_data["Value"], predictions)
        evs = explained_variance_score(climate_data["Value"], predictions)
        results.append(
            {
                "Model": model_info["Model"],
                "R² Score": r2,
                "MSE": mse,
                "MAE": mae,
                "EVS": evs,
            }
        )
    return results


def display_model_performance(results):
    results_df = pd.DataFrame(results)
    st.subheader("Model Performance")
    columns_to_display = ["Model", "R² Score", "MSE", "MAE", "EVS"]
    if (
        "CV R² Score Mean" in results_df.columns
        and "CV R² Score Std" in results_df.columns
    ):
        columns_to_display.extend(["CV R² Score Mean", "CV R² Score Std"])
    st.dataframe(results_df[columns_to_display].style.highlight_max(axis=0))
    if (
        "CV R² Score Mean" in results_df.columns
        and "CV R² Score Std" in results_df.columns
    ):
        sorted_results = sorted(
            results,
            key=lambda x: (
                -x["CV R² Score Mean"],
                -x["R² Score"],
                -x["EVS"],
                x["MSE"],
                x["MAE"],
            ),
        )
    else:
        sorted_results = sorted(
            results, key=lambda x: (-x["R² Score"], -x["EVS"], x["MSE"], x["MAE"])
        )
    best_model = sorted_results[0]
    st.success(f"Best Model Selected: {best_model['Model']}")
    return best_model


def plot_actual_comparison(historical_data, best_model_predictions, climate_data):
    fig = px.line()
    fig.add_scatter(
        x=historical_data["Year"],
        y=historical_data["Value"],
        mode="lines",
        name="Historical Data",
        line=dict(color="blue", dash="solid"),
        hovertemplate="Year: %{x}<br>Temperature: %{y}°C<br>Type: Historical",
    )
    fig.add_scatter(
        x=climate_data["Year"],
        y=best_model_predictions,
        mode="lines",
        name="Best Model Predictions",
        line=dict(color="green", dash="solid"),
        hovertemplate="Year: %{x}<br>Temperature: %{y}°C<br>Type: Prediction",
    )
    fig.add_scatter(
        x=climate_data["Year"],
        y=climate_data["Value"],
        mode="lines",
        name="Actual Projections",
        line=dict(color="red", dash="solid"),
        hovertemplate="Year: %{x}<br>Temperature: %{y}°C<br>Type: Actual",
    )
    fig.update_layout(
        title="Best Model Predictions vs Actual Climate Projections",
        xaxis_title="Year",
        yaxis_title="Temperature (°C)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    st.plotly_chart(fig)


def main():
    st.set_page_config(
        page_title="Climate Projection Model",
        page_icon="🌍🌡️",
        layout="centered",
        initial_sidebar_state="expanded",
        menu_items= None
    )
    results = []
    st.title("Climate Projection Model")
    st.sidebar.title("Model Selection")
    mode = "Auto Mode"
    country_name = st.sidebar.selectbox(
        "Select Country", [country.name for country in pycountry.countries], help= "Select the country to fetch historical climate data for."
    )
    country_code = get_country_code(country_name)
    scenario = st.sidebar.selectbox(
        "Select SSP Scenario", ("ssp126", "ssp245", "ssp370", "ssp585"), help= "Select the Shared Socioeconomic Pathway (SSP) scenario to use for climate projections."
    )
    with st.sidebar.expander("**What are SSP Scenarios?**"):
        st.markdown(
            """Shared Socioeconomic Pathways (SSPs) are a set of scenarios that explore alternative futures of societal development. Each SSP is based on a narrative storyline and describes a different future world with different levels of challenges and opportunities.
        """
        )
        st.info(
            """
            - **SSP126**: Sustainability - A world with low challenges to mitigation and adaptation.
            - **SSP245**: Middle of the Road - A world with medium challenges to mitigation and adaptation.
            - **SSP370**: Regional Rivalry - A world with high challenges to mitigation and adaptation.
            - **SSP585**: Fossil-fueled Development - A world with very high challenges to mitigation and adaptation.
            """
        )
    historical_data = fetch_historical_climate_data_for_country(country_code)
    climate_data = fetch_climate_data(scenario, country_code)
    st.sidebar.markdown("---")
    st.write("## Projections based on Historical Data")
    if mode == "Auto Mode":
        st.info(
            "Automatically selecting the best model and parameters based on multiple scoring metrics."
        )
        X = historical_data["Year"].values.reshape(-1, 1)
        y = historical_data["Value"].values
        results = evaluate_models(X, y)
        with st.expander("**View Model Selection Results**"):
            best_model = display_model_performance(results)
        if "poly_features" in best_model["Params"]:
            X_future = best_model["Params"]["poly_features"].fit_transform(
                climate_data["Year"].values.reshape(-1, 1)
            )
        else:
            X_future = climate_data["Year"].values.reshape(-1, 1)
        future_predictions = best_model["Model_Instance"].predict(X_future)
        with st.expander(f"**View Performance Metrics for {best_model['Model']}**"):
            st.subheader("Metrics for Best Model")
            st.metric(label="R² Score", value=f"{best_model['R² Score']:.4f}")
            st.metric(label="Mean Squared Error", value=f"{best_model['MSE']:.4f}")
            st.metric(label="Mean Absolute Error", value=f"{best_model['MAE']:.4f}")
            st.metric(
                label="Explained Variance Score", value=f"{best_model['EVS']:.4f}"
            )
        plot_comparison(
            historical_data, future_predictions, climate_data[["Year"]], climate_data
        )
        actual_results = evaluate_against_actual_projections(results, climate_data)
        st.divider()
        st.write(f"## Closest model to CMIP6 Projections ({scenario})")
        st.info(
            "Comparing the best performing model's predictions with actual climate projections"
        )
        with st.expander("**View Model Selection Results**"):
            display_model_performance(actual_results)
        best_actual_model = max(actual_results, key=lambda x: x["R² Score"])
        original_model_info = next(
            (
                result
                for result in results
                if result["Model"] == best_actual_model["Model"]
            ),
            None,
        )
        if original_model_info is None:
            st.error("No matching model found in the results.")
            return
        if "poly_features" in original_model_info["Params"]:
            X_future = original_model_info["Params"]["poly_features"].fit_transform(
                climate_data["Year"].values.reshape(-1, 1)
            )
        else:
            X_future = climate_data["Year"].values.reshape(-1, 1)
        best_predictions = original_model_info["Model_Instance"].predict(X_future)
        plot_actual_comparison(historical_data, best_predictions, climate_data)
        st.metric(
            label="Agreement with Projections (R² Score)",
            value=f"{best_actual_model['R² Score']:.4f}",
            help="How well the model predictions align with actual climate projections",
        )

    st.sidebar.subheader("Additional Information")
    st.sidebar.info(
        "This app allows you to predict future climate temperatures based on historical data and SSP scenarios."
    )
    st.markdown("---")
    st.caption("Data sourced from the World Bank Climate API.")


if __name__ == "__main__":
    main()
