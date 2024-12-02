# Climate Projection Model

This project is a Streamlit application that predicts future climate temperatures based on historical data and Shared Socioeconomic Pathway (SSP) scenarios. The application fetches historical climate data and future climate projections, evaluates multiple regression models, and compares the model predictions with actual climate projections.

## Features

- Fetch historical climate data for a selected country.
- Fetch future climate projections based on SSP scenarios.
- Evaluate multiple regression models (Linear, Polynomial, Ridge, Lasso).
- Automatically select the best model based on multiple scoring metrics.
- Compare model predictions with actual climate projections.
- Visualize historical data, model predictions, and actual projections.

## Project Structure

```
.
├── data
├── models
├── notebooks
├── src
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/turqutefendi/cmip6-models.git
cd cmip6-models
```

Set up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

Open your web browser and navigate to [http://localhost:8501](http://localhost:8501).

1. Select a country and an SSP scenario from the sidebar.
2. View the model selection results, performance metrics, and visualizations.

## Configuration

The application can be configured using the `.streamlit/config.toml` file. You can customize the theme, toolbar mode, and other settings.

## Data Source

The climate data is sourced from the World Bank Climate API.

## License

This project is currently not licensed.

## Acknowledgements

- Python Documentation
- Streamlit, Pandas, NumPy, Scikit-learn, plotly, pycountry
- World Bank Climate API

## Contact

For any inquiries, please contact turqutefendi@gmail.com.
