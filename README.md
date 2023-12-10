# GizaAutoML

GizaAutoML is an automated machine learning (AutoML) package designed for univariate time series forecasting. It simplifies the process of building machine learning pipelines by automating key steps, including data preprocessing, feature extraction, algorithm selection, and hyperparameter optimization.

## Table of Contents
1. [Installation](#installation)
2. [Package Overview](#package-overview)
3. [User Manual](#user-manual) 
4. [Example](#example)
5. [Web UI for Testing](#web-ui)
6. [Contributing](#contributing)
7. [License](#license)

## 1. Installation <a name="installation"></a>

To use GizaAutoML, install the package using the following command:

```bash
pip install git+https://github.com/giza-data-team/GizaAutoML.git@0.1
```
## 2. Package Overview <a name="package-overview"></a>
GizaAutoML is tailored for time series forecasting on univariate datasets. It executes the following steps:

- **Data Preprocessing:** Handles missing value imputation.
- **Feature Extraction:** Extracts features like time, trend, seasonality, and lags.
- **Algorithm Selection:** Recommends the best 3 algorithms based on the series meta features and the meta features saved in the engine knowledge base.
- **SMAC Optimization:** Applies SMAC optimization to the recommended algorithms, returning the best-performing model with minimal cost.

## 3. User Manual <a name="user-manual"></a>
Manual for the GizaAutoML package can be found [HERE](https://github.com/giza-data-team/GizaAutoML/blob/main/docs/user_maual.pdf).

## 4. Example <a name="example"></a>

``` bash
# Example usage
from GizaAutoML.auto_series_forecaster import AutoSeriesForecaster

# Load your time series data into a DataFrame called 'raw_data'
auto_ml = AutoSeriesForecaster(raw_dataframe=raw_data, optimization_metric="MAE", time_budget=10, save_results=True, random_seed=1, target_col="Target")

# Fit the model
pipeline = auto_ml.fit()

# Transform new data
new_data_transformed = auto_ml.transform(new_data)
```
### Timestamp Column Format

One crucial aspect for successful time series forecasting with GizaAutoML is the proper formatting of the timestamp column in your dataset. Follow these guidelines to ensure accurate results:

- **Format Requirement:** The timestamp column should be in the datetime format.
- **Example Conversion:** If your timestamp is not in datetime format, you can use the following code to convert it:

```bash
import pandas as pd

# Assuming 'Timestamp' is the name of your timestamp column
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
```
## 5. Web UI for Testing <a name="web-ui"></a>

GizaAutoML provides a web-based user interface for testing the engine. You can access the web UI by navigating to the following IP address: [http://172.178.116.34:8050/](http://172.178.116.34:8050/).

This web interface allows you to interactively test the capabilities of the GizaAutoML engine, providing a convenient way to explore and evaluate its functionality.
The dashboard includes informative visualization charts, offering insights into the performance of the forecasting models.

## 5. Contributing <a name="contributing"></a>
We welcome contributions! If you want to contribute to GizaAutoML, please check the **Contributing Guidelines**

## 6. License <a name="license"></a>
This project is licensed under the MIT License.

**Happy forecasting with GizaAutoML!**
